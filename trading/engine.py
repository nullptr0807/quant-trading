"""Trading engine: manages accounts, executes signals, enforces risk limits."""

from __future__ import annotations
import math
from .account import VirtualAccount
from .costs import MoomooAUCosts


class TradePersistenceError(RuntimeError):
    """The account mutation was rolled back because durable publication failed."""


class TradingEngine:
    """Manages multiple virtual accounts with risk controls."""

    def __init__(
        self,
        max_position_pct: float = 0.30,   # max 30% equity in one position
        stop_loss_pct: float = -0.03,      # -3% stop loss trigger
        max_gross_exposure_pct: float = 0.98,
        min_cash_reserve_pct: float = 0.02,
        costs: MoomooAUCosts | None = None,
        trade_callback=None,              # called with (account, trade_dict) after each trade
        clock=None,
    ):
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_gross_exposure_pct = max(0.0, min(1.0, max_gross_exposure_pct))
        self.min_cash_reserve_pct = max(0.0, min(1.0, min_cash_reserve_pct))
        if self.max_gross_exposure_pct + self.min_cash_reserve_pct > 1.000001:
            raise ValueError("gross exposure and cash reserve limits are inconsistent")
        self.costs = costs or MoomooAUCosts()
        self.accounts: dict[str, VirtualAccount] = {}
        self.trade_callback = trade_callback
        self.clock = clock

    def _lot_size(self) -> int:
        """Minimum board lot for new buy orders.

        US paper trades remain whole-share. CN uses CNCosts.lot_size=100 so new
        buys cannot create impossible A-share odd lots. Sells are not rounded
        here: full exits must be able to clear legacy odd-lot positions already
        present in the live DB.
        """
        return int(getattr(self.costs, "lot_size", 1) or 1)

    def _round_buy_shares(self, shares: float) -> int:
        lot = self._lot_size()
        whole = math.floor(float(shares))
        if lot <= 1:
            return whole
        return (whole // lot) * lot

    def buy_shares_for_budget(self, budget: float, price: float) -> int:
        """Return executable buy quantity for a cash budget and pre-slip price.

        For CN this returns 100-share board lots. If the budget cannot buy one
        lot, return 0 so callers can skip this signal and try the next ranked
        candidate instead of creating an impossible odd-lot or oversizing.
        """
        if budget <= 0 or price <= 0:
            return 0
        exec_price = self.costs.slippage(price, "buy")
        return self._round_buy_shares(budget / exec_price)

    def buy_skip_detail_for_budget(self, budget: float, price: float) -> dict | None:
        """Explain why a buy budget cannot produce an executable order.

        Returns None when at least one executable share/lot can be bought. Used
        by trading loops to persist account-level skip events for CN board-lot
        constraints without guessing from log lines.
        """
        if budget <= 0 or price <= 0:
            return {"reason": "invalid_budget_or_price", "budget": budget, "price": price}
        if self.buy_shares_for_budget(budget, price) > 0:
            return None
        lot = self._lot_size()
        exec_price = self.costs.slippage(price, "buy")
        return {
            "reason": "board_lot_unaffordable" if lot > 1 else "insufficient_budget",
            "budget": budget,
            "price": price,
            "exec_price": exec_price,
            "lot_size": lot,
            "min_lot_notional": exec_price * lot,
        }

    def create_account(self, name: str, initial_cash: float = 1000.0) -> VirtualAccount:
        acct = VirtualAccount(
            initial_cash=initial_cash, costs=self.costs, clock=self.clock
        )
        self.accounts[name] = acct
        return acct

    def get_account(self, name: str) -> VirtualAccount:
        return self.accounts[name]

    def execute_trade(
        self, account_name: str, ticker: str, side: str,
        shares: float, price: float, reason: str = "signal",
    ) -> dict | None:
        """Mutate an account and durably publish the trade, or roll it back."""
        acct = self.accounts[account_name]
        snapshot = acct.snapshot()
        try:
            if side == "buy":
                result = acct.buy(ticker, shares, price)
            elif side == "sell":
                result = acct.sell(ticker, shares, price, reason=reason)
            else:
                return None

            if result and self.trade_callback:
                try:
                    self.trade_callback(account_name, result)
                except Exception as exc:
                    raise TradePersistenceError(
                        f"failed to persist {side} {account_name}/{ticker}: {exc}"
                    ) from exc
            return result
        except Exception:
            acct.restore(snapshot)
            raise

    def execute_signal(
        self, account_name: str, ticker: str, side: str,
        shares: float, price: float, prices: dict[str, float] | None = None,
        reason: str = "signal",
    ) -> dict | None:
        """Execute a trade signal with risk checks.

        Returns trade dict on success, None if blocked by risk limits.
        """
        acct = self.accounts[account_name]
        current_prices = prices or {}

        if side == "buy":
            shares = self._round_buy_shares(shares)
            if shares <= 0:
                return None
            # Check max position concentration
            equity = acct.get_equity(current_prices)
            if equity <= 0:
                return None
            exec_price = self.costs.slippage(price, "buy")
            pos = acct.get_positions().get(ticker, 0)
            existing_value = pos * current_prices.get(ticker, exec_price)
            gross_existing = sum(
                p.shares * current_prices.get(symbol, p.avg_cost)
                for symbol, p in acct._positions.items() if p.shares > 0
            )
            position_budget = self.max_position_pct * equity - existing_value
            gross_budget = self.max_gross_exposure_pct * equity - gross_existing
            cash_budget = acct.cash - self.min_cash_reserve_pct * equity
            max_new_value = min(position_budget, gross_budget, cash_budget)
            if max_new_value <= 0:
                return None
            shares = min(shares, self._round_buy_shares(max_new_value / exec_price))
            lot = self._lot_size()
            while shares > 0:
                fees = self.costs.calculate("buy", shares, price)
                total_cost = shares * fees.get("exec_price", exec_price) + fees.get(
                    "total_fees", fees.get("total", 0.0)
                )
                if total_cost <= cash_budget + 1e-9:
                    break
                shares = self._round_buy_shares(shares - lot)
            if shares <= 0:
                return None
            result = self.execute_trade(account_name, ticker, "buy", shares, price)

        elif side == "sell":
            sellable = acct.get_sellable_shares(
                ticker, as_of=self.clock() if self.clock else None
            )
            shares = min(float(shares), sellable)
            if shares <= 0:
                return None
            result = self.execute_trade(
                account_name, ticker, "sell", shares, price, reason=reason
            )
        else:
            return None

        return result

    def check_stop_losses(
        self, account_name: str, prices: dict[str, float]
    ) -> list[dict]:
        """Trigger stop-loss sells for positions down more than threshold.

        Returns list of executed sell trades.
        """
        acct = self.accounts[account_name]
        trades = []
        for detail in acct.get_holdings_detail(prices):
            pnl_pct = detail["unrealized_pnl"] / (detail["shares"] * detail["avg_cost"])
            if pnl_pct <= self.stop_loss_pct:
                trade = self.execute_trade(
                    account_name,
                    detail["ticker"],
                    "sell",
                    detail["shares"],
                    prices[detail["ticker"]],
                    reason="stop_loss",
                )
                trades.append(trade)
        return trades
