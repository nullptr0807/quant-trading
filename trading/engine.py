"""Trading engine: manages accounts, executes signals, enforces risk limits."""

from __future__ import annotations
from .account import VirtualAccount
from .costs import MoomooAUCosts


class TradingEngine:
    """Manages multiple virtual accounts with risk controls."""

    def __init__(
        self,
        max_position_pct: float = 0.30,   # max 30% equity in one position
        stop_loss_pct: float = -0.03,      # -3% stop loss trigger
        costs: MoomooAUCosts | None = None,
        trade_callback=None,              # called with (account, trade_dict) after each trade
    ):
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct
        self.costs = costs or MoomooAUCosts()
        self.accounts: dict[str, VirtualAccount] = {}
        self.trade_callback = trade_callback

    def create_account(self, name: str, initial_cash: float = 1000.0) -> VirtualAccount:
        acct = VirtualAccount(initial_cash=initial_cash, costs=self.costs)
        self.accounts[name] = acct
        return acct

    def get_account(self, name: str) -> VirtualAccount:
        return self.accounts[name]

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
            # Check max position concentration
            equity = acct.get_equity(current_prices)
            if equity <= 0:
                return None
            exec_price = self.costs.slippage(price, "buy")
            proposed_value = shares * exec_price
            # Include existing position value
            pos = acct.get_positions().get(ticker, 0)
            existing_value = pos * current_prices.get(ticker, exec_price)
            total_position = existing_value + proposed_value
            if total_position / equity > self.max_position_pct:
                # Reduce shares to fit limit
                max_value = self.max_position_pct * equity - existing_value
                if max_value <= 0:
                    return None
                shares = int(max_value / exec_price)
                if shares <= 0:
                    return None
            result = acct.buy(ticker, shares, price)

        elif side == "sell":
            result = acct.sell(ticker, shares, price, reason=reason)
        else:
            return None

        if result and self.trade_callback:
            try:
                self.trade_callback(account_name, result)
            except Exception as e:
                import logging
                logging.getLogger("trading").warning("trade_callback failed: %s", e)
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
                trade = acct.sell(
                    detail["ticker"], detail["shares"], prices[detail["ticker"]],
                    reason="stop_loss",
                )
                if trade and self.trade_callback:
                    try:
                        self.trade_callback(account_name, trade)
                    except Exception:
                        pass
                trades.append(trade)
        return trades
