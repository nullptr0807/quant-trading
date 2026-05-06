"""Virtual trading account with position tracking and cost basis."""

from __future__ import annotations
from dataclasses import dataclass, field
from .costs import MoomooAUCosts


@dataclass
class _Position:
    shares: float = 0.0
    avg_cost: float = 0.0  # per-share average cost basis (includes fees)
    total_cost: float = 0.0


class VirtualAccount:
    """Paper trading account starting with $1000."""

    def __init__(self, initial_cash: float = 1000.0, costs: MoomooAUCosts | None = None):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.costs = costs or MoomooAUCosts()
        self._positions: dict[str, _Position] = {}
        self.trade_log: list[dict] = []

    def buy(self, ticker: str, shares: float, price: float) -> dict:
        """Buy shares. Returns trade details or raises on insufficient funds."""
        fees = self.costs.calculate("buy", shares, price)
        total_outlay = fees["amount"] + fees["total_fees"]

        if total_outlay > self.cash:
            raise ValueError(f"Insufficient cash: need ${total_outlay:.2f}, have ${self.cash:.2f}")

        self.cash -= total_outlay

        pos = self._positions.setdefault(ticker, _Position())
        pos.total_cost += total_outlay
        pos.shares += shares
        pos.avg_cost = pos.total_cost / pos.shares

        trade = {"side": "buy", "ticker": ticker, "shares": shares,
                 "price": price, **fees}
        self.trade_log.append(trade)
        return trade

    def sell(self, ticker: str, shares: float, price: float, reason: str = "signal") -> dict:
        """Sell shares. Returns trade details (incl. realized PnL)."""
        pos = self._positions.get(ticker)
        if not pos or pos.shares < shares:
            avail = pos.shares if pos else 0
            raise ValueError(f"Insufficient shares: need {shares}, have {avail}")

        avg_cost = pos.avg_cost  # capture before position is reduced
        fees = self.costs.calculate("sell", shares, price)
        proceeds = fees["amount"] - fees["total_fees"]

        self.cash += proceeds

        # Realized PnL (proceeds vs cost basis for these shares, fees included on both sides)
        cost_basis = avg_cost * shares  # avg_cost already includes buy-side fees
        pnl_dollar = proceeds - cost_basis
        pnl_pct = (pnl_dollar / cost_basis) if cost_basis > 0 else 0.0

        # Reduce position, keep avg_cost the same
        pos.total_cost -= avg_cost * shares
        pos.shares -= shares
        if pos.shares < 1e-9:
            del self._positions[ticker]

        trade = {"side": "sell", "ticker": ticker, "shares": shares,
                 "price": price, "proceeds": proceeds,
                 "pnl_dollar": pnl_dollar, "pnl_pct": pnl_pct,
                 "reason": reason, **fees}
        self.trade_log.append(trade)
        return trade

    def get_positions(self) -> dict[str, float]:
        """Return {ticker: shares} for all open positions."""
        return {t: p.shares for t, p in self._positions.items()}

    def get_equity(self, prices: dict[str, float] | None = None,
                   strict: bool = False) -> float:
        """Total equity = cash + market value of positions.

        If `strict=True` (default): missing prices for held tickers raise
        ValueError — this prevents silently marking positions at avg_cost
        (which would make equity artificially snap to ~initial_cash and
        corrupt the equity curve).

        If `strict=False`: falls back to avg_cost (old behavior). Use only
        when you know avg_cost is an acceptable proxy (e.g. fresh restore
        before first price fetch).
        """
        mkt = 0.0
        for t, p in self._positions.items():
            if prices and t in prices:
                px = prices[t]
            elif not prices or strict:
                if strict:
                    raise ValueError(
                        f"get_equity: missing price for held ticker {t!r} "
                        f"(pass prices or strict=False)"
                    )
                px = p.avg_cost
            else:
                px = p.avg_cost
            mkt += p.shares * px
        return self.cash + mkt

    def get_pnl(self, prices: dict[str, float] | None = None) -> float:
        """Profit/loss vs initial cash."""
        return self.get_equity(prices) - self.initial_cash

    def get_holdings_detail(self, prices: dict[str, float] | None = None) -> list[dict]:
        """Detailed view of each position."""
        result = []
        for ticker, pos in self._positions.items():
            mkt_price = prices.get(ticker, pos.avg_cost) if prices else pos.avg_cost
            mkt_value = pos.shares * mkt_price
            unrealized = mkt_value - pos.total_cost
            result.append({
                "ticker": ticker,
                "shares": pos.shares,
                "avg_cost": round(pos.avg_cost, 4),
                "market_price": mkt_price,
                "market_value": round(mkt_value, 2),
                "unrealized_pnl": round(unrealized, 2),
            })
        return result
