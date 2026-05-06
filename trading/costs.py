"""Moomoo Australia US stock fee schedule and slippage model."""

from dataclasses import dataclass, field


class CNCosts:
    """Adapter exposing the (slippage, calculate) interface used by
    VirtualAccount/TradingEngine, backed by config.settings.CN_FEES.

    CN_FEES.calc_fees returns {commission, transfer, stamp, slippage, total}.
    The engine only consumes 'total_fees', 'amount', 'exec_price' — we map
    accordingly. Slippage is applied on top of the price like MoomooAUCosts
    so equity impact mirrors the US path.
    """

    def __init__(self, fees=None):
        if fees is None:
            from config.settings import CN_FEES
            fees = CN_FEES
        self.fees = fees
        self.slippage_pct = fees.slippage_pct

    def slippage(self, price: float, side: str) -> float:
        if side == "buy":
            return price * (1 + self.slippage_pct)
        return price * (1 - self.slippage_pct)

    def calculate(self, side: str, shares: float, price: float) -> dict:
        exec_price = self.slippage(price, side)
        amount = shares * exec_price
        is_sell = (side == "sell")
        # CN fee schedule operates on notional; pass exec_price to match
        # what the cash side actually settles on.
        breakdown = self.fees.calc_fees(shares, exec_price, is_sell=is_sell)
        # Engine cares about exec_price/amount/total_fees; keep CN-specific
        # keys too for completeness.
        return {
            "exec_price": exec_price,
            "amount": amount,
            "commission": breakdown["commission"],
            "transfer_fee": breakdown["transfer"],
            "stamp_tax": breakdown["stamp"],
            # Note: CN fee struct already counts slippage in 'total'. We
            # subtract it so 'total_fees' here is broker fees only — the
            # actual slippage cost is already baked into exec_price.
            "total_fees": round(breakdown["total"] - breakdown["slippage"], 4),
        }


@dataclass
class MoomooAUCosts:
    """Calculate exact trading fees per moomoo Australia US stock schedule."""

    platform_fee: float = 0.99          # per order
    settlement_per_share: float = 0.003  # max 1% of trade amount
    sec_fee_rate: float = 0.0000206      # sells only, on dollar amount
    sec_fee_min: float = 0.01
    finra_taf_per_share: float = 0.000195  # sells only
    finra_taf_min: float = 0.01
    finra_taf_max: float = 9.79
    slippage_pct: float = 0.0005         # 0.05% default

    def slippage(self, price: float, side: str) -> float:
        """Return execution price after slippage."""
        if side == "buy":
            return price * (1 + self.slippage_pct)
        else:
            return price * (1 - self.slippage_pct)

    def calculate(self, side: str, shares: float, price: float) -> dict:
        """Return breakdown of all fees for an order.

        Args:
            side: 'buy' or 'sell'
            shares: number of shares
            price: price per share (before slippage)

        Returns:
            dict with fee breakdown and total_cost
        """
        exec_price = self.slippage(price, side)
        amount = shares * exec_price

        # Platform fee
        platform = self.platform_fee

        # Settlement fee: $0.003/share, capped at 1% of amount
        settlement = min(self.settlement_per_share * shares, 0.01 * amount)

        # SEC fee (sells only)
        sec = 0.0
        if side == "sell":
            sec = max(self.sec_fee_rate * amount, self.sec_fee_min)

        # FINRA TAF (sells only)
        finra = 0.0
        if side == "sell":
            finra = min(max(self.finra_taf_per_share * shares, self.finra_taf_min),
                        self.finra_taf_max)

        total = platform + settlement + sec + finra

        return {
            "exec_price": exec_price,
            "amount": amount,
            "platform_fee": platform,
            "settlement_fee": round(settlement, 4),
            "sec_fee": round(sec, 4),
            "finra_taf": round(finra, 4),
            "total_fees": round(total, 4),
        }
