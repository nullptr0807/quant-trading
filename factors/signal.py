"""Signal generation from factor scores."""

import pandas as pd
import numpy as np


# Factor subsets per strategy type
STRATEGY_FACTORS = {
    "momentum": ["ROC_5", "ROC_10", "ROC_20", "MA_RATIO_5", "MA_RATIO_10", "MA_RATIO_20",
                  "BETA_5", "BETA_10", "BETA_20"],
    "mean_reversion": ["RSV", "RSI_14", "BBPOS_5", "BBPOS_10", "BBPOS_20",
                       "KSFT", "KSFT2"],
    "volatility": ["STD_5", "STD_10", "STD_20", "BBPOS_5", "BBPOS_10", "BBPOS_20",
                    "VSTD_5", "VSTD_10", "VSTD_20"],
    "composite": None,  # use all factors
}


class SignalGenerator:
    """Rank-based composite scoring to produce buy/sell signals."""

    def __init__(self, buy_top: int = 10, sell_top: int = 10):
        """
        Args:
            buy_top: number of top-ranked tickers to flag as buy.
            sell_top: number of bottom-ranked tickers to flag as sell.
        """
        self.buy_top = buy_top
        self.sell_top = sell_top

    def generate_signals(
        self, factors_dict: dict, strategy_type: str = "momentum"
    ) -> dict:
        """Generate buy/sell signals from multi-ticker factor data.

        Args:
            factors_dict: {ticker: factors_DataFrame} — output of FactorEngine.compute_multi().
            strategy_type: one of 'momentum', 'mean_reversion', 'volatility', 'composite'.

        Returns:
            {'buy': [(ticker, score), ...], 'sell': [(ticker, score), ...]}
        """
        subset = STRATEGY_FACTORS.get(strategy_type)

        # Build cross-section: one row per ticker, one column per factor (latest bar)
        rows = {}
        for ticker, fdf in factors_dict.items():
            if fdf.empty:
                continue
            last = fdf.iloc[-1]
            if subset is not None:
                cols = [c for c in subset if c in fdf.columns]
            else:
                cols = [c for c in fdf.columns if fdf[c].dtype in (np.float64, np.float32, float)]
            if not cols:
                continue
            rows[ticker] = last[cols]

        if not rows:
            return {"buy": [], "sell": []}

        cs = pd.DataFrame(rows).T  # tickers × factors

        # V1 composite: rank each factor cross-sectionally to neutralize量纲差异 (RSI_14 ∈ [0,100]
        # vs BETA_5 ∈ [±1e-5] would otherwise let one factor dominate the mean), then equal-weight
        # sum across factors. See research/factor_composite_compare.py for the V0 vs V1 backtest.
        factor_ranks = cs.rank(axis=0, pct=True)         # per-factor cross-sectional rank, 0..1
        composite = factor_ranks.mean(axis=1, skipna=True)  # equal-weight average rank across factors
        composite = composite.dropna()
        if composite.empty:
            return {"buy": [], "sell": []}

        # Final cross-section rank for sorting (already monotonic, kept for output stability)
        ranks = composite.rank(pct=True)

        # For mean_reversion, invert: low RSI/RSV = buy signal
        if strategy_type == "mean_reversion":
            ranks = 1 - ranks

        ranked = ranks.sort_values(ascending=False)
        buy = [(t, round(float(ranked[t]), 4)) for t in ranked.index[: self.buy_top]]
        sell = [(t, round(float(ranked[t]), 4)) for t in ranked.index[-self.sell_top :]]

        return {"buy": buy, "sell": sell}
