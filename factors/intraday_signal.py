"""Intraday signal generation — cross-sectional ranking of 5-min factors.

Same philosophy as signal.py (daily layer): rank tickers by factor composite
score, return buy/sell lists. No hard rules — pure quantitative ranking.

Each account can use different intraday factor subsets via strategy config.
"""

import numpy as np
import pandas as pd


# Intraday factor subsets per strategy flavor
INTRADAY_STRATEGY_FACTORS = {
    "momentum": [
        "ROC_6", "ROC_12", "ROC_24", "MA_RATIO_6", "MA_RATIO_12",
        "SLOPE_6", "SLOPE_12", "SLOPE_24",
    ],
    "mean_reversion": [
        "RSI_6", "RSI_12", "RSI_24", "BBPOS_6", "BBPOS_12", "BBPOS_24",
        "RANGE_POS_6", "RANGE_POS_12", "DAY_RANGE_POS",
    ],
    "volatility": [
        "STD_6", "STD_12", "STD_24", "ATR_RATIO_6", "ATR_RATIO_12",
        "BBPOS_6", "BBPOS_12", "VSTD_6", "VSTD_12",
    ],
    "volume": [
        "VRATIO_6", "VRATIO_12", "VRATIO_24", "VSTD_6", "VSTD_12",
        "VOL_PACE", "VWAP_DIST", "VWAP_DIST_6", "VWAP_DIST_12",
    ],
    "composite": None,  # use all factors
}


class IntradaySignalGenerator:
    """Rank-based intraday signal generator for timing layer."""

    def __init__(self, buy_top: int = 5, sell_top: int = 5):
        self.buy_top = buy_top
        self.sell_top = sell_top

    def generate_signals(
        self,
        intraday_factors_dict: dict,
        strategy_type: str = "composite",
    ) -> dict:
        """Generate intraday buy/sell signals from cross-sectional ranking.

        Args:
            intraday_factors_dict: {ticker: factors_DataFrame} from IntradayFactorEngine
            strategy_type: one of 'momentum', 'mean_reversion', 'volatility', 'volume', 'composite'

        Returns:
            {'buy': [(ticker, score), ...], 'sell': [(ticker, score), ...]}
            Sorted by score descending for buy, ascending for sell.
        """
        subset = INTRADAY_STRATEGY_FACTORS.get(strategy_type)
        scores = {}

        for ticker, fdf in intraday_factors_dict.items():
            if fdf is None or fdf.empty:
                continue
            last = fdf.iloc[-1]
            if subset is not None:
                cols = [c for c in subset if c in fdf.columns]
            else:
                cols = [c for c in fdf.columns if fdf[c].dtype in (np.float64, np.float32, float)]
            vals = last[cols].dropna()
            if len(vals) == 0:
                continue
            scores[ticker] = float(vals.mean())

        if not scores:
            return {"buy": [], "sell": []}

        s = pd.Series(scores)
        ranks = s.rank(pct=True)

        # For mean_reversion, invert: low RSI = buy signal
        if strategy_type == "mean_reversion":
            ranks = 1 - ranks

        ranked = ranks.sort_values(ascending=False)
        buy = [(t, round(float(ranked[t]), 4)) for t in ranked.index[:self.buy_top]]
        sell = [(t, round(float(ranked[t]), 4)) for t in ranked.index[-self.sell_top:]]

        return {"buy": buy, "sell": sell}

    def generate_gp_signals(
        self,
        intraday_factors_dict: dict,
        top_n: int = 5,
    ) -> dict:
        """Generate intraday signals for GP accounts using all available factors.

        Same as GPSignalGenerator but applied to intraday factors.
        Returns {'buy': [ticker, ...], 'sell': [ticker, ...]}
        """
        ticker_scores = {}
        for ticker, fdf in intraday_factors_dict.items():
            if fdf is None or fdf.empty:
                continue
            last = fdf.iloc[-1].dropna()
            if len(last) == 0:
                continue
            ticker_scores[ticker] = last.to_dict()

        if not ticker_scores:
            return {"buy": [], "sell": []}

        scores_df = pd.DataFrame(ticker_scores).T
        ranked = scores_df.rank(axis=0, pct=True, na_option="keep")
        composite = ranked.mean(axis=1).dropna().sort_values(ascending=False)

        if len(composite) == 0:
            return {"buy": [], "sell": []}

        n = min(top_n, max(1, len(composite) // 2))
        buy_list = composite.head(n).index.tolist()
        sell_list = composite.tail(n).index.tolist()

        return {"buy": buy_list, "sell": sell_list}
