"""
GP Signal Generator — produces buy/sell signals from GP-mined alpha factors
using composite rank-based scoring.
"""

import numpy as np
import pandas as pd


class GPSignalGenerator:
    """Generate trading signals from GP factor data using cross-sectional ranking."""

    def generate_signals(
        self,
        gp_factors_dict: dict[str, pd.DataFrame],
        top_n: int = 5,
        buy_candidates: int | None = None,
    ) -> dict:
        """
        Generate buy/sell signals from GP factors.

        Args:
            gp_factors_dict: {ticker: DataFrame of GP factor values}
            top_n: number of tickers for buy and sell lists
            buy_candidates: optional larger buy-list length. Use this when
                execution constraints (e.g. CN 100-share board lots) may make a
                top-ranked ticker unbuyable and the caller wants fallback names.

        Returns:
            dict with 'buy' and 'sell' lists of ticker strings
        """
        # Get the latest factor values per ticker.
        # Walk backwards from the last row to find the most recent fully-or-partially
        # populated row — guards against intraday cycles where the freshest bar
        # has NaN factors (e.g. weekend run, pre-market, missing data).
        ticker_scores = {}
        for ticker, fdf in gp_factors_dict.items():
            if fdf is None or fdf.empty:
                continue
            last = None
            for i in range(len(fdf) - 1, max(-1, len(fdf) - 6), -1):
                row = fdf.iloc[i].dropna()
                if len(row) > 0:
                    last = row
                    break
            if last is None:
                continue
            ticker_scores[ticker] = last.to_dict()

        if not ticker_scores:
            return {"buy": [], "sell": []}

        # Build cross-sectional matrix: tickers x factors
        scores_df = pd.DataFrame(ticker_scores).T  # rows=tickers, cols=factors

        # Rank each factor cross-sectionally (higher value = higher rank)
        ranked = scores_df.rank(axis=0, pct=True, na_option="keep")

        # Composite score = equal-weight mean of percentile ranks
        composite = ranked.mean(axis=1).dropna().sort_values(ascending=False)

        if len(composite) == 0:
            return {"buy": [], "sell": []}

        n = min(top_n, max(1, len(composite) // 2))
        # Candidate-fallback list for lot/affordability constraints, but do not
        # let buy candidates overlap the sell tail on small universes.
        max_non_sell = max(n, len(composite) - n)
        buy_n = min(buy_candidates or n, max_non_sell)
        buy_list = composite.head(buy_n).index.tolist()
        sell_list = composite.tail(n).index.tolist()

        return {"buy": buy_list, "sell": sell_list}
