"""Alpha158-style factor computation using pure pandas/numpy."""

import numpy as np
import pandas as pd


class FactorEngine:
    """Computes Alpha158-style factors from OHLCV DataFrames."""

    def __init__(self):
        self.windows = [5, 10, 20]

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all factors from an OHLCV DataFrame.

        Args:
            df: DataFrame with columns [Open, High, Low, Close, Volume] indexed by date.

        Returns:
            DataFrame of all computed factors indexed by date.
        """
        df = df.copy()
        # Normalize column names to lowercase
        df.columns = [c.lower() for c in df.columns]

        factors = pd.DataFrame(index=df.index)

        # 1) KBAR factors (9)
        factors["KMID"] = (df["close"] - df["open"]) / df["open"]
        factors["KLEN"] = (df["high"] - df["low"]) / df["open"]
        factors["KMID2"] = (df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-12)
        factors["KUP"] = (df["high"] - max_oc(df)) / df["open"]
        factors["KUP2"] = (df["high"] - max_oc(df)) / (df["high"] - df["low"] + 1e-12)
        factors["KLOW"] = (min_oc(df) - df["low"]) / df["open"]
        factors["KLOW2"] = (min_oc(df) - df["low"]) / (df["high"] - df["low"] + 1e-12)
        factors["KSFT"] = (2 * df["close"] - df["high"] - df["low"]) / df["open"]
        factors["KSFT2"] = (2 * df["close"] - df["high"] - df["low"]) / (df["high"] - df["low"] + 1e-12)

        # 2) Price momentum
        for w in self.windows:
            factors[f"ROC_{w}"] = df["close"].pct_change(w)
            factors[f"MA_RATIO_{w}"] = df["close"] / df["close"].rolling(w).mean()

        # 3) Volume factors
        for w in self.windows:
            factors[f"VMOM_{w}"] = df["volume"] / df["volume"].rolling(w).mean()
        for w in self.windows:
            factors[f"VSTD_{w}"] = df["volume"].rolling(w).std() / (df["volume"].rolling(w).mean() + 1e-12)

        # 4) Volatility
        for w in self.windows:
            factors[f"STD_{w}"] = df["close"].rolling(w).std() / df["close"]
            ma = df["close"].rolling(w).mean()
            std = df["close"].rolling(w).std()
            factors[f"BBPOS_{w}"] = (df["close"] - ma) / (2 * std + 1e-12)

        # 5) Mean reversion
        low_min = df["low"].rolling(9).min()
        high_max = df["high"].rolling(9).max()
        factors["RSV"] = (df["close"] - low_min) / (high_max - low_min + 1e-12)
        factors["RSI_14"] = _rsi(df["close"], 14)

        # 6) Trend: BETA/slope via rolling regression
        for w in self.windows:
            factors[f"BETA_{w}"] = _rolling_slope(df["close"], w)

        return factors

    def compute_multi(self, data_dict: dict) -> dict:
        """Compute factors for multiple tickers.

        Args:
            data_dict: {ticker: DataFrame} from DataFetcher.

        Returns:
            {ticker: factors_DataFrame}
        """
        return {ticker: self.compute_all(df) for ticker, df in data_dict.items()}


def max_oc(df):
    return df[["open", "close"]].max(axis=1)


def min_oc(df):
    return df[["open", "close"]].min(axis=1)


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-12)
    return 100 - 100 / (1 + rs)


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Rolling OLS slope (normalized by mean price in window)."""
    def slope(y):
        if len(y) < window or np.isnan(y).any():
            return np.nan
        x = np.arange(len(y), dtype=float)
        x -= x.mean()
        y_arr = y - y.mean()
        denom = (x * x).sum()
        if denom == 0:
            return 0.0
        return (x * y_arr).sum() / denom / (y.mean() + 1e-12)

    return series.rolling(window).apply(slope, raw=True)
