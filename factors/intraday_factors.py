"""Intraday factor engine — computes 5-minute bar factors for timing layer.

Same design as alpha_factors.py: returns a DataFrame of factor values per ticker,
consumed by the signal layer for cross-sectional ranking.  No hard-coded buy/sell
rules here — just pure factor computation.

Factors are designed for 5-min bars. A "1d" period gives ~78 bars (6.5h regular session).
We use rolling windows sized for intraday: 6, 12, 24, 48 bars
(= 30min, 1h, 2h, 4h respectively).
"""

import numpy as np
import pandas as pd


# Rolling windows in number of 5-min bars
WINDOWS = [6, 12, 24, 48]  # 30m, 1h, 2h, 4h


class IntradayFactorEngine:
    """Compute intraday factors from 5-minute OHLCV bars."""

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all intraday factors.

        Args:
            df: DataFrame with columns [open, high, low, close, volume]
                indexed by datetime (5-min bars).

        Returns:
            DataFrame of factor values, same index as input.
        """
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        f = pd.DataFrame(index=df.index)

        # ── 1. VWAP factors ──────────────────────────────────────────────
        # Cumulative VWAP from session start
        typical = (df["high"] + df["low"] + df["close"]) / 3
        cum_tp_vol = (typical * df["volume"]).cumsum()
        cum_vol = df["volume"].cumsum()
        vwap = cum_tp_vol / (cum_vol + 1e-12)

        f["VWAP_DIST"] = (df["close"] - vwap) / (vwap + 1e-12)  # price vs VWAP
        for w in WINDOWS:
            rolling_tp_vol = (typical * df["volume"]).rolling(w).sum()
            rolling_vol = df["volume"].rolling(w).sum()
            rolling_vwap = rolling_tp_vol / (rolling_vol + 1e-12)
            f[f"VWAP_DIST_{w}"] = (df["close"] - rolling_vwap) / (rolling_vwap + 1e-12)

        # ── 2. Micro momentum ────────────────────────────────────────────
        for w in WINDOWS:
            f[f"ROC_{w}"] = df["close"].pct_change(w)
            f[f"MA_RATIO_{w}"] = df["close"] / (df["close"].rolling(w).mean() + 1e-12)

        # ── 3. Volume factors ────────────────────────────────────────────
        for w in WINDOWS:
            vol_ma = df["volume"].rolling(w).mean()
            f[f"VRATIO_{w}"] = df["volume"] / (vol_ma + 1e-12)
            f[f"VSTD_{w}"] = df["volume"].rolling(w).std() / (vol_ma + 1e-12)

        # Cumulative volume ratio vs expected (linear pace)
        bar_idx = np.arange(1, len(df) + 1, dtype=float)
        expected_vol = cum_vol.iloc[-1] * bar_idx / len(df) if len(df) > 0 else bar_idx
        f["VOL_PACE"] = cum_vol.values / (expected_vol + 1e-12)

        # ── 4. Volatility / range factors ────────────────────────────────
        for w in WINDOWS:
            f[f"STD_{w}"] = df["close"].rolling(w).std() / (df["close"] + 1e-12)
            ma = df["close"].rolling(w).mean()
            std = df["close"].rolling(w).std()
            f[f"BBPOS_{w}"] = (df["close"] - ma) / (2 * std + 1e-12)

        # ATR-based
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        for w in WINDOWS:
            atr = tr.rolling(w).mean()
            f[f"ATR_RATIO_{w}"] = tr / (atr + 1e-12)  # current TR vs avg

        # ── 5. Intraday range position ───────────────────────────────────
        day_high = df["high"].cummax()
        day_low = df["low"].cummin()
        f["DAY_RANGE_POS"] = (df["close"] - day_low) / (day_high - day_low + 1e-12)

        for w in WINDOWS:
            rh = df["high"].rolling(w).max()
            rl = df["low"].rolling(w).min()
            f[f"RANGE_POS_{w}"] = (df["close"] - rl) / (rh - rl + 1e-12)

        # ── 6. RSI (intraday) ────────────────────────────────────────────
        delta = df["close"].diff()
        for w in [6, 12, 24]:
            gain = delta.clip(lower=0).rolling(w).mean()
            loss = (-delta.clip(upper=0)).rolling(w).mean()
            rs = gain / (loss + 1e-12)
            f[f"RSI_{w}"] = 100 - 100 / (1 + rs)

        # ── 7. KBAR factors (same as daily, applied to 5-min bars) ───────
        f["KMID"] = (df["close"] - df["open"]) / (df["open"] + 1e-12)
        f["KLEN"] = (df["high"] - df["low"]) / (df["open"] + 1e-12)
        oc_max = df[["open", "close"]].max(axis=1)
        oc_min = df[["open", "close"]].min(axis=1)
        f["KUP"] = (df["high"] - oc_max) / (df["open"] + 1e-12)
        f["KLOW"] = (oc_min - df["low"]) / (df["open"] + 1e-12)
        f["KSFT"] = (2 * df["close"] - df["high"] - df["low"]) / (df["open"] + 1e-12)

        # ── 8. Trend / slope ─────────────────────────────────────────────
        for w in WINDOWS:
            f[f"SLOPE_{w}"] = _rolling_slope(df["close"], w)

        return f

    def compute_multi(self, data_dict: dict) -> dict:
        """Compute intraday factors for multiple tickers.

        Args:
            data_dict: {ticker: DataFrame} of 5-min bars.

        Returns:
            {ticker: factors_DataFrame}
        """
        result = {}
        for ticker, df in data_dict.items():
            if df is None or df.empty:
                continue
            try:
                result[ticker] = self.compute_all(df)
            except Exception:
                continue
        return result


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Rolling OLS slope normalized by mean."""
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
