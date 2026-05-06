"""Volatility-adaptive stop loss.

For each ticker, the stop loss percentage scales with its recent realized
volatility instead of being a fixed account-level constant. Rationale:
- TSLA σ_20 ≈ 3% means a -3% fixed stop fires almost daily on noise
- KO σ_20 ≈ 0.8% means a -3% fixed stop is asleep when the position is
  silently bleeding 4× its normal range

Formula:
    stop_pct = clip(k × σ_20, floor, cap)

where σ_20 = std of last 20 daily log returns.

Defaults (k=2.5, floor=1.5%, cap=8%) chosen to:
- match the old fixed -3% stop for an "average" S&P stock (σ≈1.2%)
- never tighter than 1.5% (avoid stop-and-restart whipsaw on low-vol)
- never looser than 8% (cap tail risk on micro-caps / event days)

If price history is insufficient, returns None — caller falls back to
account.stop_loss (the old fixed value).
"""
from __future__ import annotations
import logging
import math
import os
import sqlite3
import time
from typing import Optional

log = logging.getLogger("vol_stop")

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "trading.db")

# Tunables — calibrated against Russell 1000 σ_20 distribution:
#   median σ_20 ≈ 2.0% for "typical" S&P stock (after trimming small-caps)
#   We want stop ≈ 3% for that median, which matches the old fixed -3%.
#   → k = 1.5 hits "old default" at the median, tightens below, loosens above.
LOOKBACK_DAYS = 20      # σ window (~1 month of trading days)
K_SIGMA = 1.5           # stop = 1.5σ ~ ~10% tail under normal
FLOOR_PCT = 0.015       # 1.5% — never tighter (whipsaw protection)
CAP_PCT = 0.06          # 6% — never looser (tail-risk cap)
MIN_OBS = 10            # need 10+ daily returns to trust σ
CACHE_TTL_S = 3600      # hourly recompute

# {ticker: (computed_at_unix, stop_pct_or_None)}
_CACHE: dict[str, tuple[float, Optional[float]]] = {}


def _compute_sigma(conn: sqlite3.Connection, ticker: str,
                   lookback: int = LOOKBACK_DAYS) -> Optional[float]:
    """Std of last `lookback` daily log returns for one ticker."""
    rows = conn.execute(
        "SELECT close FROM prices WHERE interval='1d' AND ticker=? "
        "AND close IS NOT NULL AND close > 0 "
        "ORDER BY datetime DESC LIMIT ?",
        (ticker, lookback + 1),
    ).fetchall()
    if len(rows) < MIN_OBS + 1:
        return None
    closes = [r[0] for r in rows][::-1]   # chronological
    rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    if len(rets) < MIN_OBS:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var)


def get_vol_stop_pct(ticker: str,
                     fallback: float,
                     conn: Optional[sqlite3.Connection] = None,
                     k: float = K_SIGMA,
                     floor: float = FLOOR_PCT,
                     cap: float = CAP_PCT) -> float:
    """Return absolute stop-loss pct (positive number, e.g. 0.035 = -3.5%).

    Hourly-cached. Falls back to `fallback` (the old fixed strat.stop_loss)
    when σ can't be computed.
    """
    now = time.time()
    cached = _CACHE.get(ticker)
    if cached and (now - cached[0]) < CACHE_TTL_S:
        sigma_stop = cached[1]
    else:
        own_conn = conn is None
        c = conn or sqlite3.connect(DB_PATH)
        try:
            sigma = _compute_sigma(c, ticker)
        finally:
            if own_conn:
                c.close()
        sigma_stop = (k * sigma) if sigma is not None else None
        _CACHE[ticker] = (now, sigma_stop)

    if sigma_stop is None:
        return float(fallback)
    # Clip to [floor, cap]
    return max(floor, min(cap, float(sigma_stop)))


def clear_cache() -> None:
    """For tests / forced refresh."""
    _CACHE.clear()
