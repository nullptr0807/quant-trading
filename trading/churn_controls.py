"""Churn-control helpers shared by Alpha158 / GP / Qlib trading loops.

Three knobs, all default-on, each independently switchable via config:
  - STOP_COOLDOWN_HOURS: after a stop_loss sell, suppress re-buys for N hours.
  - HOLD_BAND_MULT:      hysteresis. buy band = top_n; hold band = top_n * mult.
  - MIN_HOLD_DAYS:       non-stop-loss sells require position age >= N days.

Stop-loss path is NEVER suppressed by min-hold (capital preservation > churn control).
"""

from __future__ import annotations
import sqlite3
from datetime import datetime, timedelta, timezone


def get_stop_cooldown_set(
    db_path: str, account: str, market: str, hours: int
) -> set[str]:
    """Tickers that were stop_loss-sold within the last `hours`."""
    if hours <= 0:
        return set()
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    try:
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute(
                "SELECT DISTINCT ticker FROM events "
                "WHERE category='trade' AND account=? AND market=? "
                "AND detail LIKE '%\"reason\": \"stop_loss\"%' "
                "AND ts >= ?",
                (account, market, cutoff),
            ).fetchall()
            return {r[0] for r in rows}
        finally:
            conn.close()
    except Exception:
        return set()


def get_min_hold_set(
    db_path: str, account: str, market: str, min_hold_days: int,
    held_tickers: set[str] | None = None,
) -> set[str]:
    """Tickers whose most recent BUY was within `min_hold_days` calendar days.

    These positions must NOT be sold by signal/rebalance (stop_loss still allowed
    upstream). Only checks tickers in `held_tickers` if provided (perf).
    """
    if min_hold_days <= 0:
        return set()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=min_hold_days)).isoformat()
    try:
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute(
                "SELECT DISTINCT ticker FROM events "
                "WHERE category='trade' AND account=? AND market=? "
                "AND title LIKE 'BUY %' "
                "AND ts >= ?",
                (account, market, cutoff),
            ).fetchall()
            tickers = {r[0] for r in rows}
            if held_tickers is not None:
                tickers &= held_tickers
            return tickers
        finally:
            conn.close()
    except Exception:
        return set()
