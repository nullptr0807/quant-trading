"""Research validity gates for capital-allocation backtests."""
from __future__ import annotations

import sqlite3


class ResearchValidityError(RuntimeError):
    pass


def ensure_point_in_time_universe(
    conn: sqlite3.Connection,
    *,
    market: str,
    start_date: str,
    end_date: str,
    allow_survivorship_biased: bool = False,
) -> dict:
    """Require point-in-time membership unless explicitly running biased research."""
    table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='universe_membership'"
    ).fetchone()
    covered_dates = 0
    window_days = 0
    if table:
        rows = conn.execute(
            "SELECT date,COUNT(*) FROM universe_membership "
            "WHERE market=? AND date BETWEEN ? AND ? GROUP BY date ORDER BY date",
            (market, start_date, end_date),
        ).fetchall()
        covered_dates = len(rows)
        window_days = int(conn.execute(
            "SELECT COUNT(DISTINCT substr(datetime,1,10)) FROM prices "
            "WHERE interval='1d' AND substr(datetime,1,10) BETWEEN ? AND ?",
            (start_date, end_date),
        ).fetchone()[0]) if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='prices'"
        ).fetchone() else 0
        complete = covered_dates > 0 and (window_days == 0 or covered_dates >= window_days)
    else:
        complete = False
    if not table or not complete:
        result = {
            "valid_for_capital_allocation": False,
            "reason": "missing_point_in_time_universe",
            "market": market,
            "window": [start_date, end_date],
            "membership_dates": covered_dates,
            "required_trading_dates": window_days,
        }
        if not allow_survivorship_biased:
            raise ResearchValidityError(str(result))
        return result
    return {
        "valid_for_capital_allocation": True,
        "market": market,
        "window": [start_date, end_date],
        "membership_dates": covered_dates,
    }
