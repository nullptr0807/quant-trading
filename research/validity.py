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
        membership_dates = {str(row[0]) for row in rows}
        covered_dates = len(membership_dates)
        if conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='prices'"
        ).fetchone():
            required_dates = {
                str(row[0]) for row in conn.execute(
                    "SELECT DISTINCT substr(datetime,1,10) FROM prices "
                    "WHERE interval='1d' AND substr(datetime,1,10) BETWEEN ? AND ?",
                    (start_date, end_date),
                ).fetchall()
            }
        else:
            required_dates = set()
        window_days = len(required_dates)
        missing_dates = sorted(required_dates - membership_dates)
        complete = covered_dates > 0 and not missing_dates
    else:
        missing_dates = []
        complete = False
    if not table or not complete:
        result = {
            "valid_for_capital_allocation": False,
            "reason": "missing_point_in_time_universe",
            "market": market,
            "window": [start_date, end_date],
            "membership_dates": covered_dates,
            "required_trading_dates": window_days,
            "missing_membership_dates": missing_dates,
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
