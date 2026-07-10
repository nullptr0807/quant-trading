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
    if table:
        covered_dates = int(conn.execute(
            "SELECT COUNT(DISTINCT date) FROM universe_membership "
            "WHERE market=? AND date BETWEEN ? AND ?",
            (market, start_date, end_date),
        ).fetchone()[0])
    if not table or covered_dates == 0:
        result = {
            "valid_for_capital_allocation": False,
            "reason": "missing_point_in_time_universe",
            "market": market,
            "window": [start_date, end_date],
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
