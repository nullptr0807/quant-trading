"""Research validity gates for capital-allocation backtests."""
from __future__ import annotations

import sqlite3
from typing import Any


class ResearchValidityError(RuntimeError):
    pass


def ensure_gp_provenance(
    provenance: str, *, allow_hindsight: bool = False
) -> dict[str, Any]:
    """Reject use of today's persisted GP expression as historical alpha."""
    if provenance == "walk_forward_checkpoint":
        return {
            "kind": "gp_walk_forward_checkpoint",
            "look_ahead_validity": "valid",
            "capital_allocation_valid": True,
        }
    result = {
        "kind": "gp_current_saved_expression",
        "look_ahead_validity": "invalid_hindsight",
        "capital_allocation_valid": False,
        "reason": "gp_hindsight_expression",
        "warning": (
            "Today's saved GP expression was selected with hindsight; this replay is "
            "research-only and MUST NOT be used for capital allocation."
        ),
    }
    if not allow_hindsight:
        raise ResearchValidityError(str(result))
    return result


def build_research_metadata(
    *,
    universe: dict[str, Any],
    signal_price_mode: str,
    execution_price_mode: str,
    model_provenance: dict[str, Any],
    look_ahead_validity: str,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Build the mandatory, machine-readable result-validity envelope."""
    warnings = list(warnings or [])
    capital_valid = bool(
        universe.get("valid_for_capital_allocation")
        and signal_price_mode == "adjusted"
        and execution_price_mode == "raw"
        and look_ahead_validity == "valid"
        and model_provenance.get("capital_allocation_valid", True)
    )
    return {
        "universe_mode": universe.get("mode", "unknown"),
        "signal_price_mode": signal_price_mode,
        "execution_price_mode": execution_price_mode,
        "model_provenance": model_provenance,
        "look_ahead_validity": look_ahead_validity,
        "capital_allocation_valid": capital_valid,
        "warnings": warnings,
    }


def point_in_time_members(
    conn: sqlite3.Connection, *, market: str, as_of: str
) -> set[str]:
    """Return exact-date membership; absence is never forward/back filled."""
    return {
        str(row[0])
        for row in conn.execute(
            "SELECT ticker FROM universe_membership WHERE market=? AND date=?",
            (market, str(as_of)[:10]),
        ).fetchall()
    }


def ensure_latest_cached_pit_universe(
    conn: sqlite3.Connection, *, market: str, days: int,
    allow_survivorship_biased: bool = False,
) -> dict:
    """Preflight a legacy runner before any cache/network fetch can occur."""
    if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='prices'"
    ).fetchone():
        ticker_filter = (
            "AND (ticker LIKE '%.SH' OR ticker LIKE '%.SZ' OR ticker LIKE '%.BJ')"
            if market.upper() == "CN"
            else "AND ticker NOT LIKE '%.SH' AND ticker NOT LIKE '%.SZ' AND ticker NOT LIKE '%.BJ'"
        )
        dates = [
            row[0] for row in conn.execute(
                "SELECT DISTINCT substr(datetime,1,10) AS d FROM prices "
                "WHERE interval='1d' " + ticker_filter + " ORDER BY d DESC LIMIT ?",
                (days,),
            ).fetchall()
        ]
    else:
        dates = []
    start = min(dates) if dates else "0001-01-01"
    end = max(dates) if dates else "0001-01-01"
    return ensure_point_in_time_universe(
        conn, market=market, start_date=start, end_date=end,
        allow_survivorship_biased=allow_survivorship_biased,
    )


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
            ticker_filter = (
                "AND (ticker LIKE '%.SH' OR ticker LIKE '%.SZ' OR ticker LIKE '%.BJ')"
                if market.upper() == "CN"
                else "AND ticker NOT LIKE '%.SH' AND ticker NOT LIKE '%.SZ' AND ticker NOT LIKE '%.BJ'"
            )
            required_dates = {
                str(row[0]) for row in conn.execute(
                    "SELECT DISTINCT substr(datetime,1,10) FROM prices "
                    "WHERE interval='1d' AND substr(datetime,1,10) BETWEEN ? AND ? "
                    + ticker_filter,
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
            "mode": "current_survivorship_biased",
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
        "mode": "point_in_time",
        "market": market,
        "window": [start_date, end_date],
        "membership_dates": covered_dates,
    }
