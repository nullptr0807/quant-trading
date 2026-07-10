from __future__ import annotations

import sqlite3

import pytest


def test_current_universe_research_is_not_valid_for_capital_allocation():
    from research.validity import ResearchValidityError, ensure_point_in_time_universe

    con = sqlite3.connect(":memory:")
    with pytest.raises(ResearchValidityError, match="missing_point_in_time_universe"):
        ensure_point_in_time_universe(
            con, market="US", start_date="2025-01-01", end_date="2026-01-01"
        )

    result = ensure_point_in_time_universe(
        con, market="US", start_date="2025-01-01", end_date="2026-01-01",
        allow_survivorship_biased=True,
    )
    assert result["valid_for_capital_allocation"] is False


def test_point_in_time_universe_gate_accepts_covered_membership():
    from research.validity import ensure_point_in_time_universe

    con = sqlite3.connect(":memory:")
    con.execute(
        "CREATE TABLE universe_membership (market TEXT,date TEXT,ticker TEXT,"
        "PRIMARY KEY(market,date,ticker))"
    )
    con.execute("INSERT INTO universe_membership VALUES ('US','2025-01-02','AAPL')")
    result = ensure_point_in_time_universe(
        con, market="US", start_date="2025-01-01", end_date="2026-01-01"
    )
    assert result["valid_for_capital_allocation"] is True
