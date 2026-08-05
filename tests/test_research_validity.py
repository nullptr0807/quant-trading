from __future__ import annotations

import sqlite3

import pytest
import pandas as pd
from types import SimpleNamespace


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


def test_point_in_time_universe_gate_rejects_partial_membership_coverage():
    from research.validity import ResearchValidityError, ensure_point_in_time_universe

    con = sqlite3.connect(":memory:")
    con.executescript(
        """
        CREATE TABLE universe_membership (
            market TEXT,date TEXT,ticker TEXT,PRIMARY KEY(market,date,ticker)
        );
        CREATE TABLE prices (ticker TEXT,datetime TEXT,interval TEXT);
        INSERT INTO universe_membership VALUES ('US','2025-01-02','AAPL');
        INSERT INTO prices VALUES ('AAPL','2025-01-02','1d');
        INSERT INTO prices VALUES ('AAPL','2025-01-03','1d');
        """
    )

    with pytest.raises(ResearchValidityError, match="missing_point_in_time_universe"):
        ensure_point_in_time_universe(
            con, market="US", start_date="2025-01-01", end_date="2025-01-03"
        )


def test_point_in_time_universe_gate_rejects_wrong_date_set_with_same_count():
    from research.validity import ResearchValidityError, ensure_point_in_time_universe

    con = sqlite3.connect(":memory:")
    con.executescript(
        """
        CREATE TABLE universe_membership (
            market TEXT,date TEXT,ticker TEXT,PRIMARY KEY(market,date,ticker)
        );
        CREATE TABLE prices (ticker TEXT,datetime TEXT,interval TEXT);
        INSERT INTO universe_membership VALUES ('US','2025-01-01','AAPL');
        INSERT INTO universe_membership VALUES ('US','2025-01-02','AAPL');
        INSERT INTO prices VALUES ('AAPL','2025-01-02','1d');
        INSERT INTO prices VALUES ('AAPL','2025-01-03','1d');
        """
    )

    with pytest.raises(ResearchValidityError, match="missing_point_in_time_universe"):
        ensure_point_in_time_universe(
            con, market="US", start_date="2025-01-01", end_date="2025-01-03"
        )


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


def test_hindsight_gp_is_fail_closed_and_explicit_override_is_research_only():
    from research.validity import ResearchValidityError, ensure_gp_provenance

    with pytest.raises(ResearchValidityError, match="gp_hindsight_expression"):
        ensure_gp_provenance("current_saved_expression")

    result = ensure_gp_provenance(
        "current_saved_expression", allow_hindsight=True
    )
    assert result["look_ahead_validity"] == "invalid_hindsight"
    assert result["capital_allocation_valid"] is False


def test_result_metadata_has_mandatory_validity_coordinates():
    from research.validity import build_research_metadata

    metadata = build_research_metadata(
        universe={"mode": "point_in_time", "valid_for_capital_allocation": True},
        signal_price_mode="adjusted",
        execution_price_mode="raw",
        model_provenance={"kind": "qlib_checkpoint", "as_of": "2026-07-09"},
        look_ahead_validity="valid",
    )
    assert metadata == {
        "universe_mode": "point_in_time",
        "signal_price_mode": "adjusted",
        "execution_price_mode": "raw",
        "model_provenance": {"kind": "qlib_checkpoint", "as_of": "2026-07-09"},
        "look_ahead_validity": "valid",
        "capital_allocation_valid": True,
        "warnings": [],
    }


def test_legacy_backtest_engine_calls_pit_gate_before_trading(tmp_path):
    from backtest.engine import BacktestEngine
    from research.validity import ResearchValidityError

    db = tmp_path / "fixture.db"
    sqlite3.connect(db).close()
    frame = pd.DataFrame({
        "ticker": ["AAA", "AAA"],
        "datetime": pd.to_datetime(["2026-07-09", "2026-07-10"]),
        "open": [10.0, 11.0], "high": [10.0, 11.0], "low": [10.0, 11.0],
        "close": [10.0, 11.0], "volume": [100, 100],
    })
    engine = object.__new__(BacktestEngine)
    engine.days, engine.initial_cash, engine.adaptive = 2, 1000.0, False
    engine.allow_survivorship_biased = False
    engine.allow_gp_hindsight = False
    engine.fetcher = SimpleNamespace(
        store=SimpleNamespace(db_path=str(db)), get_historical=lambda *a, **k: frame
    )
    engine._pit_cache, engine.validity = {}, {}

    with pytest.raises(ResearchValidityError, match="missing_point_in_time_universe"):
        engine.run()
