import sqlite3
from datetime import date, timedelta


def _make_db(path, *, score_date: str, price_date: str) -> None:
    con = sqlite3.connect(path)
    con.execute(
        """
        CREATE TABLE factor_values (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            factor_name TEXT NOT NULL,
            value REAL,
            factor_group TEXT NOT NULL DEFAULT 'alpha158',
            PRIMARY KEY (ticker, date, factor_name, factor_group)
        )
        """
    )
    con.execute(
        """
        CREATE TABLE prices (
            ticker TEXT NOT NULL,
            datetime TEXT NOT NULL,
            interval TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            PRIMARY KEY (ticker, datetime, interval)
        )
        """
    )
    con.execute(
        "INSERT INTO factor_values VALUES ('AAPL', ?, 'qlib_Q01_score', 1.23, 'qlib')",
        (score_date,),
    )
    con.execute(
        "INSERT INTO prices VALUES ('AAPL', ?, '1d', 1, 1, 1, 1, 1)",
        (price_date,),
    )
    con.commit()
    con.close()


def test_qlib_scores_load_when_fresh(tmp_path, monkeypatch):
    from main import QuantSystem

    db = tmp_path / "fresh.db"
    _make_db(db, score_date="2026-07-07", price_date="2026-07-07")
    emitted = []
    monkeypatch.setattr("main.emit_event", lambda *a, **k: emitted.append((a, k)))

    system = object.__new__(QuantSystem)
    system.market = "US"
    system.db_path = str(db)
    system.universe = ["AAPL"]

    assert system._load_qlib_scores("Q01") == [("AAPL", 1.23)]
    assert emitted == []


def test_qlib_scores_stale_gate_skips_and_emits_event(tmp_path, monkeypatch):
    from main import QuantSystem

    db = tmp_path / "stale.db"
    _make_db(db, score_date="2026-07-01", price_date="2026-07-07")
    emitted = []
    monkeypatch.setattr("main.emit_event", lambda *a, **k: emitted.append((a, k)))

    system = object.__new__(QuantSystem)
    system.market = "US"
    system.db_path = str(db)
    system.universe = ["AAPL"]

    assert system._load_qlib_scores("Q01") == []
    assert emitted
    assert emitted[0][0][0] == "factor"
    assert emitted[0][1]["severity"] == "error"
    assert emitted[0][1]["detail"]["lag_days"] == 6


def test_health_check_requires_factor_group_in_factor_values_pk(tmp_path):
    from scripts.health_check import check_schema

    old_db = tmp_path / "old.db"
    con = sqlite3.connect(old_db)
    con.row_factory = sqlite3.Row
    con.execute(
        """
        CREATE TABLE factor_values (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            factor_name TEXT NOT NULL,
            value REAL,
            factor_group TEXT DEFAULT 'alpha158',
            PRIMARY KEY (ticker, date, factor_name)
        )
        """
    )

    issues = check_schema(con)

    assert issues
    assert issues[0]["severity"] == "critical"
    assert issues[0]["actual_pk"] == ["ticker", "date", "factor_name"]


def test_health_check_accepts_group_pk(tmp_path):
    from scripts.health_check import check_schema

    new_db = tmp_path / "new.db"
    con = sqlite3.connect(new_db)
    con.row_factory = sqlite3.Row
    con.execute(
        """
        CREATE TABLE factor_values (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            factor_name TEXT NOT NULL,
            value REAL,
            factor_group TEXT NOT NULL DEFAULT 'alpha158',
            PRIMARY KEY (ticker, date, factor_name, factor_group)
        )
        """
    )

    assert check_schema(con) == []
