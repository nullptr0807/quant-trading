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


def test_health_check_qlib_threshold_matches_live_us_gate(tmp_path):
    from scripts.health_check import check_model_factor_freshness

    con = sqlite3.connect(tmp_path / "health.db")
    con.row_factory = sqlite3.Row
    con.executescript(
        """
        CREATE TABLE account_meta (account_id TEXT, market TEXT, "group" TEXT, status TEXT);
        CREATE TABLE account_state (account TEXT, market TEXT);
        CREATE TABLE positions (account TEXT, market TEXT);
        CREATE TABLE trades (account TEXT, market TEXT);
        CREATE TABLE accounts (name TEXT, market TEXT);
        CREATE TABLE prices (ticker TEXT, datetime TEXT, interval TEXT);
        CREATE TABLE factor_values (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            factor_name TEXT NOT NULL,
            value REAL,
            factor_group TEXT NOT NULL DEFAULT 'alpha158',
            PRIMARY KEY (ticker, date, factor_name, factor_group)
        );
        """
    )
    con.execute("INSERT INTO account_meta VALUES ('Q01','US','Q','active')")
    con.execute("INSERT INTO account_state VALUES ('Q01','US')")
    con.execute("INSERT INTO prices VALUES ('AAPL','2026-07-07','1d')")
    con.execute("INSERT INTO factor_values VALUES ('AAPL','2026-07-04','qlib_Q01_score',1.0,'qlib')")

    issues = check_model_factor_freshness(con)

    assert issues
    assert issues[0]["check"] == "qlib_factor"
    assert issues[0]["market"] == "US"
    assert issues[0]["lag_days"] == 3
    assert issues[0]["max_lag_days"] == 2


def test_health_check_qlib_threshold_allows_cn_three_day_lag(tmp_path):
    from scripts.health_check import check_model_factor_freshness

    con = sqlite3.connect(tmp_path / "health_cn.db")
    con.row_factory = sqlite3.Row
    con.executescript(
        """
        CREATE TABLE account_meta (account_id TEXT, market TEXT, "group" TEXT, status TEXT);
        CREATE TABLE account_state (account TEXT, market TEXT);
        CREATE TABLE positions (account TEXT, market TEXT);
        CREATE TABLE trades (account TEXT, market TEXT);
        CREATE TABLE accounts (name TEXT, market TEXT);
        CREATE TABLE prices (ticker TEXT, datetime TEXT, interval TEXT);
        CREATE TABLE factor_values (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            factor_name TEXT NOT NULL,
            value REAL,
            factor_group TEXT NOT NULL DEFAULT 'alpha158',
            PRIMARY KEY (ticker, date, factor_name, factor_group)
        );
        """
    )
    con.execute("INSERT INTO account_meta VALUES ('CQ01','CN','Q','active')")
    con.execute("INSERT INTO account_state VALUES ('CQ01','CN')")
    con.execute("INSERT INTO prices VALUES ('000001.SZ','2026-07-07','1d')")
    con.execute("INSERT INTO factor_values VALUES ('000001.SZ','2026-07-04','qlib_Q01_score',1.0,'qlib')")

    assert check_model_factor_freshness(con) == []
