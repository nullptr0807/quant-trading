from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _price_health_db() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.executescript(
        """
        CREATE TABLE account_meta (account_id TEXT, market TEXT, status TEXT);
        CREATE TABLE positions (
            account TEXT, market TEXT, ticker TEXT, shares REAL,
            current_price REAL, updated_at TEXT
        );
        CREATE TABLE prices (
            ticker TEXT, datetime TEXT, interval TEXT, close REAL,
            PRIMARY KEY(ticker,datetime,interval)
        );
        CREATE TABLE prices_raw (
            ticker TEXT, datetime TEXT, interval TEXT, close REAL,
            PRIMARY KEY(ticker,datetime,interval)
        );
        """
    )
    return con


def test_qlib_wrapper_records_success_failure_and_lock_timeout_lifecycle():
    wrapper = (ROOT / "scripts" / "qlib_retrain_daily.sh").read_text()

    assert "--reason lock_timeout" not in wrapper
    assert 'record_health ok 0' in wrapper
    assert 'record_health failed "$rc"' in wrapper
    assert 'record_health lock_timeout 75' in wrapper
    assert "--component qlib_retrain" in wrapper
    assert "--detail" in wrapper


def test_operational_health_writer_preserves_last_success_across_failed_attempt(tmp_path):
    from scripts.record_operational_health import record_operational_health

    db = tmp_path / "health.db"
    record_operational_health(
        db=str(db), component="qlib_retrain", market="US", status="ok",
        scheduled_at="2026-08-25T01:00:00Z", started_at="2026-08-25T01:00:01Z",
        stopped_at="2026-08-25T01:01:00Z", exit_code=0, duration=59,
        detail={"phase": "verified"}, now="2026-08-25T01:01:01+00:00",
    )
    record_operational_health(
        db=str(db), component="qlib_retrain", market="US", status="failed",
        scheduled_at="2026-08-26T01:00:00Z", started_at="2026-08-26T01:00:01Z",
        stopped_at="2026-08-26T01:00:05Z", exit_code=1, duration=4,
        detail={"reason": "retrain_or_verify_failed"}, now="2026-08-26T01:00:06+00:00",
    )
    record_operational_health(
        db=str(db), component="qlib_retrain", market="US", status="lock_timeout",
        scheduled_at="2026-08-27T01:00:00Z", started_at="2026-08-27T01:00:00Z",
        stopped_at="2026-08-27T01:00:00Z", exit_code=75, duration=0,
        detail={"reason": "lock_timeout"}, now="2026-08-27T01:00:01+00:00",
    )

    with sqlite3.connect(db) as con:
        runs = con.execute(
            "SELECT status,exit_code FROM scheduler_runs ORDER BY id"
        ).fetchall()
        health = con.execute(
            "SELECT status,success_at,details FROM operational_health "
            "WHERE component='qlib_retrain' AND market='US'"
        ).fetchone()
    assert runs == [("ok", 0), ("failed", 1), ("lock_timeout", 75)]
    assert health[0:2] == ("lock_timeout", "2026-08-25T01:01:00Z")
    assert json.loads(health[2])["reason"] == "lock_timeout"


def test_health_ledger_history_window_matches_scheduled_raw_coverage(monkeypatch):
    from scripts import health_check

    captured: list[list[str]] = []

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(
        health_check.subprocess,
        "run",
        lambda cmd, **kwargs: captured.append(cmd) or Result(),
    )

    assert health_check.run_ledger("US") == []
    command = captured[0]
    history_days = float(command[command.index("--history-days") + 1])
    assert history_days == health_check.RAW_LEDGER_SCHEDULE_DAYS
    assert history_days > 0


def test_stale_held_mark_is_critical_only_for_affected_account():
    from scripts.health_check import check_price_1d_health

    con = _price_health_db()
    con.executemany(
        "INSERT INTO account_meta VALUES (?,?,?)",
        [("A01", "US", "active"), ("A02", "US", "active")],
    )
    con.executemany(
        "INSERT INTO positions VALUES (?,?,?,?,?,?)",
        [
            ("A01", "US", "STALE", 1, 10, "2026-08-24T20:00:00+00:00"),
            ("A02", "US", "FRESH", 1, 20, "2026-08-25T20:00:00+00:00"),
        ],
    )
    con.executemany(
        "INSERT INTO prices_raw VALUES (?,?,?,?)",
        [
            ("STALE", "2026-08-24", "1d", 10),
            ("FRESH", "2026-08-25", "1d", 20),
        ],
    )

    issues = check_price_1d_health(
        con,
        universe_by_market={"US": [], "CN": []},
        target_dates={"US": "2026-08-25", "CN": "2026-08-25"},
    )

    blocking = [i for i in issues if i["check"] == "held_raw_price_freshness"]
    assert len(blocking) == 1
    assert blocking[0]["severity"] == "critical"
    assert blocking[0]["account"] == "A01"
    assert blocking[0]["market"] == "US"
    assert blocking[0]["stale_sample"] == ["STALE"]
    assert not any(i.get("account") == "A02" for i in blocking)
    assert not any(i.get("scope") == "global" for i in blocking)


def test_ledger_stale_position_mark_is_blocking():
    from scripts.ledger_watchdog import check_account

    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.executescript(
        """
        CREATE TABLE account_meta (
            account_id TEXT, market TEXT, status TEXT, initial_cash REAL,
            retired_at TEXT
        );
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY, account TEXT, market TEXT, timestamp TEXT,
            ticker TEXT, side TEXT, shares REAL, price REAL, cost REAL,
            slippage REAL
        );
        CREATE TABLE account_state (
            account TEXT, market TEXT, cash REAL
        );
        CREATE TABLE positions (
            account TEXT, market TEXT, ticker TEXT, shares REAL, total_cost REAL,
            current_price REAL, updated_at TEXT
        );
        CREATE TABLE accounts (
            id INTEGER PRIMARY KEY, name TEXT, market TEXT, cash REAL,
            equity REAL, timestamp TEXT
        );
        CREATE TABLE events (
            account TEXT, market TEXT, category TEXT, ts TEXT
        );
        """
    )
    con.execute("INSERT INTO account_meta VALUES ('A01','US','active',10000,NULL)")
    con.execute("INSERT INTO account_state VALUES ('A01','US',10000)")
    con.execute(
        "INSERT INTO positions VALUES "
        "('A01','US','STALE',1,0,10,'2020-01-01T00:00:00+00:00')"
    )
    con.execute(
        "INSERT INTO accounts VALUES "
        "(1,'A01','US',10000,10010,'2026-08-25T00:00:00+00:00')"
    )
    meta = con.execute("SELECT * FROM account_meta").fetchone()
    args = argparse.Namespace(
        negative_cash_tolerance=0.0, cash_tolerance=100.0,
        share_tolerance=1e-6, cost_tolerance=100.0,
        stale_price_hours=36.0, equity_tolerance=50.0, history_days=0.0,
        cashflow_tolerance=100.0, event_ratio_warn=0.8,
    )

    issues, _ = check_account(
        con, meta, "2026-08-25T00:00:00+00:00",
        "2026-08-26T00:00:00+00:00", args,
    )

    stale = next(i for i in issues if i.check == "stale_prices")
    assert stale.severity == "critical"
    assert stale.account == "A01"


def test_failed_corporate_action_health_is_market_scoped_and_dividend_policy_explicit():
    from scripts.health_check import (
        CASH_DIVIDEND_ACCOUNTING_POLICY,
        check_corporate_action_health,
    )

    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.execute(
        "CREATE TABLE operational_health ("
        "component TEXT, market TEXT, status TEXT, success_at TEXT, "
        "source_timestamp TEXT, details TEXT)"
    )
    con.executemany(
        "INSERT INTO operational_health VALUES (?,?,?,?,?,?)",
        [
            (
                "corporate_action_fast", "US", "failed",
                "2026-08-24T00:00:00Z", "2026-08-25T00:00:00Z",
                json.dumps({"reason": "unresolved_open_share_action", "accounts": ["A02"]}),
            ),
            (
                "corporate_action_fast", "CN", "ok",
                "2026-08-25T00:00:00Z", "2026-08-25T00:00:00Z", "{}",
            ),
        ],
    )

    issues = check_corporate_action_health(con)

    assert len(issues) == 1
    assert issues[0]["severity"] == "critical"
    assert issues[0]["market"] == "US"
    assert issues[0]["accounts"] == ["A02"]
    assert CASH_DIVIDEND_ACCOUNTING_POLICY == {
        "mode": "manual_review",
        "automatic_historical_cash_credits": False,
        "health": "policy_not_implemented",
    }
