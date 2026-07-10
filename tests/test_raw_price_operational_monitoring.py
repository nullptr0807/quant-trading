from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest


def _create_price_tables(con: sqlite3.Connection) -> None:
    for table in ("prices", "prices_raw"):
        con.execute(
            f"""
            CREATE TABLE {table} (
                ticker TEXT NOT NULL,
                datetime TEXT NOT NULL,
                interval TEXT NOT NULL,
                open REAL, high REAL, low REAL, close REAL, volume REAL,
                PRIMARY KEY (ticker, datetime, interval)
            )
            """
        )


def test_ledger_history_loader_reads_raw_execution_prices_only():
    from scripts.ledger_watchdog import _load_price_history, _price_at

    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    _create_price_tables(con)
    con.execute(
        "INSERT INTO prices VALUES ('CRWD','2026-07-09T20:00:00+00:00','1d',1,1,1,154,1)"
    )
    con.execute(
        "INSERT INTO prices_raw VALUES ('CRWD','2026-07-09T20:00:00+00:00','1d',1,1,1,616,1)"
    )

    history = _load_price_history(
        con,
        {"CRWD"},
        "2026-07-09T00:00:00+00:00",
        "2026-07-10T23:59:59+00:00",
        intervals=("1d",),
    )

    assert _price_at(history, "CRWD", "2026-07-10T00:00:00+00:00") == 616


@pytest.mark.parametrize(
    ("market", "ticker", "interval"),
    [("US", "CRWD", "5m"), ("CN", "300502.SZ", "15m")],
)
def test_history_audit_warns_and_skips_when_only_adjusted_prices_exist(
    market, ticker, interval
):
    from scripts.ledger_watchdog import history_curve_audit

    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    _create_price_tables(con)
    con.executescript(
        """
        CREATE TABLE accounts (
            id INTEGER PRIMARY KEY, name TEXT, cash REAL, equity REAL,
            timestamp TEXT, market TEXT
        );
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY, account TEXT, market TEXT, timestamp TEXT,
            ticker TEXT, side TEXT, shares REAL, price REAL
        );
        """
    )
    con.execute(
        "INSERT INTO trades VALUES (1,'A01',?,'2026-07-09T13:00:00+00:00',?,'buy',1,100)",
        (market, ticker),
    )
    con.execute(
        "INSERT INTO accounts VALUES (1,'A01',9900,10000,'2026-07-09T14:00:00+00:00',?)",
        (market,),
    )
    # Deliberately provide an adjusted bar that would make a fallback possible.
    con.execute(
        "INSERT INTO prices VALUES (?, '2026-07-09T13:55:00+00:00', ?, 1,1,1,100,1)",
        (ticker, interval),
    )
    args = argparse.Namespace(
        history_days=1.0,
        history_include_retired=False,
        history_max_points=120,
        history_equity_tolerance=250.0,
        history_equity_tolerance_pct=0.03,
        history_report_limit=8,
    )

    issues = history_curve_audit(
        con,
        "A01",
        market,
        10000.0,
        "active",
        "2026-07-09T00:00:00+00:00",
        "2026-07-10T00:00:00+00:00",
        args,
    )

    assert not any(i.check == "history_curve" for i in issues)
    warning = next(i for i in issues if i.check == "history_curve_prices")
    assert warning.severity == "warning"
    assert "raw" in warning.message.lower()
    assert "adjusted" in warning.message.lower()


def test_history_audit_warns_when_raw_price_table_is_not_initialized():
    from scripts.ledger_watchdog import history_curve_audit

    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.executescript(
        """
        CREATE TABLE prices (
            ticker TEXT, datetime TEXT, interval TEXT, close REAL
        );
        CREATE TABLE accounts (
            id INTEGER PRIMARY KEY, name TEXT, cash REAL, equity REAL,
            timestamp TEXT, market TEXT
        );
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY, account TEXT, market TEXT, timestamp TEXT,
            ticker TEXT, side TEXT, shares REAL, price REAL
        );
        INSERT INTO trades VALUES
          (1,'A01','US','2026-07-09T13:00:00+00:00','CRWD','buy',1,100);
        INSERT INTO accounts VALUES
          (1,'A01',9900,10000,'2026-07-09T14:00:00+00:00','US');
        INSERT INTO prices VALUES
          ('CRWD','2026-07-09T13:55:00+00:00','5m',100);
        """
    )
    args = argparse.Namespace(
        history_days=1.0,
        history_include_retired=False,
        history_max_points=120,
        history_equity_tolerance=250.0,
        history_equity_tolerance_pct=0.03,
        history_report_limit=8,
    )

    issues = history_curve_audit(
        con,
        "A01",
        "US",
        10000.0,
        "active",
        "2026-07-09T00:00:00+00:00",
        "2026-07-10T00:00:00+00:00",
        args,
    )

    assert [i.check for i in issues] == ["history_curve_prices"]
    assert issues[0].severity == "warning"
    assert issues[0].detail["price_table"] == "prices_raw"


def test_operational_ledger_account_discovery_defaults_to_active_only():
    from scripts.ledger_watchdog import load_accounts

    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.execute(
        "CREATE TABLE account_meta (account_id TEXT, market TEXT, status TEXT)"
    )
    con.executemany(
        "INSERT INTO account_meta VALUES (?,?,?)",
        [("A01", "US", "active"), ("A02", "US", "retired"), ("CA01", "CN", None)],
    )

    assert [r["account_id"] for r in load_accounts(con, "ALL")] == ["CA01", "A01"]
    assert [r["account_id"] for r in load_accounts(con, "US")] == ["A01"]


def _health_db() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    _create_price_tables(con)
    con.executescript(
        """
        CREATE TABLE account_meta (account_id TEXT, market TEXT, status TEXT);
        CREATE TABLE positions (account TEXT, market TEXT, ticker TEXT, shares REAL);
        """
    )
    return con


def test_price_health_checks_adjusted_universe_and_active_raw_holdings_per_market():
    from scripts.health_check import check_price_1d_health

    con = _health_db()
    con.executemany(
        "INSERT INTO account_meta VALUES (?,?,?)",
        [
            ("A01", "US", "active"),
            ("A99", "US", "retired"),
            ("CA01", "CN", "active"),
        ],
    )
    con.executemany(
        "INSERT INTO positions VALUES (?,?,?,?)",
        [
            ("A01", "US", "AAPL", 1),
            ("A99", "US", "OLD", 1),
            ("CA01", "CN", "000001.SZ", 100),
        ],
    )
    con.executemany(
        "INSERT INTO prices VALUES (?,?,?,?,?,?,?,?)",
        [
            ("AAPL", "2026-07-10", "1d", 1, 1, 1, 1, 1),
            ("000001.SZ", "2026-07-10", "1d", 1, 1, 1, 1, 1),
        ],
    )
    con.execute(
        "INSERT INTO prices_raw VALUES ('AAPL','2026-07-10','1d',1,1,1,1,1)"
    )

    issues = check_price_1d_health(
        con,
        universe_by_market={"US": ["AAPL", "MSFT"], "CN": ["000001.SZ"]},
        target_dates={"US": "2026-07-10", "CN": "2026-07-10"},
    )

    us_adjusted = next(
        i
        for i in issues
        if i["check"] == "price_1d_coverage"
        and i["market"] == "US"
        and i["price_mode"] == "adjusted"
    )
    assert us_adjusted["expected"] == 2
    assert us_adjusted["covered"] == 1
    assert us_adjusted["missing_sample"] == ["MSFT"]

    cn_raw = next(
        i
        for i in issues
        if i["check"] == "price_1d_coverage"
        and i["market"] == "CN"
        and i["price_mode"] == "raw"
    )
    assert cn_raw["expected"] == 1
    assert cn_raw["missing_sample"] == ["000001.SZ"]
    # Retired OLD is not part of operational raw coverage.
    assert not any(
        i.get("market") == "US" and i.get("price_mode") == "raw" for i in issues
    )


def test_price_health_reports_stale_1d_series_separately_from_coverage():
    from scripts.health_check import check_price_1d_health

    con = _health_db()
    con.execute("INSERT INTO account_meta VALUES ('A01','US','active')")
    con.execute("INSERT INTO positions VALUES ('A01','US','AAPL',1)")
    con.execute(
        "INSERT INTO prices VALUES ('AAPL','2026-07-09','1d',1,1,1,1,1)"
    )
    con.execute(
        "INSERT INTO prices_raw VALUES ('AAPL','2026-07-09','1d',1,1,1,1,1)"
    )

    issues = check_price_1d_health(
        con,
        universe_by_market={"US": ["AAPL"], "CN": []},
        target_dates={"US": "2026-07-10", "CN": "2026-07-10"},
    )

    assert not any(i["check"] == "price_1d_coverage" for i in issues)
    assert {
        (i["market"], i["price_mode"])
        for i in issues
        if i["check"] == "price_1d_freshness"
    } == {("US", "adjusted"), ("US", "raw")}


def test_factor_freshness_reference_remains_adjusted_prices_table():
    from scripts.health_check import latest_trading_date

    con = sqlite3.connect(":memory:")
    _create_price_tables(con)
    con.execute(
        "INSERT INTO prices VALUES ('AAPL','2026-07-08','1d',1,1,1,1,1)"
    )
    con.execute(
        "INSERT INTO prices_raw VALUES ('AAPL','2026-07-10','1d',1,1,1,1,1)"
    )

    assert latest_trading_date(con, "US") == "2026-07-08"


def test_backfill_log_stale_detection_is_pure_and_flags_unfinished_run():
    from scripts.health_check import (
        evaluate_backfill_log_events,
        parse_backfill_log_events,
    )

    text = (
        "2026-07-10 08:00:00,000 [INFO] Backfilling [CN] 300 tickers, "
        "interval=1d, days=5, price_mode=raw\n"
    )
    events = parse_backfill_log_events(text)
    issues = evaluate_backfill_log_events(
        events,
        now=datetime(2026, 7, 10, 10, 0, tzinfo=timezone.utc),
        max_success_age_seconds=24 * 3600,
        max_incomplete_seconds=30 * 60,
    )

    assert events[0]["event"] == "start"
    assert issues[0]["check"] == "backfill_log_stale"
    assert issues[0]["status"] == "incomplete"
    assert issues[0]["severity"] == "critical"


def test_backfill_process_stale_detection_is_pure_and_filters_unrelated_processes():
    from scripts.health_check import evaluate_backfill_processes, parse_process_rows

    rows = parse_process_rows(
        "101 1901 /home/gexin/quant-trading/venv/bin/python -m scripts.backfill_prices --market CN\n"
        "102 99999 python unrelated_worker.py\n"
    )
    issues = evaluate_backfill_processes(rows, max_runtime_seconds=1800)

    assert len(issues) == 1
    assert issues[0]["check"] == "backfill_process_stale"
    assert issues[0]["pid"] == 101
    assert issues[0]["elapsed_seconds"] == 1901


@pytest.mark.parametrize(
    ("summary", "expected_check"),
    [
        (
            {
                "tickers_scanned": 10,
                "fetch_errors": 0,
                "critical_open_share_actions": 1,
            },
            "open_position_share_action",
        ),
        (
            {
                "tickers_scanned": 10,
                "fetch_errors": 1,
                "critical_open_share_actions": 0,
            },
            "fetch_coverage",
        ),
    ],
)
def test_corporate_action_summary_fails_closed(summary, expected_check):
    from scripts.corporate_action_check import evaluate_audit_result

    result = evaluate_audit_result({"summary": summary, "paths": {}}, min_fetch_coverage=1.0)

    assert result["ok"] is False
    assert result["exit_code"] != 0
    assert expected_check in {i["check"] for i in result["issues"]}


def test_corporate_action_yfinance_dataframe_shape_is_normalized(monkeypatch):
    import pandas as pd
    import yfinance as yf
    from scripts.audit_corporate_actions import fetch_us_actions

    class DummyTicker:
        splits = pd.DataFrame(
            {"Stock Splits": [4.0]},
            index=pd.to_datetime(["2026-07-02 09:30:00-04:00"]),
        )
        dividends = pd.DataFrame(
            {"Dividends": [0.25]},
            index=pd.to_datetime(["2026-07-03 09:30:00-04:00"]),
        )

    monkeypatch.setattr(yf, "Ticker", lambda _ticker: DummyTicker())
    actions = fetch_us_actions("TEST", "2026-07-01", "2026-07-10")

    assert [(a.action_type, a.ratio, a.cash_per_share) for a in actions] == [
        ("split", 4.0, None),
        ("cash_dividend", None, 0.25),
    ]


def test_corporate_action_us_fetch_failure_remains_explicit(monkeypatch):
    import yfinance as yf
    from scripts.audit_corporate_actions import fetch_us_actions

    class BrokenTicker:
        @property
        def splits(self):
            raise RuntimeError("split provider failed")

        @property
        def dividends(self):
            raise RuntimeError("dividend provider failed")

    monkeypatch.setattr(yf, "Ticker", lambda _ticker: BrokenTicker())
    actions = fetch_us_actions("TEST", "2026-07-01", "2026-07-10")

    assert [a.action_type for a in actions] == ["fetch_error", "fetch_error"]


def test_corporate_action_summary_passes_without_open_share_actions_or_fetch_errors():
    from scripts.corporate_action_check import evaluate_audit_result

    result = evaluate_audit_result(
        {
            "summary": {
                "tickers_scanned": 10,
                "fetch_errors": 0,
                "critical_open_share_actions": 0,
            },
            "paths": {"md": "/tmp/report.md"},
        },
        min_fetch_coverage=1.0,
    )

    assert result["ok"] is True
    assert result["exit_code"] == 0
    assert result["issues"] == []


def test_corporate_action_daily_wrapper_has_os_lock_timeout_and_alert_path():
    wrapper = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "corporate_action_check_daily.sh"
    ).read_text()

    assert "/usr/bin/flock" in wrapper
    assert "/usr/bin/timeout" in wrapper
    assert "corporate_action_check.py" in wrapper
    assert "TELEGRAM_BOT_TOKEN" in wrapper
