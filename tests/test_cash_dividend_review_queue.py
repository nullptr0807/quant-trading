from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest


def _csv(path: Path) -> Path:
    path.write_text(
        "account,market,ticker,ex_date,action_type,ratio,cash_per_share,description,"
        "held_shares_before,open_after_action,current_db_shares,expected_current_shares,"
        "current_price,estimated_current_equity_impact,interval_start,interval_end,source,"
        "severity,repair_applied\n"
        "A01,US,AAA,2026-08-20,cash_dividend,,0.25,quarterly,10,True,10,,,,"
        "2026-08-01T00:00:00Z,,provider,info,False\n"
        "A01,US,BBB,2026-08-20,split,2,,split,10,True,20,20,,,,"
        "2026-08-01T00:00:00Z,,provider,info,False\n"
        "A02,US,CCC,2026-08-20,cash_dividend,,0.30,closed,0,False,,,,,"
        "2026-08-01T00:00:00Z,2026-08-19T00:00:00Z,provider,info,False\n"
    )
    return path


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dividend_review_sync_is_dry_by_default_and_never_mutates_cash(tmp_path):
    from scripts.sync_cash_dividend_reviews import sync_reviews

    csv_path = _csv(tmp_path / "affected.csv")
    db = tmp_path / "trading.db"
    with sqlite3.connect(db) as con:
        con.executescript(
            """
            CREATE TABLE accounts(id INTEGER PRIMARY KEY,name TEXT,cash REAL,equity REAL,timestamp TEXT,market TEXT);
            CREATE TABLE trades(id INTEGER PRIMARY KEY,account TEXT,ticker TEXT,side TEXT,shares REAL,price REAL,cost REAL,slippage REAL,timestamp TEXT,market TEXT);
            INSERT INTO accounts VALUES (1,'A01',1000,1100,'2026-08-25T00:00:00Z','US');
            """
        )
    before = _hash(db)

    result = sync_reviews(csv_path=csv_path, db=db)

    assert result == {
        "market": "ALL", "candidates": 1, "gross_amount": 2.5,
        "applied": False, "cash_mutations": 0, "trade_mutations": 0,
    }
    assert _hash(db) == before
    with sqlite3.connect(db) as con:
        assert con.execute("SELECT cash,equity FROM accounts").fetchone() == (1000, 1100)
        assert con.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 0


def test_dividend_review_sync_requires_confirmation_and_is_idempotent(tmp_path):
    from scripts.sync_cash_dividend_reviews import CONFIRM, sync_reviews

    csv_path = _csv(tmp_path / "affected.csv")
    db = tmp_path / "trading.db"
    with pytest.raises(ValueError, match="confirmation"):
        sync_reviews(csv_path=csv_path, db=db, apply=True, confirm="wrong")

    first = sync_reviews(csv_path=csv_path, db=db, apply=True, confirm=CONFIRM)
    second = sync_reviews(csv_path=csv_path, db=db, apply=True, confirm=CONFIRM)

    assert first["queue_rows"] == second["queue_rows"] == 1
    assert first["cash_mutations"] == second["cash_mutations"] == 0
    with sqlite3.connect(db) as con:
        row = con.execute(
            "SELECT account,market,ticker,ex_date,gross_amount,currency,status "
            "FROM cash_dividend_review_queue"
        ).fetchone()
    assert row == ("A01", "US", "AAA", "2026-08-20", 2.5, "USD", "pending_review")


def test_review_resync_does_not_change_approved_amount(tmp_path):
    from scripts.sync_cash_dividend_reviews import CONFIRM, sync_reviews

    csv_path = _csv(tmp_path / "affected.csv")
    db = tmp_path / "trading.db"
    sync_reviews(csv_path=csv_path, db=db, apply=True, confirm=CONFIRM)
    with sqlite3.connect(db) as con:
        con.execute(
            "UPDATE cash_dividend_review_queue SET status='approved',gross_amount=2.0"
        )
    csv_path.write_text(csv_path.read_text().replace(",10,True,10,", ",12,True,12,"))

    sync_reviews(csv_path=csv_path, db=db, apply=True, confirm=CONFIRM)

    with sqlite3.connect(db) as con:
        assert con.execute(
            "SELECT status,gross_amount FROM cash_dividend_review_queue"
        ).fetchone() == ("approved", 2.0)


def test_dividend_queue_health_requires_both_markets_and_reports_pending(tmp_path):
    from datetime import datetime, timezone
    from scripts.health_check import check_cash_dividend_review_health
    from scripts.record_operational_health import record_operational_health
    from scripts.sync_cash_dividend_reviews import CONFIRM, sync_reviews

    csv_path = _csv(tmp_path / "affected.csv")
    db = tmp_path / "trading.db"
    sync_reviews(csv_path=csv_path, db=db, apply=True, confirm=CONFIRM)
    for market in ("US", "CN"):
        record_operational_health(
            db=str(db), component="cash_dividend_review_queue", market=market,
            status="ok", scheduled_at="2026-08-25T11:15:00Z",
            started_at="2026-08-25T11:15:00Z", stopped_at="2026-08-25T11:15:01Z",
            exit_code=0, duration=1,
        )
    with sqlite3.connect(db) as con:
        con.row_factory = sqlite3.Row
        issues = check_cash_dividend_review_health(
            con, now=datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc)
        )

    assert not [i for i in issues if i["severity"] == "critical"]
    assert issues == [{
        "severity": "warning", "check": "cash_dividend_pending_review",
        "market": "US", "scope": "US", "count": 1,
        "gross_amount": 2.5, "cash_credited": False,
    }]
