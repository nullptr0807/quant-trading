from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest


def _make_db(path: Path) -> None:
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account TEXT NOT NULL, ticker TEXT NOT NULL, side TEXT NOT NULL,
            shares REAL NOT NULL, price REAL NOT NULL, cost REAL NOT NULL,
            slippage REAL DEFAULT 0, timestamp TEXT NOT NULL, market TEXT NOT NULL
        );
        CREATE TABLE positions (
            account TEXT NOT NULL, ticker TEXT NOT NULL, shares REAL NOT NULL,
            avg_cost REAL NOT NULL, total_cost REAL NOT NULL,
            current_price REAL, updated_at TEXT, market TEXT NOT NULL,
            PRIMARY KEY(account,ticker,market)
        );
        CREATE TABLE account_state (
            account TEXT NOT NULL, cash REAL NOT NULL, initial_cash REAL NOT NULL,
            updated_at TEXT NOT NULL, market TEXT NOT NULL,
            PRIMARY KEY(account,market)
        );
        CREATE TABLE accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, cash REAL NOT NULL,
            equity REAL NOT NULL, timestamp TEXT NOT NULL, market TEXT NOT NULL
        );
        CREATE TABLE positions_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT, account TEXT NOT NULL, ticker TEXT NOT NULL,
            shares REAL NOT NULL, avg_cost REAL NOT NULL, market_price REAL,
            market_value REAL, unrealized_pnl REAL, timestamp TEXT NOT NULL, market TEXT NOT NULL
        );
        CREATE TABLE prices_raw (
            ticker TEXT NOT NULL, datetime TEXT NOT NULL, interval TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            PRIMARY KEY(ticker,datetime,interval)
        );
        CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, category TEXT NOT NULL,
            severity TEXT NOT NULL, account TEXT, ticker TEXT, title TEXT NOT NULL,
            detail TEXT, market TEXT NOT NULL
        );
        """
    )
    con.executemany(
        "INSERT INTO trades(account,ticker,side,shares,price,cost,timestamp,market) VALUES (?,?,?,?,?,?,?,?)",
        [
            ("A02", "AVB", "buy", 10.0, 180.0, 8.67, "2026-08-01T14:00:00+00:00", "US"),
            ("A02", "OLD", "buy", 1.0, 10.0, 0.0, "2025-01-01T14:00:00+00:00", "US"),
            ("A02", "OLD", "sell", 1.0, 11.0, 0.0, "2025-01-02T14:00:00+00:00", "US"),
            ("A02", "REC", "buy", 1.0, 12.0, 0.0, "2026-08-12T14:00:00+00:00", "US"),
            ("A02", "REC", "sell", 1.0, 13.0, 0.0, "2026-08-13T14:00:00+00:00", "US"),
            ("CA02", "AVB", "buy", 3.0, 20.0, 0.0, "2026-08-01T14:00:00+00:00", "CN"),
        ],
    )
    con.executemany(
        "INSERT INTO positions VALUES (?,?,?,?,?,?,?,?)",
        [
            ("A02", "AVB", 10.0, 180.867, 1808.67, 181.0, "2026-08-16T20:00:00+00:00", "US"),
            ("CA02", "AVB", 3.0, 20.0, 60.0, 21.0, "2026-08-16T20:00:00+00:00", "CN"),
        ],
    )
    con.executemany(
        "INSERT INTO account_state VALUES (?,?,?,?,?)",
        [
            ("A02", 1000.0, 10000.0, "2026-08-16T20:00:00+00:00", "US"),
            ("CA02", 500.0, 10000.0, "2026-08-16T20:00:00+00:00", "CN"),
        ],
    )
    con.execute(
        "INSERT INTO accounts(name,cash,equity,timestamp,market) VALUES (?,?,?,?,?)",
        ("A02", 1000.0, 2810.0, "2026-08-16T20:00:00+00:00", "US"),
    )
    con.executemany(
        "INSERT INTO prices_raw VALUES (?,?,?,?,?,?,?,?)",
        [
            ("AVB", "2026-08-16T00:00:00+00:00", "1d", 180, 182, 179, 181, 100),
            ("AVB", "2026-08-17T00:00:00+00:00", "1d", 64, 66, 63, 65, 300),
        ],
    )
    con.commit()
    con.close()


def _verified_backup(db: Path, directory: Path) -> Path:
    from scripts.backup_trading_db import create_backup

    result = create_backup(db, directory, keep=2)
    return Path(result["path"])


def test_fast_scope_prioritizes_open_then_recent_and_omits_old_closed(tmp_path):
    from scripts.audit_corporate_actions import load_priority_tickers

    db = tmp_path / "trading.db"
    _make_db(db)
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row

    rows = load_priority_tickers(
        con, {"US"}, None, recent_since="2026-08-10T00:00:00+00:00"
    )

    assert rows == [("US", "AVB"), ("US", "REC")]
    con.close()


def test_split_preview_is_read_only_and_reports_ratio_cost_and_raw_coordinate(tmp_path):
    from scripts.repair_open_position_split import repair_open_split

    db = tmp_path / "trading.db"
    _make_db(db)
    con = sqlite3.connect(db)
    con.execute(
        "UPDATE positions SET avg_cost=180.123 "
        "WHERE account='A02' AND market='US' AND ticker='AVB'"
    )
    con.commit()
    con.close()
    before = db.read_bytes()

    result = repair_open_split(
        db_path=db, account="A02", market="US", ticker="AVB",
        ex_date="2026-08-17", ratio=2.793, source="yfinance.splits",
        apply=False,
    )

    assert db.read_bytes() == before
    assert result["mode"] == "preview"
    assert result["before"]["shares"] == 10.0
    assert result["after"]["shares"] == pytest.approx(result["before"]["shares"] * 2.793)
    assert result["after"]["avg_cost"] == pytest.approx(180.123 / 2.793)
    assert result["after"]["total_cost"] == pytest.approx(1808.67)
    assert result["after"]["current_price"] == 65.0
    assert result["raw_price_timestamp"].startswith("2026-08-17")


def test_split_apply_requires_verified_backup(tmp_path):
    from scripts.repair_open_position_split import repair_open_split

    db = tmp_path / "trading.db"
    _make_db(db)

    with pytest.raises(RuntimeError, match="verified backup"):
        repair_open_split(
            db_path=db, account="A02", market="US", ticker="AVB",
            ex_date="2026-08-17", ratio=2.793, source="yfinance.splits",
            apply=True, backup_handle=None,
        )


def test_split_rejects_backup_manifest_from_a_different_canonical_source(tmp_path):
    from scripts.repair_open_position_split import repair_open_split

    db = tmp_path / "trading.db"
    other = tmp_path / "other.db"
    _make_db(db)
    _make_db(other)
    backup = _verified_backup(other, tmp_path / "backups")

    with pytest.raises(RuntimeError, match="canonical source"):
        repair_open_split(
            db_path=db, account="A02", market="US", ticker="AVB",
            ex_date="2026-08-17", ratio=2.793, source="yfinance.splits",
            apply=True, backup_handle=backup,
        )


def test_split_rejects_artifact_and_checksum_without_updated_manifest(tmp_path):
    from scripts.repair_open_position_split import repair_open_split

    db = tmp_path / "trading.db"
    _make_db(db)
    backup = _verified_backup(db, tmp_path / "backups")
    Path(str(backup) + ".manifest.json").unlink()

    with pytest.raises(RuntimeError, match="manifest"):
        repair_open_split(
            db_path=db, account="A02", market="US", ticker="AVB",
            ex_date="2026-08-17", ratio=2.793, source="yfinance.splits",
            apply=True, backup_handle=backup,
        )


def test_split_rejects_backup_when_unrelated_pre_state_changed(tmp_path):
    from scripts.repair_open_position_split import repair_open_split

    db = tmp_path / "trading.db"
    _make_db(db)
    backup = _verified_backup(db, tmp_path / "backups")
    con = sqlite3.connect(db)
    con.execute("UPDATE account_state SET cash=cash+1 WHERE account='CA02' AND market='CN'")
    con.commit()
    con.close()

    with pytest.raises(RuntimeError, match="full pre-state fingerprint"):
        repair_open_split(
            db_path=db, account="A02", market="US", ticker="AVB",
            ex_date="2026-08-17", ratio=2.793, source="yfinance.splits",
            apply=True, backup_handle=backup,
        )


@pytest.mark.parametrize("side", ["buy", "sell"])
def test_split_rejects_post_ex_date_trades(tmp_path, side):
    from scripts.repair_open_position_split import repair_open_split

    db = tmp_path / "trading.db"
    _make_db(db)
    con = sqlite3.connect(db)
    con.execute(
        "INSERT INTO trades(account,ticker,side,shares,price,cost,timestamp,market) "
        "VALUES ('A02','AVB',?,1,65,0,'2026-08-18T14:00:00+00:00','US')",
        (side,),
    )
    con.commit()
    con.close()

    with pytest.raises(RuntimeError, match="post-ex-date trade"):
        repair_open_split(
            db_path=db, account="A02", market="US", ticker="AVB",
            ex_date="2026-08-17", ratio=2.793, source="yfinance.splits",
        )


def test_split_rejects_position_not_equal_to_ex_date_entitlement(tmp_path):
    from scripts.repair_open_position_split import repair_open_split

    db = tmp_path / "trading.db"
    _make_db(db)
    con = sqlite3.connect(db)
    con.execute(
        "UPDATE positions SET shares=9,avg_cost=total_cost/9 "
        "WHERE account='A02' AND market='US' AND ticker='AVB'"
    )
    con.commit()
    con.close()

    with pytest.raises(RuntimeError, match="ex-date entitlement"):
        repair_open_split(
            db_path=db, account="A02", market="US", ticker="AVB",
            ex_date="2026-08-17", ratio=2.793, source="yfinance.splits",
        )


def test_audit_impact_uses_post_split_raw_price_not_stale_position_mark(tmp_path):
    from scripts.audit_corporate_actions import estimate_current_impact, post_action_raw_price

    db = tmp_path / "trading.db"
    _make_db(db)
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    pos = con.execute(
        "SELECT * FROM positions WHERE account='A02' AND market='US' AND ticker='AVB'"
    ).fetchone()

    raw_price = post_action_raw_price(con, "AVB", "2026-08-17")
    expected, price, impact = estimate_current_impact(pos, 2.793, raw_price)

    assert expected == pytest.approx(27.93)
    assert price == 65.0
    assert impact == pytest.approx((27.93 - 10.0) * 65.0)
    con.close()


def test_split_apply_is_atomic_audited_idempotent_and_market_isolated(tmp_path):
    from scripts.repair_open_position_split import repair_open_split

    db = tmp_path / "trading.db"
    _make_db(db)
    backup = _verified_backup(db, tmp_path / "backups")
    applied_at = datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc)

    first = repair_open_split(
        db_path=db, account="A02", market="US", ticker="AVB",
        ex_date="2026-08-17", ratio=2.793, source="yfinance.splits",
        raw={"date": "2026-08-17", "ratio": 2.793}, apply=True,
        backup_handle=backup, now=applied_at,
    )
    second = repair_open_split(
        db_path=db, account="A02", market="US", ticker="AVB",
        ex_date="2026-08-17", ratio=2.793, source="yfinance.splits",
        apply=True, backup_handle=backup,
        now=datetime(2026, 8, 25, 13, 0, tzinfo=timezone.utc),
    )

    assert first["changed"] is True
    assert second["changed"] is False
    assert second["already_applied"] is True

    con = sqlite3.connect(db)
    us = con.execute(
        "SELECT shares,avg_cost,total_cost,current_price FROM positions WHERE account='A02' AND market='US' AND ticker='AVB'"
    ).fetchone()
    assert us[0] == pytest.approx(27.93)
    assert us[1] == pytest.approx(1808.67 / 27.93)
    assert us[2] == pytest.approx(1808.67)
    assert us[3] == 65.0
    assert con.execute(
        "SELECT shares,avg_cost,total_cost,current_price FROM positions WHERE account='CA02' AND market='CN' AND ticker='AVB'"
    ).fetchone() == (3.0, 20.0, 60.0, 21.0)
    assert con.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 6
    assert con.execute("SELECT COUNT(*) FROM corporate_action_repairs").fetchone()[0] == 1
    audit = con.execute(
        "SELECT before_state,after_state,backup_handle,source FROM corporate_action_repairs"
    ).fetchone()
    assert json.loads(audit[0])["shares"] == 10.0
    assert json.loads(audit[1])["shares"] == pytest.approx(27.93)
    assert audit[2] == str(backup.resolve())
    assert audit[3] == "yfinance.splits"
    event = con.execute(
        "SELECT category,severity,account,ticker,market,detail FROM events WHERE category='corporate_action_repair'"
    ).fetchone()
    assert event[:5] == ("corporate_action_repair", "warning", "A02", "AVB", "US")
    assert json.loads(event[5])["ratio"] == 2.793
    history = con.execute(
        "SELECT shares,avg_cost,market_price,market_value,unrealized_pnl FROM positions_history WHERE account='A02' AND market='US'"
    ).fetchone()
    assert history[0] == pytest.approx(27.93)
    assert history[2] == 65.0
    assert history[3] == pytest.approx(27.93 * 65.0)
    latest = con.execute(
        "SELECT cash,equity,timestamp FROM accounts WHERE name='A02' AND market='US' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert latest[:2] == pytest.approx((1000.0, 1000.0 + 27.93 * 65.0))
    assert latest[2] == applied_at.isoformat()
    con.close()


def test_watchdog_replays_audited_split_in_current_and_historical_state(tmp_path):
    from scripts.ledger_watchdog import _replay_series, compare_positions, replay_account
    from scripts.repair_open_position_split import repair_open_split

    db = tmp_path / "trading.db"
    _make_db(db)
    con = sqlite3.connect(db)
    con.execute(
        "UPDATE positions SET total_cost=1801.92,avg_cost=180.192 "
        "WHERE account='A02' AND market='US' AND ticker='AVB'"
    )
    con.commit()
    con.close()
    backup = _verified_backup(db, tmp_path / "backups")
    applied_at = datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc)
    con = sqlite3.connect(db)
    trades_before = con.execute("SELECT * FROM trades ORDER BY id").fetchall()
    con.close()
    repair_open_split(
        db_path=db, account="A02", market="US", ticker="AVB",
        ex_date="2026-08-17", ratio=2.793, source="yfinance.splits",
        apply=True, backup_handle=backup, now=applied_at,
    )

    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    current = replay_account(
        con, "A02", "US", 10000.0,
        "2026-08-25T00:00:00+00:00", "2026-08-26T00:00:00+00:00",
    )
    account_rows = con.execute(
        "SELECT * FROM accounts WHERE name='A02' AND market='US' ORDER BY timestamp,id"
    ).fetchall()
    series = _replay_series(con, "A02", "US", 10000.0, account_rows)

    assert current.positions["AVB"]["shares"] == pytest.approx(27.93)
    assert current.positions["AVB"]["total_cost"] == pytest.approx(1801.92)
    assert series[0][2]["AVB"]["shares"] == pytest.approx(10.0)
    assert series[-1][2]["AVB"]["shares"] == pytest.approx(27.93)
    db_positions = con.execute(
        "SELECT * FROM positions WHERE account='A02' AND market='US'"
    ).fetchall()
    mismatches, _ = compare_positions(current.positions, db_positions, 1e-9, 1e-6)
    assert mismatches == []
    assert [tuple(row) for row in con.execute(
        "SELECT * FROM trades ORDER BY id"
    ).fetchall()] == trades_before
    con.close()


def test_completed_repair_is_recognized_by_subsequent_gate(tmp_path):
    from scripts.audit_corporate_actions import has_applied_share_repair
    from scripts.repair_open_position_split import repair_open_split

    db = tmp_path / "trading.db"
    _make_db(db)
    backup = _verified_backup(db, tmp_path / "backups")
    repair_open_split(
        db_path=db, account="A02", market="US", ticker="AVB",
        ex_date="2026-08-17", ratio=2.793, source="yfinance.splits",
        apply=True, backup_handle=backup,
    )
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row

    assert has_applied_share_repair(
        con, account="A02", market="US", ticker="AVB",
        ex_date="2026-08-17", action_type="split", ratio=2.793,
    ) is True
    assert has_applied_share_repair(
        con, account="A02", market="CN", ticker="AVB",
        ex_date="2026-08-17", action_type="split", ratio=2.793,
    ) is False
    con.close()


def test_split_rejects_missing_post_split_raw_price(tmp_path):
    from scripts.repair_open_position_split import repair_open_split

    db = tmp_path / "trading.db"
    _make_db(db)
    con = sqlite3.connect(db)
    con.execute("DELETE FROM prices_raw WHERE datetime >= '2026-08-17'")
    con.commit()
    con.close()

    with pytest.raises(RuntimeError, match="post-split raw price"):
        repair_open_split(
            db_path=db, account="A02", market="US", ticker="AVB",
            ex_date="2026-08-17", ratio=2.793, source="yfinance.splits",
            apply=False,
        )


def test_daily_wrapper_publishes_scheduler_health_and_keeps_full_scan_secondary():
    wrapper = (Path(__file__).parents[1] / "scripts" / "corporate_action_check_daily.sh").read_text()

    assert "run_scheduled_job.sh" in wrapper
    assert "--scope fast" in wrapper
    assert "--scope full" in wrapper
    assert "QUANT_CORPORATE_ACTION_FULL_SCAN" in wrapper
