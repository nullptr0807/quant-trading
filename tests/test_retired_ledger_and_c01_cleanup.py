from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEDGER_SCRIPT = PROJECT_ROOT / "scripts" / "ledger_watchdog.py"


def _make_watchdog_db(path: Path) -> None:
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE account_meta (
            account_id TEXT PRIMARY KEY,
            market TEXT NOT NULL,
            status TEXT,
            initial_cash REAL,
            retired_at TEXT
        );
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY,
            account TEXT,
            market TEXT,
            timestamp TEXT,
            ticker TEXT,
            side TEXT,
            shares REAL,
            price REAL,
            cost REAL,
            slippage REAL
        );
        CREATE TABLE accounts (
            id INTEGER PRIMARY KEY,
            name TEXT,
            market TEXT,
            cash REAL,
            equity REAL,
            timestamp TEXT
        );
        CREATE TABLE positions (
            account TEXT,
            market TEXT,
            ticker TEXT,
            shares REAL,
            total_cost REAL,
            current_price REAL,
            updated_at TEXT
        );
        CREATE TABLE account_state (
            account TEXT,
            market TEXT,
            cash REAL
        );
        CREATE TABLE events (
            id INTEGER PRIMARY KEY,
            ts TEXT,
            category TEXT,
            severity TEXT,
            account TEXT,
            ticker TEXT,
            title TEXT,
            detail TEXT,
            market TEXT
        );
        """
    )
    con.executemany(
        "INSERT INTO account_meta VALUES (?,?,?,?,?)",
        [
            ("A01", "US", "active", 10000.0, None),
            ("B03", "US", "retired", 10000.0, "2026-07-10T00:00:00+00:00"),
        ],
    )
    # Deliberately dirty frozen history: this must remain visible when explicitly
    # requested, but it must never make the operational/daily check red.
    con.execute(
        "INSERT INTO trades VALUES (1,'B03','US','2026-07-09T12:00:00+00:00',"
        "'AAPL','sell',1,100,0,0)"
    )
    con.commit()
    con.close()


def _run_watchdog(db: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(LEDGER_SCRIPT),
            "--db",
            str(db),
            "--market",
            "US",
            "--date",
            "2026-07-09",
            "--history-days",
            "0",
            "--quiet-ok",
            "--fail-on-critical",
            *extra,
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_watchdog_defaults_to_active_and_retired_findings_are_archival(tmp_path):
    db = tmp_path / "watchdog.db"
    _make_watchdog_db(db)

    operational = _run_watchdog(db)
    assert operational.returncode == 0, operational.stdout + operational.stderr
    assert operational.stdout == ""

    archival = _run_watchdog(db, "--history-include-retired")
    assert archival.returncode == 0, archival.stdout + archival.stderr
    assert "B03" in archival.stdout
    assert "Archival findings" in archival.stdout
    assert "non-operational" in archival.stdout
    assert "Operational issues: critical=0" in archival.stdout


def test_load_accounts_can_explicitly_include_retired():
    from scripts.ledger_watchdog import load_accounts

    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.execute(
        "CREATE TABLE account_meta (account_id TEXT, market TEXT, status TEXT)"
    )
    con.executemany(
        "INSERT INTO account_meta VALUES (?,?,?)",
        [
            ("A01", "US", "active"),
            ("A02", "US", "retired"),
            ("CA01", "CN", None),
        ],
    )

    assert [r["account_id"] for r in load_accounts(con, "ALL")] == ["CA01", "A01"]
    assert [
        r["account_id"]
        for r in load_accounts(con, "ALL", include_retired=True)
    ] == ["CA01", "A01", "A02"]


def _make_cleanup_db(path: Path, *, with_activity: bool = False) -> None:
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE account_meta (
            account_id TEXT PRIMARY KEY,
            strategy_name TEXT,
            status TEXT,
            market TEXT NOT NULL,
            retired_at TEXT,
            retire_reason TEXT
        );
        CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            category TEXT NOT NULL,
            severity TEXT NOT NULL,
            account TEXT,
            ticker TEXT,
            title TEXT NOT NULL,
            detail TEXT,
            market TEXT NOT NULL
        );
        CREATE TABLE account_state (account TEXT, market TEXT);
        CREATE TABLE positions (account TEXT, market TEXT);
        CREATE TABLE trades (account TEXT, market TEXT);
        CREATE TABLE accounts (name TEXT, market TEXT);
        CREATE TABLE positions_history (account TEXT, market TEXT);
        """
    )
    con.execute(
        "INSERT INTO account_meta VALUES ('C01','测试策略','active','US',NULL,NULL)"
    )
    if with_activity:
        con.execute("INSERT INTO trades VALUES ('C01','US')")
    con.commit()
    con.close()


def _backup_tables(con: sqlite3.Connection) -> list[str]:
    return [
        r[0]
        for r in con.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name LIKE 'account_meta_backup_c01_cleanup_%' "
            "ORDER BY name"
        ).fetchall()
    ]


def test_c01_cleanup_dry_run_is_read_only(tmp_path):
    from scripts.cleanup_c01_placeholder import cleanup_c01

    db = tmp_path / "trading.db"
    _make_cleanup_db(db)
    result = cleanup_c01(
        db,
        apply=False,
        now=datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc),
    )

    con = sqlite3.connect(db)
    assert con.execute(
        "SELECT status,retired_at,retire_reason FROM account_meta WHERE account_id='C01'"
    ).fetchone() == ("active", None, None)
    assert con.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
    assert _backup_tables(con) == []
    con.close()
    assert list(tmp_path.glob("trading.db.bak_c01_cleanup_*")) == []
    assert result["mode"] == "dry-run"
    assert result["would_change"] is True


def test_c01_cleanup_apply_backs_up_then_retires_and_is_idempotent(tmp_path):
    from scripts.cleanup_c01_placeholder import REASON, cleanup_c01

    db = tmp_path / "trading.db"
    _make_cleanup_db(db)
    first_now = datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc)
    first = cleanup_c01(db, apply=True, now=first_now)

    assert first["changed"] is True
    backup_path = Path(first["database_backup"])
    assert backup_path.exists()

    # The file backup is a pre-mutation snapshot.
    backup_con = sqlite3.connect(backup_path)
    assert backup_con.execute(
        "SELECT status,retired_at,retire_reason FROM account_meta WHERE account_id='C01'"
    ).fetchone() == ("active", None, None)
    assert _backup_tables(backup_con) == []
    backup_con.close()

    con = sqlite3.connect(db)
    row = con.execute(
        "SELECT status,retired_at,retire_reason FROM account_meta WHERE account_id='C01'"
    ).fetchone()
    assert row == ("retired", first_now.isoformat(), REASON)
    tables = _backup_tables(con)
    assert tables == [first["account_meta_backup_table"]]
    assert con.execute(
        f'SELECT status,retired_at,retire_reason FROM "{tables[0]}" WHERE account_id=\'C01\''
    ).fetchone() == ("active", None, None)
    event = con.execute(
        "SELECT category,severity,account,title,detail,market FROM events WHERE account='C01'"
    ).fetchone()
    assert event[:3] == ("lifecycle", "warning", "C01")
    assert event[5] == "US"
    assert json.loads(event[4])["reason"] == REASON
    con.close()

    before_backups = sorted(tmp_path.glob("trading.db.bak_c01_cleanup_*"))
    second = cleanup_c01(
        db,
        apply=True,
        now=datetime(2026, 7, 10, 13, 0, tzinfo=timezone.utc),
    )
    assert second["changed"] is False
    assert second["already_clean"] is True
    assert sorted(tmp_path.glob("trading.db.bak_c01_cleanup_*")) == before_backups

    con = sqlite3.connect(db)
    assert _backup_tables(con) == tables
    assert con.execute("SELECT COUNT(*) FROM events WHERE account='C01'").fetchone()[0] == 1
    assert con.execute(
        "SELECT retired_at FROM account_meta WHERE account_id='C01'"
    ).fetchone()[0] == first_now.isoformat()
    con.close()


def test_c01_cleanup_refuses_non_inert_account(tmp_path):
    from scripts.cleanup_c01_placeholder import cleanup_c01

    db = tmp_path / "trading.db"
    _make_cleanup_db(db, with_activity=True)

    with pytest.raises(RuntimeError, match="activity"):
        cleanup_c01(
            db,
            apply=True,
            now=datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc),
        )

    assert list(tmp_path.glob("trading.db.bak_c01_cleanup_*")) == []
    con = sqlite3.connect(db)
    assert con.execute(
        "SELECT status FROM account_meta WHERE account_id='C01'"
    ).fetchone()[0] == "active"
    assert _backup_tables(con) == []
    assert con.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
    con.close()
