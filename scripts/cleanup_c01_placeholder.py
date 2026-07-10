#!/usr/bin/env python3
"""Safely retire the inert legacy C01 test metadata row.

Dry-run by default. --apply requires C01 to have no state, positions, trades,
snapshots, events, or position history. It creates both a pre-mutation SQLite
backup and an in-database account_meta backup table before changing status.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

REASON = "inert legacy test placeholder; no operational ledger activity"
_ACTIVITY_TABLES = (
    ("account_state", "account"),
    ("positions", "account"),
    ("trades", "account"),
    ("accounts", "name"),
    ("positions_history", "account"),
)


def _stamp(now: datetime) -> str:
    return now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def cleanup_c01(
    db_path: str | Path,
    *,
    apply: bool = False,
    now: datetime | None = None,
) -> dict:
    db = Path(db_path).expanduser().resolve()
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    try:
        row = con.execute(
            "SELECT * FROM account_meta WHERE account_id='C01' AND market='US'"
        ).fetchone()
        if row is None:
            return {"mode": "apply" if apply else "dry-run", "would_change": False, "changed": False, "already_clean": True}
        if (row["status"] or "active") == "retired":
            return {"mode": "apply" if apply else "dry-run", "would_change": False, "changed": False, "already_clean": True}

        activity = {}
        for table, column in _ACTIVITY_TABLES:
            try:
                activity[table] = int(con.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE {column}='C01' AND market='US'"
                ).fetchone()[0])
            except sqlite3.OperationalError:
                activity[table] = 0
        try:
            activity["events"] = int(con.execute(
                "SELECT COUNT(*) FROM events WHERE account='C01' AND market='US'"
            ).fetchone()[0])
        except sqlite3.OperationalError:
            activity["events"] = 0
        if any(activity.values()):
            raise RuntimeError(f"C01 has operational activity and cannot be auto-retired: {activity}")

        if not apply:
            return {
                "mode": "dry-run",
                "would_change": True,
                "changed": False,
                "account": "C01",
                "activity": activity,
                "reason": REASON,
            }
    finally:
        con.close()

    stamp = _stamp(now)
    backup_path = db.with_name(f"{db.name}.bak_c01_cleanup_{stamp}")
    shutil.copy2(db, backup_path)
    backup_table = f"account_meta_backup_c01_cleanup_{stamp}"

    con = sqlite3.connect(db)
    try:
        con.execute("BEGIN IMMEDIATE")
        con.execute(
            f'CREATE TABLE "{backup_table}" AS '
            "SELECT * FROM account_meta WHERE account_id='C01' AND market='US'"
        )
        con.execute(
            "UPDATE account_meta SET status='retired',retired_at=?,retire_reason=? "
            "WHERE account_id='C01' AND market='US' AND COALESCE(status,'active')='active'",
            (now.isoformat(), REASON),
        )
        if con.total_changes != 1:
            raise RuntimeError("C01 cleanup expected exactly one metadata update")
        detail = json.dumps({
            "reason": REASON,
            "database_backup": str(backup_path),
            "account_meta_backup_table": backup_table,
        }, ensure_ascii=False)
        con.execute(
            "INSERT INTO events (ts,category,severity,account,ticker,title,detail,market) "
            "VALUES (?,'lifecycle','warning','C01',NULL,'🟡 C01 legacy test placeholder retired',?,'US')",
            (now.isoformat(), detail),
        )
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
    return {
        "mode": "apply",
        "would_change": True,
        "changed": True,
        "already_clean": False,
        "database_backup": str(backup_path),
        "account_meta_backup_table": backup_table,
        "reason": REASON,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / "quant-trading/data/trading.db"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(cleanup_c01(args.db, apply=args.apply), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
