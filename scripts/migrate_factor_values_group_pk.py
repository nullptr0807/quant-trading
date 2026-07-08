#!/usr/bin/env python3
"""Migrate factor_values PK to include factor_group.

Old schema PRIMARY KEY(ticker,date,factor_name) made GP/F/Q groups overwrite each
other whenever factor_name collided. New schema uses
PRIMARY KEY(ticker,date,factor_name,factor_group).

Idempotent and non-destructive: creates a DB file backup and a table backup first.
"""
from __future__ import annotations

import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "trading.db"


def _pk_cols(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [r[1] for r in sorted((r for r in rows if r[5]), key=lambda r: r[5])]


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    file_backup = DB_PATH.with_name(f"trading.db.bak_factor_values_pk_{stamp}")
    shutil.copy2(DB_PATH, file_backup)
    print(f"db_backup={file_backup}")

    conn = sqlite3.connect(DB_PATH, timeout=60)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=60000")
        pk = _pk_cols(conn, "factor_values")
        print(f"current_pk={pk}")
        if pk == ["ticker", "date", "factor_name", "factor_group"]:
            print("already_migrated=true")
            return 0
        if pk != ["ticker", "date", "factor_name"]:
            raise RuntimeError(f"unexpected factor_values PK: {pk}")

        table_backup = f"factor_values_backup_group_pk_{stamp}"
        conn.execute("BEGIN")
        conn.execute(f"CREATE TABLE {table_backup} AS SELECT * FROM factor_values")
        conn.execute("""CREATE TABLE factor_values_new (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            factor_name TEXT NOT NULL,
            value REAL,
            factor_group TEXT NOT NULL DEFAULT 'alpha158',
            PRIMARY KEY (ticker, date, factor_name, factor_group)
        )""")
        conn.execute("""INSERT OR REPLACE INTO factor_values_new
            (ticker, date, factor_name, value, factor_group)
            SELECT ticker, date, factor_name, value, COALESCE(factor_group, 'alpha158')
            FROM factor_values
        """)
        old_count = conn.execute(f"SELECT COUNT(*) FROM {table_backup}").fetchone()[0]
        new_count = conn.execute("SELECT COUNT(*) FROM factor_values_new").fetchone()[0]
        conn.execute("DROP TABLE factor_values")
        conn.execute("ALTER TABLE factor_values_new RENAME TO factor_values")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fv_date ON factor_values(date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fv_group_name_date_ticker ON factor_values(factor_group, factor_name, date, ticker)")
        conn.commit()
        print(f"table_backup={table_backup}")
        print(f"rows_old={old_count} rows_new={new_count}")
        print(f"new_pk={_pk_cols(conn, 'factor_values')}")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
