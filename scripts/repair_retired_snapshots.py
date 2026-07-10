#!/usr/bin/env python3
"""Repair frozen retired-account terminal snapshots from durable frozen state."""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

TARGETS = ("B03", "B07", "B09")
REASON = "retired frozen terminal snapshot reconciled to account_state cash plus positions current_price"


def repair(db_path: str | Path, *, apply: bool = False, now: datetime | None = None) -> dict:
    db = Path(db_path).resolve()
    now = now or datetime.now(timezone.utc)
    stamp = now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    repairs = []
    for account in TARGETS:
        meta = con.execute(
            "SELECT status,retired_at FROM account_meta WHERE account_id=? AND market='US'",
            (account,),
        ).fetchone()
        if not meta or meta["status"] != "retired":
            raise RuntimeError(f"{account} is not retired")
        after = con.execute(
            "SELECT COUNT(*) FROM trades WHERE account=? AND market='US' AND timestamp>?",
            (account, meta["retired_at"]),
        ).fetchone()[0]
        if after:
            raise RuntimeError(f"{account} has {after} post-retirement trades")
        state = con.execute(
            "SELECT cash FROM account_state WHERE account=? AND market='US'",
            (account,),
        ).fetchone()
        latest = con.execute(
            "SELECT id,cash,equity,timestamp FROM accounts WHERE name=? AND market='US' "
            "ORDER BY timestamp DESC,id DESC LIMIT 1",
            (account,),
        ).fetchone()
        if not state or not latest:
            raise RuntimeError(f"{account} missing durable state/snapshot")
        missing = con.execute(
            "SELECT COUNT(*) FROM positions WHERE account=? AND market='US' "
            "AND shares>0 AND (current_price IS NULL OR current_price<=0)",
            (account,),
        ).fetchone()[0]
        if missing:
            raise RuntimeError(f"{account} has {missing} unpriced positions")
        mv = con.execute(
            "SELECT COALESCE(SUM(shares*current_price),0) FROM positions "
            "WHERE account=? AND market='US' AND shares>0",
            (account,),
        ).fetchone()[0]
        computed = float(state["cash"]) + float(mv or 0)
        repairs.append({
            "account": account,
            "latest_id": int(latest["id"]),
            "old_cash": float(latest["cash"]),
            "old_equity": float(latest["equity"]),
            "new_cash": float(state["cash"]),
            "new_equity": computed,
        })
    con.close()
    if not apply:
        return {"mode": "dry-run", "repairs": repairs}

    backup = db.with_name(f"{db.name}.bak_retired_snapshot_{stamp}")
    shutil.copy2(db, backup)
    table = f"accounts_backup_retired_snapshot_{stamp}"
    con = sqlite3.connect(db)
    try:
        con.execute("BEGIN IMMEDIATE")
        placeholders = ",".join("?" * len(TARGETS))
        con.execute(
            f'CREATE TABLE "{table}" AS SELECT * FROM accounts '
            f"WHERE name IN ({placeholders}) AND market='US'",
            TARGETS,
        )
        for item in repairs:
            con.execute(
                "UPDATE accounts SET cash=?,equity=? WHERE id=?",
                (item["new_cash"], item["new_equity"], item["latest_id"]),
            )
            con.execute(
                "INSERT INTO events (ts,category,severity,account,ticker,title,detail,market) "
                "VALUES (?,'repair','warning',?,NULL,?,?, 'US')",
                (
                    now.isoformat(), item["account"],
                    f"🧹 {item['account']} retired terminal snapshot reconciled",
                    json.dumps({**item, "reason": REASON, "backup": str(backup), "backup_table": table}),
                ),
            )
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
    return {"mode": "apply", "backup": str(backup), "backup_table": table, "repairs": repairs}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(Path.home() / "quant-trading/data/trading.db"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(repair(args.db, apply=args.apply), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
