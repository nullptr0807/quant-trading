#!/usr/bin/env python3
"""Repair CN ticker trades that were historically stored with market='US'.

This script is intentionally narrow and auditable:
- It only touches rows where ticker has .SH/.SZ/.BJ suffix and trades.market='US'.
- It only updates the market column to 'CN'.
- It creates backup tables first.
- It also fixes matching trade events if any exist.

It does NOT change cash, positions, shares, prices, or accounts snapshots.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path.home() / "quant-trading" / "data" / "trading.db"
PRED = "ticker LIKE '%.SH' OR ticker LIKE '%.SZ' OR ticker LIKE '%.BJ'"


def q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def main() -> int:
    ap = argparse.ArgumentParser(description="Repair CN ticker trades mislabeled as US")
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--yes", action="store_true", help="actually apply repair; default is dry-run")
    args = ap.parse_args()

    db = Path(args.db).expanduser()
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            f"SELECT * FROM trades WHERE market='US' AND ({PRED}) ORDER BY account,timestamp,id"
        ).fetchall()
        event_rows = con.execute(
            f"SELECT * FROM events WHERE market='US' AND ({PRED}) ORDER BY account,ts,id"
        ).fetchall()
        summary = {
            "db": str(db),
            "dry_run": not args.yes,
            "mislabeled_trade_rows": len(rows),
            "mislabeled_event_rows": len(event_rows),
            "accounts": {},
        }
        for r in rows:
            acct = r["account"]
            info = summary["accounts"].setdefault(acct, {"trades": 0, "min_ts": r["timestamp"], "max_ts": r["timestamp"]})
            info["trades"] += 1
            info["min_ts"] = min(info["min_ts"], r["timestamp"])
            info["max_ts"] = max(info["max_ts"], r["timestamp"])
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if not args.yes:
            return 0
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        b_trades = f"trades_backup_market_mislabel_{stamp}"
        b_events = f"events_backup_market_mislabel_{stamp}"
        con.execute("BEGIN")
        con.execute(
            f"CREATE TABLE {q(b_trades)} AS SELECT * FROM trades WHERE market='US' AND ({PRED})"
        )
        con.execute(
            f"CREATE TABLE {q(b_events)} AS SELECT * FROM events WHERE market='US' AND ({PRED})"
        )
        con.execute(f"UPDATE trades SET market='CN' WHERE market='US' AND ({PRED})")
        trade_updated = con.total_changes
        # total_changes includes CREATE TABLE row inserts, so measure event rowcount explicitly.
        cur = con.execute(f"UPDATE events SET market='CN' WHERE market='US' AND ({PRED})")
        event_updated = cur.rowcount
        # Emit repair marker.
        detail = {
            "repair": "cn_ticker_market_mislabel",
            "backup_trades": b_trades,
            "backup_events": b_events,
            "trade_rows": len(rows),
            "event_rows": len(event_rows),
            "accounts": summary["accounts"],
        }
        con.execute(
            "INSERT INTO events (ts, category, severity, account, ticker, title, detail, market) "
            "VALUES (?, 'system', 'warn', NULL, NULL, ?, ?, 'CN')",
            (
                datetime.now(timezone.utc).isoformat(),
                f"🧹 Repaired {len(rows)} CN ticker trade rows mislabeled as US",
                json.dumps(detail, ensure_ascii=False),
            ),
        )
        con.commit()
        print(json.dumps({
            "applied": True,
            "trade_rows": len(rows),
            "event_rows": len(event_rows),
            "backup_trades": b_trades,
            "backup_events": b_events,
            "event_update_rowcount": event_updated,
        }, ensure_ascii=False, indent=2))
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
