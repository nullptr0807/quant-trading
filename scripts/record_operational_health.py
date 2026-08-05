#!/usr/bin/env python3
"""Small, dependency-free scheduler/alert observability writer."""
from __future__ import annotations
import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=os.environ.get("QUANT_DB_PATH", "data/trading.db"))
    p.add_argument("--component", required=True)
    p.add_argument("--market", default="ALL")
    p.add_argument("--status", required=True)
    p.add_argument("--scheduled-at", required=True)
    p.add_argument("--started-at")
    p.add_argument("--stopped-at")
    p.add_argument("--exit-code", type=int)
    p.add_argument("--duration", type=float)
    p.add_argument("--detail")
    args = p.parse_args()
    now = datetime.now(timezone.utc).isoformat()
    detail = json.loads(args.detail) if args.detail else {}
    con = sqlite3.connect(args.db, timeout=30)
    con.execute("PRAGMA busy_timeout=30000")
    con.execute("""CREATE TABLE IF NOT EXISTS scheduler_runs (
      id INTEGER PRIMARY KEY AUTOINCREMENT, component TEXT NOT NULL, market TEXT NOT NULL,
      scheduled_at TEXT NOT NULL, actual_start TEXT, stopped_at TEXT, exit_code INTEGER,
      duration_seconds REAL, status TEXT NOT NULL, detail TEXT, recorded_at TEXT NOT NULL)""")
    con.execute("""CREATE TABLE IF NOT EXISTS operational_health (
      component TEXT NOT NULL, market TEXT NOT NULL, status TEXT NOT NULL,
      success_at TEXT NOT NULL, source_timestamp TEXT, details TEXT,
      PRIMARY KEY(component,market))""")
    row = con.execute(
        "SELECT success_at FROM operational_health WHERE component=? AND market=?",
        (args.component, args.market),
    ).fetchone()
    success_at = (args.stopped_at or now) if args.status == "ok" else (row[0] if row else args.scheduled_at)
    stop_scan = None
    if args.component == "update_prices" and args.status == "ok":
        # A successful updater run completed the protective-sell scan even when
        # no stop fired; event timestamps only exist when a trade occurred.
        stop_scan = args.stopped_at or now
    detail.update({
        "scheduled_at": args.scheduled_at, "actual_start": args.started_at,
        "stopped_at": args.stopped_at, "exit_code": args.exit_code,
        "duration_seconds": args.duration, "last_successful_stop_scan": stop_scan,
    })
    packed = json.dumps(detail, ensure_ascii=False, sort_keys=True)
    con.execute(
        "INSERT INTO scheduler_runs(component,market,scheduled_at,actual_start,stopped_at,exit_code,duration_seconds,status,detail,recorded_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (args.component,args.market,args.scheduled_at,args.started_at,args.stopped_at,args.exit_code,args.duration,args.status,packed,now),
    )
    con.execute(
        "INSERT OR REPLACE INTO operational_health(component,market,status,success_at,source_timestamp,details) VALUES (?,?,?,?,?,?)",
        (args.component,args.market,args.status,success_at,args.stopped_at or args.started_at,packed),
    )
    con.commit(); con.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
