#!/usr/bin/env python3
"""Small, dependency-free scheduler/alert observability writer."""
from __future__ import annotations
import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone


def record_operational_health(
    *,
    db: str,
    component: str,
    market: str,
    status: str,
    scheduled_at: str,
    started_at: str | None = None,
    stopped_at: str | None = None,
    exit_code: int | None = None,
    duration: float | None = None,
    detail: dict | None = None,
    source_timestamp: str | None = None,
    now: str | None = None,
) -> None:
    """Persist one scheduler attempt and its latest component health."""
    recorded_at = now or datetime.now(timezone.utc).isoformat()
    detail = dict(detail or {})
    con = sqlite3.connect(db, timeout=30)
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
        (component, market),
    ).fetchone()
    last_success_at = row[0] if row and row[0] else None
    success_at = (
        (stopped_at or recorded_at)
        if status == "ok"
        else (last_success_at or "")
    )
    stop_scan = None
    if component == "update_prices" and status == "ok":
        # A successful updater run completed the protective-sell scan even when
        # no stop fired; event timestamps only exist when a trade occurred.
        stop_scan = stopped_at or recorded_at
    detail.update({
        "scheduled_at": scheduled_at, "actual_start": started_at,
        "stopped_at": stopped_at, "attempt_stopped_at": stopped_at,
        "exit_code": exit_code, "last_success_at": success_at or None,
        "duration_seconds": duration, "last_successful_stop_scan": stop_scan,
    })
    packed = json.dumps(detail, ensure_ascii=False, sort_keys=True)
    con.execute(
        "INSERT INTO scheduler_runs(component,market,scheduled_at,actual_start,stopped_at,exit_code,duration_seconds,status,detail,recorded_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (component, market, scheduled_at, started_at, stopped_at, exit_code,
         duration, status, packed, recorded_at),
    )
    con.execute(
        "INSERT OR REPLACE INTO operational_health(component,market,status,success_at,source_timestamp,details) VALUES (?,?,?,?,?,?)",
        (
            component, market, status, success_at,
            source_timestamp or stopped_at or started_at, packed,
        ),
    )
    con.commit()
    con.close()


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
    p.add_argument("--source-timestamp")
    args = p.parse_args()
    detail = json.loads(args.detail) if args.detail else {}
    record_operational_health(
        db=args.db, component=args.component, market=args.market,
        status=args.status, scheduled_at=args.scheduled_at,
        started_at=args.started_at, stopped_at=args.stopped_at,
        exit_code=args.exit_code, duration=args.duration, detail=detail,
        source_timestamp=args.source_timestamp,
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
