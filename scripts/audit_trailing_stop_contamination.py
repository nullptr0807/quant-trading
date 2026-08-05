#!/usr/bin/env python3
"""Counterfactual audit of historical US trailing-stop fills.

Default mode opens SQLite read-only. ``--apply`` only adds independently keyed
quarantine records and audit events; it never updates/deletes trades or fills.
"""
from __future__ import annotations

import argparse
import json
import os

import sqlite3
from datetime import datetime, time, timezone
from pathlib import Path

from config import settings
from trading.risk_regime import _portfolio_drawdown

CONFIRMATION = "QUARANTINE TRAILING STOP AUDIT"


def _parse_ts(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _counterfactual_state(conn: sqlite3.Connection, at: datetime) -> tuple[str, float]:
    dates = [
        row[0] for row in conn.execute(
            "SELECT DISTINCT date(timestamp) FROM accounts "
            "WHERE UPPER(market)='US' AND timestamp<=? ORDER BY 1", (at.isoformat(),)
        )
    ]
    state, streak, last_dd = "DISARMED", 0, 0.0
    for day in dates:
        day_end = datetime.combine(
            datetime.fromisoformat(day).date(), time.max, tzinfo=timezone.utc
        )
        check_at = min(day_end, at)
        last_dd = _portfolio_drawdown(conn, market="US", as_of=check_at)
        if state == "DISARMED" and last_dd >= settings.AUTO_ARM_DD_PCT:
            state, streak = "ARMED", 0
        elif state == "ARMED":
            if last_dd <= settings.AUTO_DISARM_DD_PCT:
                streak += 1
                if streak >= settings.AUTO_DISARM_DAYS:
                    state, streak = "DISARMED", 0
            else:
                streak = 0
    return state, last_dd


def classify(db_path: str) -> list[dict]:
    uri = f"file:{os.path.abspath(db_path)}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        candidates = []
        for event in conn.execute(
            "SELECT id,ts,account,ticker,detail FROM events "
            "WHERE category='trade' AND UPPER(market)='US' ORDER BY ts,id"
        ):
            try:
                detail = json.loads(event["detail"] or "{}")
            except (TypeError, json.JSONDecodeError):
                continue
            if detail.get("reason") != "trailing_stop":
                continue
            trade = conn.execute(
                "SELECT id FROM trades WHERE account=? AND ticker=? AND side='sell' "
                "AND timestamp=? AND UPPER(market)='US' ORDER BY id LIMIT 1",
                (event["account"], event["ticker"], event["ts"]),
            ).fetchone()
            if not trade:
                continue
            state, drawdown = _counterfactual_state(conn, _parse_ts(event["ts"]))
            candidates.append({
                "trade_id": trade["id"], "event_id": event["id"],
                "timestamp": event["ts"], "account": event["account"],
                "ticker": event["ticker"], "counterfactual_state": state,
                "us_only_drawdown": drawdown, "polluted": state != "ARMED",
            })
        return candidates
    finally:
        conn.close()


def apply_quarantine(
    db_path: str, rows: list[dict], *, backup_path: str, confirmation: str,
) -> int:
    if confirmation != CONFIRMATION:
        raise ValueError(f"confirmation must exactly equal: {CONFIRMATION}")
    source = Path(db_path).resolve()
    backup = Path(backup_path).resolve()
    if source == backup:
        raise ValueError("backup path must differ from database")
    if backup.exists():
        raise FileExistsError(f"refusing to overwrite backup: {backup}")
    backup.parent.mkdir(parents=True, exist_ok=True)
    # SQLite's online backup API includes committed WAL pages; a plain file copy
    # can silently omit them even while the writer is paused between commits.
    source_conn = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    backup_conn = sqlite3.connect(backup)
    try:
        source_conn.backup(backup_conn)
    finally:
        backup_conn.close()
        source_conn.close()

    conn = sqlite3.connect(source)
    inserted = 0
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trailing_stop_quarantine (
                trade_id INTEGER PRIMARY KEY,
                source_event_id INTEGER NOT NULL,
                market TEXT NOT NULL CHECK(market='US'),
                classification TEXT NOT NULL,
                counterfactual_state TEXT NOT NULL,
                us_only_drawdown REAL NOT NULL,
                audited_at TEXT NOT NULL,
                audit_version INTEGER NOT NULL DEFAULT 1
            )
        """)
        now = datetime.now(timezone.utc).isoformat()
        for row in rows:
            if not row["polluted"]:
                continue
            cursor = conn.execute(
                "INSERT OR IGNORE INTO trailing_stop_quarantine "
                "(trade_id,source_event_id,market,classification,counterfactual_state,"
                "us_only_drawdown,audited_at) VALUES(?,?,'US','mixed_market_risk_regime',?,?,?)",
                (row["trade_id"], row["event_id"], row["counterfactual_state"],
                 row["us_only_drawdown"], now),
            )
            if cursor.rowcount != 1:
                continue
            inserted += 1
            conn.execute(
                "INSERT INTO events(ts,category,severity,account,ticker,title,detail,market) "
                "VALUES(?,'audit','warning',?,?,?,?,'US')",
                (now, row["account"], row["ticker"],
                 f"Quarantined trailing-stop trade #{row['trade_id']} for analysis",
                 json.dumps(row, sort_keys=True)),
            )
        conn.commit()
        return inserted
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup")
    parser.add_argument("--confirm")
    args = parser.parse_args()
    rows = classify(args.db)
    print(json.dumps({"rows": rows, "polluted": sum(r["polluted"] for r in rows)}, indent=2))
    if args.apply:
        if not args.backup:
            parser.error("--apply requires --backup")
        inserted = apply_quarantine(
            args.db, rows, backup_path=args.backup, confirmation=args.confirm or ""
        )
        print(f"quarantined={inserted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
