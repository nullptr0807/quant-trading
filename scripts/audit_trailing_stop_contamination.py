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
from datetime import datetime, time, timedelta, timezone
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
    """Classify all US trailing stops in one chronological pass.

    The old implementation replayed the full 30-day snapshot window once per
    event (O(events × snapshots)). This version streams snapshots once, updates
    the intended daily state machine after each completed snapshot timestamp,
    and classifies events against the state available before that event.
    """
    absolute = os.path.abspath(db_path)
    wal = Path(absolute + "-wal")
    if wal.exists() and wal.stat().st_size > 0:
        raise RuntimeError(
            "read-only audit requires a checkpointed database (non-empty WAL present)"
        )
    uri = f"file:{absolute}?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    conn.execute("PRAGMA query_only=ON")
    conn.row_factory = sqlite3.Row
    try:
        raw_events = []
        for event in conn.execute(
            "SELECT id,ts,account,ticker,detail FROM events "
            "WHERE category='trade' AND UPPER(market)='US' ORDER BY ts,id"
        ):
            try:
                detail = json.loads(event["detail"] or "{}")
            except (TypeError, json.JSONDecodeError):
                continue
            if detail.get("reason") == "trailing_stop":
                raw_events.append(event)
        if not raw_events:
            return []

        trade_map = {}
        for trade in conn.execute(
            "SELECT id,account,ticker,timestamp FROM trades "
            "WHERE side='sell' AND UPPER(market)='US' ORDER BY id"
        ):
            trade_map.setdefault(
                (trade["account"], trade["ticker"], trade["timestamp"]), trade["id"]
            )

        first_at = _parse_ts(raw_events[0]["ts"])
        last_at = _parse_ts(raw_events[-1]["ts"])
        snapshots = iter(conn.execute(
            "SELECT a.timestamp,a.name,a.equity FROM accounts a "
            "JOIN account_meta m ON m.account_id=a.name AND UPPER(m.market)='US' "
            "WHERE UPPER(a.market)='US' AND m.status='active' "
            "AND a.timestamp>=? AND a.timestamp<? ORDER BY a.timestamp,a.name",
            ((first_at - timedelta(days=30)).isoformat(), last_at.isoformat()),
        ))
        pending = next(snapshots, None)
        latest: dict[str, float] = {}
        daily_totals: dict[str, float] = {}
        state, streak, last_recovery_day = "DISARMED", 0, None
        last_dd = 0.0

        def evaluate_snapshot_group(ts_text: str, rows: list[sqlite3.Row]) -> None:
            nonlocal state, streak, last_recovery_day, last_dd
            for row in rows:
                if row["equity"] is not None and float(row["equity"]) > 0:
                    latest[str(row["name"])] = float(row["equity"])
            at = _parse_ts(ts_text)
            day = at.date().isoformat()
            if latest:
                daily_totals[day] = sum(latest.values())
            start = (at.date() - timedelta(days=30)).isoformat()
            window = [total for d, total in daily_totals.items() if d >= start]
            if not window:
                return
            peak = max(window)
            last_dd = max(0.0, (peak - window[-1]) / peak) if peak > 0 else 0.0
            if state == "DISARMED" and last_dd >= settings.AUTO_ARM_DD_PCT:
                state, streak, last_recovery_day = "ARMED", 0, None
            elif state == "ARMED":
                if last_dd <= settings.AUTO_DISARM_DD_PCT:
                    if last_recovery_day != day:
                        streak += 1
                        last_recovery_day = day
                    if streak >= settings.AUTO_DISARM_DAYS:
                        state, streak, last_recovery_day = "DISARMED", 0, None
                else:
                    streak, last_recovery_day = 0, None

        results = []
        for event in raw_events:
            event_at = _parse_ts(event["ts"])
            while pending is not None and _parse_ts(pending["timestamp"]) < event_at:
                group_ts = pending["timestamp"]
                group = [pending]
                pending = next(snapshots, None)
                while pending is not None and pending["timestamp"] == group_ts:
                    group.append(pending)
                    pending = next(snapshots, None)
                evaluate_snapshot_group(group_ts, group)
            trade_id = trade_map.get((event["account"], event["ticker"], event["ts"]))
            if trade_id is None:
                continue
            results.append({
                "trade_id": trade_id, "event_id": event["id"],
                "timestamp": event["ts"], "account": event["account"],
                "ticker": event["ticker"], "counterfactual_state": state,
                "us_only_drawdown": last_dd, "polluted": state != "ARMED",
            })
        return results
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
