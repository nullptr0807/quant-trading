#!/usr/bin/env python3
"""Audit and quarantine economically impossible CN T+1 sells.

Historical same-day sells are counterfactual: the actual next executable price
and downstream strategy decisions are not uniquely observable. Therefore this
script NEVER rewrites live cash/positions or deletes trades. It archives exact
rows, records a data-quality quarantine range/account list, and inserts audit
events. Default is dry-run; --apply requires an existing full DB backup path.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data/trading.db"


def _shanghai_date(value: str) -> str:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(ZoneInfo("Asia/Shanghai")).date().isoformat()


def find_violations(con: sqlite3.Connection) -> list[dict]:
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT id,account,ticker,side,shares,price,cost,slippage,timestamp,market "
        "FROM trades WHERE market='CN' ORDER BY account,ticker,timestamp,id"
    ).fetchall()
    state: dict[tuple[str, str], dict] = {}
    bad = []
    for row in rows:
        key = (row["account"], row["ticker"])
        day = _shanghai_date(row["timestamp"])
        item = state.setdefault(key, {"day": day, "held": 0.0, "settled": 0.0})
        if day != item["day"]:
            item["day"] = day
            item["settled"] = max(0.0, item["held"])
        qty = float(row["shares"])
        if str(row["side"]).lower() == "buy":
            item["held"] += qty
        elif str(row["side"]).lower() == "sell":
            illegal = max(0.0, qty - item["settled"])
            if illegal > 1e-9:
                bad.append({**dict(row), "settled_before": item["settled"], "illegal_qty": illegal})
            item["settled"] = max(0.0, item["settled"] - qty)
            item["held"] -= qty
    return bad


def run(*, apply: bool, backup: str | None) -> dict:
    con = sqlite3.connect(DB, timeout=60)
    bad = find_violations(con)
    result = {
        "violations": len(bad),
        "accounts": sorted({r["account"] for r in bad}),
        "first": min((r["timestamp"] for r in bad), default=None),
        "last": max((r["timestamp"] for r in bad), default=None),
        "repairable_exactly": False,
        "policy": "archive_and_quarantine_only",
    }
    if not apply:
        con.close()
        return result
    if not backup or not Path(backup).exists():
        con.close()
        raise RuntimeError("--apply requires --backup pointing to an existing full DB backup")
    existing_table = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='data_quality_quarantine'"
    ).fetchone()
    if existing_table:
        existing = con.execute(
            "SELECT archive_table,row_count FROM data_quality_quarantine "
            "WHERE issue='cn_t_plus_one_counterfactual' AND market='CN' "
            "AND start_ts IS ? AND end_ts IS ? ORDER BY id DESC LIMIT 1",
            (result["first"], result["last"]),
        ).fetchone()
        if existing and int(existing[1]) == len(bad):
            con.close()
            result.update({
                "applied": True,
                "already_applied": True,
                "archive_table": existing[0],
                "backup": backup,
            })
            return result
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    table = f"trades_quarantine_cn_t1_{stamp}"
    now = datetime.now(timezone.utc).isoformat()
    try:
        con.execute("BEGIN IMMEDIATE")
        con.execute("""CREATE TABLE IF NOT EXISTS data_quality_quarantine (
            id INTEGER PRIMARY KEY AUTOINCREMENT, issue TEXT NOT NULL,
            market TEXT NOT NULL, account TEXT, start_ts TEXT, end_ts TEXT,
            row_count INTEGER NOT NULL, repairable_exactly INTEGER NOT NULL,
            archive_table TEXT, backup_path TEXT, detail TEXT, created_at TEXT NOT NULL
        )""")
        con.execute(f'CREATE TABLE "{table}" AS SELECT * FROM trades WHERE 0')
        if bad:
            ids = [int(r["id"]) for r in bad]
            for start in range(0, len(ids), 500):
                batch = ids[start:start+500]
                marks = ",".join("?" * len(batch))
                con.execute(f'INSERT INTO "{table}" SELECT * FROM trades WHERE id IN ({marks})', batch)
        for account in result["accounts"]:
            rows = [r for r in bad if r["account"] == account]
            detail = {"trade_ids": [r["id"] for r in rows], "reason": "A-share same-day buy quantity was sold before T+1 settlement"}
            con.execute(
                "INSERT INTO data_quality_quarantine "
                "(issue,market,account,start_ts,end_ts,row_count,repairable_exactly,archive_table,backup_path,detail,created_at) "
                "VALUES ('cn_t_plus_one_counterfactual','CN',?,?,?,?,0,?,?,?,?)",
                (account, min(r["timestamp"] for r in rows), max(r["timestamp"] for r in rows),
                 len(rows), table, str(Path(backup).resolve()), json.dumps(detail), now),
            )
            con.execute(
                "INSERT INTO events (ts,category,severity,account,ticker,title,detail,market) "
                "VALUES (?,'data_quality','warning',?,NULL,?,?, 'CN')",
                (now, account, f"⚠️ {account} history quarantined: impossible CN T+1 sells", json.dumps(detail)),
            )
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
    result.update({"applied": True, "archive_table": table, "backup": backup})
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true")
    p.add_argument("--backup")
    args = p.parse_args()
    print(json.dumps(run(apply=args.apply, backup=args.backup), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
