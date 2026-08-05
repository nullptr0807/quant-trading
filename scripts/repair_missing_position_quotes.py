"""Repair NULL valuation fields in historical position snapshots.

The repair is deliberately narrow and auditable:
- candidates must have NULL market_price, market_value, and unrealized_pnl;
- prices come from the nearest raw intraday bar at or before the snapshot;
- the bar must be from the same UTC date and no older than --max-age-min;
- --apply creates both a full SQLite backup and a row-level backup table;
- current positions/account state/trades/accounts snapshots are never changed.

Run:
    python -m scripts.repair_missing_position_quotes --dry-run
    python -m scripts.repair_missing_position_quotes --apply
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "trading.db"


def _epoch(ts: str) -> float:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


def _same_utc_date(left: str, right: str) -> bool:
    return datetime.fromisoformat(left.replace("Z", "+00:00")).date() == datetime.fromisoformat(
        right.replace("Z", "+00:00")
    ).date()


def _price_for_snapshot(conn: sqlite3.Connection, ticker: str, ts: str, max_age_sec: float):
    """Return a recent raw 5m/1h price at or before ts, preferring 5m."""
    comparable_ts = ts.replace("T", " ")
    for interval in ("5m", "1h"):
        row = conn.execute(
            """
            SELECT datetime, close
            FROM prices_raw
            WHERE ticker=? AND interval=? AND datetime<=?
              AND close IS NOT NULL AND close>0
            ORDER BY datetime DESC LIMIT 1
            """,
            (ticker, interval, comparable_ts),
        ).fetchone()
        if not row or not _same_utc_date(row["datetime"], ts):
            continue
        age = _epoch(ts) - _epoch(row["datetime"])
        if -1e-6 <= age <= max_age_sec:
            return float(row["close"]), row["datetime"], interval, age / 60.0
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="perform the audited repair")
    mode.add_argument("--dry-run", action="store_true", help="print proposed changes only")
    ap.add_argument("--max-age-min", type=float, default=90.0)
    args = ap.parse_args()
    if not args.apply:
        args.dry_run = True

    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    try:
        candidates = conn.execute(
            """
            SELECT id, account, ticker, shares, avg_cost, timestamp, market
            FROM positions_history
            WHERE market_price IS NULL
              AND market_value IS NULL
              AND unrealized_pnl IS NULL
            ORDER BY timestamp, account, ticker
            """
        ).fetchall()
        repairs = []
        skipped = []
        max_age_sec = args.max_age_min * 60.0
        for r in candidates:
            quote = _price_for_snapshot(conn, r["ticker"], r["timestamp"], max_age_sec)
            if quote is None:
                skipped.append({"id": r["id"], "account": r["account"], "ticker": r["ticker"], "timestamp": r["timestamp"]})
                continue
            price, price_ts, interval, age_min = quote
            shares = float(r["shares"])
            avg_cost = float(r["avg_cost"])
            repairs.append({
                "id": int(r["id"]),
                "account": r["account"],
                "ticker": r["ticker"],
                "timestamp": r["timestamp"],
                "market": r["market"],
                "price": price,
                "market_value": shares * price,
                "unrealized_pnl": shares * (price - avg_cost),
                "price_ts": price_ts,
                "interval": interval,
                "age_min": round(age_min, 3),
            })

        print(f"candidates={len(candidates)} repairs={len(repairs)} skipped={len(skipped)}")
        by_ticker: dict[str, int] = {}
        for r in repairs:
            by_ticker[r["ticker"]] = by_ticker.get(r["ticker"], 0) + 1
        print("by_ticker", json.dumps(by_ticker, sort_keys=True))
        for r in repairs[:12]:
            print(r)
        if len(repairs) > 12:
            print("...")
            for r in repairs[-6:]:
                print(r)
        if skipped:
            print("skipped_examples", skipped[:10])

        if not args.apply:
            print("DRY RUN — no positions_history rows changed")
            return
        if not repairs:
            print("No repairable rows; nothing applied")
            return

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        now_iso = datetime.now(timezone.utc).isoformat()
        db_backup = f"{DB_PATH}.bak_missing_position_quotes_{stamp}"
        shutil.copy2(DB_PATH, db_backup)
        backup_table = f"positions_history_backup_missing_quotes_{stamp}"
        ids = [r["id"] for r in repairs]
        placeholders = ",".join("?" for _ in ids)

        conn.execute("BEGIN IMMEDIATE")
        conn.execute(f"CREATE TABLE {backup_table} AS SELECT * FROM positions_history WHERE 0")
        conn.execute(
            f"INSERT INTO {backup_table} SELECT * FROM positions_history WHERE id IN ({placeholders})",
            ids,
        )
        for r in repairs:
            conn.execute(
                """
                UPDATE positions_history
                SET market_price=?, market_value=?, unrealized_pnl=?
                WHERE id=? AND market_price IS NULL
                         AND market_value IS NULL AND unrealized_pnl IS NULL
                """,
                (r["price"], r["market_value"], r["unrealized_pnl"], r["id"]),
            )
        detail = {
            "reason": "repair NULL historical position valuations from nearest raw intraday bar",
            "rows_repaired": len(repairs),
            "rows_skipped": len(skipped),
            "tickers": by_ticker,
            "max_age_min": args.max_age_min,
            "backup_table": backup_table,
            "db_backup": db_backup,
        }
        conn.execute(
            """
            INSERT INTO events (ts, category, severity, account, ticker, title, detail, market)
            VALUES (?, 'data', 'warn', NULL, NULL, ?, ?, 'US')
            """,
            (now_iso, f"🧹 历史持仓缺失报价已修复：{len(repairs)} 条", json.dumps(detail, ensure_ascii=False)),
        )
        conn.commit()
        print("APPLIED")
        print("db_backup", db_backup)
        print("backup_table", backup_table)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
