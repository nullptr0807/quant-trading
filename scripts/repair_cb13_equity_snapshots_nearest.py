"""Second-pass repair for CB13 historical avg-cost fallback snapshots.

Uses the nearest positions_history snapshot within a configurable time window
(before OR after the account snapshot). This catches CN session-open bad rows
(e.g. 01:30/05:00 UTC) where the first valid positions_history sample appears a
few minutes later, plus sparse snapshot gaps. Non-destructive: backs up rows.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
from bisect import bisect_left
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "trading.db")
ACCOUNT = "CB13"
MARKET = "CN"
BAD_LOW = 99000.0
BAD_HIGH = 101000.0
MAX_CASH_FOR_INVESTED = 5000.0


def _ts_epoch(ts: str) -> float:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def _load_position_snapshots(conn: sqlite3.Connection) -> tuple[list[float], dict[float, float]]:
    rows = conn.execute(
        """
        SELECT timestamp, SUM(COALESCE(market_value, shares * market_price, 0)) AS mv
        FROM positions_history
        WHERE account=?
        GROUP BY timestamp
        ORDER BY timestamp
        """,
        (ACCOUNT,),
    ).fetchall()
    by_epoch: dict[float, float] = {}
    for r in rows:
        ep = _ts_epoch(r["timestamp"])
        if ep > 0 and r["mv"] is not None:
            by_epoch[ep] = float(r["mv"])
    epochs = sorted(by_epoch)
    return epochs, by_epoch


def _nearest_epoch(epochs: list[float], ep: float) -> float | None:
    if not epochs:
        return None
    i = bisect_left(epochs, ep)
    cand = []
    if i < len(epochs): cand.append(epochs[i])
    if i > 0: cand.append(epochs[i - 1])
    return min(cand, key=lambda x: abs(x - ep)) if cand else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-gap-min", type=float, default=240.0)
    args = ap.parse_args()
    if not args.apply and not args.dry_run:
        args.dry_run = True

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        epochs, mv_by_epoch = _load_position_snapshots(conn)
        max_gap = args.max_gap_min * 60.0
        candidates = conn.execute(
            """
            SELECT id, timestamp, cash, equity
            FROM accounts
            WHERE name=? AND market=?
              AND cash < ?
              AND equity BETWEEN ? AND ?
            ORDER BY timestamp
            """,
            (ACCOUNT, MARKET, MAX_CASH_FOR_INVESTED, BAD_LOW, BAD_HIGH),
        ).fetchall()
        repairs = []
        skipped_gap = skipped_small = 0
        for r in candidates:
            ep = _ts_epoch(r["timestamp"])
            nep = _nearest_epoch(epochs, ep)
            if nep is None or abs(nep - ep) > max_gap:
                skipped_gap += 1; continue
            new_equity = float(r["cash"]) + mv_by_epoch[nep]
            if abs(new_equity - float(r["equity"])) < 500.0:
                skipped_small += 1; continue
            repairs.append({
                "id": int(r["id"]),
                "timestamp": r["timestamp"],
                "old_cash": float(r["cash"]),
                "old_equity": float(r["equity"]),
                "new_equity": round(new_equity, 4),
                "pos_gap_min": round((nep - ep) / 60.0, 2),
            })
        print(f"candidates={len(candidates)} repairs={len(repairs)} skipped_gap={skipped_gap} skipped_small={skipped_small}")
        for x in repairs[:12]: print(x)
        if len(repairs) > 12:
            print('...')
            for x in repairs[-8:]: print(x)
        if not args.apply:
            print('DRY RUN — no DB changes')
            return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        now_iso = datetime.now(timezone.utc).isoformat()
        db_backup = f"{DB_PATH}.bak_cb13_repair_nearest_{stamp}"
        shutil.copy2(DB_PATH, db_backup)
        backup_table = f"accounts_backup_cb13_repair_nearest_{stamp}"
        conn.execute(f"CREATE TABLE {backup_table} AS SELECT *, ? AS backup_ts FROM accounts WHERE 0", (now_iso,))
        ids = [r["id"] for r in repairs]
        if ids:
            placeholders = ','.join(['?'] * len(ids))
            conn.execute(
                f"INSERT INTO {backup_table} SELECT *, ? AS backup_ts FROM accounts WHERE id IN ({placeholders})",
                [now_iso, *ids],
            )
            for r in repairs:
                conn.execute("UPDATE accounts SET equity=? WHERE id=?", (r["new_equity"], r["id"]))
        conn.execute(
            """
            INSERT INTO events (ts, category, severity, account, ticker, title, detail, market)
            VALUES (?, 'data', 'warn', ?, NULL, ?, ?, ?)
            """,
            (now_iso, ACCOUNT, "🧹 CB13 equity 快照二次修复", json.dumps({
                "reason": "second-pass repair using nearest positions_history market values",
                "rows_repaired": len(repairs),
                "backup_table": backup_table,
                "db_backup": db_backup,
                "max_abs_gap_min": args.max_gap_min,
            }, ensure_ascii=False), MARKET),
        )
        conn.commit()
        print('APPLIED')
        print('db_backup', db_backup)
        print('backup_table', backup_table)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
