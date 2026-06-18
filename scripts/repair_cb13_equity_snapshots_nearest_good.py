"""Third-pass repair for CB13 equity snapshots using nearest good account row.

Some historical CB13 rows have no nearby positions_history sample, but the same
session contains adjacent `accounts` rows with real market-priced equity. This
script repairs only obvious avg-cost fallback rows (cash low, equity ~100k) by
copying the nearest non-fallback CB13 equity within a short time window.

Non-destructive: backs up the rows before updating.
"""
from __future__ import annotations

import argparse, json, os, shutil, sqlite3
from bisect import bisect_left
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "trading.db")
ACCOUNT = "CB13"
MARKET = "CN"
BAD_LOW = 99000.0
BAD_HIGH = 101000.0
MAX_CASH_FOR_INVESTED = 5000.0


def _ep(ts: str) -> float:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def _is_bad(r) -> bool:
    return float(r["cash"]) < MAX_CASH_FOR_INVESTED and BAD_LOW <= float(r["equity"]) <= BAD_HIGH


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-gap-min", type=float, default=90.0)
    args = ap.parse_args()
    if not args.apply and not args.dry_run:
        args.dry_run = True

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = [dict(r) for r in conn.execute(
            "SELECT id,timestamp,cash,equity FROM accounts WHERE name=? AND market=? ORDER BY timestamp",
            (ACCOUNT, MARKET),
        )]
        good = []
        for r in rows:
            if _is_bad(r):
                continue
            # good repair anchors must be invested rows, not all-cash plateaus
            if float(r["cash"]) < MAX_CASH_FOR_INVESTED and float(r["equity"]) > BAD_HIGH:
                ep = _ep(r["timestamp"])
                if ep > 0:
                    good.append((ep, float(r["equity"]), r["timestamp"]))
        epochs = [g[0] for g in good]
        candidates = [r for r in rows if _is_bad(r)]
        max_gap = args.max_gap_min * 60.0
        repairs = []
        skipped_gap = skipped_small = 0
        for r in candidates:
            ep = _ep(r["timestamp"])
            i = bisect_left(epochs, ep)
            cand = []
            if i < len(good): cand.append(good[i])
            if i > 0: cand.append(good[i-1])
            if not cand:
                skipped_gap += 1; continue
            nep, neq, nts = min(cand, key=lambda x: abs(x[0]-ep))
            gap = nep - ep
            if abs(gap) > max_gap:
                skipped_gap += 1; continue
            if abs(neq - float(r["equity"])) < 500.0:
                skipped_small += 1; continue
            repairs.append({
                "id": int(r["id"]),
                "timestamp": r["timestamp"],
                "old_cash": float(r["cash"]),
                "old_equity": float(r["equity"]),
                "new_equity": round(neq, 4),
                "anchor_ts": nts,
                "anchor_gap_min": round(gap/60.0, 2),
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
        db_backup = f"{DB_PATH}.bak_cb13_repair_goodrow_{stamp}"
        shutil.copy2(DB_PATH, db_backup)
        backup_table = f"accounts_backup_cb13_repair_goodrow_{stamp}"
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
            (now_iso, ACCOUNT, "🧹 CB13 equity 快照三次修复", json.dumps({
                "reason": "repair residual avg-cost fallback rows using nearest good account equity row",
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
