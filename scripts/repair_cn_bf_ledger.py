#!/usr/bin/env python3
"""Repair CN B/F current ledger by writing explicit replay baselines.

The affected CN B/F accounts have a mixed-world historical trade ledger: current
account_state/positions are cash-constrained/live, while replaying all old trade
rows from initial cash implies negative cash and stale positions. We preserve the
raw trade audit trail and add ledger_repair_baselines rows. The watchdog starts
future replay from the baseline cash+positions and only applies trades after the
baseline timestamp.

Default is dry-run. Pass --apply to mutate data.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "trading.db"
MARKET = "CN"
REASON = "cn-bf-mixed-world-ledger-baseline-20260625"
TARGET_ACCOUNTS = [
    "CB01", "CB03", "CB05", "CB08", "CB11", "CB12", "CB13", "CB14", "CB15", "CB16",
    "CF11", "CF13", "CF14", "CF15", "CF16",
]


def ensure_baseline_table(cur: sqlite3.Cursor) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ledger_repair_baselines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            account TEXT NOT NULL,
            market TEXT NOT NULL,
            baseline_ts TEXT NOT NULL,
            cash REAL NOT NULL,
            positions_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            reason TEXT NOT NULL,
            detail TEXT
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_ledger_repair_baselines_acct "
        "ON ledger_repair_baselines(account, market, baseline_ts)"
    )


def has_table(cur: sqlite3.Cursor, name: str) -> bool:
    return cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def position_payload(cur: sqlite3.Cursor, acct: str) -> dict[str, dict[str, float]]:
    rows = cur.execute(
        "SELECT ticker,shares,total_cost FROM positions WHERE market=? AND account=? ORDER BY ticker",
        (MARKET, acct),
    ).fetchall()
    return {
        r["ticker"]: {"shares": float(r["shares"] or 0.0), "total_cost": float(r["total_cost"] or 0.0)}
        for r in rows
        if abs(float(r["shares"] or 0.0)) > 1e-9
    }


def snapshot_counts(cur: sqlite3.Cursor, accounts: list[str]) -> dict[str, int]:
    ph = ",".join("?" for _ in accounts)
    counts = {
        "trades": cur.execute(f"SELECT COUNT(*) FROM trades WHERE market=? AND account IN ({ph})", [MARKET, *accounts]).fetchone()[0],
        "events": cur.execute(f"SELECT COUNT(*) FROM events WHERE market=? AND account IN ({ph})", [MARKET, *accounts]).fetchone()[0],
        "positions": cur.execute(f"SELECT COUNT(*) FROM positions WHERE market=? AND account IN ({ph})", [MARKET, *accounts]).fetchone()[0],
    }
    counts["baselines"] = cur.execute(
        f"SELECT COUNT(*) FROM ledger_repair_baselines WHERE market=? AND account IN ({ph})",
        [MARKET, *accounts],
    ).fetchone()[0] if has_table(cur, "ledger_repair_baselines") else 0
    return counts


def existing_replay_problem(cur: sqlite3.Cursor, acct: str) -> dict[str, Any]:
    # Lightweight summary only; authoritative verification is ledger_watchdog.
    st = cur.execute("SELECT cash FROM account_state WHERE market=? AND account=?", (MARKET, acct)).fetchone()
    pos_n = cur.execute("SELECT COUNT(*) FROM positions WHERE market=? AND account=?", (MARKET, acct)).fetchone()[0]
    tr = cur.execute(
        "SELECT COUNT(*) n, MIN(timestamp) first_ts, MAX(timestamp) last_ts FROM trades WHERE market=? AND account=?",
        (MARKET, acct),
    ).fetchone()
    return {
        "account": acct,
        "db_cash": float(st["cash"]) if st else None,
        "position_count": pos_n,
        "trade_count_preserved": tr["n"],
        "first_trade": tr["first_ts"],
        "last_trade": tr["last_ts"],
    }


def trade_event_title_and_detail(t: sqlite3.Row) -> tuple[str, dict[str, Any]]:
    side = str(t["side"]).lower()
    ticker = t["ticker"]
    shares = float(t["shares"])
    price = float(t["price"])
    sym = "¥"
    detail: dict[str, Any]
    if side == "buy":
        title = f"BUY {ticker} ×{int(shares)} @ {sym}{price:.2f}"
        detail = {"shares": shares, "price": price, "fees": float(t["cost"] or 0.0)}
    else:
        title = f"SELL {ticker} ×{int(shares)} @ {sym}{price:.2f} · 🛑 stop-loss"
        detail = {
            "shares": shares,
            "price": price,
            "reason": "stop_loss",
            "fees": float(t["cost"] or 0.0),
            "slippage": float(t["slippage"] or 0.0),
        }
    detail["source"] = "repair_cn_bf_ledger.backfill_missing_trade_event"
    detail["repair_reason"] = REASON
    return title, detail


def backfill_missing_trade_events(cur: sqlite3.Cursor, accounts: list[str]) -> int:
    total = 0
    for acct in accounts:
        rows = cur.execute(
            """
            SELECT * FROM trades t
            WHERE t.market=? AND t.account=?
              AND t.timestamp >= '2026-06-25T00:00:00+00:00'
              AND t.timestamp <  '2026-06-26T00:00:00+00:00'
              AND NOT EXISTS (
                SELECT 1 FROM events e
                WHERE e.market=t.market AND e.account=t.account AND e.category='trade'
                  AND e.ticker=t.ticker AND e.ts=t.timestamp
              )
            ORDER BY t.timestamp,t.id
            """,
            (MARKET, acct),
        ).fetchall()
        for t in rows:
            title, detail = trade_event_title_and_detail(t)
            cur.execute(
                "INSERT INTO events (ts,category,severity,account,ticker,title,detail,market) VALUES (?,?,?,?,?,?,?,?)",
                (t["timestamp"], "trade", "info", acct, t["ticker"], title, json.dumps(detail, ensure_ascii=False), MARKET),
            )
            total += 1
    return total


def count_missing_trade_events(cur: sqlite3.Cursor, accounts: list[str]) -> int:
    total = 0
    for acct in accounts:
        total += cur.execute(
            """
            SELECT COUNT(*) FROM trades t
            WHERE t.market=? AND t.account=?
              AND t.timestamp >= '2026-06-25T00:00:00+00:00'
              AND t.timestamp <  '2026-06-26T00:00:00+00:00'
              AND NOT EXISTS (
                SELECT 1 FROM events e
                WHERE e.market=t.market AND e.account=t.account AND e.category='trade'
                  AND e.ticker=t.ticker AND e.ts=t.timestamp
              )
            """,
            (MARKET, acct),
        ).fetchone()[0]
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--accounts", nargs="*", default=TARGET_ACCOUNTS)
    args = ap.parse_args()

    con = sqlite3.connect(Path(args.db).expanduser())
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    accounts = args.accounts
    now_iso = datetime.now(timezone.utc).isoformat()
    before = snapshot_counts(cur, accounts)
    plan = [existing_replay_problem(cur, acct) for acct in accounts]
    missing_events = count_missing_trade_events(cur, accounts)

    summary = {
        "mode": "apply" if args.apply else "dry_run",
        "reason": REASON,
        "repair_method": "baseline_no_trade_deletion",
        "baseline_ts": now_iso,
        "accounts": accounts,
        "before": before,
        "missing_trade_events_to_backfill": missing_events,
        "plan": plan,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not args.apply:
        con.close()
        return 0

    try:
        cur.execute("BEGIN")
        ensure_baseline_table(cur)
        for item in plan:
            acct = item["account"]
            st = cur.execute("SELECT cash FROM account_state WHERE market=? AND account=?", (MARKET, acct)).fetchone()
            if not st:
                raise RuntimeError(f"missing account_state for {acct}")
            payload = position_payload(cur, acct)
            detail = {**item, "positions": payload}
            cur.execute(
                """
                INSERT INTO ledger_repair_baselines
                (account,market,baseline_ts,cash,positions_json,created_at,reason,detail)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    acct, MARKET, now_iso, float(st["cash"]), json.dumps(payload, ensure_ascii=False),
                    now_iso, REASON, json.dumps(detail, ensure_ascii=False),
                ),
            )
        backfilled = backfill_missing_trade_events(cur, accounts)
        cur.execute(
            "INSERT INTO events (ts,category,severity,account,ticker,title,detail,market) VALUES (?,?,?,?,?,?,?,?)",
            (
                now_iso,
                "system",
                "warn",
                None,
                None,
                "🧹 CN B/F ledger replay baselines written",
                json.dumps({"reason": REASON, "baseline_ts": now_iso, "accounts": plan, "backfilled_trade_events": backfilled}, ensure_ascii=False),
                MARKET,
            ),
        )
        con.commit()
    except Exception:
        con.rollback()
        raise

    after = snapshot_counts(cur, accounts)
    print(json.dumps({"applied": True, "before": before, "after": after}, ensure_ascii=False, indent=2))
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
