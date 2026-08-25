#!/usr/bin/env python3
"""Persist audited cash-dividend candidates for review without changing cash.

The corporate-action provider currently proves ex-date and gross amount only; it
does not prove pay-date, withholding tax, currency conversion, or broker cash
posting.  This queue is therefore metadata-only and intentionally has no code
path that updates accounts, trades, positions, or historical snapshots.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

CONFIRM = "SYNC CASH DIVIDEND REVIEW QUEUE"


def _decimal(value: object) -> Decimal:
    try:
        result = Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid decimal: {value!r}") from exc
    if not result.is_finite() or result <= 0:
        raise ValueError(f"decimal must be positive: {value!r}")
    return result


def _key(account: str, market: str, ticker: str, ex_date: str, cash_per_share: Decimal) -> str:
    payload = "|".join(
        [account, market, ticker, ex_date, format(cash_per_share.normalize(), "f")]
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def load_candidates(csv_path: str | Path, market: str | None = None) -> list[dict]:
    path = Path(csv_path)
    rows: dict[str, dict] = {}
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            if str(raw.get("action_type") or "") != "cash_dividend":
                continue
            row_market = str(raw.get("market") or "").upper()
            if market and row_market != market.upper():
                continue
            account = str(raw.get("account") or "").strip()
            ticker = str(raw.get("ticker") or "").strip().upper()
            ex_date = str(raw.get("ex_date") or "")[:10]
            if not account or not ticker or len(ex_date) != 10 or row_market not in {"US", "CN"}:
                raise ValueError(f"invalid dividend identity: {raw}")
            cps = _decimal(raw.get("cash_per_share"))
            try:
                shares = Decimal(str(raw.get("held_shares_before") or "0").strip())
            except (InvalidOperation, ValueError) as exc:
                raise ValueError(
                    f"invalid held_shares_before: {raw.get('held_shares_before')!r}"
                ) from exc
            if not shares.is_finite() or shares < 0:
                raise ValueError(
                    f"held_shares_before must be non-negative: {raw.get('held_shares_before')!r}"
                )
            if shares == 0:
                # The audit includes closed intervals for context. A zero
                # ex-date holding has no dividend entitlement and must not
                # create a zero-value review item.
                continue
            gross = cps * shares
            key = _key(account, row_market, ticker, ex_date, cps)
            candidate = {
                "action_key": key,
                "account": account,
                "market": row_market,
                "ticker": ticker,
                "ex_date": ex_date,
                "cash_per_share": float(cps),
                "held_shares_before": float(shares),
                "gross_amount": float(gross),
                "currency": "USD" if row_market == "US" else "CNY",
                "source": str(raw.get("source") or ""),
                "detail": json.dumps(
                    {
                        "description": raw.get("description"),
                        "interval_start": raw.get("interval_start"),
                        "interval_end": raw.get("interval_end"),
                        "open_after_action": raw.get("open_after_action"),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            }
            existing = rows.get(key)
            if existing and (
                existing["held_shares_before"] != candidate["held_shares_before"]
                or existing["gross_amount"] != candidate["gross_amount"]
            ):
                raise ValueError(f"conflicting duplicate dividend candidate: {key}")
            rows[key] = candidate
    return sorted(rows.values(), key=lambda r: (r["market"], r["account"], r["ex_date"], r["ticker"]))


def _ensure_table(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS cash_dividend_review_queue (
          action_key TEXT PRIMARY KEY,
          account TEXT NOT NULL,
          market TEXT NOT NULL,
          ticker TEXT NOT NULL,
          ex_date TEXT NOT NULL,
          cash_per_share REAL NOT NULL,
          held_shares_before REAL NOT NULL,
          gross_amount REAL NOT NULL,
          currency TEXT NOT NULL,
          source TEXT,
          status TEXT NOT NULL DEFAULT 'pending_review'
            CHECK(status IN ('pending_review','approved','rejected')),
          first_seen_at TEXT NOT NULL,
          last_seen_at TEXT NOT NULL,
          detail TEXT
        )
        """
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_cash_dividend_review_status "
        "ON cash_dividend_review_queue(status,market,ex_date)"
    )


def sync_reviews(
    *, csv_path: str | Path, db: str | Path, market: str | None = None,
    apply: bool = False, confirm: str = "",
) -> dict:
    candidates = load_candidates(csv_path, market)
    summary = {
        "market": market.upper() if market else "ALL",
        "candidates": len(candidates),
        "gross_amount": round(sum(float(r["gross_amount"]) for r in candidates), 8),
        "applied": bool(apply),
        "cash_mutations": 0,
        "trade_mutations": 0,
    }
    if not apply:
        return summary
    if confirm != CONFIRM:
        raise ValueError(f"confirmation must equal {CONFIRM}")
    now = datetime.now(timezone.utc).isoformat()
    con = sqlite3.connect(str(db), timeout=30)
    try:
        con.execute("PRAGMA busy_timeout=30000")
        _ensure_table(con)
        for row in candidates:
            con.execute(
                """
                INSERT INTO cash_dividend_review_queue(
                  action_key,account,market,ticker,ex_date,cash_per_share,
                  held_shares_before,gross_amount,currency,source,status,
                  first_seen_at,last_seen_at,detail
                ) VALUES (?,?,?,?,?,?,?,?,?,?,'pending_review',?,?,?)
                ON CONFLICT(action_key) DO UPDATE SET
                  held_shares_before=CASE WHEN cash_dividend_review_queue.status='pending_review'
                    THEN excluded.held_shares_before ELSE cash_dividend_review_queue.held_shares_before END,
                  gross_amount=CASE WHEN cash_dividend_review_queue.status='pending_review'
                    THEN excluded.gross_amount ELSE cash_dividend_review_queue.gross_amount END,
                  source=excluded.source,
                  last_seen_at=excluded.last_seen_at,
                  detail=excluded.detail
                """,
                (
                    row["action_key"], row["account"], row["market"], row["ticker"],
                    row["ex_date"], row["cash_per_share"], row["held_shares_before"],
                    row["gross_amount"], row["currency"], row["source"], now, now,
                    row["detail"],
                ),
            )
        con.commit()
        summary["queue_rows"] = con.execute(
            "SELECT COUNT(*) FROM cash_dividend_review_queue"
        ).fetchone()[0]
    finally:
        con.close()
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--db", default="data/trading.db")
    parser.add_argument("--market", choices=("US", "CN"))
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()
    result = sync_reviews(
        csv_path=args.csv, db=args.db, market=args.market,
        apply=args.apply, confirm=args.confirm,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
