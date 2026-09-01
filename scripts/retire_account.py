"""Retire a trading account: freeze it from further trading & equity updates,
preserve historical data, emit a dashboard event.

Usage:
    python -m scripts.retire_account --account B02 --reason "公式与B01同质化(IC=0.105)"
    python -m scripts.retire_account --account B02 --reason "..." --market US
    python -m scripts.retire_account --account B02 --unretire   # reverse

Effects:
    - account_meta.status = 'retired'
    - account_meta.retired_at = now (UTC ISO)
    - account_meta.retire_reason = <reason>
    - emits an event (severity=warning, category=lifecycle) so the LiveStream
      shows a yellow notice
"""
from __future__ import annotations
import argparse
import os
import sqlite3
import sys
from datetime import datetime, timezone

# Make project importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DB_PATH = os.path.join(PROJECT_ROOT, "data", "trading.db")


def _emit_event(conn: sqlite3.Connection, *, category: str, severity: str,
                account: str | None, title: str, detail: str, market: str):
    conn.execute(
        "INSERT INTO events (ts, category, severity, account, ticker, title, detail, market) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(), category, severity, account,
         None, title, detail, market),
    )


def retire(account_id: str, reason: str, market: str = "US") -> int:
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            "SELECT status, strategy_name FROM account_meta "
            "WHERE account_id=? AND market=?",
            (account_id, market),
        ).fetchone()
        if not row:
            print(f"❌ Account not found: {account_id} (market={market})")
            return 1
        status, strat = row
        if status == "retired":
            print(f"⚠️  {account_id} is already retired — updating reason only")
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE account_meta SET status='retired', retired_at=?, retire_reason=? "
            "WHERE account_id=? AND market=?",
            (now, reason, account_id, market),
        )
        _emit_event(
            conn, category="lifecycle", severity="warning",
            account=account_id,
            title=f"🟡 {account_id} 已退役",
            detail=f'{{"reason": {reason!r}, "retired_at": {now!r}}}',
            market=market,
        )
        conn.commit()
        print(f"✅ Retired {account_id} [{market}] — {strat}")
        print(f"   reason: {reason}")
        print(f"   at:     {now}")
        return 0
    finally:
        conn.close()


def unretire(account_id: str, market: str = "US") -> int:
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            "SELECT status FROM account_meta WHERE account_id=? AND market=?",
            (account_id, market),
        ).fetchone()
        if not row:
            print(f"❌ Account not found: {account_id}")
            return 1
        if row[0] != "retired":
            print(f"⚠️  {account_id} is not retired (status={row[0]})")
            return 0
        conn.execute(
            "UPDATE account_meta SET status='active', retired_at=NULL, retire_reason=NULL "
            "WHERE account_id=? AND market=?",
            (account_id, market),
        )
        _emit_event(
            conn, category="lifecycle", severity="warning",
            account=account_id, title=f"♻️  {account_id} 复活，重新激活",
            detail="{}", market=market,
        )
        conn.commit()
        print(f"✅ Unretired {account_id} [{market}]")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", required=True, help="Account ID (e.g. B02)")
    ap.add_argument("--reason", default="", help="Why this account is being retired")
    ap.add_argument("--market", default="US", choices=["US", "CN"])
    ap.add_argument("--unretire", action="store_true", help="Reactivate")
    args = ap.parse_args()
    if args.unretire:
        sys.exit(unretire(args.account, args.market))
    if not args.reason:
        print("❌ --reason is required when retiring")
        sys.exit(1)
    sys.exit(retire(args.account, args.reason, args.market))
