#!/usr/bin/env python3
"""Audited, idempotent migration of an open position across a 1:1 ticker rename.

Dry-run is the default. Historical trades are retained verbatim; ledger replay
uses config.security_master to resolve the identifier at each trade timestamp.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone

from config.security_master import SYMBOL_CHANGES
from data.store import DB_PATH


def migrate(db_path: str, market: str, old: str, *, apply: bool = False) -> dict:
    market, old = market.upper(), old.upper()
    change = SYMBOL_CHANGES.get((market, old))
    if not change:
        raise ValueError(f"no audited symbol change configured for {market}/{old}")
    new = str(change["new_ticker"])
    con = sqlite3.connect(db_path, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=30000")
    rows = con.execute(
        "SELECT * FROM positions WHERE market=? AND ticker=? ORDER BY account", (market, old)
    ).fetchall()
    conflicts = [
        r["account"] for r in rows
        if con.execute(
            "SELECT 1 FROM positions WHERE account=? AND market=? AND ticker=?",
            (r["account"], market, new),
        ).fetchone()
    ]
    result = {
        "dry_run": not apply, "market": market, "old_ticker": old,
        "new_ticker": new, "ratio": change["ratio"],
        "accounts": [r["account"] for r in rows], "conflicts": conflicts,
        "evidence": change["evidence"],
    }
    if conflicts:
        con.close()
        raise RuntimeError(f"destination position already exists for: {conflicts}")
    if not apply or not rows:
        con.close()
        return result

    now = datetime.now(timezone.utc).isoformat()
    with con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS security_symbol_migrations (
              id INTEGER PRIMARY KEY AUTOINCREMENT, migrated_at TEXT NOT NULL,
              market TEXT NOT NULL, account TEXT NOT NULL, old_ticker TEXT NOT NULL,
              new_ticker TEXT NOT NULL, shares REAL NOT NULL, avg_cost REAL,
              total_cost REAL, prior_current_price REAL, prior_updated_at TEXT,
              effective_at TEXT NOT NULL, evidence TEXT NOT NULL,
              UNIQUE(market,account,old_ticker,new_ticker,effective_at)
            )
            """
        )
        for r in rows:
            con.execute(
                """
                INSERT OR IGNORE INTO security_symbol_migrations
                (migrated_at,market,account,old_ticker,new_ticker,shares,avg_cost,
                 total_cost,prior_current_price,prior_updated_at,effective_at,evidence)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (now, market, r["account"], old, new, r["shares"], r["avg_cost"],
                 r["total_cost"], r["current_price"], r["updated_at"],
                 change["effective_at"], change["evidence"]),
            )
            con.execute(
                "UPDATE positions SET ticker=? WHERE account=? AND market=? AND ticker=?",
                (new, r["account"], market, old),
            )
            try:
                con.execute(
                    """INSERT INTO events(ts,category,severity,account,ticker,title,detail,market)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    (now, "corporate_action", "info", r["account"], new,
                     f"Ticker migrated {old} → {new}", json.dumps(result), market),
                )
            except sqlite3.OperationalError:
                pass
    con.close()
    result["migrated"] = len(rows)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--market", required=True, choices=["US", "CN"])
    p.add_argument("--old", required=True)
    p.add_argument("--db", default=DB_PATH)
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    print(json.dumps(migrate(args.db, args.market, args.old, apply=args.apply), indent=2))


if __name__ == "__main__":
    main()
