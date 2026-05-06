"""Add `market` column ('US'|'CN') to account-scoped tables. Idempotent.

After this migration:
- All existing rows are tagged market='US' (default).
- New CN rows must explicitly set market='CN'.
- DataStore reads should filter by market to keep US/CN isolated.
"""
import sqlite3
from pathlib import Path

DB = Path.home() / "quant-trading" / "data" / "trading.db"

TABLES = [
    "account_meta",
    "account_state",
    "accounts",
    "events",
    "positions",
    "positions_history",
    "trades",
    "adaptive_state",
]


def has_col(con: sqlite3.Connection, table: str, col: str) -> bool:
    return any(r[1] == col for r in con.execute(f"PRAGMA table_info({table})"))


def main() -> None:
    con = sqlite3.connect(DB)
    con.execute("PRAGMA foreign_keys=ON")
    added: list[str] = []
    skipped: list[str] = []
    for t in TABLES:
        if not has_col(con, t, "market"):
            con.execute(
                f"ALTER TABLE {t} ADD COLUMN market TEXT NOT NULL DEFAULT 'US'"
            )
            added.append(t)
        else:
            skipped.append(t)

    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_trades_market_ts ON trades(market, timestamp)"
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_events_market_ts ON events(market, ts)"
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_positions_market ON positions(market, account)"
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_account_state_market ON account_state(market, account)"
    )
    con.commit()
    con.close()

    print(f"added market column to: {added or '(none — already present)'}")
    print(f"skipped (already had column): {skipped}")
    print("indexes ensured: idx_trades_market_ts, idx_events_market_ts, "
          "idx_positions_market, idx_account_state_market")
    print("done")


if __name__ == "__main__":
    main()
