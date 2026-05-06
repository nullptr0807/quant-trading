"""Event logging — durable, queryable system events for the dashboard.

Events are written to the shared SQLite (`~/quant-trading/data/trading.db`)
in the `events` table, then surfaced by trading-dashboard's /api/events.

Categories: data | factor | rebalance | trade | system
Severities: info | warn | error
"""
from __future__ import annotations

import json
import os
import sqlite3
import logging
from datetime import datetime, timezone
from typing import Optional

DB_PATH = os.path.expanduser("~/quant-trading/data/trading.db")
log = logging.getLogger("events")

_INIT_DONE = False


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _ensure_table(conn: sqlite3.Connection) -> None:
    global _INIT_DONE
    if _INIT_DONE:
        return
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            category TEXT NOT NULL,
            severity TEXT NOT NULL DEFAULT 'info',
            account TEXT,
            ticker TEXT,
            title TEXT NOT NULL,
            detail TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_events_cat ON events(category)")
    conn.commit()
    _INIT_DONE = True


def emit(
    category: str,
    title: str,
    *,
    severity: str = "info",
    account: Optional[str] = None,
    ticker: Optional[str] = None,
    detail: Optional[dict] = None,
    market: str = "US",
) -> None:
    """Insert an event row. Best-effort — never raises into caller.

    `market` defaults to 'US' for backward compat with US-only call sites.
    CN call sites must pass market='CN' explicitly.
    """
    try:
        conn = _connect()
        _ensure_table(conn)
        ts = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO events (ts, category, severity, account, ticker, title, detail, market) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ts,
                category,
                severity,
                account,
                ticker,
                title,
                json.dumps(detail, ensure_ascii=False) if detail else None,
                market,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:  # pragma: no cover
        log.warning("event emit failed: %s", e)


# convenience wrappers
def data_event(title: str, **kw): emit("data", title, **kw)
def factor_event(title: str, **kw): emit("factor", title, **kw)
def rebalance_event(title: str, **kw): emit("rebalance", title, **kw)
def trade_event(title: str, **kw): emit("trade", title, **kw)
def system_event(title: str, **kw): emit("system", title, **kw)
