"""Market-scoped trailing-stop risk regime with fail-safe legacy migration."""
from __future__ import annotations

import json
import logging
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

from config import settings

log = logging.getLogger("risk_regime")
DB_PATH = "/home/gexin/quant-trading/data/trading.db"


def _validate_market(market: str) -> str:
    value = str(market or "").upper()
    if value not in {"US", "CN"}:
        raise ValueError("market must be explicitly 'US' or 'CN'")
    return value


def _create_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS risk_regime (
            market TEXT PRIMARY KEY CHECK (market IN ('US','CN')),
            state TEXT NOT NULL CHECK (state IN ('DISARMED','ARMED')),
            armed_at TEXT,
            recovery_streak INTEGER NOT NULL DEFAULT 0,
            last_drawdown REAL,
            last_check_at TEXT
        )
    """)


def _record_event(
    conn: sqlite3.Connection, kind: str, message: str, *, market: str,
    severity: str = "warn",
) -> None:
    try:
        title = message.splitlines()[0][:120]
        conn.execute(
            "INSERT INTO events (ts,category,severity,title,detail,market) "
            "VALUES (?,'risk',?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), severity, title, message, market),
        )
    except Exception as exc:
        log.warning("could not write event row: %s", exc)


def _ensure_table(conn: sqlite3.Connection) -> None:
    """Create v2 schema; migrate v1 without trusting its mixed-market state."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(risk_regime)")}
    if columns and "market" not in columns:
        suffix = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        backup = f"risk_regime_legacy_{suffix}"
        conn.execute(f'ALTER TABLE risk_regime RENAME TO "{backup}"')
        _create_table(conn)
        now = datetime.now(timezone.utc).isoformat()
        markets = [
            row[0] for row in conn.execute(
                "SELECT DISTINCT UPPER(market) FROM account_meta "
                "WHERE status='active' AND UPPER(market) IN ('US','CN')"
            )
        ]
        recomputed = {}
        for market in markets:
            dd = _portfolio_drawdown(conn, market=market)
            state = "ARMED" if dd >= settings.AUTO_ARM_DD_PCT else "DISARMED"
            conn.execute(
                "INSERT INTO risk_regime(market,state,armed_at,recovery_streak,last_drawdown,last_check_at) "
                "VALUES(?,?,?,?,?,?)",
                (market, state, now if state == "ARMED" else None, 0, dd, now),
            )
            recomputed[market] = {"state": state, "drawdown": dd}
        _record_event(
            conn,
            "risk_regime_schema_migrated",
            "Risk regime schema migration to per-market state\n"
            + json.dumps({
                "legacy_backup_table": backup,
                "legacy state was not copied": True,
                "recomputed": recomputed,
            }, ensure_ascii=False, sort_keys=True),
            market="ALL",
            severity="warning",
        )
        conn.commit()
        return
    _create_table(conn)
    conn.commit()


def _portfolio_drawdown(
    conn: sqlite3.Connection, *, market: str, lookback_days: int = 30,
    as_of: datetime | None = None,
) -> float:
    """Drawdown of active live accounts in exactly one market.

    Account metadata is authoritative: research/retired/missing metadata rows are
    excluded. Account snapshots must independently carry the requested market,
    preventing equal account names in another market from leaking into totals.
    """
    market = _validate_market(market)
    end = as_of or datetime.now(timezone.utc)
    cutoff = (end - timedelta(days=lookback_days)).isoformat()
    rows = conn.execute("""
        SELECT name, d, equity FROM (
            SELECT a.name, date(a.timestamp) AS d, a.equity,
                   ROW_NUMBER() OVER (
                       PARTITION BY a.name, date(a.timestamp)
                       ORDER BY a.timestamp DESC
                   ) AS rn
              FROM accounts a
             WHERE a.timestamp >= ? AND a.timestamp <= ?
               AND UPPER(a.market)=?
               AND EXISTS (
                   SELECT 1 FROM account_meta m
                    WHERE m.account_id=a.name AND UPPER(m.market)=?
                      AND m.status='active'
               )
        ) WHERE rn=1 ORDER BY name,d
    """, (cutoff, end.isoformat(), market, market)).fetchall()
    per_acct: dict[str, dict[str, float]] = defaultdict(dict)
    all_days: set[str] = set()
    for name, day, equity in rows:
        if equity is not None and equity > 0:
            per_acct[name][day] = float(equity)
            all_days.add(day)
    if len(all_days) < 2 or not per_acct:
        return 0.0
    days = sorted(all_days)
    last_by_account: dict[str, float] = {}
    totals: list[float] = []
    for day in days:
        for name, series in per_acct.items():
            if day in series:
                last_by_account[name] = series[day]
        total = sum(last_by_account.values())
        if total > 0:
            totals.append(total)
    if len(totals) < 2:
        return 0.0
    peak = max(totals)
    return max(0.0, (peak - totals[-1]) / peak) if peak > 0 else 0.0


def _send_telegram(message: str) -> None:
    try:
        from reports.telegram import TelegramReporter
        TelegramReporter().send_message(message)
    except Exception as exc:
        log.warning("telegram send failed: %s", exc)


def get_state(
    *, market: str, conn: Optional[sqlite3.Connection] = None,
    db_path: str = DB_PATH,
) -> dict:
    market = _validate_market(market)
    own = conn is None
    c = conn or sqlite3.connect(db_path)
    try:
        _ensure_table(c)
        c.execute(
            "INSERT OR IGNORE INTO risk_regime(market,state,recovery_streak) "
            "VALUES(?,'DISARMED',0)", (market,),
        )
        c.commit()
        row = c.execute(
            "SELECT state,armed_at,recovery_streak,last_drawdown,last_check_at "
            "FROM risk_regime WHERE market=?", (market,),
        ).fetchone()
    finally:
        if own:
            c.close()
    return {
        "market": market, "state": row[0], "armed_at": row[1],
        "recovery_streak": row[2] or 0, "last_drawdown": row[3] or 0.0,
        "last_check_at": row[4],
    }


def evaluate_and_update(*, market: str, db_path: str = DB_PATH) -> dict:
    market = _validate_market(market)
    conn = sqlite3.connect(db_path)
    try:
        _ensure_table(conn)
        dd = _portfolio_drawdown(conn, market=market)
        now_iso = datetime.now(timezone.utc).isoformat()
        current = get_state(market=market, conn=conn)
        old_state = current["state"]
        state = old_state
        streak = current["recovery_streak"]
        message = None
        if state == "DISARMED" and dd >= settings.AUTO_ARM_DD_PCT:
            state, streak = "ARMED", 0
            message = (
                f"🚨 *RISK REGIME [{market}]: AUTO-ARMED*\n"
                f"{market} drawdown {dd*100:.2f}% ≥ {settings.AUTO_ARM_DD_PCT*100:.1f}%."
            )
        elif state == "ARMED":
            if dd <= settings.AUTO_DISARM_DD_PCT:
                streak += 1
                if streak >= settings.AUTO_DISARM_DAYS:
                    state, streak = "DISARMED", 0
                    message = f"✅ *RISK REGIME [{market}]: AUTO-DISARMED*\nDrawdown {dd*100:.2f}%."
            else:
                streak = 0
        if message:
            _record_event(conn, f"risk_regime_{state.lower()}", message, market=market)
        conn.execute("""
            UPDATE risk_regime SET state=?,armed_at=?,recovery_streak=?,
                last_drawdown=?,last_check_at=? WHERE market=?
        """, (
            state,
            now_iso if state == "ARMED" and old_state == "DISARMED" else current["armed_at"],
            streak, dd, now_iso, market,
        ))
        conn.commit()
        if message:
            _send_telegram(message)
        return {
            "market": market, "state": state, "drawdown": dd,
            "recovery_streak": streak, "transitioned": state != old_state,
        }
    finally:
        conn.close()


def get_effective_trailing_stop(*, market: str, db_path: str = DB_PATH) -> Optional[float]:
    market = _validate_market(market)
    if settings.TRAILING_STOP_PCT is not None:
        return float(settings.TRAILING_STOP_PCT)
    return (
        float(settings.AUTO_ARM_TRAIL_PCT)
        if get_state(market=market, db_path=db_path)["state"] == "ARMED"
        else None
    )


def status_line(*, market: str, db_path: str = DB_PATH) -> str:
    market = _validate_market(market)
    state = get_state(market=market, db_path=db_path)
    if settings.TRAILING_STOP_PCT is not None:
        return f"🛡 [{market}] Trailing stop: *MANUAL ON* ({settings.TRAILING_STOP_PCT*100:.1f}%, DD={state['last_drawdown']*100:.2f}%)"
    if state["state"] == "ARMED":
        return f"🚨 [{market}] Trailing stop: *AUTO-ARMED* ({settings.AUTO_ARM_TRAIL_PCT*100:.1f}%, DD={state['last_drawdown']*100:.2f}%)"
    return f"✓ [{market}] Trailing stop: off (DD={state['last_drawdown']*100:.2f}%)"
