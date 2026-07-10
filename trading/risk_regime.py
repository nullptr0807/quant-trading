"""Risk-regime manager: auto-arm / auto-disarm trailing stop based on
portfolio drawdown, with persistent state and audit trail.

State machine:
    DISARMED  →  ARMED  : portfolio drawdown >= AUTO_ARM_DD_PCT (one tick)
    ARMED     →  DISARMED : drawdown <= AUTO_DISARM_DD_PCT for AUTO_DISARM_DAYS
                              consecutive days

State persists in `risk_regime` table (created on first call). Every state
transition writes an `events` row + sends a Telegram alert.

Reading effective trailing-stop pct:
    get_effective_trailing_stop() → returns float or None
        - If config.TRAILING_STOP_PCT is set manually, returns it (override)
        - Else if state == ARMED, returns config.AUTO_ARM_TRAIL_PCT
        - Else returns None (disabled)
"""
from __future__ import annotations
import logging
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Optional

from config import settings

log = logging.getLogger("risk_regime")

DB_PATH = "/home/gexin/quant-trading/data/trading.db"


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS risk_regime (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            state TEXT NOT NULL,             -- 'DISARMED' or 'ARMED'
            armed_at TEXT,                   -- ISO timestamp when last armed
            recovery_streak INTEGER DEFAULT 0,
            last_drawdown REAL,
            last_check_at TEXT
        )
    """)
    conn.execute(
        "INSERT OR IGNORE INTO risk_regime (id, state, recovery_streak) "
        "VALUES (1, 'DISARMED', 0)"
    )
    conn.commit()


def _portfolio_drawdown(conn: sqlite3.Connection, lookback_days: int = 30) -> float:
    """Total-equity drawdown from rolling N-day peak.

    IMPORTANT: naive `SUM(equity) GROUP BY date` is unsafe — if any account
    has no snapshot on a given day (CN holiday, cron skip, paused fetcher),
    its equity drops out of the sum and the daily total appears to crash.
    This produced a spurious 93% drawdown on 2026-05-04 (CN Labour Day) and
    auto-armed the trailing stop.

    Fix: for each (account, day) get the latest snapshot, then **forward-fill
    per account** across the lookback window using the last known equity.
    A missing-snapshot day inherits yesterday's equity, so the total stays
    coherent and only reflects real P&L moves.

    Returns a non-negative float (0.045 = 4.5% below peak).
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
    rows = conn.execute("""
        SELECT name, d, equity FROM (
            SELECT name, date(timestamp) AS d, equity,
                   ROW_NUMBER() OVER (
                       PARTITION BY name, date(timestamp)
                       ORDER BY timestamp DESC
                   ) AS rn
              FROM accounts
             WHERE timestamp >= ?
               AND name NOT IN (
                   SELECT account_id FROM account_meta WHERE status='retired'
               )
        ) WHERE rn=1
        ORDER BY name, d
    """, (cutoff,)).fetchall()
    if not rows:
        return 0.0

    # Build per-account series, then forward-fill across the union of all days
    from collections import defaultdict
    per_acct: dict[str, dict[str, float]] = defaultdict(dict)
    all_days: set[str] = set()
    for name, d, eq in rows:
        if eq is None or eq <= 0:
            continue
        per_acct[name][d] = float(eq)
        all_days.add(d)
    if len(all_days) < 2 or not per_acct:
        return 0.0

    sorted_days = sorted(all_days)
    daily_total: list[float] = []
    for d in sorted_days:
        total = 0.0
        for name, series in per_acct.items():
            # Most recent equity for this account on or before day d.
            # Skip account entirely until its first snapshot (it didn't exist yet).
            last_eq = None
            for sd in sorted_days:
                if sd > d:
                    break
                if sd in series:
                    last_eq = series[sd]
            if last_eq is not None:
                total += last_eq
        if total > 0:
            daily_total.append(total)

    if len(daily_total) < 2:
        return 0.0
    peak = max(daily_total)
    last = daily_total[-1]
    if peak <= 0:
        return 0.0
    return max(0.0, (peak - last) / peak)


def _record_event(conn: sqlite3.Connection, kind: str, message: str) -> None:
    try:
        # events schema: id, ts, category, severity, account, ticker, title, detail, market
        title = message.splitlines()[0][:120]
        conn.execute(
            "INSERT INTO events (ts, category, severity, title, detail) "
            "VALUES (?, 'risk', 'warn', ?, ?)",
            (datetime.now(timezone.utc).isoformat(), title, message),
        )
    except Exception as e:
        log.warning("could not write event row: %s", e)


def _send_telegram(message: str) -> None:
    """Best-effort Telegram notification. Never raises."""
    try:
        from reports.telegram import TelegramReporter
        TelegramReporter().send_message(message)
        return
    except Exception as e:
        log.warning("telegram send failed: %s", e)


def get_state(
    conn: Optional[sqlite3.Connection] = None,
    *,
    db_path: str = DB_PATH,
) -> dict:
    own = conn is None
    c = conn or sqlite3.connect(db_path)
    try:
        _ensure_table(c)
        r = c.execute(
            "SELECT state, armed_at, recovery_streak, last_drawdown, last_check_at "
            "FROM risk_regime WHERE id=1"
        ).fetchone()
    finally:
        if own:
            c.close()
    return dict(state=r[0], armed_at=r[1], recovery_streak=r[2] or 0,
                last_drawdown=r[3] or 0.0, last_check_at=r[4])


def evaluate_and_update(*, db_path: str = DB_PATH) -> dict:
    """Compute drawdown, run state machine, persist + alert on transitions.
    Call once per day (e.g. after equity-snapshot update). Idempotent within
    a day — re-runs just refresh `last_drawdown`."""
    conn = sqlite3.connect(db_path)
    try:
        _ensure_table(conn)
        dd = _portfolio_drawdown(conn)
        now_iso = datetime.now(timezone.utc).isoformat()
        st = get_state(conn)
        cur_state = st["state"]
        new_state = cur_state
        streak = st["recovery_streak"]

        if cur_state == "DISARMED":
            if dd >= settings.AUTO_ARM_DD_PCT:
                new_state = "ARMED"
                streak = 0
                msg = (f"🚨 *RISK REGIME: AUTO-ARMED*\n"
                       f"Portfolio drawdown {dd*100:.2f}% ≥ "
                       f"{settings.AUTO_ARM_DD_PCT*100:.1f}% threshold.\n"
                       f"Trailing stop = {settings.AUTO_ARM_TRAIL_PCT*100:.1f}% "
                       f"now active.\nDisarms after {settings.AUTO_DISARM_DAYS}d "
                       f"recovery below {settings.AUTO_DISARM_DD_PCT*100:.1f}%.")
                log.warning(msg.replace("*", ""))
                _record_event(conn, "risk_regime_armed", msg)
                _send_telegram(msg)
        else:  # ARMED
            if dd <= settings.AUTO_DISARM_DD_PCT:
                streak += 1
                if streak >= settings.AUTO_DISARM_DAYS:
                    new_state = "DISARMED"
                    streak = 0
                    msg = (f"✅ *RISK REGIME: AUTO-DISARMED*\n"
                           f"Drawdown recovered to {dd*100:.2f}% for "
                           f"{settings.AUTO_DISARM_DAYS} consecutive days.\n"
                           f"Trailing stop deactivated, returning to fixed stop.")
                    log.info(msg.replace("*", ""))
                    _record_event(conn, "risk_regime_disarmed", msg)
                    _send_telegram(msg)
            else:
                streak = 0   # any day above threshold resets recovery counter

        conn.execute("""
            UPDATE risk_regime
               SET state=?, armed_at=COALESCE(?, armed_at),
                   recovery_streak=?, last_drawdown=?, last_check_at=?
             WHERE id=1
        """, (
            new_state,
            now_iso if (new_state == "ARMED" and cur_state == "DISARMED") else None,
            streak, dd, now_iso,
        ))
        conn.commit()
        return dict(state=new_state, drawdown=dd, recovery_streak=streak,
                    transitioned=(new_state != cur_state))
    finally:
        conn.close()


def get_effective_trailing_stop(*, db_path: str = DB_PATH) -> Optional[float]:
    """The single source of truth used by trading logic.
    Priority: manual override > auto-armed > disabled."""
    if settings.TRAILING_STOP_PCT is not None:
        return float(settings.TRAILING_STOP_PCT)
    st = get_state(db_path=db_path)
    if st["state"] == "ARMED":
        return float(settings.AUTO_ARM_TRAIL_PCT)
    return None


def status_line() -> str:
    """One-line human-readable status, for daily Telegram digest."""
    st = get_state()
    eff = get_effective_trailing_stop()
    if settings.TRAILING_STOP_PCT is not None:
        return (f"🛡 Trailing stop: *MANUAL ON* "
                f"({settings.TRAILING_STOP_PCT*100:.1f}%, "
                f"DD={st['last_drawdown']*100:.2f}%)")
    if st["state"] == "ARMED":
        return (f"🚨 Trailing stop: *AUTO-ARMED* "
                f"({settings.AUTO_ARM_TRAIL_PCT*100:.1f}%, "
                f"DD={st['last_drawdown']*100:.2f}%, "
                f"recovery streak {st['recovery_streak']}/"
                f"{settings.AUTO_DISARM_DAYS}d)")
    return f"✓ Trailing stop: off (DD={st['last_drawdown']*100:.2f}%)"
