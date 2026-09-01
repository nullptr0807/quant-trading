#!/usr/bin/env python3
"""Daily ledger watchdog for quant-trading.

Reconciles trades → cash/positions → account_state/positions/accounts/events.
Designed to catch B11-style dirty state before a bad equity curve is mistaken
for a bad strategy.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path.home() / "quant-trading" / "data" / "trading.db"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trading.costs import CNCosts, MoomooAUCosts  # noqa: E402
from config.security_master import canonical_ticker  # noqa: E402
from data.us_market_calendar import is_us_regular_session  # noqa: E402

_CN_SUFFIXES = (".SH", ".SZ", ".BJ")


def is_cn_ticker(ticker: str) -> bool:
    return str(ticker or "").upper().endswith(_CN_SUFFIXES)


@dataclass
class Issue:
    severity: str  # critical | warning | info
    account: str
    check: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class AccountReplay:
    cash: float
    positions: dict[str, dict[str, float]]
    oversells: list[dict[str, Any]]
    buys_notional_day: float = 0.0
    sells_notional_day: float = 0.0
    fees_day: float = 0.0
    slippage_day: float = 0.0
    trade_count_day: int = 0


def _load_repair_baseline(
    conn: sqlite3.Connection,
    account: str,
    market: str,
) -> tuple[str, float, dict[str, dict[str, float]]] | None:
    """Return latest explicit ledger repair baseline, if present.

    A baseline means old trades before `baseline_ts` are preserved for audit but
    are not the authoritative current ledger. Replay starts from the recorded
    cash+positions, then applies later trades only.
    """
    try:
        row = conn.execute(
            """
            SELECT baseline_ts,cash,positions_json
            FROM ledger_repair_baselines
            WHERE account=? AND market=?
            ORDER BY baseline_ts DESC,id DESC
            LIMIT 1
            """,
            (account, market),
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    if not row:
        return None
    raw = json.loads(row["positions_json"] or "{}")
    positions = {
        str(t): {"shares": float(p.get("shares", 0.0)), "total_cost": float(p.get("total_cost", 0.0))}
        for t, p in raw.items()
        if abs(float(p.get("shares", 0.0))) > 1e-9
    }
    return str(row["baseline_ts"]), float(row["cash"]), positions


def parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    s = str(ts).replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        # SQLite date-only fallback
        try:
            dt = datetime.fromisoformat(s[:19])
        except Exception:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def iso_day_bounds(day: str) -> tuple[str, str]:
    start = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return start.isoformat(), end.isoformat()


def money(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "N/A"
    return f"${x:,.2f}"


def rowdict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def _load_corporate_action_repairs(
    conn: sqlite3.Connection, account: str, market: str, after_ts: str | None = None,
) -> list[dict[str, Any]]:
    """Load audited share repairs as immutable replay events."""
    try:
        rows = conn.execute(
            """
            SELECT action_key,ticker,ex_date,ratio,action_type,applied_at
            FROM corporate_action_repairs
            WHERE account=? AND market=?
              AND action_type IN ('split','bonus_or_transfer')
            ORDER BY ex_date,applied_at,action_key
            """,
            (account, market),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc):
            return []
        raise
    actions = [dict(row) for row in rows]
    if after_ts is not None:
        baseline_dt = parse_ts(after_ts)
        if baseline_dt is not None:
            pending = []
            for action in actions:
                applied_dt = parse_ts(str(action["applied_at"]))
                if applied_dt is None or applied_dt <= baseline_dt:
                    continue
                # A later repair of an older action is not represented in the
                # baseline. Apply it immediately after that authoritative reset.
                action["replay_ts"] = max(
                    _corporate_action_ts(action), baseline_dt + timedelta(microseconds=1)
                ).isoformat()
                pending.append(action)
            actions = pending
    return actions


def _corporate_action_ts(action: dict[str, Any]) -> datetime:
    replay_ts = parse_ts(action.get("replay_ts"))
    if replay_ts is not None:
        return replay_ts
    return datetime.strptime(str(action["ex_date"]), "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _apply_corporate_action(
    positions: dict[str, dict[str, float]], action: dict[str, Any],
    violations: list[dict[str, Any]], market: str,
) -> None:
    ticker = canonical_ticker(str(action["ticker"]), market, str(action["ex_date"]))
    ratio = float(action["ratio"])
    position = positions.get(ticker)
    if (
        position is None or float(position.get("shares", 0.0)) <= 0
        or not math.isfinite(ratio) or ratio <= 0
    ):
        violations.append({
            "action_key": action["action_key"], "ex_date": action["ex_date"],
            "ticker": ticker, "ratio": ratio,
            "error": "invalid_corporate_action_replay_state",
        })
        return
    # Split changes inventory coordinates only. Total lot cost is immutable;
    # therefore replayed average cost is inversely adjusted by the same ratio.
    position["shares"] *= ratio


def _canonicalize_position_symbols(
    positions: dict[str, dict[str, float]], market: str, at: str,
) -> dict[str, dict[str, float]]:
    """Apply audited identifier changes to replay inventory at the audit cutoff."""
    out: dict[str, dict[str, float]] = {}
    for ticker, position in positions.items():
        canonical = canonical_ticker(ticker, market, at)
        target = out.setdefault(canonical, {"shares": 0.0, "total_cost": 0.0})
        target["shares"] += float(position.get("shares", 0.0))
        target["total_cost"] += float(position.get("total_cost", 0.0))
    return out


def replay_account(
    conn: sqlite3.Connection,
    account: str,
    market: str,
    initial_cash: float,
    day_start: str,
    day_end: str,
) -> AccountReplay:
    costs = CNCosts() if market == "CN" else MoomooAUCosts()
    baseline = _load_repair_baseline(conn, account, market)
    baseline_ts: str | None = None
    if baseline:
        baseline_ts, cash, pos = baseline
        rows = conn.execute(
            """
            SELECT id,timestamp,ticker,side,shares,price,cost,COALESCE(slippage,0) AS slippage
            FROM trades
            WHERE account=? AND market=? AND timestamp>?
            ORDER BY timestamp,id
            """,
            (account, market, baseline_ts),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT id,timestamp,ticker,side,shares,price,cost,COALESCE(slippage,0) AS slippage
            FROM trades
            WHERE account=? AND market=?
            ORDER BY timestamp,id
            """,
            (account, market),
        ).fetchall()
        cash = float(initial_cash)
        pos: dict[str, dict[str, float]] = {}
    oversells: list[dict[str, Any]] = []
    actions = _load_corporate_action_repairs(
        conn, account, market, baseline_ts if baseline else None
    )
    action_index = 0
    buys_notional_day = sells_notional_day = fees_day = slippage_day = 0.0
    trade_count_day = 0

    for r in rows:
        trade_ts = parse_ts(str(r["timestamp"]))
        while (
            action_index < len(actions)
            and trade_ts is not None
            and _corporate_action_ts(actions[action_index]) <= trade_ts
        ):
            _apply_corporate_action(pos, actions[action_index], oversells, market)
            action_index += 1
        side = str(r["side"]).lower()
        ticker = canonical_ticker(str(r["ticker"]), market, str(r["timestamp"]))
        shares = float(r["shares"])
        price = float(r["price"])
        fees = costs.calculate(side, shares, price) if side in ("buy", "sell") else None
        total_fees = float(fees["total_fees"]) if fees else float(r["cost"] or 0.0)
        exec_price = float(fees["exec_price"]) if fees else price
        slip_amount = abs(exec_price - price) * shares if fees else float(r["slippage"] or 0.0)
        notional = shares * price
        in_day = day_start <= str(r["timestamp"]) < day_end
        if in_day:
            trade_count_day += 1
            fees_day += total_fees
            slippage_day += slip_amount
            if side == "buy":
                buys_notional_day += notional
            elif side == "sell":
                sells_notional_day += notional

        p = pos.setdefault(ticker, {"shares": 0.0, "total_cost": 0.0})
        if side == "buy":
            # Recompute cash impact from the canonical fee/slippage model, not
            # from stored trades.cost/slippage. Older main.py trade rows store
            # quote price and often slippage=0 even though account cash used
            # exec_price.
            outlay = float(fees["amount"]) + total_fees  # type: ignore[index]
            cash -= outlay
            p["shares"] += shares
            p["total_cost"] += outlay
        elif side == "sell":
            proceeds = float(fees["amount"]) - total_fees  # type: ignore[index]
            cash += proceeds
            before = p["shares"]
            if before + 1e-9 < shares:
                oversells.append(
                    {
                        "trade_id": r["id"],
                        "timestamp": r["timestamp"],
                        "ticker": ticker,
                        "sell_shares": shares,
                        "held_before": before,
                    }
                )
                # Continue replay without letting negative inventory poison
                # total_cost. The mismatch will be reported separately.
                p["shares"] = before - shares
                continue
            avg = p["total_cost"] / before if before else 0.0
            p["shares"] -= shares
            p["total_cost"] -= avg * shares
            if abs(p["shares"]) < 1e-9:
                p["shares"] = 0.0
                p["total_cost"] = 0.0
        else:
            oversells.append(
                {
                    "trade_id": r["id"],
                    "timestamp": r["timestamp"],
                    "ticker": ticker,
                    "side": side,
                    "error": "unknown_side",
                }
            )

    while action_index < len(actions):
        _apply_corporate_action(pos, actions[action_index], oversells, market)
        action_index += 1

    # Daily flow stats are independent of replay baselines: a baseline may be
    # written after today's trades to repair current state, but the watchdog
    # report must still count those real trades/events for the day.
    day_rows = conn.execute(
        """
        SELECT id,timestamp,ticker,side,shares,price,cost,COALESCE(slippage,0) AS slippage
        FROM trades
        WHERE account=? AND market=? AND timestamp>=? AND timestamp<?
        ORDER BY timestamp,id
        """,
        (account, market, day_start, day_end),
    ).fetchall()
    buys_notional_day = sells_notional_day = fees_day = slippage_day = 0.0
    trade_count_day = 0
    for r in day_rows:
        side = str(r["side"]).lower()
        shares = float(r["shares"])
        price = float(r["price"])
        fees = costs.calculate(side, shares, price) if side in ("buy", "sell") else None
        total_fees = float(fees["total_fees"]) if fees else float(r["cost"] or 0.0)
        exec_price = float(fees["exec_price"]) if fees else price
        slip_amount = abs(exec_price - price) * shares if fees else float(r["slippage"] or 0.0)
        notional = shares * price
        trade_count_day += 1
        fees_day += total_fees
        slippage_day += slip_amount
        if side == "buy":
            buys_notional_day += notional
        elif side == "sell":
            sells_notional_day += notional

    canonical_pos = _canonicalize_position_symbols(pos, market, day_end)
    clean_pos = {
        t: p for t, p in canonical_pos.items()
        if abs(p.get("shares", 0.0)) > 1e-9
    }
    return AccountReplay(
        cash=cash,
        positions=clean_pos,
        oversells=oversells,
        buys_notional_day=buys_notional_day,
        sells_notional_day=sells_notional_day,
        fees_day=fees_day,
        slippage_day=slippage_day,
        trade_count_day=trade_count_day,
    )


def compare_positions(
    replay_pos: dict[str, dict[str, float]],
    db_pos_rows: list[sqlite3.Row],
    share_tol: float,
    cost_tol: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    db = {r["ticker"]: dict(r) for r in db_pos_rows}
    tickers = sorted(set(replay_pos) | set(db))
    for t in tickers:
        rp = replay_pos.get(t, {"shares": 0.0, "total_cost": 0.0})
        dp = db.get(t, {"shares": 0.0, "total_cost": 0.0})
        rs = float(rp.get("shares", 0.0))
        ds = float(dp.get("shares", 0.0) or 0.0)
        rc = float(rp.get("total_cost", 0.0))
        dc = float(dp.get("total_cost", 0.0) or 0.0)
        if abs(rs - ds) > share_tol or abs(rc - dc) > cost_tol:
            issues.append(
                {
                    "ticker": t,
                    "replay_shares": rs,
                    "db_shares": ds,
                    "replay_total_cost": rc,
                    "db_total_cost": dc,
                    "share_diff": ds - rs,
                    "cost_diff": dc - rc,
                }
            )
    return issues, list(db.values())


def latest_account_row(conn: sqlite3.Connection, account: str, market: str) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT * FROM accounts
        WHERE name=? AND market=?
        ORDER BY timestamp DESC,id DESC LIMIT 1
        """,
        (account, market),
    ).fetchone()


def previous_and_last_cash_for_day(
    conn: sqlite3.Connection,
    account: str,
    market: str,
    day_start: str,
    day_end: str,
) -> tuple[sqlite3.Row | None, sqlite3.Row | None]:
    prev = conn.execute(
        """
        SELECT * FROM accounts WHERE name=? AND market=? AND timestamp < ?
        ORDER BY timestamp DESC,id DESC LIMIT 1
        """,
        (account, market, day_start),
    ).fetchone()
    last = conn.execute(
        """
        SELECT * FROM accounts WHERE name=? AND market=? AND timestamp < ?
        ORDER BY timestamp DESC,id DESC LIMIT 1
        """,
        (account, market, day_end),
    ).fetchone()
    return prev, last


def _trade_cash_model(market: str):
    return CNCosts() if market == "CN" else MoomooAUCosts()


def _replay_series(
    conn: sqlite3.Connection,
    account: str,
    market: str,
    initial_cash: float,
    account_rows: list[sqlite3.Row],
) -> list[tuple[sqlite3.Row, float, dict[str, dict[str, float]], list[dict[str, Any]]]]:
    """Incrementally replay trades to every account snapshot timestamp."""
    costs = _trade_cash_model(market)
    baseline = _load_repair_baseline(conn, account, market)
    baseline_ts: str | None = None
    if baseline:
        baseline_ts, cash, positions = baseline
        trades = conn.execute(
            """
            SELECT id,timestamp,ticker,side,shares,price
            FROM trades WHERE account=? AND market=? AND timestamp>?
            ORDER BY timestamp,id
            """,
            (account, market, baseline_ts),
        ).fetchall()
    else:
        trades = conn.execute(
            """
            SELECT id,timestamp,ticker,side,shares,price
            FROM trades WHERE account=? AND market=? ORDER BY timestamp,id
            """,
            (account, market),
        ).fetchall()
        cash = float(initial_cash)
        positions: dict[str, dict[str, float]] = {}
    out = []
    ti = 0
    oversells: list[dict[str, Any]] = []
    actions = _load_corporate_action_repairs(conn, account, market, baseline_ts)
    ai = 0

    def apply_trade(r: sqlite3.Row):
        nonlocal cash
        side = str(r["side"]).lower()
        ticker = canonical_ticker(str(r["ticker"]), market, str(r["timestamp"]))
        shares = float(r["shares"])
        price = float(r["price"])
        fees = costs.calculate(side, shares, price) if side in ("buy", "sell") else None
        p = positions.setdefault(ticker, {"shares": 0.0, "total_cost": 0.0})
        if side == "buy":
            outlay = float(fees["amount"]) + float(fees["total_fees"])  # type: ignore[index]
            cash -= outlay
            p["shares"] += shares
            p["total_cost"] += outlay
        elif side == "sell":
            proceeds = float(fees["amount"]) - float(fees["total_fees"])  # type: ignore[index]
            cash += proceeds
            before = p["shares"]
            if before + 1e-9 < shares:
                oversells.append({
                    "trade_id": r["id"], "timestamp": r["timestamp"],
                    "ticker": ticker, "sell_shares": shares, "held_before": before,
                })
                p["shares"] = before - shares
                return
            avg = p["total_cost"] / before if before else 0.0
            p["shares"] -= shares
            p["total_cost"] -= avg * shares
            if abs(p["shares"]) < 1e-9:
                p["shares"] = 0.0
                p["total_cost"] = 0.0

    for row in account_rows:
        ts = str(row["timestamp"])
        snapshot_dt = parse_ts(ts)
        while snapshot_dt is not None:
            trade_dt = parse_ts(str(trades[ti]["timestamp"])) if ti < len(trades) else None
            action_dt = _corporate_action_ts(actions[ai]) if ai < len(actions) else None
            trade_ready = trade_dt is not None and trade_dt <= snapshot_dt
            action_ready = action_dt is not None and action_dt <= snapshot_dt
            if (
                action_ready and action_dt is not None
                and (not trade_ready or trade_dt is None or action_dt <= trade_dt)
            ):
                _apply_corporate_action(positions, actions[ai], oversells, market)
                ai += 1
            elif trade_ready:
                apply_trade(trades[ti])
                ti += 1
            else:
                break
        canonical_snapshot = _canonicalize_position_symbols(positions, market, ts)
        snap_pos = {
            t: dict(p) for t, p in canonical_snapshot.items()
            if abs(p.get("shares", 0.0)) > 1e-9
        }
        out.append((row, cash, snap_pos, list(oversells)))
    return out


_CN_SUFFIXES = (".SH", ".SZ", ".BJ")


def _is_cn_ticker(ticker: str) -> bool:
    return str(ticker).upper().endswith(_CN_SUFFIXES)


def _is_benchmark_account(account: str, market: str) -> bool:
    return account in ({"IDX1", "IDX2"} if market == "US" else {"IDX3"})


def _benchmark_ticker(account: str, market: str) -> str | None:
    if market == "CN" and account == "IDX3":
        return "000300.SH"
    if market == "US" and account in {"IDX1", "IDX2"}:
        return {"IDX1": "QQQ", "IDX2": "SPY"}[account]
    return None


def _latest_price_timestamp(
    conn: sqlite3.Connection,
    tickers: set[str],
    intervals: tuple[str, ...],
) -> datetime | None:
    if not tickers:
        return None
    placeholders = ",".join("?" for _ in tickers)
    interval_placeholders = ",".join("?" for _ in intervals)
    row = conn.execute(
        f"""
        SELECT MAX(datetime) AS max_dt
        FROM prices_raw
        WHERE ticker IN ({placeholders}) AND interval IN ({interval_placeholders})
        """,
        [*sorted(tickers), *intervals],
    ).fetchone()
    if not row or not row["max_dt"]:
        return None
    return parse_ts(str(row["max_dt"]))


def _load_price_history(
    conn: sqlite3.Connection,
    tickers: set[str],
    start_ts: str,
    end_ts: str,
    intervals: tuple[str, ...] = ("1h", "1d"),
) -> dict[str, list[tuple[float, float, str]]]:
    if not tickers:
        return {}
    placeholders = ",".join("?" for _ in tickers)
    interval_placeholders = ",".join("?" for _ in intervals)
    start_dt = parse_ts(start_ts) or datetime.now(timezone.utc)
    start_buf = (start_dt - timedelta(days=7)).isoformat()
    try:
        rows = conn.execute(
            f"""
            SELECT ticker, datetime, interval, close
            FROM prices_raw
            WHERE ticker IN ({placeholders}) AND interval IN ({interval_placeholders})
              AND datetime >= ? AND datetime <= ?
            ORDER BY ticker, datetime, CASE interval WHEN '1d' THEN 0 ELSE 1 END
            """,
            [*sorted(tickers), *intervals, start_buf, end_ts],
        ).fetchall()
    except sqlite3.OperationalError as exc:
        if "no such table: prices_raw" in str(exc):
            return {}
        raise
    out: dict[str, list[tuple[float, float, str]]] = {}
    for r in rows:
        dt = parse_ts(str(r["datetime"]))
        if dt is None:
            continue
        out.setdefault(r["ticker"], []).append((dt.timestamp(), float(r["close"]), str(r["interval"])))
    return out


def _price_at_meta(
    history: dict[str, list[tuple[float, float, str]]],
    ticker: str,
    ts: str,
    max_age_by_interval: dict[str, float] | None = None,
) -> tuple[float, str] | None:
    rows = history.get(ticker) or []
    if not rows:
        return None
    dt = parse_ts(ts)
    if dt is None:
        return None
    keys = [x[0] for x in rows]
    idx = bisect_right(keys, dt.timestamp()) - 1
    if idx < 0:
        return None
    price_ts, price, interval = rows[idx]
    max_age = (max_age_by_interval or {}).get(interval)
    if max_age is not None and dt.timestamp() - price_ts > max_age:
        return None
    return price, interval


def _price_at(history: dict[str, list[tuple[float, float, str]]], ticker: str, ts: str) -> float | None:
    meta = _price_at_meta(history, ticker, ts)
    return meta[0] if meta else None


def history_curve_audit(
    conn: sqlite3.Connection,
    account: str,
    market: str,
    initial_cash: float,
    status: str,
    audit_start: str,
    audit_end: str,
    args: argparse.Namespace,
) -> list[Issue]:
    """Compare recent accounts history rows to replayed cash+positions+prices."""
    if args.history_days <= 0:
        return []
    if status == "retired" and not args.history_include_retired:
        return []
    rows = conn.execute(
        """
        SELECT id,name,cash,equity,timestamp FROM accounts
        WHERE name=? AND market=? AND timestamp>=? AND timestamp<?
        ORDER BY timestamp,id
        """,
        (account, market, audit_start, audit_end),
    ).fetchall()
    if not rows:
        return []
    is_cn_history = market == "CN"
    if market == "US":
        # The persisted raw 5m cache is an RTH audit source. Yahoo quote
        # metadata may produce sparse pre/post snapshots that have no matching
        # historical bar; treating those expected gaps as audit failures creates
        # systematic noise. Audit RTH rows only, while current-ledger and latest
        # snapshot checks continue to cover the full account state.
        rows = [
            row for row in rows
            if (parse_ts(str(row["timestamp"])) is not None
                and is_us_regular_session(parse_ts(str(row["timestamp"]))))
        ]
        if not rows:
            return []
    raw_price_warning = Issue(
        "warning", account, "history_curve_prices",
        "historical equity audit skipped where raw execution-price coverage is insufficient; adjusted prices were not used as fallback",
        {"audit_start": audit_start, "audit_end": audit_end, "price_table": "prices_raw"},
    )
    if _is_benchmark_account(account, market):
        bt = _benchmark_ticker(account, market)
        tickers = {bt} if bt else set()
    else:
        trades = conn.execute(
            "SELECT DISTINCT ticker FROM trades WHERE account=? AND market=? AND timestamp<=?",
            (account, market, audit_end),
        ).fetchall()
        tickers = {r["ticker"] for r in trades}

    if is_cn_history:
        cn_intervals = ("15m", "5m", "1m", "1h")
        latest_intraday = _latest_price_timestamp(conn, tickers, cn_intervals) if tickers else None
        audit_start_dt = parse_ts(audit_start)
        audit_end_dt = parse_ts(audit_end)
        # CN account snapshots are intraday realtime rows. If we do not have
        # intraday bars overlapping this audit window, pricing them with stale
        # daily closes creates thousands of false critical mismatches. Keep the
        # authoritative current-ledger checks and silently skip this historical
        # curve audit until raw intraday history is backfilled/maintained. Never
        # substitute adjusted/qfq bars for the execution coordinate.
        if tickers and (
            latest_intraday is None
            or (audit_start_dt is not None and latest_intraday < audit_start_dt)
            or (audit_end_dt is not None and latest_intraday < audit_end_dt - timedelta(days=1))
        ):
            return [raw_price_warning]
    if len(rows) > args.history_max_points:
        step = len(rows) / float(args.history_max_points)
        rows = [rows[int(i * step)] for i in range(args.history_max_points)]
    price_intervals = ("15m", "5m", "1m", "1h", "1d") if is_cn_history else ("5m", "15m", "30m", "1h", "1d")
    price_hist = _load_price_history(conn, tickers, audit_start, audit_end, intervals=price_intervals)
    mismatches: list[dict[str, Any]] = []
    missing_price_rows = 0
    checked = 0
    for row, cash, pos, oversells in _replay_series(conn, account, market, initial_cash, rows):
        if oversells:
            continue
        eq = cash
        missing = []
        for ticker, ppos in pos.items():
            price_meta = _price_at_meta(
                price_hist, ticker, str(row["timestamp"]),
                max_age_by_interval={"5m": 600.0, "15m": 1800.0, "30m": 3600.0},
            )
            if price_meta is None:
                missing.append(ticker)
                continue
            px, px_interval = price_meta
            if market == "US" and px_interval in {"1d", "1h"}:
                # US snapshots are written from realtime quotes every few
                # minutes. Daily/hourly bars are too coarse for historical
                # minute-level equity rows and create false history_curve
                # criticals. If no sub-hour intraday bar exists at/before this
                # timestamp, mark the row unauditable instead of comparing
                # different price grains.
                missing.append(ticker)
                continue
            eq += float(ppos["shares"]) * px
        if missing:
            missing_price_rows += 1
            continue
        checked += 1
        diff = float(row["equity"] or 0.0) - eq
        threshold = max(float(args.history_equity_tolerance), float(initial_cash) * float(args.history_equity_tolerance_pct))
        if abs(diff) > threshold:
            mismatches.append({
                "id": row["id"],
                "timestamp": row["timestamp"],
                "stored_equity": float(row["equity"] or 0.0),
                "replayed_equity": round(eq, 4),
                "diff": round(diff, 4),
            })
            if len(mismatches) >= args.history_report_limit:
                break
    issues: list[Issue] = []
    if mismatches:
        issues.append(Issue(
            "critical", account, "history_curve",
            f"{len(mismatches)} recent accounts history row(s) disagree with replayed equity",
            {
                "audit_start": audit_start,
                "audit_end": audit_end,
                "checked_rows": checked,
                "tolerance_abs": args.history_equity_tolerance,
                "tolerance_pct": args.history_equity_tolerance_pct,
                "sample_mismatches": mismatches,
            },
        ))

    if missing_price_rows and market == "US":
        # US minute-level history rows are realtime-quote snapshots. The raw
        # price cache intentionally does not maintain complete sub-hour bars for
        # every held/legacy ticker and benchmark row, so missing fine-grain
        # history is a coverage limitation rather than a ledger issue. Never
        # fall back to adjusted prices; current replay, cash/positions, and
        # latest-equity checks remain authoritative.
        issues.append(raw_price_warning)
        return issues
    if missing_price_rows and checked == 0:
        # If every sampled row lacks raw price resolution fine enough for this
        # audit, the watchdog has learned only "history curve audit not
        # possible", not "ledger is dirty". Emit a clear operational warning
        # and skip rather than crossing into adjusted price coordinates.
        issues.append(raw_price_warning)
        return issues
    if missing_price_rows:
        issues.append(Issue(
            "warning", account, "history_curve_prices",
            f"{missing_price_rows} sampled history row(s) could not be audited due to missing raw execution prices; adjusted prices were not used as fallback",
            {"audit_start": audit_start, "audit_end": audit_end, "price_table": "prices_raw"},
        ))
    return issues


def check_account(
    conn: sqlite3.Connection,
    meta: sqlite3.Row,
    day_start: str,
    day_end: str,
    args: argparse.Namespace,
) -> tuple[list[Issue], dict[str, Any]]:
    account = meta["account_id"]
    market = meta["market"]
    status = meta["status"] or "active"
    initial_cash = float(meta["initial_cash"] or (100000.0 if market == "CN" else 10000.0))
    issues: list[Issue] = []

    replay = replay_account(conn, account, market, initial_cash, day_start, day_end)
    if replay.oversells:
        issues.append(Issue("critical", account, "oversell", f"{len(replay.oversells)} oversell / invalid-side trade(s)", {"violations": replay.oversells[:10]}))

    state = conn.execute(
        "SELECT * FROM account_state WHERE account=? AND market=?",
        (account, market),
    ).fetchone()
    if not state:
        activity = conn.execute(
            """
            SELECT
              (SELECT COUNT(*) FROM trades WHERE account=? AND market=?) AS trades,
              (SELECT COUNT(*) FROM accounts WHERE name=? AND market=?) AS snapshots,
              (SELECT COUNT(*) FROM positions WHERE account=? AND market=?) AS positions
            """,
            (account, market, account, market, account, market),
        ).fetchone()
        detail = {"trades": activity["trades"], "snapshots": activity["snapshots"], "positions": activity["positions"]} if activity else {}
        if detail and not any(detail.values()):
            # Ignore inert legacy metadata rows (e.g. C01 test account) so the
            # daily watchdog stays quiet when the real ledgers pass. If a row
            # ever has trades/snapshots/positions, missing state remains critical.
            return issues, {
                "account": account,
                "market": market,
                "status": status,
                "trade_count_day": 0,
                "buys_notional_day": 0.0,
                "sells_notional_day": 0.0,
                "fees_day": 0.0,
                "slippage_day": 0.0,
                "position_count": 0,
                "latest_equity": None,
                "state_cash": None,
                "daily_actual_cash_delta": None,
                "daily_expected_cash_delta": 0.0,
            }
        issues.append(Issue("critical", account, "account_state", "missing account_state row", detail))
        state_cash = None
    else:
        state_cash = float(state["cash"])
        if status != "retired" and state_cash < -args.negative_cash_tolerance:
            issues.append(
                Issue(
                    "critical", account, "negative_cash",
                    f"active account_state.cash is negative: {money(state_cash)}",
                    {"db_cash": state_cash, "tolerance": args.negative_cash_tolerance},
                )
            )
        cash_diff = state_cash - replay.cash
        if abs(cash_diff) > args.cash_tolerance:
            issues.append(
                Issue(
                    "critical", account, "cash_replay",
                    f"account_state.cash differs from replay by {money(cash_diff)}",
                    {"db_cash": state_cash, "replay_cash": replay.cash, "diff": cash_diff},
                )
            )

    db_pos_rows = conn.execute(
        "SELECT * FROM positions WHERE account=? AND market=? ORDER BY ticker",
        (account, market),
    ).fetchall()
    pos_mismatches, db_positions = compare_positions(
        replay.positions, db_pos_rows, args.share_tolerance, args.cost_tolerance
    )
    if pos_mismatches:
        issues.append(
            Issue(
                "critical", account, "positions_replay",
                f"positions table differs from replay for {len(pos_mismatches)} ticker(s)",
                {"mismatches": pos_mismatches[:12]},
            )
        )

    stale_positions: list[dict[str, Any]] = []
    missing_prices: list[str] = []
    now = datetime.now(timezone.utc)
    mv = 0.0
    for p in db_positions:
        ticker = p["ticker"]
        px = p.get("current_price")
        if px is None or float(px) <= 0:
            missing_prices.append(ticker)
        else:
            mv += float(p["shares"] or 0.0) * float(px)
        upd = parse_ts(p.get("updated_at"))
        if status != "retired":
            if upd is None:
                stale_positions.append({
                    "ticker": ticker,
                    "updated_at": p.get("updated_at"),
                    "age_hours": None,
                    "reason": "missing_mark_timestamp",
                })
            else:
                age_h = (now - upd).total_seconds() / 3600.0
                if age_h > args.stale_price_hours:
                    stale_positions.append({"ticker": ticker, "updated_at": p.get("updated_at"), "age_hours": round(age_h, 1)})
    if missing_prices:
        issues.append(Issue("critical", account, "position_prices", f"{len(missing_prices)} current position(s) have missing/non-positive current_price", {"tickers": missing_prices}))
    if stale_positions:
        issues.append(Issue("critical", account, "stale_prices", f"{len(stale_positions)} held position price(s) stale > {args.stale_price_hours}h; account must remain trade-blocked", {"market": market, "positions": stale_positions[:12]}))

    latest = latest_account_row(conn, account, market)
    if latest and state_cash is not None and not missing_prices:
        computed_equity = state_cash + mv
        equity_diff = float(latest["equity"]) - computed_equity
        if abs(equity_diff) > args.equity_tolerance:
            issues.append(
                Issue(
                    "critical", account, "equity_snapshot",
                    f"latest accounts.equity differs from account_state cash + positions MV by {money(equity_diff)}",
                    {
                        "latest_equity": float(latest["equity"]),
                        "computed_equity": computed_equity,
                        "diff": equity_diff,
                        "latest_timestamp": latest["timestamp"],
                    },
                )
            )
    elif status != "retired":
        issues.append(Issue("warning", account, "equity_snapshot", "missing latest accounts snapshot"))

    if args.history_days > 0:
        audit_end_dt = datetime.fromisoformat(day_end.replace('Z', '+00:00'))
        audit_start = (audit_end_dt - timedelta(days=args.history_days)).isoformat()
        issues.extend(history_curve_audit(
            conn, account, market, initial_cash, status, audit_start, day_end, args
        ))

    mislabeled_trades = conn.execute(
        """
        SELECT id,ticker,side,shares,price,timestamp,market
        FROM trades
        WHERE account=? AND market='US'
          AND (ticker LIKE '%.SH' OR ticker LIKE '%.SZ' OR ticker LIKE '%.BJ')
        ORDER BY timestamp,id
        LIMIT 20
        """,
        (account,),
    ).fetchall()
    if mislabeled_trades:
        issues.append(
            Issue(
                "critical", account, "trade_market_mislabel",
                f"{len(mislabeled_trades)} CN ticker trade row(s) stored with market='US'",
                {"rows": [dict(r) for r in mislabeled_trades[:12]]},
            )
        )

    # Daily cash-flow check based on snapshots. This is a smoke test; all-time
    # replay above is authoritative.
    prev_snap, last_snap = previous_and_last_cash_for_day(conn, account, market, day_start, day_end)
    daily_expected_delta = -replay.buys_notional_day + replay.sells_notional_day - replay.fees_day - replay.slippage_day
    daily_actual_delta = None
    has_day_snapshot = bool(
        last_snap and day_start <= str(last_snap["timestamp"]) < day_end
    )
    if replay.trade_count_day and not has_day_snapshot:
        issues.append(
            Issue(
                "warning", account, "daily_snapshot_missing",
                "trades occurred but no account snapshot was persisted during the trading day",
                {
                    "trades_day": replay.trade_count_day,
                    "day_start": day_start,
                    "day_end": day_end,
                    "last_snapshot": last_snap["timestamp"] if last_snap else None,
                },
            )
        )
    elif prev_snap and last_snap and replay.trade_count_day:
        daily_actual_delta = float(last_snap["cash"]) - float(prev_snap["cash"])
        delta_diff = daily_actual_delta - daily_expected_delta
        if abs(delta_diff) > args.cashflow_tolerance:
            issues.append(
                Issue(
                    "warning", account, "daily_cashflow",
                    f"daily snapshot cash change differs from today's trades by {money(delta_diff)}",
                    {
                        "snapshot_cash_delta": daily_actual_delta,
                        "trade_cash_delta": daily_expected_delta,
                        "diff": delta_diff,
                        "prev_snapshot": prev_snap["timestamp"],
                        "last_snapshot": last_snap["timestamp"],
                    },
                )
            )

    retired_at = parse_ts(meta["retired_at"])
    if status == "retired" and retired_at is not None:
        n_after = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE account=? AND market=? AND timestamp > ?",
            (account, market, retired_at.isoformat()),
        ).fetchone()[0]
        if n_after:
            issues.append(Issue("critical", account, "retired_trade", f"retired account has {n_after} trade(s) after retired_at", {"retired_at": meta["retired_at"]}))

    # Event stream sanity for the day.
    trade_events_day = conn.execute(
        """
        SELECT COUNT(*) FROM events
        WHERE account=? AND market=? AND category='trade' AND ts>=? AND ts<?
        """,
        (account, market, day_start, day_end),
    ).fetchone()[0]
    if replay.trade_count_day and trade_events_day < replay.trade_count_day:
        ratio = trade_events_day / max(replay.trade_count_day, 1)
        severity = "warning" if ratio >= args.event_ratio_warn else "critical"
        issues.append(
            Issue(
                severity, account, "trade_events",
                f"trade events missing for day: trades={replay.trade_count_day}, events={trade_events_day}",
                {"trades_day": replay.trade_count_day, "trade_events_day": trade_events_day, "ratio": ratio},
            )
        )

    stats = {
        "account": account,
        "market": market,
        "status": status,
        "trade_count_day": replay.trade_count_day,
        "buys_notional_day": replay.buys_notional_day,
        "sells_notional_day": replay.sells_notional_day,
        "fees_day": replay.fees_day,
        "slippage_day": replay.slippage_day,
        "position_count": len(db_positions),
        "latest_equity": float(latest["equity"]) if latest else None,
        "state_cash": state_cash,
        "daily_actual_cash_delta": daily_actual_delta,
        "daily_expected_cash_delta": daily_expected_delta,
    }
    return issues, stats


def load_accounts(
    conn: sqlite3.Connection,
    market: str,
    *,
    include_retired: bool = False,
) -> list[sqlite3.Row]:
    """Load active operational accounts, optionally including retired archives."""
    status_clause = "" if include_retired else " AND COALESCE(status,'active')='active'"
    if market == "ALL":
        where = "WHERE 1=1" + status_clause
        return conn.execute(
            f"SELECT * FROM account_meta {where} ORDER BY market, account_id"
        ).fetchall()
    return conn.execute(
        f"SELECT * FROM account_meta WHERE market=?{status_clause} ORDER BY account_id",
        (market,),
    ).fetchall()


def render_report(day: str, market: str, issues: list[Issue], stats: list[dict[str, Any]], quiet_ok: bool) -> str:
    crit = [i for i in issues if i.severity == "critical"]
    warn = [i for i in issues if i.severity == "warning"]
    info = [i for i in issues if i.severity == "info"]
    accounts = len(stats)
    active = sum(1 for s in stats if s["status"] != "retired")
    retired = accounts - active
    trades_day = sum(s["trade_count_day"] for s in stats)
    buys = sum(s["buys_notional_day"] for s in stats)
    sells = sum(s["sells_notional_day"] for s in stats)
    fees = sum(s["fees_day"] for s in stats)
    slip = sum(s["slippage_day"] for s in stats)

    if not issues and quiet_ok:
        return ""

    status_icon = "✅" if not crit and not warn else ("🚨" if crit else "⚠️")
    lines = [
        f"{status_icon} Daily Ledger Watchdog — {market} {day}",
        "",
        f"Accounts checked: {accounts} active={active} retired={retired}",
        f"Issues: critical={len(crit)} warning={len(warn)} info={len(info)}",
        f"Today's trades: {trades_day} | buys={money(buys)} sells={money(sells)} fees={money(fees)} slippage={money(slip)}",
    ]
    if not issues:
        lines.append("All ledger reconciliation checks passed: oversell=0, cash mismatch=0, position mismatch=0, equity mismatch=0.")
        return "\n".join(lines)

    def add_section(title: str, items: list[Issue], limit: int = 20):
        if not items:
            return
        lines.append("")
        lines.append(title)
        for i, issue in enumerate(items[:limit], 1):
            lines.append(f"{i}. [{issue.account}] {issue.check}: {issue.message}")
            if issue.detail:
                compact = json.dumps(issue.detail, ensure_ascii=False, default=str)
                if len(compact) > 900:
                    compact = compact[:900] + "…"
                lines.append(f"   detail: {compact}")
        if len(items) > limit:
            lines.append(f"... {len(items)-limit} more")

    add_section("🚨 Critical", crit)
    add_section("⚠️ Warnings", warn)
    add_section("ℹ️ Info", info)

    # Top daily cash-flow accounts for context.
    moved = sorted(
        [s for s in stats if s["trade_count_day"]],
        key=lambda s: s["buys_notional_day"] + s["sells_notional_day"],
        reverse=True,
    )[:8]
    if moved:
        lines.append("")
        lines.append("📊 Largest daily trade flows")
        for s in moved:
            lines.append(
                f"- {s['account']}: trades={s['trade_count_day']} buys={money(s['buys_notional_day'])} sells={money(s['sells_notional_day'])} cash={money(s['state_cash'])} equity={money(s['latest_equity'])}"
            )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Reconcile quant-trading ledger state.")
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--market", default="ALL", choices=["ALL", "US", "CN"])
    ap.add_argument("--date", default=datetime.now(timezone.utc).date().isoformat(), help="UTC date YYYY-MM-DD")
    ap.add_argument("--cash-tolerance", type=float, default=100.0)
    ap.add_argument("--cashflow-tolerance", type=float, default=100.0)
    ap.add_argument("--equity-tolerance", type=float, default=50.0)
    ap.add_argument("--negative-cash-tolerance", type=float, default=0.0,
                    help="Active accounts may not have cash below this value; default 0 forbids any negative cash")
    ap.add_argument("--share-tolerance", type=float, default=1e-6)
    ap.add_argument("--cost-tolerance", type=float, default=100.0)
    ap.add_argument("--stale-price-hours", type=float, default=36.0)
    ap.add_argument("--event-ratio-warn", type=float, default=0.8, help="Below this ratio missing trade events are critical instead of warning")
    ap.add_argument("--history-days", type=float, default=0.0,
                    help="Audit recent accounts history rows against replayed equity; 0 disables")
    ap.add_argument("--history-equity-tolerance", type=float, default=250.0)
    ap.add_argument("--history-equity-tolerance-pct", type=float, default=0.03,
                    help="History curve tolerance as a fraction of initial cash; combined with absolute tolerance via max()")
    ap.add_argument("--history-max-points", type=int, default=120,
                    help="Max account history rows sampled per account per run")
    ap.add_argument("--history-report-limit", type=int, default=8)
    ap.add_argument("--history-include-retired", action="store_true")
    ap.add_argument("--quiet-ok", action="store_true", help="Print nothing when all checks pass")
    ap.add_argument("--fail-on-critical", action="store_true")
    args = ap.parse_args()

    db = Path(args.db).expanduser()
    if not db.exists():
        raise SystemExit(f"DB not found: {db}")
    day_start, day_end = iso_day_bounds(args.date)
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        metas = load_accounts(
            conn,
            args.market,
            include_retired=args.history_include_retired,
        )
        all_issues: list[Issue] = []
        all_stats: list[dict[str, Any]] = []
        for meta in metas:
            issues, stats = check_account(conn, meta, day_start, day_end, args)
            all_issues.extend(issues)
            all_stats.append(stats)
    finally:
        conn.close()
    operational_issues = [
        issue for issue in all_issues
        if next((s for s in all_stats if s["account"] == issue.account), {}).get("status") != "retired"
    ]
    archival_issues = [issue for issue in all_issues if issue not in operational_issues]
    if args.history_include_retired and archival_issues:
        # Retired ledgers remain auditable, but they are frozen/non-operational
        # and must not page the live risk monitor.
        print(
            f"Archival findings (non-operational): {len(archival_issues)}; "
            f"Operational issues: critical={sum(i.severity == 'critical' for i in operational_issues)}"
        )
        for issue in archival_issues:
            print(f"- [{issue.account}] {issue.check}: {issue.message}")
    report = render_report(args.date, args.market, operational_issues, all_stats, args.quiet_ok)
    if report:
        print(report)
    if args.fail_on_critical and any(i.severity == "critical" for i in operational_issues):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
