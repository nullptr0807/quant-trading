#!/usr/bin/env python3
"""In-place repair for historical accounts snapshots.

Recomputes existing `accounts` rows from trades + raw prices + corporate actions
and updates the rows in place after creating a backup table.

Safety choices:
- Does NOT touch trades, account_state, positions, or today's live snapshots.
- Updates only accounts rows with timestamp < --cutoff (default: current UTC day).
- Creates accounts_backup_inplace_corrected_<stamp> before any UPDATE.
- Uses ledger_repair_baselines as reset points when present.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

DB_PATH = Path.home() / "quant-trading" / "data" / "trading.db"
ACTIONS_CSV = Path("/tmp/hermes/corporate_action_audit_fast/affected_account_intervals_fast.csv")


def q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def parse_ts(s: str) -> pd.Timestamp:
    return pd.to_datetime(str(s), utc=True)


def day(s: str | pd.Timestamp) -> str:
    return str(s)[:10]


def utc_today_start() -> str:
    return datetime.now(timezone.utc).date().isoformat() + "T00:00:00+00:00"


@dataclass
class State:
    cash: float
    positions: dict[str, dict[str, float]]


class CostModel:
    def __init__(self, market: str):
        if market == "CN":
            from trading.costs import CNCosts
            self.impl = CNCosts()
        else:
            from trading.costs import MoomooAUCosts
            self.impl = MoomooAUCosts()
    def calc(self, side: str, shares: float, price: float) -> tuple[float, float, float]:
        fees = self.impl.calculate(side, shares, price)
        exec_price = float(fees["exec_price"])
        total_fees = float(fees["total_fees"])
        amount = shares * exec_price
        return exec_price, total_fees, amount


def load_actions() -> pd.DataFrame:
    if not ACTIONS_CSV.exists():
        return pd.DataFrame(columns=["account","ticker","ex_date","action_type","ratio","cash_per_share"])
    df = pd.read_csv(ACTIONS_CSV)
    needed = ["account","ticker","ex_date","action_type","ratio","cash_per_share"]
    for col in needed:
        if col not in df.columns:
            df[col] = None
    # Rows with held_shares_before=0 in the audit are informational artifacts; skip.
    if "held_shares_before" in df.columns:
        df = df[pd.to_numeric(df["held_shares_before"], errors="coerce").fillna(0) > 0]
    return df[needed]


def load_price_map(conn: sqlite3.Connection, tickers: set[str], start: str, end: str) -> dict[str, tuple[list[str], list[float]]]:
    if not tickers:
        return {}
    ph = ",".join("?" * len(tickers))
    rows = conn.execute(
        f"""
        SELECT ticker, substr(datetime,1,10) AS d, close
        FROM prices_raw
        WHERE interval='1d' AND ticker IN ({ph})
          AND substr(datetime,1,10) <= ?
        ORDER BY ticker, datetime
        """,
        [*sorted(tickers), end],
    ).fetchall()
    out: dict[str, tuple[list[str], list[float]]] = {}
    for r in rows:
        if r["close"] is None:
            continue
        dates, vals = out.setdefault(str(r["ticker"]), ([], []))
        dates.append(str(r["d"]))
        vals.append(float(r["close"]))
    return out


def price_on(price_map: dict[str, tuple[list[str], list[float]]], ticker: str, ts: str) -> float | None:
    pair = price_map.get(ticker)
    if pair is None:
        return None
    dates, vals = pair
    if not dates:
        return None
    idx = bisect_right(dates, day(ts)) - 1
    if idx < 0:
        return None
    return vals[idx]


def mark_to_market(st: State, price_map: dict[str, tuple[list[str], list[float]]], ts: str) -> tuple[float, list[str]]:
    mv = 0.0
    missing: list[str] = []
    for t, p in st.positions.items():
        sh = float(p.get("shares", 0.0))
        if sh <= 1e-9:
            continue
        px = price_on(price_map, t, ts)
        if px is None:
            missing.append(t)
            px = float(p.get("total_cost", 0.0)) / sh if sh else 0.0
        mv += sh * px
    return st.cash + mv, missing


def apply_trade(st: State, trade: sqlite3.Row, costs: CostModel) -> dict[str, Any] | None:
    side = str(trade["side"]).lower()
    t = str(trade["ticker"])
    sh = float(trade["shares"])
    px = float(trade["price"])
    _, fee, amount = costs.calc(side, sh, px)
    p = st.positions.setdefault(t, {"shares": 0.0, "total_cost": 0.0})
    if side == "buy":
        outlay = amount + fee
        st.cash -= outlay
        p["shares"] += sh
        p["total_cost"] += outlay
        return None
    if side == "sell":
        proceeds = amount - fee
        st.cash += proceeds
        before = float(p.get("shares", 0.0))
        if before + 1e-9 < sh:
            # Don't poison replay; sell what exists and report.
            sold = max(0.0, before)
            err = {"trade_id": trade["id"], "ticker": t, "sell": sh, "held": before, "timestamp": trade["timestamp"]}
        else:
            sold = sh
            err = None
        avg = float(p.get("total_cost", 0.0)) / before if before > 1e-12 else 0.0
        p["shares"] = before - sold
        p["total_cost"] = float(p.get("total_cost", 0.0)) - avg * sold
        if p["shares"] <= 1e-9:
            p["shares"] = 0.0
            p["total_cost"] = 0.0
        return err
    return {"trade_id": trade["id"], "ticker": t, "side": side, "timestamp": trade["timestamp"]}


def apply_action(st: State, action: dict[str, Any]) -> dict[str, Any] | None:
    t = str(action["ticker"])
    p = st.positions.get(t)
    sh = 0.0 if not p else float(p.get("shares", 0.0))
    if sh <= 1e-9:
        return None
    typ = str(action["action_type"])
    if typ in ("split", "bonus_or_transfer"):
        ratio = float(action.get("ratio") or 0.0)
        if not ratio or not math.isfinite(ratio):
            return None
        old_sh = sh
        p["shares"] = old_sh * ratio
        # total_cost unchanged, avg_cost changes implicitly.
        return {"type": typ, "ticker": t, "ratio": ratio, "old_shares": old_sh, "new_shares": p["shares"], "timestamp": action["ex_date"]}
    if typ == "cash_dividend":
        cps = float(action.get("cash_per_share") or 0.0)
        if not cps or not math.isfinite(cps):
            return None
        amount = sh * cps
        st.cash += amount
        return {"type": typ, "ticker": t, "shares": sh, "cash_per_share": cps, "amount": amount, "timestamp": action["ex_date"]}
    return None


def baseline_events(conn: sqlite3.Connection, account: str, market: str) -> list[dict[str, Any]]:
    try:
        rows = conn.execute(
            "SELECT baseline_ts,cash,positions_json,id FROM ledger_repair_baselines WHERE account=? AND market=? ORDER BY baseline_ts,id",
            (account, market),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    out = []
    for r in rows:
        raw = json.loads(r["positions_json"] or "{}")
        pos = {str(t): {"shares": float(p.get("shares",0.0)), "total_cost": float(p.get("total_cost",0.0))} for t,p in raw.items()}
        out.append({"kind": "baseline", "ts": r["baseline_ts"], "cash": float(r["cash"]), "positions": pos, "id": r["id"]})
    return out


def repair_account(conn: sqlite3.Connection, meta: sqlite3.Row, actions_df: pd.DataFrame, cutoff: str, dry_run: bool) -> dict[str, Any]:
    acct = meta["account_id"]
    market = meta["market"] or "US"
    initial = float(meta["initial_cash"] or (100000.0 if market == "CN" else 10000.0))
    account_rows = conn.execute(
        "SELECT id,timestamp,cash,equity FROM accounts WHERE name=? AND market=? AND timestamp<? ORDER BY timestamp,id",
        (acct, market, cutoff),
    ).fetchall()
    if not account_rows:
        return {"account": acct, "market": market, "rows": 0, "updated": 0}
    trades = conn.execute(
        "SELECT id,timestamp,ticker,side,shares,price FROM trades WHERE account=? AND market=? AND timestamp<? ORDER BY timestamp,id",
        (acct, market, cutoff),
    ).fetchall()
    acts = []
    for _, r in actions_df[(actions_df.account == acct)].iterrows():
        ts = str(r["ex_date"]) + "T00:00:00+00:00"
        if ts < cutoff:
            acts.append({"kind": "action", "ts": ts, **r.to_dict()})
    bases = baseline_events(conn, acct, market)
    bases = [b for b in bases if b["ts"] < cutoff]
    tickers = {str(t["ticker"]) for t in trades} | {str(a["ticker"]) for a in acts}
    for b in bases:
        tickers |= set(b["positions"].keys())
    price_map = load_price_map(conn, tickers, str(account_rows[0]["timestamp"])[:10], cutoff[:10])
    events: list[dict[str, Any]] = []
    for tr in trades:
        events.append({"kind": "trade", "ts": tr["timestamp"], "row": tr})
    events.extend(acts)
    events.extend(bases)
    # stable order: baseline before actions/trades at same timestamp; actions before trades.
    order = {"baseline": 0, "action": 1, "trade": 2}
    events.sort(key=lambda e: (str(e["ts"]), order[e["kind"]], e["row"]["id"] if e["kind"] == "trade" else 0))
    st = State(cash=initial, positions={})
    costs = CostModel(market)
    ev_i = 0
    updates: list[tuple[float, float, int]] = []
    missing_points = 0
    overs: list[dict[str, Any]] = []
    applied_actions: list[dict[str, Any]] = []
    for ar in account_rows:
        ts = ar["timestamp"]
        while ev_i < len(events) and str(events[ev_i]["ts"]) <= str(ts):
            ev = events[ev_i]
            if ev["kind"] == "baseline":
                st = State(cash=float(ev["cash"]), positions={t: dict(p) for t,p in ev["positions"].items()})
            elif ev["kind"] == "action":
                got = apply_action(st, ev)
                if got:
                    applied_actions.append(got)
            else:
                err = apply_trade(st, ev["row"], costs)
                if err:
                    overs.append(err)
            ev_i += 1
        eq, missing = mark_to_market(st, price_map, ts)
        if missing:
            missing_points += 1
        if math.isfinite(eq) and math.isfinite(st.cash):
            updates.append((round(st.cash, 6), round(eq, 6), int(ar["id"])))
    if not dry_run and updates:
        conn.executemany("UPDATE accounts SET cash=?, equity=? WHERE id=?", updates)
    diffs = []
    for (cash, eq, row_id), ar in zip(updates, account_rows):
        diffs.append(eq - float(ar["equity"]))
    return {
        "account": acct,
        "market": market,
        "rows": len(account_rows),
        "updated": len(updates),
        "max_abs_diff": max((abs(x) for x in diffs), default=0.0),
        "net_last_diff": diffs[-1] if diffs else 0.0,
        "missing_price_points": missing_points,
        "oversells": len(overs),
        "actions_applied": len(applied_actions),
        "actions_cash": sum(a.get("amount", 0.0) for a in applied_actions),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--cutoff", default=utc_today_start(), help="update rows with timestamp < cutoff; default current UTC day start")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()
    db = Path(args.db).expanduser()
    conn = sqlite3.connect(db, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    actions = load_actions()
    metas = conn.execute("SELECT * FROM account_meta ORDER BY market, account_id").fetchall()
    dry = not args.yes
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_name = f"accounts_backup_inplace_corrected_{stamp}"
    summaries = []
    try:
        if not dry:
            conn.execute("BEGIN")
            conn.execute(
                f"CREATE TABLE {q(backup_name)} AS SELECT * FROM accounts WHERE timestamp<?",
                (args.cutoff,),
            )
        for meta in metas:
            summaries.append(repair_account(conn, meta, actions, args.cutoff, dry))
        if not dry:
            detail = {"backup_table": backup_name, "cutoff": args.cutoff, "accounts": summaries}
            conn.execute(
                "INSERT INTO events (ts, category, severity, account, ticker, title, detail, market) VALUES (?, 'system', 'warn', NULL, NULL, ?, ?, 'ALL')",
                (datetime.now(timezone.utc).isoformat(), f"🧹 In-place repaired accounts history before {args.cutoff}", json.dumps(detail, ensure_ascii=False)),
            )
            conn.commit()
    except Exception:
        if not dry:
            conn.rollback()
        raise
    finally:
        conn.close()
    out = {
        "dry_run": dry,
        "cutoff": args.cutoff,
        "backup_table": None if dry else backup_name,
        "accounts": len(summaries),
        "rows": sum(s.get("rows",0) for s in summaries),
        "updated": sum(s.get("updated",0) for s in summaries),
        "missing_price_points": sum(s.get("missing_price_points",0) for s in summaries),
        "oversells": sum(s.get("oversells",0) for s in summaries),
        "actions_applied": sum(s.get("actions_applied",0) for s in summaries),
        "top_diffs": sorted([s for s in summaries if s.get("updated",0)], key=lambda x: x.get("max_abs_diff",0), reverse=True)[:20],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
