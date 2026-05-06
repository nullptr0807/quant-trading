#!/usr/bin/env python3
"""One-shot historical replay for the Q-group US accounts (Q01-Q10).

Why this script exists:
    Last night (2026-04-30) the US Q-group did not trade because of a bug in
    `main._load_qlib_scores` — when CN's qlib retrain wrote 'qlib_QXX_score'
    rows for 2026-04-30, MAX(date) shadowed the US scores from 2026-04-29.
    The bug is patched, but Q-group missed a day's exposure vs A/B groups.

What this script does:
    For each Q01..Q10, reproduce the trades that *would* have run on 04-30:
      * Read scores from factor_values WHERE factor_name='qlib_QXX_score'
        AND date='2026-04-29' AND ticker IN US universe.
      * Use 04-29 close as the entry price (same data the model saw — this
        keeps signal/execution aligned; pre-open of 04-30 is the next event
        the model could have acted on).
      * Apply MoomooAUCosts slippage (+0.05% on buy) and broker fees, exactly
        like the live engine would.
      * Backdate trade/event/snapshot/positions_history timestamps to
        2026-04-30T13:30:00+00:00 (US market open) so equity-curve sits
        on the same day as A/B group trades.
      * Stamp every event with category='replay_backfill' and a detail note
        for audit.
      * Refuse to run twice (checks for any pre-existing Q US trades and
        any positions; uses a transaction).

Usage:
    python -m scripts.replay_qlib_us_20260430 --dry-run   # print plan
    python -m scripts.replay_qlib_us_20260430 --execute   # commit changes
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone

PROJECT_ROOT = os.path.expanduser("~/quant-trading")
sys.path.insert(0, PROJECT_ROOT)

from config.settings import STOCK_UNIVERSE, INITIAL_CASH
from accounts.qlib_strategies import QLIB_STRATEGIES
from trading.costs import MoomooAUCosts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("replay_qlib_us")

DB_PATH = os.path.join(PROJECT_ROOT, "data", "trading.db")
MARKET = "US"
SCORE_DATE = "2026-04-29"
REPLAY_TS = "2026-04-30T13:30:00+00:00"   # US market open UTC
NOW_TS = datetime.now(timezone.utc).isoformat()
REPLAY_TAG = "replay_qlib_us_20260430"


def load_prices(conn: sqlite3.Connection, universe: list[str]) -> dict[str, float]:
    ph = ",".join("?" * len(universe))
    rows = conn.execute(
        f"SELECT ticker, close FROM prices WHERE interval='1d' "
        f"AND date(datetime)=? AND ticker IN ({ph})",
        (SCORE_DATE, *universe),
    ).fetchall()
    return {t: float(c) for t, c in rows if c is not None}


def load_scores(conn: sqlite3.Connection, base_id: str, universe: list[str]) -> list[tuple[str, float]]:
    factor_name = f"qlib_{base_id}_score"
    ph = ",".join("?" * len(universe))
    rows = conn.execute(
        f"SELECT ticker, value FROM factor_values "
        f"WHERE factor_name=? AND date=? AND value IS NOT NULL "
        f"AND ticker IN ({ph})",
        (factor_name, SCORE_DATE, *universe),
    ).fetchall()
    out = [(t, float(v)) for t, v in rows]
    out.sort(key=lambda x: x[1], reverse=True)
    return out


def plan_buys(strat, scores: list[tuple[str, float]], prices: dict[str, float],
              cash: float, costs: MoomooAUCosts) -> list[dict]:
    """Mirror the BUY logic in main._trade_qlib_account."""
    initial_equity = cash   # no holdings yet
    target_per_position = initial_equity * strat.max_position_pct
    plan = []
    remaining_cash = cash
    for ticker, score in scores[:strat.top_n]:
        price = prices.get(ticker)
        if not price or price <= 0:
            continue
        budget = min(target_per_position, remaining_cash * 0.95)
        if budget < 5:
            continue
        shares = int(budget / price)
        if shares <= 0:
            continue
        # Compute fees w/ slippage
        fee_info = costs.calculate("buy", shares, price)
        exec_price = fee_info["exec_price"]
        amount = shares * exec_price
        total_cost = amount + fee_info["total_fees"]
        if total_cost > remaining_cash:
            # trim
            shares = int(remaining_cash / exec_price * 0.99)
            if shares <= 0:
                continue
            fee_info = costs.calculate("buy", shares, price)
            exec_price = fee_info["exec_price"]
            amount = shares * exec_price
            total_cost = amount + fee_info["total_fees"]
            if total_cost > remaining_cash:
                continue
        # Engine's max_position_pct check (proposed_value/equity <= max_pct)
        if amount / initial_equity > strat.max_position_pct:
            max_amt = strat.max_position_pct * initial_equity
            shares = int(max_amt / exec_price)
            if shares <= 0:
                continue
            fee_info = costs.calculate("buy", shares, price)
            exec_price = fee_info["exec_price"]
            amount = shares * exec_price
            total_cost = amount + fee_info["total_fees"]
        remaining_cash -= total_cost
        plan.append({
            "ticker": ticker, "score": score, "shares": shares,
            "price": price, "exec_price": exec_price,
            "amount": amount, "fees": fee_info["total_fees"],
            "total_cost": total_cost,
            "slippage_cost": shares * (exec_price - price),
        })
    return plan


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    conn = sqlite3.connect(DB_PATH)
    conn.execute("BEGIN")

    # Safety: refuse if Q-group already has any US trades or positions
    existing_trades = conn.execute(
        "SELECT COUNT(*) FROM trades WHERE account LIKE 'Q%' AND market='US'"
    ).fetchone()[0]
    existing_positions = conn.execute(
        "SELECT COUNT(*) FROM positions WHERE account LIKE 'Q%' AND market='US'"
    ).fetchone()[0]
    if existing_trades or existing_positions:
        log.error("Refusing: Q US already has %d trades, %d positions. "
                  "This script is one-shot.", existing_trades, existing_positions)
        conn.rollback(); sys.exit(2)

    universe = list(STOCK_UNIVERSE)
    log.info("US universe: %d tickers", len(universe))
    prices = load_prices(conn, universe)
    log.info("04-29 close prices loaded: %d tickers", len(prices))
    if len(prices) < 0.9 * len(universe):
        log.error("Too few prices on %s (%d/%d). Aborting.",
                  SCORE_DATE, len(prices), len(universe))
        conn.rollback(); sys.exit(3)

    costs = MoomooAUCosts()
    initial_cash = float(INITIAL_CASH[MARKET])
    grand_total_trades = 0
    grand_total_spent = 0.0

    for strat in QLIB_STRATEGIES:
        scored = load_scores(conn, strat.id, universe)
        if not scored:
            log.warning("[%s] no scores on %s — skip", strat.id, SCORE_DATE)
            continue
        plan = plan_buys(strat, scored, prices, initial_cash, costs)
        log.info("[%s] (%s, top_n=%d, max_pos=%.0f%%) -> %d trades, $%.2f spent",
                 strat.id, strat.model_class, strat.top_n,
                 strat.max_position_pct * 100, len(plan),
                 sum(p["total_cost"] for p in plan))
        for p in plan:
            log.info("    BUY %-6s ×%-5d @ $%.2f  (score=%.4f, fees=$%.2f)",
                     p["ticker"], p["shares"], p["price"], p["score"], p["fees"])
        grand_total_trades += len(plan)
        grand_total_spent += sum(p["total_cost"] for p in plan)

        if not args.execute:
            continue

        _PLAN_CACHE[strat.id] = plan

        # ----- Persist trades, positions, snapshot, account_state -----
        cash = initial_cash
        for p in plan:
            conn.execute(
                "INSERT INTO trades (account,ticker,side,shares,price,cost,slippage,timestamp,market) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (strat.id, p["ticker"], "buy", p["shares"], p["exec_price"],
                 p["fees"], p["slippage_cost"], REPLAY_TS, MARKET),
            )
            # event row
            conn.execute(
                "INSERT INTO events (ts,category,severity,account,ticker,title,detail,market) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (REPLAY_TS, "trade", "info", strat.id, p["ticker"],
                 f"BUY {p['ticker']} ×{p['shares']} @ ${p['exec_price']:.2f}",
                 json.dumps({"shares": p["shares"], "price": p["exec_price"],
                             "fees": p["fees"], "score": p["score"],
                             "replay_tag": REPLAY_TAG}),
                 MARKET),
            )
            cash -= p["total_cost"]

        # rebalance event for the account (so dashboard sees one)
        conn.execute(
            "INSERT INTO events (ts,category,severity,account,ticker,title,detail,market) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (REPLAY_TS, "rebalance", "info", strat.id, None,
             f"🔄 Rebalance {strat.id} (qlib/{strat.model_class}, replay backfill)",
             json.dumps({"replay_tag": REPLAY_TAG, "n_trades": len(plan),
                         "score_date": SCORE_DATE}),
             MARKET),
        )

        # positions table (current state)
        equity = cash
        for p in plan:
            current_px = p["price"]   # use 04-29 close as MTM (latest data we have)
            equity += p["shares"] * current_px
            conn.execute(
                "INSERT OR REPLACE INTO positions "
                "(account,ticker,shares,avg_cost,total_cost,current_price,updated_at,market) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (strat.id, p["ticker"], p["shares"], p["exec_price"],
                 p["amount"] + p["fees"], current_px, REPLAY_TS, MARKET),
            )
            # positions_history
            conn.execute(
                "INSERT INTO positions_history "
                "(account,ticker,shares,avg_cost,market_price,market_value,unrealized_pnl,timestamp,market) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (strat.id, p["ticker"], p["shares"], p["exec_price"],
                 current_px, p["shares"] * current_px,
                 (current_px - p["exec_price"]) * p["shares"],
                 REPLAY_TS, MARKET),
            )

        # account_state — update cash but keep initial_cash
        conn.execute(
            "UPDATE account_state SET cash=?, updated_at=? WHERE account=? AND market=?",
            (cash, REPLAY_TS, strat.id, MARKET),
        )
        # accounts snapshot row at REPLAY_TS for equity curve
        conn.execute(
            "INSERT INTO accounts (name,cash,equity,timestamp,market) VALUES (?,?,?,?,?)",
            (strat.id, cash, equity, REPLAY_TS, MARKET),
        )

    log.info("=" * 60)
    log.info("PLAN TOTAL: %d trades, $%.2f gross spent", grand_total_trades, grand_total_spent)

    if args.execute:
        # ----- Tail backfill: fix any update_prices snapshots that were written
        # between REPLAY_TS and "now" while account_state was still at initial
        # cash (equity ≈ initial_cash, no positions). Without this step, the
        # equity curve has a flat plateau at $10000 from REPLAY_TS until the
        # next update_prices tick after we commit. We mark-to-market each bad
        # snapshot using yfinance 5m bars; out-of-window timestamps fall back
        # to the previous available close. -----
        try:
            backfill_post_replay_snapshots(conn, plan_by_acct=_PLAN_CACHE,
                                           initial_cash=initial_cash)
        except Exception as e:
            log.error("tail backfill failed: %s — committing core data anyway", e)
        conn.commit()
        log.info("✅ committed.")
    else:
        conn.rollback()
        log.info("dry-run only — nothing written. Re-run with --execute.")


# Stash plans so the tail backfill can recompute equity per account without
# re-running plan_buys (which is non-deterministic across runs since it uses
# costs / floor() etc).
_PLAN_CACHE: dict[str, list[dict]] = {}


def backfill_post_replay_snapshots(conn: sqlite3.Connection,
                                    plan_by_acct: dict[str, list[dict]],
                                    initial_cash: float) -> None:
    """Mark-to-market every `accounts` row with name LIKE 'Q%' / market='US'
    written between REPLAY_TS and now whose equity is still ≈ initial_cash.
    Uses yfinance 5m bars for the position tickers; outside the bar coverage
    window (after-hours / dates with no data) falls back to the latest
    available close at-or-before the snapshot timestamp.
    """
    import yfinance as yf
    import pandas as pd

    if not plan_by_acct:
        log.info("backfill: no plans cached, skipping")
        return

    # Collect all unique tickers across all Q accounts' plans.
    tickers = sorted({p["ticker"] for plan in plan_by_acct.values() for p in plan})
    if not tickers:
        return

    # Window: REPLAY_TS .. now
    win_start = REPLAY_TS
    win_end = datetime.now(timezone.utc).isoformat()

    # Fetch 5m bars covering REPLAY_TS day +1
    yf_t = " ".join(t.replace(".", "-") for t in tickers)
    df = yf.download(yf_t, start="2026-04-30", end="2026-05-02",
                     interval="5m", progress=False, auto_adjust=False,
                     group_by="ticker", threads=True)

    series_by_t: dict[str, "pd.Series"] = {}
    if isinstance(df.columns, pd.MultiIndex):
        for t in tickers:
            try:
                s = df[t.replace(".", "-")]["Close"].dropna()
                if not s.empty:
                    if s.index.tz is not None:
                        s.index = s.index.tz_convert("UTC").tz_localize(None)
                    series_by_t[t] = s
            except Exception:
                continue
    else:
        s = df["Close"].dropna()
        if s.index.tz is not None:
            s.index = s.index.tz_convert("UTC").tz_localize(None)
        if not s.empty:
            series_by_t[tickers[0]] = s

    log.info("backfill: yfinance 5m for %d/%d tickers", len(series_by_t), len(tickers))

    def price_at(ticker: str, dt_naive_utc: datetime, fallback: float) -> float:
        s = series_by_t.get(ticker)
        if s is None or s.empty:
            return fallback
        loc = s.index.searchsorted(dt_naive_utc, side="right") - 1
        if loc < 0:
            return float(s.iloc[0])
        return float(s.iloc[loc])

    # Per-account static cash (after replay) — taken from plan_by_acct.
    cash_by_acct: dict[str, float] = {}
    for acct, plan in plan_by_acct.items():
        cash_by_acct[acct] = initial_cash - sum(p["total_cost"] for p in plan)

    rows = conn.execute(
        "SELECT id, name, timestamp FROM accounts "
        "WHERE name LIKE 'Q%' AND market='US' "
        "AND timestamp > ? AND timestamp < ? "
        "AND ABS(equity - ?) < 0.01",
        (win_start, win_end, initial_cash),
    ).fetchall()
    log.info("backfill: %d snapshot rows to repair", len(rows))

    updates = []
    for rid, name, ts in rows:
        ts_iso = ts.replace("Z", "+00:00") if ts.endswith("Z") else ts
        try:
            dt = datetime.fromisoformat(ts_iso)
        except ValueError:
            continue
        if dt.tzinfo:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        plan = plan_by_acct.get(name, [])
        cash = cash_by_acct.get(name, initial_cash)
        equity = cash
        for p in plan:
            equity += p["shares"] * price_at(p["ticker"], dt, p["price"])
        updates.append((round(cash, 4), round(equity, 4), rid))

    if updates:
        conn.executemany(
            "UPDATE accounts SET cash=?, equity=? WHERE id=?", updates
        )
        log.info("backfill: updated %d snapshots", len(updates))


if __name__ == "__main__":
    main()
