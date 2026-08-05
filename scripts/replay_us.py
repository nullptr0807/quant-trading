#!/usr/bin/env python3
"""Historical replay of the US paper-trading market (A-group only).

Mirrors `replay_cn.py` but stripped of CN-specific guards (no ±10% limit,
no T+1, no lot=100, no 停牌 by-zero-volume). Runs at daily granularity using
the 1d cache in `prices` table.

A-group accounts only: A01-A10. B/Q groups skipped — they have their own
training/serving cycles. IDX1 (QQQ) and IDX2 (SPY) buy-and-hold benchmarks
are bought at the first timepoint and held.

The replay is look-ahead safe: signals use data through T close and orders fill
at T+1 open. It also fails closed for capital-allocation use unless a
point-in-time universe_membership table covers the requested window; pass
--allow-survivorship-biased only for explicitly labelled exploratory research.

Usage:
    python scripts/replay_us.py --start 2026-04-01 --end 2026-05-22 \\
        [--accounts A03,A05] [--label baseline]

The `--label` value is written into `trades.market` so you can A/B compare
two replays in the same DB (e.g. `US_baseline` vs `US_hyst`). Default
label is `US_replay`, which is INTENTIONALLY DIFFERENT from production's
`US` market tag so this script CANNOT clobber live data.
"""
from __future__ import annotations

import os
import sys
import json
import math
import argparse
import sqlite3
import logging
from pathlib import Path
from copy import deepcopy
from datetime import datetime, timezone, date
from zoneinfo import ZoneInfo

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from config.settings import STOCK_UNIVERSE, BENCHMARKS_BY_MARKET, INITIAL_CASH
from factors.alpha_factors import FactorEngine
from factors.signal import SignalGenerator
from accounts.strategies import STRATEGIES
from trading.costs import MoomooAUCosts
from research.validity import build_research_metadata

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("replay_us")

ET = ZoneInfo("America/New_York")
UTC = timezone.utc
DB_PATH = os.path.join(PROJECT_ROOT, "data", "trading.db")
DEFAULT_LABEL = "US_replay"     # market tag — must NOT equal live "US"
QQQ = "QQQ"
SPY = "SPY"


class ReplayCoverageError(RuntimeError):
    """Broker-like replay input coverage is incomplete."""


# --------------------------------------------------------------------------
# Position bookkeeping (replay-local; mirrors CN's CNPos/CNAccount)
# --------------------------------------------------------------------------
class USPos:
    __slots__ = ("shares", "avg_cost", "total_cost")
    def __init__(self):
        self.shares = 0.0
        self.avg_cost = 0.0
        self.total_cost = 0.0


class USAccount:
    def __init__(self, acct_id: str, cash: float):
        self.id = acct_id
        self.cash = cash
        self.initial_cash = cash
        self.positions: dict[str, USPos] = {}

    def equity(self, prices: dict[str, float]) -> float:
        eq = self.cash
        for tk, p in self.positions.items():
            if p.shares > 0:
                px = prices.get(tk, p.avg_cost)
                eq += p.shares * px
        return eq

    def held_qty(self, ticker: str) -> float:
        p = self.positions.get(ticker)
        return p.shares if p else 0.0


def apply_corporate_actions(account: USAccount, actions: list[dict]) -> None:
    """Apply ex-date share/cash actions before that day's opening auction."""
    for action in actions:
        pos = account.positions.get(str(action["ticker"]))
        if not pos or pos.shares <= 0:
            continue
        kind = str(action["action_type"])
        if kind in {"split", "bonus_or_transfer"}:
            ratio = float(action.get("ratio") or 0)
            if ratio <= 0:
                raise ReplayCoverageError(f"invalid corporate-action ratio: {action}")
            pos.shares *= ratio
            pos.avg_cost /= ratio
            # total_cost is unchanged by a share-count action.
        elif kind == "cash_dividend":
            account.cash += pos.shares * float(action.get("cash_per_share") or 0)
        else:
            raise ReplayCoverageError(f"unsupported corporate action: {kind}")


# --------------------------------------------------------------------------
# Replay engine
# --------------------------------------------------------------------------
class USReplay:
    def __init__(self, conn: sqlite3.Connection, *,
                 label: str, accounts_filter: set[str] | None,
                 hysteresis_mult: int, decorrelate: bool = True,
                 execution_mode: str = "broker",
                 universe_validity: dict | None = None):
        self.conn = conn
        self.label = label
        self.accounts_filter = accounts_filter
        self.hysteresis_mult = hysteresis_mult     # 1 = no hysteresis (legacy)
        self.decorrelate = decorrelate
        self.execution_mode = execution_mode
        self.universe_validity = universe_validity or {}
        self.universe = list(STOCK_UNIVERSE)
        self.costs = MoomooAUCosts()

        # Pre-load 1d cache
        self.daily: dict[str, pd.DataFrame] = self._load_daily()
        self.raw_daily: dict[str, pd.DataFrame] = (
            self._load_daily(table="prices_raw") if execution_mode == "broker" else {}
        )
        # Pre-build a set of trading dates from QQQ (proxy for "market open")
        self.trading_dates: list[date] = self._build_trading_dates()

        self.factor_engine = FactorEngine()
        # buy_top must cover top_n * hysteresis_mult for the largest top_n
        max_top_n = max((s.top_n for s in STRATEGIES if s.id.startswith("A")), default=10)
        self.signal_gen = SignalGenerator(
            buy_top=max(30, max_top_n * hysteresis_mult * 2),
            sell_top=10,
            decorrelate=decorrelate,
        )

        # Only A-group strategies, optionally filtered
        self.strategies = [
            s for s in STRATEGIES
            if s.id.startswith("A") and (accounts_filter is None or s.id in accounts_filter)
        ]

        # Accounts: A01..A10 + IDX1 (QQQ) + IDX2 (SPY)
        self.accounts: dict[str, USAccount] = {}
        for s in self.strategies:
            self.accounts[s.id] = USAccount(s.id, INITIAL_CASH["US"])
        # Benchmarks (always create — they're cheap)
        for bm in BENCHMARKS_BY_MARKET["US"]:
            if accounts_filter is None or bm["id"] in accounts_filter:
                self.accounts[bm["id"]] = USAccount(bm["id"], INITIAL_CASH["US"])

        self.trade_count = 0
        self.churn_stats: dict[str, list[str]] = {}   # account -> ['ticker:side' ...]
        self.last_rebalance: dict[str, datetime] = {}

    # ---- Cache loaders ----------------------------------------------------
    def _load_daily(self, table: str = "prices") -> dict[str, pd.DataFrame]:
        if not self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone():
            if table == "prices_raw":
                return {}
            raise ReplayCoverageError(f"missing required table {table}")
        df = pd.read_sql_query(
            "SELECT ticker, datetime, open, high, low, close, volume "
            f"FROM {table} WHERE interval='1d' "
            "AND ticker NOT LIKE '%.SH' AND ticker NOT LIKE '%.SZ'",
            self.conn,
        )
        df["datetime"] = pd.to_datetime(df["datetime"])
        out: dict[str, pd.DataFrame] = {}
        for tk, g in df.groupby("ticker"):
            gg = g.sort_values("datetime").reset_index(drop=True)
            gg = gg.set_index("datetime")
            out[tk] = gg
        log.info("Loaded %s daily 1d for %d US tickers", table, len(out))
        return out

    def _build_trading_dates(self) -> list[date]:
        if QQQ in self.daily:
            dates = sorted({d.date() for d in self.daily[QQQ].index})
        else:
            # fallback: union of all tickers
            s = set()
            for df in self.daily.values():
                s |= {d.date() for d in df.index}
            dates = sorted(s)
        return dates

    # ---- Helpers ----------------------------------------------------------
    def _slice_history(self, t_date: date) -> dict[str, pd.DataFrame]:
        """All bars strictly before execution day T (signal known at T-1 close)."""
        out: dict[str, pd.DataFrame] = {}
        cutoff = pd.Timestamp(t_date)
        for tk, df in self.daily.items():
            sl = df[df.index < cutoff]
            if not sl.empty:
                out[tk] = sl
        return out

    def _current_prices(self, history: dict[str, pd.DataFrame], t_date: date) -> dict[str, float]:
        """T+1 execution uses the execution day's open, never the signal close."""
        prices = {}
        day = pd.Timestamp(t_date)
        execution_mode = getattr(self, "execution_mode", "signal-only")
        source = self.raw_daily if execution_mode == "broker" else self.daily
        for tk, full in source.items():
            row = full[full.index == day]
            if not row.empty:
                prices[tk] = float(row["open"].iloc[0])
        configured_universe = getattr(self, "universe", None)
        if getattr(self, "universe_validity", {}).get("mode") == "point_in_time":
            expected_scope = {
                row[0] for row in self.conn.execute(
                    "SELECT ticker FROM universe_membership WHERE market='US' AND date=?",
                    (t_date.isoformat(),),
                ).fetchall()
            } | {QQQ, SPY}
        else:
            expected_scope = (
                set(configured_universe) | {QQQ, SPY}
                if configured_universe is not None else set(self.daily)
            )
        expected = {
            tk for tk, full in self.daily.items()
            if tk in expected_scope and not full[full.index == day].empty
        }
        missing = sorted(expected - set(prices))
        if execution_mode == "broker" and missing:
            raise ReplayCoverageError(
                f"missing raw execution price on {t_date}: {missing[:10]}"
            )
        return prices

    def _actions_for_date(self, t_date: date) -> list[dict]:
        if self.execution_mode != "broker":
            return []
        rows = self.conn.execute(
            "SELECT ticker,action_type,ratio,cash_per_share FROM corporate_actions "
            "WHERE market='US' AND ex_date=? ORDER BY ticker,action_type",
            (t_date.isoformat(),),
        ).fetchall()
        return [dict(zip(("ticker", "action_type", "ratio", "cash_per_share"), row)) for row in rows]

    def _ensure_broker_coverage(self, dates: list[date]) -> None:
        if self.execution_mode != "broker":
            return
        if not self.raw_daily:
            raise ReplayCoverageError("raw execution price table has no US daily data")
        for t_date in dates:
            day = pd.Timestamp(t_date)
            if self.universe_validity.get("mode") == "point_in_time":
                scope = {
                    row[0] for row in self.conn.execute(
                        "SELECT ticker FROM universe_membership WHERE market='US' AND date=?",
                        (t_date.isoformat(),),
                    ).fetchall()
                } | {QQQ, SPY}
            else:
                scope = set(self.universe) | {QQQ, SPY}
            expected = {
                ticker for ticker in scope
                if ticker in self.daily
                and not self.daily[ticker][self.daily[ticker].index == day].empty
            }
            available = {
                ticker for ticker in expected
                if ticker in self.raw_daily
                and not self.raw_daily[ticker][self.raw_daily[ticker].index == day].empty
            }
            missing_raw = sorted(expected - available)
            if missing_raw:
                raise ReplayCoverageError(
                    f"missing raw execution price on {t_date}: {missing_raw[:10]}"
                )
        required_tables = {"corporate_actions", "corporate_action_coverage"}
        found = {
            row[0] for row in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if not required_tables <= found:
            raise ReplayCoverageError(
                "corporate-action coverage unavailable; required tables: "
                + ", ".join(sorted(required_tables - found))
            )
        start, end = dates[0].isoformat(), dates[-1].isoformat()
        covered = {
            row[0] for row in self.conn.execute(
                "SELECT ticker FROM corporate_action_coverage WHERE market='US' "
                "AND status='complete' AND start_date<=? AND end_date>=?",
                (start, end),
            ).fetchall()
        }
        if self.universe_validity.get("mode") == "point_in_time":
            required = {
                row[0] for row in self.conn.execute(
                    "SELECT DISTINCT ticker FROM universe_membership "
                    "WHERE market='US' AND date BETWEEN ? AND ?",
                    (start, end),
                ).fetchall()
            } | {QQQ, SPY}
        else:
            required = set(self.universe) | {QQQ, SPY}
        missing = sorted(required - covered)
        if missing:
            raise ReplayCoverageError(
                f"corporate-action coverage incomplete for {len(missing)} ticker(s): {missing[:10]}"
            )

    # ---- DB writers -------------------------------------------------------
    def _t_to_iso(self, t_date: date) -> str:
        # Stamp at 14:00 ET (≈ midday) for realism
        return datetime.combine(t_date, datetime.min.time(), tzinfo=ET).replace(hour=14).astimezone(UTC).isoformat()

    def _write_trade(self, acct: str, ticker: str, side: str, shares: float,
                     price: float, fees: dict, t_date: date):
        ts = self._t_to_iso(t_date)
        self.conn.execute(
            "INSERT INTO trades (account,ticker,side,shares,price,cost,slippage,timestamp,market) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (acct, ticker, side, shares, price,
             fees.get("total_fees", fees.get("total", 0.0)),
             fees.get("slippage", 0.0), ts, self.label),
        )
        self.trade_count += 1
        self.churn_stats.setdefault(acct, []).append(f"{ticker}:{side}")

    def _write_snapshot(self, acct: USAccount, prices: dict[str, float], t_date: date):
        ts = self._t_to_iso(t_date)
        eq = acct.equity(prices)
        self.conn.execute(
            "INSERT INTO accounts (name,cash,equity,timestamp,market) VALUES (?,?,?,?,?)",
            (acct.id, acct.cash, eq, ts, self.label),
        )
        for tk, p in acct.positions.items():
            if p.shares <= 0:
                continue
            px = prices.get(tk, p.avg_cost)
            mv = p.shares * px
            upnl = (px - p.avg_cost) * p.shares
            self.conn.execute(
                "INSERT INTO positions_history (account,ticker,shares,avg_cost,"
                "market_price,market_value,unrealized_pnl,timestamp,market) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (acct.id, tk, p.shares, p.avg_cost, px, mv, upnl, ts, self.label),
            )

    # ---- Trade execution (US — minimal guards) ---------------------------
    def execute(self, acct: USAccount, ticker: str, side: str,
                desired_shares: float, price: float, t_date: date) -> bool:
        if price <= 0 or desired_shares <= 0:
            return False

        if side == "buy":
            qty = math.floor(desired_shares)
            if qty < 1:
                return False
            fees = self.costs.calculate("buy", qty, price)
            exec_price = fees["exec_price"]
            cost_total = qty * exec_price + fees["total_fees"]
            if cost_total > acct.cash + 1e-6:
                new_qty = math.floor(acct.cash * 0.999 / price)
                if new_qty < 1:
                    return False
                qty = new_qty
                fees = self.costs.calculate("buy", qty, price)
                exec_price = fees["exec_price"]
                cost_total = qty * exec_price + fees["total_fees"]
            acct.cash -= cost_total
            p = acct.positions.setdefault(ticker, USPos())
            new_total_cost = p.total_cost + qty * exec_price + fees["total_fees"]
            new_shares = p.shares + qty
            p.shares = new_shares
            p.total_cost = new_total_cost
            p.avg_cost = new_total_cost / new_shares if new_shares else 0.0
            self._write_trade(acct.id, ticker, "buy", qty, exec_price, fees, t_date)
            return True

        else:  # sell
            p = acct.positions.get(ticker)
            if not p or p.shares <= 0:
                return False
            qty = min(desired_shares, p.shares)
            if qty <= 0:
                return False
            fees = self.costs.calculate("sell", qty, price)
            exec_price = fees["exec_price"]
            proceeds = qty * exec_price - fees["total_fees"]
            acct.cash += proceeds
            cost_removed = p.avg_cost * qty
            p.shares -= qty
            p.total_cost = max(0.0, p.total_cost - cost_removed)
            if p.shares < 1e-9:
                p.shares = 0.0
                p.avg_cost = 0.0
                p.total_cost = 0.0
            self._write_trade(acct.id, ticker, "sell", qty, exec_price, fees, t_date)
            return True

    # ---- Per-account A-group trading logic --------------------------------
    def _factors_dict_from_db(self, t_date: date, factor_names: list[str]) -> dict:
        """Build {ticker: 1-row DataFrame[factor_names]} from factor_values for date <= t_date.

        Uses the most recent value per (ticker, factor) on or before t_date.
        Much faster than recomputing with FactorEngine.compute_multi.
        """
        placeholders = ",".join(["?"] * len(factor_names))
        df = pd.read_sql_query(
            f"""
            SELECT ticker, factor_name, value, date
            FROM factor_values
            WHERE factor_name IN ({placeholders})
              AND date = (
                SELECT MAX(date) FROM factor_values fv2
                WHERE fv2.ticker = factor_values.ticker
                  AND fv2.factor_name = factor_values.factor_name
                  AND fv2.date < ?
              )
            """,
            self.conn,
            params=[*factor_names, t_date.isoformat()],
        )
        if df.empty:
            return {}
        if self.universe_validity.get("mode") == "point_in_time":
            pairs = {
                (row[0], row[1]) for row in self.conn.execute(
                    "SELECT date,ticker FROM universe_membership WHERE market='US' AND date<?",
                    (t_date.isoformat(),),
                ).fetchall()
            }
            df = df[df.apply(lambda row: (str(row["date"]), row["ticker"]) in pairs, axis=1)]
        df = df.drop(columns=["date"])
        wide = df.pivot(index="ticker", columns="factor_name", values="value")
        out = {}
        for tkr, row in wide.iterrows():
            out[tkr] = pd.DataFrame([row])
        return out

    def _trade_alpha(self, strat, t_date: date,
                     factors_dict: dict, prices: dict[str, float]):
        acct = self.accounts[strat.id]
        signals = self.signal_gen.generate_signals(factors_dict, strat.strategy_type)

        # Rank hysteresis (P0 fix mirror of main.py change)
        buy_tickers = {t for t, _ in signals["buy"][:strat.top_n]}
        hold_n = strat.top_n * self.hysteresis_mult
        hold_tickers = {t for t, _ in signals["buy"][:hold_n]}

        # SELL
        for ticker in list(acct.positions.keys()):
            p = acct.positions[ticker]
            if p.shares <= 0:
                continue
            price = prices.get(ticker)
            if not price:
                continue
            if p.avg_cost > 0:
                pnl_pct = (price - p.avg_cost) / p.avg_cost
                if pnl_pct <= -strat.stop_loss:
                    self.execute(acct, ticker, "sell", p.shares, price, t_date)
                    continue
            if ticker not in hold_tickers:
                self.execute(acct, ticker, "sell", p.shares, price, t_date)

        # BUY
        equity = acct.equity(prices)
        target_per_pos = equity * strat.max_position_pct
        for ticker, _score in signals["buy"][:strat.top_n]:
            price = prices.get(ticker)
            if not price or price <= 0:
                continue
            held_val = acct.held_qty(ticker) * price
            if held_val >= target_per_pos * 0.9:
                continue
            budget = min(target_per_pos - held_val, acct.cash * 0.95)
            if budget < price:        # can't even afford 1 share
                continue
            shares = math.floor(budget / price)
            if shares >= 1:
                self.execute(acct, ticker, "buy", shares, price, t_date)

    def _trade_benchmark(self, bm_id: str, ticker: str, t_date: date,
                        prices: dict[str, float], is_first: bool):
        if not is_first:
            return
        acct = self.accounts.get(bm_id)
        if acct is None:
            return
        price = prices.get(ticker)
        if not price or price <= 0:
            log.warning("%s: no opening price for %s at %s", bm_id, ticker, t_date)
            return
        shares = math.floor(acct.cash * 0.999 / price)
        if shares <= 0:
            return
        self.execute(acct, ticker, "buy", shares, price, t_date)

    # ---- Main loop --------------------------------------------------------
    def run(self, start_date: date, end_date: date):
        dates = [d for d in self.trading_dates if start_date <= d <= end_date]
        if not dates:
            raise SystemExit(f"No trading days in [{start_date}, {end_date}]")
        log.info("Replay window: %s … %s (%d trading days)",
                 dates[0], dates[-1], len(dates))
        self._ensure_broker_coverage(dates)

        # Wipe prior rows for this label
        log.info("Wiping prior label=%r rows…", self.label)
        cur = self.conn.cursor()
        cur.execute("DELETE FROM trades WHERE market=?", (self.label,))
        cur.execute("DELETE FROM accounts WHERE market=?", (self.label,))
        cur.execute("DELETE FROM positions_history WHERE market=?", (self.label,))
        self.conn.commit()

        first_date = dates[0]
        for i, t_date in enumerate(dates):
            actions = self._actions_for_date(t_date)
            for account in self.accounts.values():
                apply_corporate_actions(account, actions)
            history = self._slice_history(t_date)
            if not history:
                continue
            prices = self._current_prices(history, t_date)

            # Build per-strategy factors_dict on demand (uses each strat's factor_names)
            factors_cache: dict[tuple, dict] = {}  # cache by factor name tuple
            def get_factors_for(strat):
                key = tuple(sorted(strat.factor_names))
                if key not in factors_cache:
                    factors_cache[key] = self._factors_dict_from_db(t_date, list(strat.factor_names))
                return factors_cache[key]

            # Benchmarks (first day only)
            self._trade_benchmark("IDX1", QQQ, t_date, prices, i == 0)
            self._trade_benchmark("IDX2", SPY, t_date, prices, i == 0)

            # A-group rebalance with cadence honoured
            t_dt = datetime.combine(t_date, datetime.min.time(), tzinfo=ET).replace(hour=14)
            for strat in self.strategies:
                last = self.last_rebalance.get(strat.id)
                hours_elapsed = (t_dt - last).total_seconds() / 3600.0 if last else 1e9
                if hours_elapsed < strat.rebalance_hours:
                    continue
                try:
                    self._trade_alpha(strat, t_date, get_factors_for(strat), prices)
                except Exception as e:
                    log.exception("[%s] alpha trade fail at %s: %s", strat.id, t_date, e)
                self.last_rebalance[strat.id] = t_dt

            # Snapshots for all accounts
            for acct in self.accounts.values():
                self._write_snapshot(acct, prices, t_date)

            self.conn.commit()
            if (i + 1) % 5 == 0 or i == len(dates) - 1:
                log.info("[%d/%d] %s — trades=%d", i+1, len(dates), t_date, self.trade_count)

        warnings = []
        if self.execution_mode == "signal-only":
            warnings.append(
                "Signal-only replay executes on adjusted prices; not broker-like and not valid for capital allocation."
            )
        self.result_metadata = build_research_metadata(
            universe=self.universe_validity,
            signal_price_mode="adjusted",
            execution_price_mode="raw" if self.execution_mode == "broker" else "adjusted",
            model_provenance={"kind": "alpha158_persisted_factors", "capital_allocation_valid": True},
            look_ahead_validity="valid",
            warnings=warnings,
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS research_result_metadata ("
            "label TEXT PRIMARY KEY, created_at TEXT NOT NULL, metadata_json TEXT NOT NULL)"
        )
        self.conn.execute(
            "INSERT OR REPLACE INTO research_result_metadata VALUES (?,?,?)",
            (self.label, datetime.now(UTC).isoformat(), json.dumps(self.result_metadata, sort_keys=True)),
        )
        self.conn.commit()


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="US A-group historical replay")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end",   required=True, help="YYYY-MM-DD")
    ap.add_argument("--accounts", default=None,
                    help="Comma-separated subset (e.g. A03,A05). Default: all A-group + IDX1/IDX2.")
    ap.add_argument("--label", default=DEFAULT_LABEL,
                    help=f"market tag in DB (default: {DEFAULT_LABEL!r}). Must NOT equal 'US' (live).")
    ap.add_argument("--hysteresis", type=int, default=3,
                    help="hold band multiplier (top_n * H). 1 = no hysteresis (legacy). Default: 3.")
    ap.add_argument("--decorrelate", action="store_true", default=True,
                    help="Apply PCA whitening to remove factor redundancy (V2). Default: ON.")
    ap.add_argument("--no-decorrelate", action="store_false", dest="decorrelate",
                    help="Disable PCA whitening (legacy V1 plain rank-mean composite).")
    ap.add_argument("--allow-survivorship-biased", action="store_true",
                    help="Exploratory only: use today's universe and mark output invalid for capital allocation")
    ap.add_argument(
        "--execution-mode", choices=["broker", "signal-only"], default="broker",
        help=("broker (default): adjusted signals + raw opens + complete corporate actions; "
              "fails closed on coverage gaps. signal-only: adjusted prices throughout, research-only."),
    )
    args = ap.parse_args()

    if args.label == "US":
        raise SystemExit("Refusing to use label='US' (would clobber live data).")

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end   = datetime.strptime(args.end,   "%Y-%m-%d").date()
    accounts = set(a.strip() for a in args.accounts.split(",")) if args.accounts else None

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    from research.validity import ensure_point_in_time_universe
    validity = ensure_point_in_time_universe(
        conn, market="US", start_date=start.isoformat(), end_date=end.isoformat(),
        allow_survivorship_biased=args.allow_survivorship_biased,
    )
    print("RESEARCH_VALIDITY", json.dumps(validity, ensure_ascii=False))

    replay = USReplay(conn,
                      label=args.label,
                      accounts_filter=accounts,
                      hysteresis_mult=args.hysteresis,
                      decorrelate=args.decorrelate,
                      execution_mode=args.execution_mode,
                      universe_validity=validity)
    replay.run(start, end)
    print("RESULT_METADATA", json.dumps(replay.result_metadata, ensure_ascii=False))

    log.info("=== Replay complete ===")
    log.info("Total trades: %d", replay.trade_count)

    rows = conn.execute(
        "SELECT name, cash, equity FROM accounts WHERE market=? "
        "AND timestamp = (SELECT MAX(timestamp) FROM accounts WHERE market=?) "
        "ORDER BY name", (args.label, args.label),
    ).fetchall()
    print()
    print(f"{'Account':<8} {'Cash':>12} {'Equity':>12}  PnL%")
    print("-" * 50)
    for name, cash, eq in rows:
        pnl_pct = (eq - INITIAL_CASH["US"]) / INITIAL_CASH["US"] * 100
        print(f"{name:<8} {cash:>12,.2f} {eq:>12,.2f}  {pnl_pct:+.2f}%")

    # Per-account churn
    print()
    print("Trades per account:")
    for acct, evs in sorted(replay.churn_stats.items()):
        # count direction flips per ticker
        per_tkr: dict[str, list[str]] = {}
        for ev in evs:
            tk, side = ev.split(":")
            per_tkr.setdefault(tk, []).append(side)
        flips = sum(
            sum(1 for i in range(1, len(v)) if v[i] != v[i-1])
            for v in per_tkr.values()
        )
        print(f"  {acct}: {len(evs)} trades, {flips} direction flips, "
              f"{len(per_tkr)} unique tickers")

    conn.close()


if __name__ == "__main__":
    main()
