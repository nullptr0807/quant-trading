#!/usr/bin/env python3
"""Historical replay of the CN paper-trading market.

Replays 2026-04-17 09:30 → 2026-04-23 15:00 CST at 15-min granularity, writing
trades / events / positions / equity-curve snapshots with HISTORICAL timestamps
(the bar's clock time), exactly as if the live system had been running that week.

Bar-timestamp convention: SINA 15m bars are end-stamped (a bar dated 09:45
covers 09:30→09:45). Our grid follows the same convention:
  morning   : 09:45, 10:00, …, 11:30  (8 bars)
  afternoon : 13:15, 13:30, …, 15:00  (8 bars)
  → 16 bars × 5 trading days = 80 timepoints

For each timepoint t we:
  • load history with `datetime <= t` (no look-ahead beyond what spec allows);
  • compute Alpha158 factors on CN universe (CA01-CA10);
  • compute GP factors using factors/mined_alphas_per_account.json mapped B→CB
    (CB01-CB10) — formulas reused verbatim, applied to CN data;
  • IDX3 buys 000300.SH at first timepoint and holds;
  • execute orders subject to CN-specific guards:
      lot=100, T+1, ±10% 涨跌停 (uniform; no special-case for 创业板/科创板/北交所
      tickers starting 30/68/8 — yagni per spec), 停牌 detection, CN_FEES;
  • record trades / events / position-history / equity snapshot at ts=t (CST→UTC).

Adaptive cadence: each strategy's natural rebalance_hours is respected via
in-memory `last_rebalance` tracking.

We DO NOT touch `account_state` until the very end of the replay so its
"last live state" semantics are preserved (final cash mirrors final replay state
once the loop completes).
"""
from __future__ import annotations

import os
import sys
import json
import math
import sqlite3
import logging
from copy import deepcopy
from datetime import datetime, timedelta, timezone, date
from zoneinfo import ZoneInfo

PROJECT_ROOT = os.path.expanduser("~/quant-trading")
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from config.settings import CN_FEES, CN_UNIVERSE, BENCHMARKS_BY_MARKET, INITIAL_CASH
from data.store import DataStore
from factors.alpha_factors import FactorEngine
from factors.signal import SignalGenerator
from factors.gp_signal import GPSignalGenerator
from factors.gp_miner import GPAlphaMiner
from accounts.strategies import STRATEGIES
from accounts.gp_strategies import GP_STRATEGIES
from dataclasses import replace as _dc_replace

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("replay_cn")

CST = ZoneInfo("Asia/Shanghai")
UTC = timezone.utc
LOT = 100
MARKET = "CN"
DB_PATH = os.path.join(PROJECT_ROOT, "data", "trading.db")
INDEX_TICKER = "000300.SH"

# --------------------------------------------------------------------------
# Time grid
# --------------------------------------------------------------------------
def gen_timepoints(start_date: date, end_date: date) -> list[datetime]:
    """Bar-end timepoints in CST, 16/day, Mon-Fri."""
    out = []
    d = start_date
    while d <= end_date:
        if d.weekday() < 5:
            for h, m in [(9,45),(10,0),(10,15),(10,30),(10,45),(11,0),(11,15),(11,30)]:
                out.append(datetime(d.year, d.month, d.day, h, m, tzinfo=CST))
            for h, m in [(13,15),(13,30),(13,45),(14,0),(14,15),(14,30),(14,45),(15,0)]:
                out.append(datetime(d.year, d.month, d.day, h, m, tzinfo=CST))
        d += timedelta(days=1)
    return out


def t_to_db_str(t_cst: datetime) -> str:
    """Bar timestamps are stored as 'YYYY-MM-DD HH:MM:SS' (matching SINA format)."""
    return t_cst.strftime("%Y-%m-%d %H:%M:%S")


def t_to_utc_iso(t_cst: datetime) -> str:
    return t_cst.astimezone(UTC).isoformat()


# --------------------------------------------------------------------------
# Position bookkeeping (replay-local)
# --------------------------------------------------------------------------
class CNPos:
    __slots__ = ("shares", "avg_cost", "total_cost", "bought_date")
    def __init__(self):
        self.shares = 0.0
        self.avg_cost = 0.0
        self.total_cost = 0.0
        # YYYY-MM-DD CST string of most recent buy (T+1 sell guard)
        self.bought_date: str | None = None


class CNAccount:
    def __init__(self, acct_id: str, cash: float):
        self.id = acct_id
        self.cash = cash
        self.initial_cash = cash
        self.positions: dict[str, CNPos] = {}

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


# --------------------------------------------------------------------------
# Replay engine
# --------------------------------------------------------------------------
class CNReplay:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.store = DataStore(DB_PATH)
        self.universe = list(CN_UNIVERSE)
        # Pre-load the entire 1d cache once (used only for prev-close lookup).
        self.daily_close: dict[str, pd.Series] = self._load_daily_close()
        # Pre-load the entire 15m cache once into memory; we'll slice by t in-loop.
        self.intraday: dict[str, pd.DataFrame] = self._load_intraday()

        self.factor_engine = FactorEngine()
        self.signal_gen = SignalGenerator(buy_top=10, sell_top=10)
        self.gp_signal_gen = GPSignalGenerator()
        self.gp_miner = GPAlphaMiner()

        # Strategies — prefix C
        self.strategies = [_dc_replace(s, id=f"C{s.id}") for s in STRATEGIES]
        self.gp_strategies = [_dc_replace(g, id=f"C{g.id}") for g in GP_STRATEGIES]

        # Map US-mined factors to CN account ids: B01 → CB01
        raw_mined = GPAlphaMiner.load_per_account_factors()
        self.mined_per_account: dict[str, list[dict]] = {}
        for k, v in raw_mined.items():
            if k.startswith("B"):
                self.mined_per_account[f"C{k}"] = v

        # Accounts (in-memory; we reset positions to empty, cash from account_state)
        self.accounts: dict[str, CNAccount] = {}
        for s in self.strategies:
            self.accounts[s.id] = CNAccount(s.id, INITIAL_CASH[MARKET])
        for g in self.gp_strategies:
            self.accounts[g.id] = CNAccount(g.id, INITIAL_CASH[MARKET])
        for bm in BENCHMARKS_BY_MARKET[MARKET]:
            self.accounts[bm["id"]] = CNAccount(bm["id"], INITIAL_CASH[MARKET])

        # Counters / guard stats
        self.guard_counts: dict[str, int] = {}
        self.trade_count = 0
        self.last_rebalance: dict[str, datetime] = {}

    # ---- Cache loaders ----------------------------------------------------
    def _load_daily_close(self) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        df = pd.read_sql_query(
            "SELECT ticker, datetime, close FROM prices WHERE interval='1d' "
            "AND (ticker LIKE '%.SH' OR ticker LIKE '%.SZ') ORDER BY ticker, datetime",
            self.conn,
        )
        for tk, g in df.groupby("ticker"):
            s = g.set_index("datetime")["close"]
            # date-only index for fast lookups
            s.index = pd.to_datetime(s.index).date
            out[tk] = s
        log.info("Loaded daily close for %d tickers", len(out))
        return out

    def _load_intraday(self) -> dict[str, pd.DataFrame]:
        df = pd.read_sql_query(
            "SELECT ticker, datetime, open, high, low, close, volume FROM prices "
            "WHERE interval='15m'", self.conn,
        )
        out: dict[str, pd.DataFrame] = {}
        for tk, g in df.groupby("ticker"):
            gg = g.sort_values("datetime").reset_index(drop=True)
            # Index by datetime string for O(1) lookup
            gg["dt_key"] = gg["datetime"]
            out[tk] = gg
        log.info("Loaded intraday 15m for %d tickers (rows=%d)",
                 len(out), len(df))
        return out

    # ---- Helpers ----------------------------------------------------------
    def _get_bar(self, ticker: str, t_str: str) -> dict | None:
        df = self.intraday.get(ticker)
        if df is None or df.empty:
            return None
        row = df[df["dt_key"] == t_str]
        if row.empty:
            return None
        r = row.iloc[0]
        return {"open": float(r["open"]), "high": float(r["high"]),
                "low": float(r["low"]), "close": float(r["close"]),
                "volume": float(r["volume"] or 0.0)}

    def _prev_close(self, ticker: str, t_cst: datetime) -> float | None:
        """Most recent 1d close strictly BEFORE t_cst's date."""
        s = self.daily_close.get(ticker)
        if s is None or s.empty:
            return None
        today = t_cst.date()
        prev = s[s.index < today]
        if prev.empty:
            return None
        return float(prev.iloc[-1])

    def _has_volume_today(self, ticker: str, t_cst: datetime) -> bool:
        df = self.intraday.get(ticker)
        if df is None or df.empty:
            return False
        day_prefix = t_cst.strftime("%Y-%m-%d")
        today_bars = df[df["dt_key"].str.startswith(day_prefix)]
        if today_bars.empty:
            return False
        return float(today_bars["volume"].fillna(0).sum()) > 0

    def _bump(self, key: str, n: int = 1):
        self.guard_counts[key] = self.guard_counts.get(key, 0) + n

    # ---- DB writers (batched per-timepoint commit) ------------------------
    def _write_trade(self, acct: str, ticker: str, side: str, shares: float,
                     price: float, fees: dict, t_cst: datetime):
        ts = t_to_utc_iso(t_cst)
        self.conn.execute(
            "INSERT INTO trades (account,ticker,side,shares,price,cost,slippage,timestamp,market) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (acct, ticker, side, shares, price,
             fees["total"], fees["slippage"], ts, MARKET),
        )
        self.trade_count += 1

    def _write_event(self, category: str, title: str, *, account=None, ticker=None,
                     detail=None, t_cst: datetime, severity="info"):
        self.conn.execute(
            "INSERT INTO events (ts,category,severity,account,ticker,title,detail,market) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (t_to_utc_iso(t_cst), category, severity, account, ticker,
             title, json.dumps(detail) if detail else None, MARKET),
        )

    def _write_snapshot(self, acct: CNAccount, prices: dict[str, float],
                        t_cst: datetime):
        ts = t_to_utc_iso(t_cst)
        eq = acct.equity(prices)
        self.conn.execute(
            "INSERT INTO accounts (name,cash,equity,timestamp,market) VALUES (?,?,?,?,?)",
            (acct.id, acct.cash, eq, ts, MARKET),
        )
        # positions_history rows for each open position
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
                (acct.id, tk, p.shares, p.avg_cost, px, mv, upnl, ts, MARKET),
            )

    def _flush_positions_table(self, prices: dict[str, float], t_cst: datetime):
        """Replace `positions` rows with current state (current snapshot only)."""
        ts = t_to_utc_iso(t_cst)
        self.conn.execute("DELETE FROM positions WHERE market=?", (MARKET,))
        for acct in self.accounts.values():
            for tk, p in acct.positions.items():
                if p.shares <= 0:
                    continue
                px = prices.get(tk, p.avg_cost)
                self.conn.execute(
                    "INSERT INTO positions (account,ticker,shares,avg_cost,total_cost,"
                    "current_price,updated_at,market) VALUES (?,?,?,?,?,?,?,?)",
                    (acct.id, tk, p.shares, p.avg_cost, p.total_cost, px, ts, MARKET),
                )

    # ---- Core trade execution with CN guards ------------------------------
    def execute(self, acct: CNAccount, ticker: str, side: str,
                desired_shares: float, t_cst: datetime,
                allow_partial_lot: bool = False) -> bool:
        """Try to place an order; honour all CN-specific guards.
        Returns True if a trade was recorded.
        """
        bar = self._get_bar(ticker, t_to_db_str(t_cst))
        # 停牌: no bar at this timepoint, OR zero volume across the whole day
        if bar is None or not self._has_volume_today(ticker, t_cst):
            self._bump("停牌")
            self._write_event("guard", f"停牌 {ticker}", account=acct.id,
                              ticker=ticker, t_cst=t_cst,
                              detail={"reason": "no bar / zero volume"})
            return False

        price = bar["open"]
        if not (price and price > 0):
            self._bump("bad_price")
            return False

        prev_close = self._prev_close(ticker, t_cst)
        if prev_close and prev_close > 0:
            # Uniform ±10% limit. Simplification: ignore the ±20% rule that
            # applies to ChiNext (300xxx)/STAR (688xxx)/BSE (8xxxxx) tickers
            # — yagni; the universe is CSI300 (mostly main board) so impact
            # is small and over-restrictive direction (won't enable bad fills).
            limit_up = prev_close * 1.10
            limit_down = prev_close * 0.90
            if side == "buy" and price >= limit_up - 1e-6:
                self._bump("涨停买不到")
                self._write_event("guard", f"涨停买不到 {ticker}", account=acct.id,
                                  ticker=ticker, t_cst=t_cst,
                                  detail={"open": price, "limit_up": limit_up})
                return False
            if side == "sell" and price <= limit_down + 1e-6:
                self._bump("跌停卖不掉")
                self._write_event("guard", f"跌停卖不掉 {ticker}", account=acct.id,
                                  ticker=ticker, t_cst=t_cst,
                                  detail={"open": price, "limit_down": limit_down})
                return False

        if side == "buy":
            qty = desired_shares
            if not allow_partial_lot:
                qty = math.floor(qty / LOT) * LOT
            if qty < (1 if allow_partial_lot else LOT):
                self._bump("below_lot_size")
                self._write_event("guard", f"below lot size {ticker}",
                                  account=acct.id, ticker=ticker, t_cst=t_cst,
                                  detail={"desired": desired_shares})
                return False
            fees = CN_FEES.calc_fees(shares=qty, price=price, is_sell=False)
            cost_total = qty * price + fees["total"]
            if cost_total > acct.cash + 1e-6:
                # Trim qty down to what cash allows (in lots).
                if allow_partial_lot:
                    new_qty = math.floor(acct.cash / price * 0.999)
                else:
                    new_qty = math.floor(acct.cash * 0.999 / price / LOT) * LOT
                if new_qty < (1 if allow_partial_lot else LOT):
                    self._bump("insufficient_cash")
                    return False
                qty = new_qty
                fees = CN_FEES.calc_fees(shares=qty, price=price, is_sell=False)
                cost_total = qty * price + fees["total"]
            acct.cash -= cost_total
            p = acct.positions.setdefault(ticker, CNPos())
            new_total_cost = p.total_cost + qty * price + fees["total"]
            new_shares = p.shares + qty
            p.shares = new_shares
            p.total_cost = new_total_cost
            p.avg_cost = new_total_cost / new_shares if new_shares else 0.0
            p.bought_date = t_cst.strftime("%Y-%m-%d")
            self._write_trade(acct.id, ticker, "buy", qty, price, fees, t_cst)
            return True

        else:  # sell
            p = acct.positions.get(ticker)
            if not p or p.shares <= 0:
                return False
            # T+1: can't sell shares bought today.
            today_str = t_cst.strftime("%Y-%m-%d")
            if p.bought_date and p.bought_date >= today_str:
                self._bump("T+1")
                self._write_event("guard", f"T+1阻止 {ticker}", account=acct.id,
                                  ticker=ticker, t_cst=t_cst,
                                  detail={"bought_date": p.bought_date})
                return False
            qty = min(desired_shares, p.shares)
            if qty <= 0:
                return False
            fees = CN_FEES.calc_fees(shares=qty, price=price, is_sell=True)
            proceeds = qty * price - fees["total"]
            acct.cash += proceeds
            # Reduce shares; total_cost shrinks proportionally (FIFO/avg cost).
            cost_removed = p.avg_cost * qty
            p.shares -= qty
            p.total_cost = max(0.0, p.total_cost - cost_removed)
            if p.shares < 1e-9:
                p.shares = 0.0
                p.avg_cost = 0.0
                p.total_cost = 0.0
                p.bought_date = None
            self._write_trade(acct.id, ticker, "sell", qty, price, fees, t_cst)
            return True

    # ---- Per-account trading logic ----------------------------------------
    def _trade_alpha(self, strat, t_cst: datetime,
                     factors_dict: dict, prices: dict[str, float]):
        acct = self.accounts[strat.id]
        signals = self.signal_gen.generate_signals(factors_dict, strat.strategy_type)
        buy_set = {t for t, _ in signals["buy"][:strat.top_n]}

        # --- SELL: positions not in buy list, or stop-loss ---
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
                    self.execute(acct, ticker, "sell", p.shares, t_cst)
                    continue
            if ticker not in buy_set:
                self.execute(acct, ticker, "sell", p.shares, t_cst)

        # --- BUY ---
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
            if budget < price * LOT:
                # Can't even afford one lot — skip silently (not a guard violation).
                continue
            shares = budget / price
            self.execute(acct, ticker, "buy", shares, t_cst)

    def _trade_gp(self, gp_strat, t_cst: datetime,
                  history_data: dict, prices: dict[str, float]):
        acct = self.accounts[gp_strat.id]
        mined = self.mined_per_account.get(gp_strat.id, [])
        if not mined:
            return
        # Compute GP factor values for current history snapshot
        gp_factors = self.gp_miner.compute_gp_factors(history_data, mined)
        # Filter per strategy spec (mirrors main._filter_gp_factors)
        sorted_by_ic = sorted(mined, key=lambda f: abs(f["ic"]), reverse=True)
        all_names = [f["name"] for f in sorted_by_ic]
        if gp_strat.factor_selection == "top5":
            keep = set(all_names[:5])
        elif gp_strat.factor_selection == "top10":
            keep = set(all_names[:10])
        elif gp_strat.factor_selection == "bottom5":
            keep = set(all_names[-5:]) if len(all_names) >= 5 else set(all_names)
        else:
            keep = set(all_names)
        if gp_strat.scoring_method == "top3_only":
            keep = keep & set(all_names[:3]) if len(all_names) >= 3 else keep
        filtered = {}
        for tk, fdf in gp_factors.items():
            if fdf is None or fdf.empty:
                filtered[tk] = fdf
                continue
            cols = [c for c in fdf.columns if c in keep]
            filtered[tk] = fdf[cols] if cols else fdf

        signals = self.gp_signal_gen.generate_signals(filtered, top_n=gp_strat.top_n)
        buy_set = set(signals["buy"])
        sell_set = set(signals["sell"])

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
                if pnl_pct <= -gp_strat.stop_loss:
                    self.execute(acct, ticker, "sell", p.shares, t_cst)
                    continue
            if ticker not in buy_set or ticker in sell_set:
                self.execute(acct, ticker, "sell", p.shares, t_cst)

        # BUY
        equity = acct.equity(prices)
        target_per_pos = equity * gp_strat.max_position_pct
        for ticker in signals["buy"][:gp_strat.top_n]:
            price = prices.get(ticker)
            if not price or price <= 0:
                continue
            held_val = acct.held_qty(ticker) * price
            if held_val >= target_per_pos * 0.9:
                continue
            budget = min(target_per_pos - held_val, acct.cash * 0.95)
            if budget < price * LOT:
                continue
            shares = budget / price
            self.execute(acct, ticker, "buy", shares, t_cst)

    def _trade_idx3(self, t_cst: datetime, prices: dict[str, float],
                    is_first_timepoint: bool):
        if not is_first_timepoint:
            return
        acct = self.accounts["IDX3"]
        bar = self._get_bar(INDEX_TICKER, t_to_db_str(t_cst))
        if bar is None:
            log.warning("IDX3: no opening bar for %s at %s", INDEX_TICKER, t_cst)
            return
        price = bar["open"]
        # Buy as much as possible — paper-only, no lot constraint.
        shares = math.floor(acct.cash * 0.999 / price)
        if shares <= 0:
            return
        self.execute(acct, INDEX_TICKER, "buy", shares, t_cst, allow_partial_lot=True)

    # ---- Build snapshot of price data accessible at t ---------------------
    def _build_history(self, t_cst: datetime) -> dict[str, pd.DataFrame]:
        """Return {ticker: df} with all 15m bars dt <= t_cst (1y warm-up included).
        Indexed by datetime, columns open/high/low/close/volume.
        """
        t_str = t_to_db_str(t_cst)
        out: dict[str, pd.DataFrame] = {}
        for tk, df in self.intraday.items():
            sl = df[df["dt_key"] <= t_str]
            if sl.empty:
                continue
            sl2 = sl[["datetime", "open", "high", "low", "close", "volume"]].copy()
            sl2["datetime"] = pd.to_datetime(sl2["datetime"])
            sl2 = sl2.set_index("datetime")
            out[tk] = sl2
        return out

    def _current_prices(self, history: dict[str, pd.DataFrame]) -> dict[str, float]:
        """Latest close from each ticker's history slice."""
        prices = {}
        for tk, df in history.items():
            if df is not None and not df.empty:
                prices[tk] = float(df["close"].iloc[-1])
        return prices

    # ---- Main loop --------------------------------------------------------
    def run(self, timepoints: list[datetime]):
        # Wipe replay-target data first (clean curve).
        log.info("Wiping prior CN trades/events/positions/snapshots/positions_history…")
        cur = self.conn.cursor()
        cur.execute("DELETE FROM trades WHERE market='CN'")
        cur.execute("DELETE FROM events WHERE market='CN'")
        cur.execute("DELETE FROM positions WHERE market='CN'")
        cur.execute("DELETE FROM positions_history WHERE market='CN'")
        cur.execute("DELETE FROM accounts WHERE market='CN'")
        self.conn.commit()

        first_t = timepoints[0]

        for i, t_cst in enumerate(timepoints):
            t_str = t_to_db_str(t_cst)
            history = self._build_history(t_cst)
            if not history:
                log.warning("[%d/%d] %s: no history available, skipping", i+1, len(timepoints), t_str)
                continue
            prices = self._current_prices(history)
            # Override with bar-open for any ticker that has a bar at t (more
            # accurate "live now" mid-bar quote than prior bar's close).
            for tk in self.intraday:
                bar = self._get_bar(tk, t_str)
                if bar:
                    prices[tk] = bar["close"]   # mark-to-market with current bar's close

            # Compute Alpha158 once per t (shared across CA01-CA10)
            factors_dict = self.factor_engine.compute_multi(history)

            # --- IDX3 first-timepoint init buy ---
            self._trade_idx3(t_cst, prices, is_first_timepoint=(i == 0))

            # --- CA accounts ---
            for strat in self.strategies:
                last = self.last_rebalance.get(strat.id)
                hours_elapsed = (t_cst - last).total_seconds() / 3600.0 if last else 1e9
                if hours_elapsed < strat.rebalance_hours:
                    continue
                try:
                    self._trade_alpha(strat, t_cst, factors_dict, prices)
                except Exception as e:
                    log.exception("[%s] alpha trade fail at %s: %s", strat.id, t_str, e)
                self.last_rebalance[strat.id] = t_cst

            # --- CB accounts ---
            for gp_strat in self.gp_strategies:
                last = self.last_rebalance.get(gp_strat.id)
                hours_elapsed = (t_cst - last).total_seconds() / 3600.0 if last else 1e9
                if hours_elapsed < gp_strat.rebalance_hours:
                    continue
                try:
                    self._trade_gp(gp_strat, t_cst, history, prices)
                except Exception as e:
                    log.exception("[%s] gp trade fail at %s: %s", gp_strat.id, t_str, e)
                self.last_rebalance[gp_strat.id] = t_cst

            # --- Snapshots for ALL accounts (incl. IDX3) ---
            for acct in self.accounts.values():
                self._write_snapshot(acct, prices, t_cst)
            self._flush_positions_table(prices, t_cst)

            self.conn.commit()
            if (i + 1) % 8 == 0 or i == len(timepoints) - 1:
                log.info("[%d/%d] %s — trades so far=%d", i+1, len(timepoints),
                         t_str, self.trade_count)

        # Final: update account_state to reflect post-replay live cash
        log.info("Updating account_state to final replay state…")
        ts_now = datetime.now(UTC).isoformat()
        for acct in self.accounts.values():
            self.conn.execute(
                "INSERT OR REPLACE INTO account_state (account, cash, initial_cash, updated_at, market) "
                "VALUES (?,?,?,?,?)",
                (acct.id, acct.cash, acct.initial_cash, ts_now, MARKET),
            )
        self.conn.commit()


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    start = date(2026, 4, 17)
    end = date(2026, 4, 23)
    tps = gen_timepoints(start, end)
    log.info("Generated %d timepoints (%s … %s)", len(tps), tps[0], tps[-1])
    assert len(tps) == 80, f"expected 80 timepoints, got {len(tps)}"

    replay = CNReplay(conn)
    replay.run(tps)

    log.info("=== Replay complete ===")
    log.info("Total trades: %d", replay.trade_count)
    log.info("Guard counts: %s", replay.guard_counts)
    # Per-account final equity
    rows = conn.execute(
        "SELECT name, cash, equity FROM accounts WHERE market='CN' "
        "AND timestamp = (SELECT MAX(timestamp) FROM accounts WHERE market='CN') "
        "ORDER BY name"
    ).fetchall()
    for name, cash, eq in rows:
        log.info("  %s  cash=%.2f  equity=%.2f  pnl=%+.2f%%",
                 name, cash, eq, (eq - 100000) / 1000.0)

    conn.close()


if __name__ == "__main__":
    main()
