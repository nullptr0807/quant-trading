#!/usr/bin/env python3
"""
美股量化交易系统 - 主调度器
Orchestrates: data fetch -> factor compute -> signal generation -> trading -> reporting
"""
from dotenv import load_dotenv
load_dotenv()

import os
import sys
import time
import json
import logging
import traceback
from datetime import datetime, timezone, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd

# Ensure this checkout's project root is first on sys.path. Using ~/quant-trading
# here made an isolated worktree silently import production modules.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.settings import (
    STOCK_UNIVERSE, FINNHUB_API_KEY, FEES, ACCOUNTS, ADAPTIVE_REBALANCE,
    MARKETS, DEFAULT_MARKET, UNIVERSES, INITIAL_CASH, FEES_BY_MARKET,
    BENCHMARKS_BY_MARKET, ACCOUNT_PREFIX, CHURN_CONTROLS,
)
from trading.churn_controls import get_stop_cooldown_set, get_min_hold_set
from data.fetcher import DataFetcher
from data.cn_fetcher import get_fetcher_for
from data.store import DataStore
from factors.alpha_factors import FactorEngine
from factors.signal import SignalGenerator
from trading.engine import TradingEngine, TradePersistenceError
from trading.costs import MoomooAUCosts, CNCosts
from accounts.strategies import STRATEGIES
from accounts.gp_strategies import active_gp_strategies_for_market
from accounts.qlib_strategies import QLIB_STRATEGIES
from factors.gp_miner import GPAlphaMiner, FEATURE_COLS
from data.quotes import (
    prices_from_quotes,
    validate_required_quotes,
    valid_quote_tickers,
)
from factors.factor_miner_gp import FactorMinerGPBackend
from dataclasses import replace as _dc_replace

# Factor formula descriptions for reporting (strict math)
FACTOR_FORMULAS = {
    "KMID": "(Close - Open) / Open",
    "KLEN": "(High - Low) / Open",
    "KMID2": "(Close - Open) / (High - Low)",
    "KUP": "(High - max(Open, Close)) / Open",
    "KUP2": "(High - max(Open, Close)) / (High - Low)",
    "KLOW": "(min(Open, Close) - Low) / Open",
    "KLOW2": "(min(Open, Close) - Low) / (High - Low)",
    "KSFT": "(2*Close - High - Low) / Open",
    "KSFT2": "(2*Close - High - Low) / (High - Low)",
    "ROC_5": "Close(t) / Close(t-5) - 1",
    "ROC_10": "Close(t) / Close(t-10) - 1",
    "ROC_20": "Close(t) / Close(t-20) - 1",
    "MA_RATIO_5": "Close(t) / SMA(Close, 5)",
    "MA_RATIO_10": "Close(t) / SMA(Close, 10)",
    "MA_RATIO_20": "Close(t) / SMA(Close, 20)",
    "VMOM_5": "Volume(t) / SMA(Volume, 5)",
    "VMOM_10": "Volume(t) / SMA(Volume, 10)",
    "VMOM_20": "Volume(t) / SMA(Volume, 20)",
    "VSTD_5": "Std(Volume, 5) / SMA(Volume, 5)",
    "VSTD_10": "Std(Volume, 10) / SMA(Volume, 10)",
    "VSTD_20": "Std(Volume, 20) / SMA(Volume, 20)",
    "STD_5": "Std(Close, 5) / Close",
    "STD_10": "Std(Close, 10) / Close",
    "STD_20": "Std(Close, 20) / Close",
    "BBPOS_5": "(Close - SMA(Close,5)) / (2 * Std(Close,5))",
    "BBPOS_10": "(Close - SMA(Close,10)) / (2 * Std(Close,10))",
    "BBPOS_20": "(Close - SMA(Close,20)) / (2 * Std(Close,20))",
    "RSV": "(Close - Min(Low,9)) / (Max(High,9) - Min(Low,9))",
    "RSI_14": "100 - 100 / (1 + SMA(Gain,14) / SMA(Loss,14))",
    "BETA_5": "OLS_Slope(Close, 5) / SMA(Close, 5)",
    "BETA_10": "OLS_Slope(Close, 10) / SMA(Close, 10)",
    "BETA_20": "OLS_Slope(Close, 20) / SMA(Close, 20)",
}

# GP function to math notation mapping
GP_FUNC_MATH = {
    "add": "({0} + {1})",
    "sub": "({0} - {1})",
    "mul": "({0} * {1})",
    "div": "({0} / {1})",  # protected: returns 0 when |{1}|<1e-10
    "sqrt_abs": "sqrt(|{0}|)",
    "log_abs1": "ln(|{0}| + 1)",
    "neg": "-({0})",
    "inv": "(1 / {0})",  # protected: returns 0 when |{0}|<1e-10
    "max2": "max({0}, {1})",
    "min2": "min({0}, {1})",
}

# GP variable to math notation
GP_VAR_MATH = {
    "o_c": "(Open-Close)/Close",
    "h_c": "(High-Close)/Close",
    "l_c": "(Low-Close)/Close",
    "v_vma20": "Volume/SMA(Volume,20)",
    "ma_5": "SMA(Close,5)/Close",
    "ma_10": "SMA(Close,10)/Close",
    "ma_20": "SMA(Close,20)/Close",
    "std_5": "Std(Close,5)/Close",
    "std_10": "Std(Close,10)/Close",
    "std_20": "Std(Close,20)/Close",
    "ret_1": "Close(t)/Close(t-1)-1",
    "ret_5": "Close(t)/Close(t-5)-1",
    "ret_10": "Close(t)/Close(t-10)-1",
}


def gp_expr_to_math(expr_str: str) -> str:
    """Convert a gplearn expression string to strict mathematical notation.
    E.g. 'max2(X11, log_abs1(X10))' -> 'max(Close(t)/Close(t-5)-1, ln(|Close(t)/Close(t-1)-1| + 1))'
    """
    import re

    # Replace X{i} with variable names first
    def replace_vars(s):
        def _repl(m):
            idx = int(m.group(1))
            if idx < len(FEATURE_COLS):
                var_name = FEATURE_COLS[idx]
                return GP_VAR_MATH.get(var_name, var_name)
            return m.group(0)
        return re.sub(r'X(\d+)', _repl, s)

    # Recursive parser for nested function calls
    tokens = []
    s = replace_vars(expr_str)

    def parse(s):
        s = s.strip()
        # Check if it's a function call: fname(args...)
        m = re.match(r'^([a-z_][a-z_0-9]*)\((.+)\)$', s)
        if m:
            fname = m.group(1)
            # Parse arguments respecting parentheses
            inner = m.group(2)
            args = _split_args(inner)
            parsed_args = [parse(a) for a in args]
            fmt = GP_FUNC_MATH.get(fname)
            if fmt:
                return fmt.format(*parsed_args)
            else:
                return f"{fname}({', '.join(parsed_args)})"
        return s

    def _split_args(s):
        """Split comma-separated args respecting nested parens."""
        args = []
        depth = 0
        current = []
        for ch in s:
            if ch == '(':
                depth += 1
                current.append(ch)
            elif ch == ')':
                depth -= 1
                current.append(ch)
            elif ch == ',' and depth == 0:
                args.append(''.join(current).strip())
                current = []
            else:
                current.append(ch)
        if current:
            args.append(''.join(current).strip())
        return args

    return parse(s)
from factors.gp_signal import GPSignalGenerator
from reports.generator import ReportGenerator
from reports.telegram import TelegramReporter
from core.events import emit as emit_event

# ---------------------------------------------------------------------------
# Benchmark (buy-and-hold) configuration — kept as a US default for any
# legacy import path that still references the global. Market-aware code
# should use BENCHMARKS_BY_MARKET[market] from config.settings instead.
# ---------------------------------------------------------------------------
BENCHMARKS = BENCHMARKS_BY_MARKET["US"]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "trading.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("quant")

# ---------------------------------------------------------------------------
# US market hours helper (EDT: Mar-Nov, EST: Nov-Mar)
# ---------------------------------------------------------------------------
# ZoneInfo objects used by both is_market_hours() and is_market_hours_cn().
# Defined at module level (above the helpers that use them) so import order
# is irrelevant — the previous layout had _ET defined AFTER is_market_hours()
# referenced it, which would NameError on the new ET-based path.
try:
    from zoneinfo import ZoneInfo
    _CST = ZoneInfo("Asia/Shanghai")
    _ET = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    _CST = None
    _ET = None


def is_market_hours() -> bool:
    """Check if current time is within extended US market hours (pre+regular+after).
    Pre-market: 04:00-09:30 ET, Regular: 09:30-16:00 ET, After: 16:00-20:00 ET
    => 04:00-20:00 ET, Mon-Fri.

    IMPORTANT: must use ET weekday/hour, NOT UTC. The previous UTC-based
    simplification (`weekday<5 and (hour>=8 or hour<1)`) has a fatal edge
    case at the Sunday→Monday boundary: at 00:00 UTC Mon, ET is still
    Sun 20:00 ET (market closed), but UTC weekday=Mon and hour=0 satisfied
    `hour<1` → returned True. This silently let trading cycles fire on
    Sunday evenings ET (e.g. 2026-05-04 00:03 UTC = 2026-05-03 20:03 ET),
    creating ~30 phantom B-account trades. DST-aware via ZoneInfo.
    """
    if _ET is None:
        # Fallback only when zoneinfo unavailable.
        now = datetime.now(timezone.utc)
        if now.weekday() >= 5:
            return False
        # Use a slightly conservative window that excludes the 00:00-01:00 UTC
        # ambiguity entirely; this trades a tiny EST-tail loss for safety.
        return 8 <= now.hour < 24
    now_et = datetime.now(_ET)
    if now_et.weekday() >= 5:
        return False
    m = now_et.hour * 60 + now_et.minute
    return 4 * 60 <= m < 20 * 60


def is_us_rth() -> bool:
    """True only while the NYSE/Nasdaq regular equity session is open.

    Uses the exchange calendar rather than a weekday/time approximation, so
    holidays, DST and special closes (half-days) fail closed correctly.
    """
    from data.us_market_calendar import is_us_regular_session

    return is_us_regular_session(_utc_now())


# CN A-share session: 09:30-11:30 + 13:00-15:00 Asia/Shanghai, Mon-Fri.
# (_CST / _ET are defined at the top of this section.)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def is_market_hours_cn() -> bool:
    if _CST is None:
        return False
    now = datetime.now(_CST)
    if now.weekday() >= 5:
        return False
    m = now.hour * 60 + now.minute
    return (9 * 60 + 30 <= m < 11 * 60 + 30) or (13 * 60 <= m < 15 * 60)


def is_market_hours_for(market: str) -> bool:
    # US execution is regular-session only. Extended-hours collection remains
    # available through update_prices --no-trades.
    return is_us_rth() if market == "US" else is_market_hours_cn()


# ---------------------------------------------------------------------------
# Core system
# ---------------------------------------------------------------------------
class QuantSystem:
    """Main orchestrator for the quantitative trading system."""

    def __init__(self, market: str = "US"):
        if market not in MARKETS:
            raise ValueError(f"Unknown market: {market!r}. Allowed: {MARKETS}")
        self.market = market
        log.info("Initializing Quant Trading System (market=%s)...", market)

        # Resolve market-specific config
        self.universe = UNIVERSES[market]
        self.benchmarks = BENCHMARKS_BY_MARKET[market]
        self.initial_cash = INITIAL_CASH[market]

        # Derive market-scoped strategy lists. US keeps original A01-A10/B01-B10 ids;
        # CN gets CA01-CA10/CB01-CB10 by prefixing 'C' (per ACCOUNT_PREFIX[market]).
        prefix = ACCOUNT_PREFIX.get(market, "")
        if prefix:
            self.strategies = [_dc_replace(s, id=f"{prefix}{s.id}") for s in STRATEGIES]
            self.gp_strategies = [_dc_replace(g, id=f"{prefix}{g.id}") for g in active_gp_strategies_for_market(market)]
        else:
            self.strategies = list(STRATEGIES)
            self.gp_strategies = list(active_gp_strategies_for_market(market))
        # Qlib Q-accounts: US uses Q01-Q10 (no prefix); CN mirrors as CQ01-CQ10
        # via the same ACCOUNT_PREFIX mechanism (phase 2 — gated on CN qlib data).
        if prefix:
            self.qlib_strategies = [_dc_replace(q, id=f"{prefix}{q.id}") for q in QLIB_STRATEGIES]
        else:
            self.qlib_strategies = list(QLIB_STRATEGIES)

        # Init database. DataStore owns schema initialization.
        self.db_path = os.path.join(PROJECT_ROOT, "data", "trading.db")
        self.store = DataStore(self.db_path)

        # Data — market-aware fetcher
        self.fetcher = get_fetcher_for(market)

        # Factors
        self.factor_engine = FactorEngine()
        # decorrelate=False (V2 PCA whitening tested in replay 4/24–5/22, full A-group
        # avg PnL: V1 +8.4% vs V2-with-decorrelate -0.5%. Whitening on small (3-5)
        # high-corr factor sets amplifies noise PCs. Keep V2 plumbing for future experiments
        # but ship with it off.)
        self.signal_gen = SignalGenerator(buy_top=30, sell_top=10, decorrelate=False)

        # Trading — market-aware costs
        if market == "CN":
            self.costs = CNCosts()
        else:
            self.costs = MoomooAUCosts()
        self.engine = TradingEngine(costs=self.costs, trade_callback=self._on_trade)

        # Create strategy accounts (A01-A10 / CA01-CA10)
        for strat in self.strategies:
            self.engine.create_account(strat.id, initial_cash=self.initial_cash)
            log.info(f"  Account {strat.id}: {strat.name} ({strat.strategy_type})")

        # Create GP accounts (B01-B10 / CB01-CB10)
        for gp_strat in self.gp_strategies:
            self.engine.create_account(gp_strat.id, initial_cash=self.initial_cash)
            log.info(f"  GP Account {gp_strat.id}: {gp_strat.name}")

        # Create Qlib accounts (Q01-Q10) — US only in phase 1
        for q_strat in self.qlib_strategies:
            self.engine.create_account(q_strat.id, initial_cash=self.initial_cash)
            log.info(f"  Qlib Account {q_strat.id}: {q_strat.name} ({q_strat.model_class})")

        # Create benchmark (buy-and-hold) accounts
        for bm in self.benchmarks:
            self.engine.create_account(bm["id"], initial_cash=self.initial_cash)
            log.info(f"  Benchmark {bm['id']}: {bm['name']} (hold {bm['ticker']})")

        # Bootstrap account_meta rows for this market (idempotent).
        self._ensure_account_meta_rows()

        # GP factor mining — per-account
        self.gp_miner = GPAlphaMiner()
        self.factor_miner_gp = FactorMinerGPBackend()
        self.gp_signal_gen = GPSignalGenerator()
        self._per_account_gp_factors = {}   # {account_id: {ticker: DataFrame}}
        self._legacy_gp_mined = GPAlphaMiner.load_per_account_factors()  # B-family {account_id: [factor_list]}
        self._factor_miner_mined = FactorMinerGPBackend.load_factors()   # F-family {account_id: [factor_list]}
        self._per_account_mined = {**self._legacy_gp_mined, **self._factor_miner_mined}
        self._gp_mining_done = len(self._per_account_mined) > 0

        # Reports — only US sends Telegram. CN runs silent (per spec).
        if market == "US":
            try:
                self.reporter = TelegramReporter()
            except Exception:
                log.warning("TelegramReporter unavailable (no TELEGRAM_BOT_TOKEN), reports will be local only")
                self.reporter = None
        else:
            self.reporter = None
        self.report_gen = ReportGenerator(self.db_path)

        # State
        self._historical_data = {}
        self._factors_dict = {}
        self._prepared_alpha_signals = {}
        self._prepared_gp_signals = {}
        self._prepared_qlib_scores = {}
        self._fast_live_mode = False
        self._last_data_fetch = None
        self._last_report = None
        self._realtime_prices = {}  # {ticker: float} latest real-time prices
        self._realtime_quote_details = {}

        # 自适应换仓状态
        self._adaptive_enabled = ADAPTIVE_REBALANCE.get("enabled", False)
        self._market_returns = []  # 每日市场平均收益率
        self._sigma_ref = 0.01     # 参考波动率
        self._last_rebalance = {}  # {account_id: datetime} 上次换仓时间
        self._rebalance_hours_cache = {}  # {account_id: float} 当前自适应周期
        self._lot_skip_seen: dict[str, set[tuple[str, str, float, float]]] = defaultdict(set)

        # 从DB恢复状态
        self._restore_state()

        log.info("System initialized [%s] with %d+%d accounts, %d tickers (adaptive=%s)",
                 market, len(self.strategies), len(self.gp_strategies), len(self.universe),
                 "ON" if self._adaptive_enabled else "OFF")

    # -- retired account filter ----------------------------------------------
    def _retired_accounts(self) -> set[str]:
        """Return set of account_ids whose status='retired' for this market.
        Re-queried each trading cycle so retiring an account via CLI takes
        effect on the very next cron tick — no process restart required.
        """
        import sqlite3
        try:
            conn = sqlite3.connect(self.db_path)
            rows = conn.execute(
                "SELECT account_id FROM account_meta "
                "WHERE status='retired' AND market=?",
                (self.market,),
            ).fetchall()
            conn.close()
            return {r[0] for r in rows}
        except Exception as e:
            log.warning("Failed to load retired accounts: %s", e)
            return set()

    def _record_lot_skip(
        self,
        account: str,
        ticker: str,
        *,
        strategy_kind: str,
        budget: float,
        price: float,
        target_per_position: float,
        held_value: float,
        rank: int | None = None,
        score: float | None = None,
    ) -> None:
        """Persist an account-level event when a CN signal is skipped by lot sizing.

        Dedupe within the running process by (account, ticker, reason, rounded
        price/budget) so repeated candidate passes don't flood the dashboard.
        """
        detail = self.engine.buy_skip_detail_for_budget(budget, price)
        if not detail:
            return
        key = (ticker, detail.get("reason", "unknown"), round(float(budget), 2), round(float(price), 4))
        seen = self._lot_skip_seen[account]
        if key in seen:
            return
        seen.add(key)
        detail.update({
            "strategy_kind": strategy_kind,
            "target_per_position": target_per_position,
            "held_value": held_value,
            "rank": rank,
            "score": score,
            "source": "lot_sizing",
        })
        if detail.get("reason") == "board_lot_unaffordable":
            title = (
                f"⏭️ Skip {ticker}: 1手买不起/超仓位 "
                f"(budget ¥{budget:.0f}, 1手≈¥{detail.get('min_lot_notional', 0):.0f})"
            )
        else:
            title = f"⏭️ Skip {ticker}: buy sizing failed ({detail.get('reason')})"
        emit_event(
            "system",
            title,
            severity="info",
            account=account,
            ticker=ticker,
            detail=detail,
            market=self.market,
        )

    # -- account meta bootstrap ------------------------------------------------
    def _ensure_account_meta_rows(self):
        """Idempotently insert account_meta rows for this market's accounts."""
        import sqlite3
        try:
            from analysis.account_analyzer import _ensure_account_meta
        except Exception:
            _ensure_account_meta = None
        conn = sqlite3.connect(self.db_path)
        try:
            if _ensure_account_meta:
                _ensure_account_meta(conn)
            now = datetime.now(timezone.utc).isoformat()
            entries = []
            for s in self.strategies:
                entries.append((s.id, s.name, getattr(s, "strategy_type", ""), "A",
                                ",".join(getattr(s, "factor_names", []) or [])))
            for g in self.gp_strategies:
                # Build a descriptive factors string from GP params so the
                # dashboard strategy-factor panel is never blank.
                parts = [f"seed={g.gp_seed}", f"pop={g.gp_population}", f"gen={g.gp_generations}"]
                y = getattr(g, "gp_y_target", "next_1d_ret")
                if y != "next_1d_ret":
                    parts.append(f"y={y}")
                feat = getattr(g, "gp_feature_subset", None)
                if feat:
                    parts.append(f"feat={'/'.join(feat)}")
                backend = getattr(g, "mining_backend", "gplearn")
                prefix = "FMGP" if backend == "factor_miner_gp" else "GP"
                gp_factors_str = f"{prefix}({','.join(parts)})"
                entries.append((g.id, g.name, g.description, getattr(g, "family", "B"), gp_factors_str))
            for q in self.qlib_strategies:
                qfactors = f"qlib_{q.id}_score ({q.model_class})"
                entries.append((q.id, q.name, q.description, "Q", qfactors))
            for bm in self.benchmarks:
                entries.append((bm["id"], bm["name"], "benchmark", bm.get("group", "IDX"),
                                bm.get("ticker", "")))
            for acct_id, strat_name, desc, grp, factors in entries:
                exists = conn.execute(
                    "SELECT 1 FROM account_meta WHERE account_id=? AND market=?",
                    (acct_id, self.market),
                ).fetchone()
                if exists:
                    # Sync code-of-record fields so dashboard never drifts from
                    # accounts/strategies.py (factors/name/description/group).
                    # initial_cash, created_at, status, retired_at, retire_reason
                    # are operational state and intentionally preserved.
                    try:
                        conn.execute(
                            "UPDATE account_meta SET strategy_name=?, description=?, "
                            "\"group\"=?, factors=? "
                            "WHERE account_id=? AND market=?",
                            (strat_name, desc, grp, factors, acct_id, self.market),
                        )
                    except Exception:
                        pass
                    continue
                try:
                    conn.execute(
                        "INSERT INTO account_meta (account_id, strategy_name, description, "
                        "\"group\", factors, initial_cash, created_at, status, market) "
                        "VALUES (?,?,?,?,?,?,?,?,?)",
                        (acct_id, strat_name, desc, grp, factors,
                         self.initial_cash, now, "active", self.market),
                    )
                    # Emit an inception event so the LiveStream surfaces every
                    # new account as a green dot at birth (mirrors retire's yellow).
                    # Best-effort: never break account bootstrap on event failure.
                    try:
                        import json as _json
                        detail = {
                            "name": strat_name,
                            "group": grp,
                            "factors": factors,
                            "initial_cash": self.initial_cash,
                            "market": self.market,
                        }
                        conn.execute(
                            "INSERT INTO events (ts, category, severity, account, ticker, "
                            "title, detail, market) VALUES (?,?,?,?,?,?,?,?)",
                            (now, "inception", "info", acct_id, None,
                             f"🟢 {acct_id} 已创建 ({strat_name})",
                             _json.dumps(detail, ensure_ascii=False), self.market),
                        )
                    except Exception as _ev_e:
                        log.warning("inception event emit failed for %s: %s", acct_id, _ev_e)
                except sqlite3.IntegrityError:
                    # account_id is the table PK in older schemas; ignore
                    # if the same id already exists in another market.
                    pass
            conn.commit()
        finally:
            conn.close()

    # -- State persistence ----------------------------------------------------
    def _execute_live_signal(
        self, account: str, ticker: str, side: str, shares: float, price: float,
        current_prices: dict[str, float], *, reason: str = "signal",
    ):
        """Apply production execution-state guards before mutating the paper book."""
        from config.security_master import ticker_lifecycle_block_reason
        symbol_block = ticker_lifecycle_block_reason(ticker, self.market, _utc_now())
        if symbol_block:
            emit_event(
                "guard", f"⛔ {account}/{ticker} blocked: {symbol_block}",
                severity="critical", account=account, ticker=ticker,
                detail={"reason": symbol_block}, market=self.market,
            )
            return None
        if self.market == "CN":
            detail = getattr(self, "_realtime_quote_details", {}).get(ticker)
            block = None
            if not isinstance(detail, dict):
                block = "execution_state_unknown"
            else:
                try:
                    prev_close = float(detail["prev_close"])
                    volume = float(detail["volume"])
                    quote_price = float(detail.get("price", price))
                    if volume <= 0:
                        block = "cn_suspended_or_no_volume"
                    elif side == "sell" and quote_price <= prev_close * 0.90 + 1e-9:
                        block = "cn_limit_down"
                    elif side == "buy" and quote_price >= prev_close * 1.10 - 1e-9:
                        block = "cn_limit_up"
                except (KeyError, TypeError, ValueError):
                    block = "execution_state_unknown"
            if block:
                emit_event(
                    "guard", f"⛔ {account}/{ticker} blocked: {block}",
                    severity="warning", account=account, ticker=ticker,
                    detail={"reason": block, "quote": detail}, market=self.market,
                )
                return None
        return self.engine.execute_signal(
            account, ticker, side, shares, price, current_prices, reason=reason
        )

    def _on_trade(self, account_name: str, trade: dict):
        """Atomically persist a trade, its event, and the resulting account state."""
        side = trade["side"]
        ticker = trade["ticker"]
        shares = trade["shares"]
        price = trade["price"]
        reason = trade.get("reason", "signal")
        from config.settings import CURRENCY_SYMBOL
        sym = CURRENCY_SYMBOL.get(self.market, "$")
        if side == "buy":
            title = f"BUY {ticker} ×{int(shares)} @ {sym}{price:.2f}"
            detail = {"shares": shares, "price": price, "fees": trade.get("total_fees", 0)}
        else:
            pnl_pct = trade.get("pnl_pct")
            pnl_dollar = trade.get("pnl_dollar")
            reason_label = {"stop_loss": "🛑 stop-loss", "signal": "📉 signal-exit",
                            "take_profit": "🎯 take-profit"}.get(reason, reason)
            pnl_str = ""
            if pnl_pct is not None:
                pnl_str = f" PnL {pnl_pct*100:+.2f}%"
                if pnl_dollar is not None:
                    pnl_str += f" ({sym}{pnl_dollar:+.2f})"
            title = f"SELL {ticker} ×{int(shares)} @ {sym}{price:.2f} · {reason_label}{pnl_str}"
            detail = {"shares": shares, "price": price, "reason": reason,
                      "pnl_pct": pnl_pct, "pnl_dollar": pnl_dollar}

        account = self.engine.get_account(account_name)
        durable_prices = self.store.load_position_prices(account_name, market=self.market)
        positions = [
            {
                "ticker": held_ticker,
                "shares": position.shares,
                "avg_cost": position.avg_cost,
                "total_cost": position.total_cost,
                # Preserve a durable mark for each still-open holding. The
                # triggering quote is safe for both buys and partial sells;
                # untouched positions keep their previous mark.
                "current_price": (
                    float(price)
                    if held_ticker == ticker
                    else durable_prices.get(held_ticker)
                ),
            }
            for held_ticker, position in account._positions.items()
            if position.shares > 0
        ]
        self.store.save_trade_with_state(
            account=account_name,
            trade=trade,
            cash=account.cash,
            initial_cash=account.initial_cash,
            positions=positions,
            event={"title": title, "detail": detail},
            market=self.market,
        )

    def _restore_state(self):
        """Restore account cash, positions, and adaptive state from DB."""
        restored = 0
        all_accounts = (
            [(s.id, s) for s in self.strategies]
            + [(g.id, g) for g in self.gp_strategies]
            + [(q.id, q) for q in self.qlib_strategies]
            + [(bm["id"], bm) for bm in self.benchmarks]
        )
        for acct_id, _ in all_accounts:
            state = self.store.load_account_state(acct_id, market=self.market)
            if state is None:
                continue
            acct = self.engine.get_account(acct_id)
            acct.cash = state["cash"]
            acct.initial_cash = state["initial_cash"]

            # Restore positions. CN T+1 acquisition lots are reconstructed from
            # the durable trade log; mismatches fail closed (zero sellable) rather
            # than inventing settlement dates.
            positions = self.store.load_positions(acct_id, market=self.market)
            trade_lots = (
                self.store.load_trade_lots(acct_id, market=self.market)
                if self.market == "CN" else {}
            )
            for p in positions:
                from trading.account import _Position
                lots = trade_lots.get(p["ticker"]) if self.market == "CN" else None
                if self.market == "CN":
                    explained = sum(float(lot.get("shares", 0)) for lot in (lots or []))
                    if abs(explained - float(p["shares"])) > 1e-6:
                        log.error(
                            "[%s/%s] T+1 lot replay mismatch for %s: positions=%s trades=%s; fail-closed",
                            self.market, acct_id, p["ticker"], p["shares"], explained,
                        )
                        lots = []
                acct._positions[p["ticker"]] = _Position(
                    shares=p["shares"],
                    avg_cost=p["avg_cost"],
                    total_cost=p["total_cost"],
                    lots=lots,
                )
            restored += 1
            log.info("  Restored %s: $%.2f cash, %d positions",
                     acct_id, acct.cash, len(positions))

        # Restore adaptive state
        if self._adaptive_enabled:
            adaptive_states = self.store.load_all_adaptive_state(market=self.market)
            for acct_id, astate in adaptive_states.items():
                if astate["last_rebalance"]:
                    try:
                        self._last_rebalance[acct_id] = datetime.fromisoformat(astate["last_rebalance"])
                    except (ValueError, TypeError):
                        pass
                if astate["rebalance_hours"]:
                    self._rebalance_hours_cache[acct_id] = astate["rebalance_hours"]
                if astate["sigma_ref"]:
                    self._sigma_ref = astate["sigma_ref"]

            # Restore market returns
            mkt_rets = self.store.load_market_returns()
            if mkt_rets:
                self._market_returns = mkt_rets
                log.info("  Restored %d market returns, sigma_ref=%.6f",
                         len(mkt_rets), self._sigma_ref)

        if restored > 0:
            log.info("Restored state for %d/%d accounts from DB", restored, len(all_accounts))

    def _save_all_state(self):
        """Persist all account states, positions, and adaptive state."""
        current_prices = self._get_current_prices(
            fetch_missing_benchmarks=not getattr(self, "_fast_live_mode", False)
        )
        retired = self._retired_accounts()

        all_accounts = (
            [(s.id,) for s in self.strategies]
            + [(g.id,) for g in self.gp_strategies]
            + [(q.id,) for q in self.qlib_strategies]
            + [(bm["id"],) for bm in self.benchmarks]
        )
        if getattr(self, "_fast_live_mode", False):
            missing_marks = sorted(
                {
                    ticker
                    for (acct_id,) in all_accounts
                    if acct_id not in retired
                    for ticker, position in self.engine.get_account(acct_id)._positions.items()
                    if position.shares > 0 and ticker not in current_prices
                }
            )
            if missing_marks:
                log.warning(
                    "Fast live state save will skip accounts with blocked/missing marks: %s",
                    ",".join(missing_marks[:20]),
                )
        for (acct_id,) in all_accounts:
            if acct_id in retired:
                continue  # frozen — preserve last-known state, no equity drift
            acct = self.engine.get_account(acct_id)

            # Build the position payload from the in-memory account FIRST, then
            # persist cash+positions atomically. If one held ticker is missing a
            # quote, skip only the equity snapshot; never save new cash while
            # leaving old positions behind (B11 2026-06-15/17 cliff bug).
            held_tickers = [t for t, p in acct._positions.items() if p.shares > 0]
            missing_prices = [t for t in held_tickers if t not in current_prices]
            positions = []
            for ticker, pos in acct._positions.items():
                if pos.shares <= 0:
                    continue
                price = current_prices.get(ticker)
                item = {
                    "ticker": ticker,
                    "shares": pos.shares,
                    "avg_cost": pos.avg_cost,
                    "total_cost": pos.total_cost,
                    "current_price": price,
                }
                if price is not None:
                    item["market_value"] = pos.shares * price
                    item["unrealized_pnl"] = pos.shares * (price - pos.avg_cost)
                positions.append(item)

            verified_equity = None
            if missing_prices and getattr(self, "_fast_live_mode", False):
                log.warning(
                    "Skipping all state persistence for %s: blocked/missing realtime prices "
                    "for %d/%d held tickers: %s",
                    acct_id, len(missing_prices), len(held_tickers),
                    ",".join(missing_prices[:8]),
                )
                continue
            if missing_prices:
                log.error(
                    "Saving cash+positions for %s but skipping equity snapshot: "
                    "missing prices for %d/%d held tickers: %s",
                    acct_id, len(missing_prices), len(held_tickers), ",".join(missing_prices[:8]),
                )
            else:
                try:
                    verified_equity = acct.get_equity(current_prices, strict=True)
                except ValueError as e:
                    log.error("Skipping snapshot for %s: %s", acct_id, e)
                except Exception as e:
                    log.warning("Snapshot equity compute failed for %s: %s", acct_id, e)

            try:
                self.store.save_account_full_state(
                    acct_id, acct.cash, acct.initial_cash, positions,
                    equity=verified_equity, market=self.market,
                )
            except Exception as e:
                log.warning("Full-state save failed for %s: %s", acct_id, e)

        # Save adaptive state
        if self._adaptive_enabled:
            for (acct_id,) in all_accounts:
                last_rb = self._last_rebalance.get(acct_id)
                self.store.save_adaptive_state(
                    acct_id,
                    last_rebalance=last_rb.isoformat() if last_rb else None,
                    rebalance_hours=self._rebalance_hours_cache.get(acct_id),
                    sigma_ref=self._sigma_ref,
                    market=self.market,
                )

        log.info("Saved state for %d accounts to DB", len(all_accounts))

    # -- Real-time prices ---------------------------------------------------
    def _fetch_realtime_prices(self, *, strict_execution: bool = False) -> dict[str, float]:
        """Fetch real-time prices via the market-aware fetcher.

        US fetcher uses yfinance fast_info → Finnhub fallback; CN fetcher
        uses akshare/Sina. Both routes converge here so main.py and
        scripts/update_prices.py write equity off the SAME price source.

        Important: live portfolios can contain legacy/non-universe holdings
        after universe rotation (e.g. ARM after Russell-1000 refresh).  Fetch
        universe + active held tickers + benchmarks, otherwise the trading
        cycle can successfully trade from universe prices but skip equity
        snapshots for accounts holding legacy symbols.
        """
        tickers = set(self.universe)
        retired = self._retired_accounts()
        for acct_id, acct in self.engine.accounts.items():
            if acct_id in retired:
                continue
            for ticker, pos in acct._positions.items():
                if pos.shares > 0:
                    tickers.add(ticker)
        for bm in self.benchmarks:
            tickers.add(bm["ticker"])
        requested = sorted(tickers)
        getter = getattr(self.fetcher, "get_realtime_quote_metadata", None)
        if getter is None:
            if strict_execution:
                raise RuntimeError(
                    f"[{self.market}] execution quote gate requires metadata; "
                    "legacy scalar quotes are display-only"
                )
            try:
                return self.fetcher.get_realtime_quotes(requested)
            except Exception as e:
                log.warning("realtime quote fetch failed: %s", e)
                return {}
        try:
            quotes = getter(requested)
        except Exception as e:
            if strict_execution:
                raise RuntimeError(
                    f"[{self.market}] execution quote metadata fetch failed"
                ) from e
            log.warning("realtime quote metadata fetch failed: %s", e)
            return {}
        prices = prices_from_quotes(quotes)
        if strict_execution:
            now = _utc_now()
            max_age = float(os.environ.get("QUANT_MAX_QUOTE_AGE_SECONDS", "180"))
            validation = validate_required_quotes(
                quotes, requested, now=now, max_age_seconds=max_age
            )
            if not validation["ok"]:
                raise RuntimeError(
                    f"[{self.market}] execution quote gate failed: "
                    f"missing={validation['missing']} invalid={validation['invalid']} "
                    f"stale={validation['stale']} untradable={validation['untradable']}"
                )
            self._realtime_quote_details = {
                ticker: {
                    "price": quote.price,
                    "prev_close": quote.prev_close,
                    "volume": quote.volume,
                }
                for ticker, quote in quotes.items()
            }
        if not prices:
            log.warning("Failed to fetch any real-time prices")
        return prices

    def _fetch_realtime_prices_for(self, tickers) -> dict[str, float]:
        """Fetch quotes for an explicit, already-bounded fast-cycle universe."""
        requested = sorted({str(t) for t in tickers if t})
        if not requested:
            return {}
        try:
            prices = self.fetcher.get_realtime_quotes(requested)
        except Exception as e:
            log.warning("bounded realtime quote fetch failed: %s", e)
            return {}
        if not prices:
            log.warning("Bounded realtime quote fetch returned no prices")
            return {}
        return {
            ticker: float(price)
            for ticker, price in prices.items()
            if ticker in requested and price is not None and float(price) > 0
        }

    # -- Persisted-signal fast live path -------------------------------------
    def _latest_market_price_date(self, conn) -> str | None:
        """Latest cached daily price date for this market's configured universe."""
        universe = list(self.universe)
        if not universe:
            return None
        placeholders = ",".join("?" * len(universe))
        row = conn.execute(
            f"SELECT MAX(substr(datetime,1,10)) FROM prices "
            f"WHERE interval='1d' AND ticker IN ({placeholders})",
            universe,
        ).fetchone()
        return row[0] if row else None

    def _emit_persisted_signal_gate(
        self,
        *,
        account_id: str,
        factor_group: str,
        reason: str,
        factor_date: str | None,
        latest_price_date: str | None,
        covered_tickers: int,
        universe_size: int,
        coverage: float,
        required_coverage: float,
        lag_days: int | None,
        max_lag_days: int,
    ) -> None:
        detail = {
            "reason": reason,
            "factor_group": factor_group,
            "factor_date": factor_date,
            "latest_price_date": latest_price_date,
            "lag_days": lag_days,
            "max_lag_days": max_lag_days,
            "covered_tickers": covered_tickers,
            "universe_size": universe_size,
            "coverage": round(coverage, 4),
            "required_coverage": required_coverage,
        }
        log.error(
            "[%s] persisted signal gate failed: group=%s reason=%s date=%s "
            "latest_price=%s lag=%s coverage=%d/%d (%.1f%%; need %.1f%%); no-op",
            account_id, factor_group, reason, factor_date, latest_price_date,
            lag_days, covered_tickers, universe_size, coverage * 100,
            required_coverage * 100,
        )
        emit_event(
            "factor",
            f"⚠️ Persisted signal skipped: {account_id} ({reason})",
            severity="error",
            account=account_id,
            detail=detail,
            market=self.market,
        )

    def _load_latest_persisted_factor_frames(
        self,
        *,
        account_id: str,
        factor_group: str,
        factor_names: list[str],
        min_coverage: float = 0.80,
        max_lag_days: int | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Load one account's newest persisted factor cross-section, fail closed.

        Every requested factor must come from the same latest factor date.  The
        account is allowed to trade only when that date is near the market's
        latest cached daily bar and enough of the configured universe has at
        least one usable requested factor.  A failed gate returns an empty dict;
        callers treat it as a technical no-op and never liquidate positions.
        """
        import sqlite3

        names = list(dict.fromkeys(name for name in factor_names if name))
        universe = list(self.universe)
        if max_lag_days is None:
            max_lag_days = 3 if self.market == "CN" else 2
        if not names or not universe:
            self._emit_persisted_signal_gate(
                account_id=account_id,
                factor_group=factor_group,
                reason="empty_config",
                factor_date=None,
                latest_price_date=None,
                covered_tickers=0,
                universe_size=len(universe),
                coverage=0.0,
                required_coverage=min_coverage,
                lag_days=None,
                max_lag_days=max_lag_days,
            )
            return {}

        try:
            conn = sqlite3.connect(self.db_path)
            ticker_ph = ",".join("?" * len(universe))
            factor_ph = ",".join("?" * len(names))
            latest_row = conn.execute(
                f"SELECT MAX(date) FROM factor_values "
                f"WHERE factor_group=? AND factor_name IN ({factor_ph}) "
                f"AND ticker IN ({ticker_ph}) AND value IS NOT NULL",
                (factor_group, *names, *universe),
            ).fetchone()
            factor_date = latest_row[0] if latest_row else None
            latest_price_date = self._latest_market_price_date(conn)
            rows = []
            if factor_date:
                rows = conn.execute(
                    f"SELECT ticker,factor_name,value FROM factor_values "
                    f"WHERE factor_group=? AND date=? AND value IS NOT NULL "
                    f"AND factor_name IN ({factor_ph}) "
                    f"AND ticker IN ({ticker_ph})",
                    (factor_group, factor_date, *names, *universe),
                ).fetchall()
            conn.close()
        except Exception as e:
            log.warning("[%s] persisted factor load failed for %s: %s", account_id, factor_group, e)
            return {}

        covered = len({ticker for ticker, _name, _value in rows})
        coverage = covered / len(universe) if universe else 0.0
        lag_days = None
        if factor_date and latest_price_date:
            try:
                from datetime import date as _date
                lag_days = (
                    _date.fromisoformat(str(latest_price_date)[:10])
                    - _date.fromisoformat(str(factor_date)[:10])
                ).days
            except (TypeError, ValueError):
                lag_days = None

        reason = None
        if not factor_date or not rows:
            reason = "missing"
        elif latest_price_date and (lag_days is None or lag_days > max_lag_days):
            reason = "stale"
        elif coverage < min_coverage:
            reason = "coverage"
        if reason:
            self._emit_persisted_signal_gate(
                account_id=account_id,
                factor_group=factor_group,
                reason=reason,
                factor_date=factor_date,
                latest_price_date=latest_price_date,
                covered_tickers=covered,
                universe_size=len(universe),
                coverage=coverage,
                required_coverage=min_coverage,
                lag_days=lag_days,
                max_lag_days=max_lag_days,
            )
            return {}

        by_ticker: dict[str, dict[str, float]] = defaultdict(dict)
        for ticker, factor_name, value in rows:
            by_ticker[ticker][factor_name] = float(value)
        return {
            ticker: pd.DataFrame(
                [[values.get(name, np.nan) for name in names]],
                index=[pd.Timestamp(str(factor_date)[:10])],
                columns=names,
            )
            for ticker, values in by_ticker.items()
        }

    @staticmethod
    def _signals_from_persisted_alpha_frames(
        frames: dict[str, pd.DataFrame],
        *,
        strategy_type: str,
        buy_top: int,
        sell_top: int = 10,
    ) -> dict:
        """Rank exactly the account-declared persisted factors cross-sectionally."""
        rows = {}
        for ticker, frame in frames.items():
            if frame is None or frame.empty:
                continue
            rows[ticker] = frame.iloc[-1]
        if not rows:
            return {"buy": [], "sell": []}
        cross_section = pd.DataFrame(rows).T
        composite = cross_section.rank(axis=0, pct=True).mean(axis=1, skipna=True).dropna()
        if composite.empty:
            return {"buy": [], "sell": []}
        ranks = composite.rank(pct=True)
        if strategy_type == "mean_reversion":
            ranks = 1 - ranks
        ranked = ranks.sort_values(ascending=False)
        return {
            "buy": [(ticker, round(float(score), 4)) for ticker, score in ranked.head(buy_top).items()],
            "sell": [(ticker, round(float(score), 4)) for ticker, score in ranked.tail(sell_top).items()],
        }

    @staticmethod
    def _signal_tickers(signals: dict | None, hold_count: int) -> set[str]:
        """Extract candidate + hold-band tickers from A or GP signal shapes."""
        if not signals:
            return set()
        buy = signals.get("buy") or []
        out = set()
        for item in buy[:max(0, hold_count)]:
            ticker = item[0] if isinstance(item, (tuple, list)) else item
            if ticker:
                out.add(str(ticker))
        return out

    def _active_holding_tickers(self) -> set[str]:
        retired = self._retired_accounts()
        return {
            ticker
            for account_id, account in self.engine.accounts.items()
            if account_id not in retired
            for ticker, position in account._positions.items()
            if position.shares > 0
        }

    def prepare_fast_live_cycle(self) -> set[str]:
        """Load persisted A/GP/F/Q signals and return the bounded quote universe."""
        cc = CHURN_CONTROLS if CHURN_CONTROLS.get("enabled", True) else {}
        hold_mult = int(cc.get("hold_band_mult", 1) or 1)
        quote_tickers = self._active_holding_tickers()
        quote_tickers.update(bm["ticker"] for bm in self.benchmarks)
        self._prepared_alpha_signals = {}
        self._prepared_gp_signals = {}
        self._prepared_qlib_scores = {}
        self._factors_dict = {}
        self._per_account_gp_factors = {}

        # Alpha accounts use exactly the factors declared by each account.  A06
        # is the legacy "all factors" account; derive its active names from the
        # latest alpha158 date rather than loading arbitrary historical rows.
        all_alpha_names = None
        for strat in self.strategies:
            names = list(getattr(strat, "factor_names", None) or [])
            if not names:
                if all_alpha_names is None:
                    import sqlite3
                    with sqlite3.connect(self.db_path) as conn:
                        universe = list(self.universe)
                        if universe:
                            ph = ",".join("?" * len(universe))
                            row = conn.execute(
                                f"SELECT MAX(date) FROM factor_values "
                                f"WHERE factor_group='alpha158' AND ticker IN ({ph})",
                                universe,
                            ).fetchone()
                            latest = row[0] if row else None
                            if latest:
                                all_alpha_names = [
                                    r[0] for r in conn.execute(
                                        f"SELECT DISTINCT factor_name FROM factor_values "
                                        f"WHERE factor_group='alpha158' AND date=? "
                                        f"AND ticker IN ({ph}) ORDER BY factor_name",
                                        (latest, *universe),
                                    ).fetchall()
                                ]
                        if all_alpha_names is None:
                            all_alpha_names = []
                names = all_alpha_names
            frames = self._load_latest_persisted_factor_frames(
                account_id=strat.id,
                factor_group="alpha158",
                factor_names=names,
            )
            # Persist per-account signals because A configurations may select
            # different factor names even when strategy_type is the same.
            signals = (
                self._signals_from_persisted_alpha_frames(
                    frames,
                    strategy_type=strat.strategy_type,
                    buy_top=max(30, strat.top_n * hold_mult),
                )
                if frames else {"buy": [], "sell": []}
            )
            self._prepared_alpha_signals[strat.id] = signals
            quote_tickers.update(
                self._signal_tickers(signals, strat.top_n * hold_mult)
            )

        for gp_strat in self.gp_strategies:
            mined = [
                f for f in self._per_account_mined.get(gp_strat.id, [])
                if self._is_active_mined_factor(f)
            ]
            names = [f.get("name") for f in mined if f.get("name")]
            prefix = "fmgp" if getattr(gp_strat, "family", "B") == "F" else "gp"
            frames = self._load_latest_persisted_factor_frames(
                account_id=gp_strat.id,
                factor_group=f"{prefix}_{gp_strat.id}",
                factor_names=names,
            )
            self._per_account_gp_factors[gp_strat.id] = frames
            filtered = self._filter_gp_factors(gp_strat, frames) if frames else {}
            signals = (
                self.gp_signal_gen.generate_signals(
                    filtered,
                    top_n=gp_strat.top_n,
                    buy_candidates=max(gp_strat.top_n * 5, 30),
                )
                if filtered else {"buy": [], "sell": []}
            )
            self._prepared_gp_signals[gp_strat.id] = signals
            quote_tickers.update(
                self._signal_tickers(signals, gp_strat.top_n * hold_mult)
            )

        for q_strat in self.qlib_strategies:
            scored = self._load_qlib_scores(q_strat.id)
            self._prepared_qlib_scores[q_strat.id] = scored
            quote_tickers.update(
                ticker for ticker, _score in scored[:q_strat.top_n * hold_mult]
            )

        log.info(
            "Prepared persisted signals [%s]: A=%d GP/F=%d Q=%d bounded_quotes=%d "
            "(universe=%d)",
            self.market,
            sum(bool(v.get("buy")) for v in self._prepared_alpha_signals.values()),
            sum(bool(v.get("buy")) for v in self._prepared_gp_signals.values()),
            sum(bool(v) for v in self._prepared_qlib_scores.values()),
            len(quote_tickers), len(self.universe),
        )
        return quote_tickers

    def run_fast_live_cycle(
        self,
        prepared_path: str | None = None,
        *,
        dry_run: bool = False,
    ):
        """Trade from persisted signals without universe history/factor work."""
        if not is_market_hours_for(self.market) and not dry_run:
            log.info("Outside %s market window — fast live cycle skipped", self.market)
            return None

        self._fast_live_mode = True
        if prepared_path:
            prepared = json.loads(open(prepared_path).read())
            if prepared.get("market") != self.market:
                raise RuntimeError(
                    f"prepared fast-cycle market mismatch: {prepared.get('market')} != {self.market}"
                )
            prepared_at = prepared.get("prepared_at")
            if prepared_at is None:
                raise RuntimeError("prepared fast-cycle artifact missing prepared_at")
            try:
                prepared_dt = datetime.fromisoformat(
                    str(prepared_at).replace("Z", "+00:00")
                )
                if prepared_dt.tzinfo is None:
                    prepared_dt = prepared_dt.replace(tzinfo=timezone.utc)
                age = (_utc_now() - prepared_dt.astimezone(timezone.utc)).total_seconds()
            except (TypeError, ValueError):
                raise RuntimeError("prepared fast-cycle artifact has invalid prepared_at")
            max_age = float(os.environ.get("QUANT_FAST_PREPARED_MAX_AGE_SECONDS", "180"))
            if age < -30 or age > max_age:
                raise RuntimeError(
                    f"prepared fast-cycle artifact stale: age={age:.1f}s max={max_age:.1f}s"
                )
            quote_tickers = set(prepared.get("tickers") or [])
            self._prepared_alpha_signals = prepared.get("prepared_alpha_signals") or {}
            self._prepared_gp_signals = prepared.get("prepared_gp_signals") or {}
            self._prepared_qlib_scores = {
                account: [(str(ticker), float(score)) for ticker, score in scores]
                for account, scores in (prepared.get("prepared_qlib_scores") or {}).items()
            }
            # The fast trade phase no longer needs factor frames; prepared GP
            # rankings are authoritative for this cycle.
            self._per_account_gp_factors = {
                account: {} for account in self._prepared_gp_signals
            }
        else:
            t0 = time.time()
            quote_tickers = self.prepare_fast_live_cycle()
            prepare_seconds = time.time() - t0
            if prepare_seconds > 90:
                log.error(
                    "[%s] fast live cycle aborted: persisted-signal preparation took %.1fs (>90s)",
                    self.market, prepare_seconds,
                )
                return None
        held = self._active_holding_tickers()
        missing_held = sorted(held - quote_tickers)
        if missing_held:
            raise RuntimeError(
                "prepared fast-cycle artifact missing active holdings: "
                + ",".join(missing_held[:20])
            )
        getter = getattr(self.fetcher, "get_realtime_quote_metadata", None)
        if getter is None:
            raise RuntimeError(
                f"[{self.market}] fast live execution requires quote metadata; "
                "legacy scalar quotes are display-only"
            )
        else:
            quotes = getter(sorted(quote_tickers))
        # Every quote that can reach a buy/sell path must be positive, fresh,
        # timestamped, and tradable. Validating only held names still allowed a
        # stale/untradable candidate quote to create a new position.
        max_quote_age = float(os.environ.get("QUANT_MAX_QUOTE_AGE_SECONDS", "180"))
        validation = validate_required_quotes(
            quotes, quote_tickers, now=_utc_now(),
            max_age_seconds=max_quote_age,
        )
        valid_tickers = valid_quote_tickers(
            quotes, quote_tickers, now=_utc_now(),
            max_age_seconds=max_quote_age,
        )
        blocked_tickers = sorted(quote_tickers - valid_tickers)
        if blocked_tickers:
            log.warning(
                "Fast-cycle quote gate blocked %d/%d tickers individually: %s",
                len(blocked_tickers), len(quote_tickers), ",".join(blocked_tickers[:20]),
            )
        # Invalid/stale/untradable quotes are excluded per ticker. Accounts with
        # missing held marks are skipped by each cycle and by _save_all_state;
        # unaffected accounts continue instead of turning one stale symbol into
        # a market-wide execution outage.
        self._realtime_prices = {
            ticker: price
            for ticker, price in prices_from_quotes(quotes).items()
            if ticker in valid_tickers
        }
        self._realtime_quote_details = {
            ticker: {
                "price": quote.price,
                "prev_close": quote.prev_close,
                "volume": quote.volume,
            }
            for ticker, quote in quotes.items()
            if ticker in valid_tickers
        }
        if not self._realtime_prices:
            raise RuntimeError(f"[{self.market}] fast live cycle: no bounded realtime quotes")
        if dry_run:
            log.info(
                "Fast live dry-run [%s] OK: prepared=%d quoted=%d held=%d; no trades/state writes",
                self.market, len(quote_tickers), len(self._realtime_prices), len(held),
            )
            return None

        self.initialize_benchmarks()
        self.run_trading_cycle()
        self.run_gp_trading_cycle()
        self.run_qlib_trading_cycle()
        try:
            self._save_all_state()
        except Exception:
            log.exception("Failed to save state after fast live cycle")
            raise
        return None

    def _get_current_prices(self, fetch_missing_benchmarks: bool = True) -> dict[str, float]:
        """Get best available prices: real-time > historical close.

        Fast live cycles pass ``fetch_missing_benchmarks=False`` so this helper
        cannot silently expand the bounded quote request with historical calls.
        """
        # Base: historical close prices
        prices = {}
        for ticker, df in self._historical_data.items():
            if df is not None and not df.empty:
                prices[ticker] = float(df["close"].iloc[-1])
        # Override with real-time where available
        prices.update(self._realtime_prices)
        # Ensure benchmark tickers have prices
        for bm in self.benchmarks:
            t = bm["ticker"]
            if t not in prices and fetch_missing_benchmarks:
                # Realtime via the market-aware fetcher (no direct yfinance).
                try:
                    q = self.fetcher.get_realtime_quotes([t])
                    if q.get(t):
                        prices[t] = float(q[t])
                        continue
                except Exception:
                    pass
                # Fallback: latest cached close
                try:
                    df = self.fetcher.get_historical([t], days=5)
                    if df is not None and not df.empty:
                        prices[t] = float(df["close"].iloc[-1])
                except Exception:
                    pass
        return prices

    # -- Step 1: Fetch data -------------------------------------------------
    def _alpha_history_days(self) -> int:
        """History length for live Alpha158 factor computation.

        Alpha158 includes 20-trading-day rolling factors (ROC_20,
        MA_RATIO_20, STD_20, BBPOS_20, BETA_20, etc.). The old 30 calendar-day
        window can shrink to ~19-21 trading bars after weekends/holidays/data
        gaps, so 20d factors silently became NaN and stopped updating. Use 90
        calendar days for a stable warm-up while keeping the live cycle cheap.
        """
        return 90

    def fetch_data(self):
        """Download historical data with enough warm-up for Alpha158 factors."""
        days = self._alpha_history_days()
        log.info("Fetching %d-day historical data for %d tickers...", days, len(self.universe))
        raw_df = self.fetcher.get_historical(self.universe, days=days)

        # Split concatenated DataFrame into per-ticker dict
        self._historical_data = {}
        if not raw_df.empty and "ticker" in raw_df.columns:
            for ticker, group in raw_df.groupby("ticker"):
                df = group.drop(columns=["ticker"]).set_index("datetime").sort_index()
                self._historical_data[ticker] = df

        log.info("Got data for %d/%d tickers", len(self._historical_data), len(self.universe))
        self._last_data_fetch = datetime.now(timezone.utc)
        emit_event(
            "data",
            f"📡 Data refresh [{self.market}]: {len(self._historical_data)}/{len(self.universe)} tickers ({days}d)",
            detail={"got": len(self._historical_data), "total": len(self.universe), "days": days},
            market=self.market,
        )

        # 更新自适应换仓的市场波动率数据
        if self._adaptive_enabled:
            self._update_market_returns()

    # -- Step 2: Compute factors --------------------------------------------
    def compute_factors(self):
        """Compute Alpha158-style factors for all tickers."""
        log.info("Computing factors...")
        self._factors_dict = self.factor_engine.compute_multi(self._historical_data)
        good = sum(1 for v in self._factors_dict.values() if v is not None and not v.empty)
        log.info("Computed factors for %d tickers", good)
        emit_event(
            "factor",
            f"🧮 Alpha158 cross-section recomputed [{self.market}]: {good} tickers",
            detail={"tickers": good, "group": "alpha158"},
            market=self.market,
        )

        # 持久化因子值到DB
        try:
            self.store.save_factor_df(self._factors_dict, group="alpha158")
        except Exception as e:
            log.warning("Failed to save factor values: %s", e)

    # -- Step 2b: Mine and compute GP factors --------------------------------
    def _gp_compute_history_days(self) -> int:
        """History length for applying mined GP/F-family expressions.

        This does not change the paper/article premise: factors are still mined
        on past data, then evaluated deterministically. It only makes the live
        application window long enough for FactorMiner terminals such as
        vol_of_vol_20, skew_20, kurt_20, trend_r2_20, and pv_corr_20 to exist.
        The previous 30 calendar days yielded ~21 trading bars, so F13's nested
        20-day risk features were all-NaN at runtime despite being mineable on
        the 120-day training window.
        """
        needs_expanded = any(
            getattr(g, "mining_backend", "gplearn") == "factor_miner_gp"
            for g in self.gp_strategies
        )
        return 90 if needs_expanded else 30

    def _load_gp_compute_history(self) -> dict:
        days = self._gp_compute_history_days()
        if days <= 30:
            return self._historical_data

        # fetch_data() now uses the same 90d warm-up required by FactorMiner's
        # nested 20d terminals, so the GP/F application path should normally
        # reuse it.  Avoid a second 1000-ticker yfinance pull inside the same
        # cycle; that duplicate fetch was a major reason cycles overran before
        # factor writes could be observed clearly.
        min_rows = max(1, int(days * 0.45))
        if self._historical_data:
            lengths = [len(df) for df in self._historical_data.values() if df is not None]
            # A few stale/delisted/legacy symbols can have shorter histories; do
            # not let them force a second 1000-ticker download. If the universe
            # as a whole has a 90d-ish window, GP/F should reuse it and simply
            # skip/NaN sparse tickers during factor construction.
            good = sum(1 for n in lengths if n >= min_rows)
            if lengths and good >= max(1, int(len(self.universe) * 0.90)):
                log.info(
                    "Reusing %d-day fetch_data history for GP/F-family factor application "
                    "(%d/%d tickers >= %d rows; min=%d median=%d)",
                    days, good, len(self.universe), min_rows, min(lengths),
                    sorted(lengths)[len(lengths) // 2],
                )
                return self._historical_data

        try:
            log.info("Fetching %d-day history for GP/F-family factor application...", days)
            raw_df = self.fetcher.get_historical(self.universe, days=days)
            hist = {}
            if raw_df is not None and not raw_df.empty and "ticker" in raw_df.columns:
                for ticker, group in raw_df.groupby("ticker"):
                    hist[ticker] = group.drop(columns=["ticker"]).set_index("datetime").sort_index()
            if hist:
                log.info("Got %d tickers for GP/F-family factor application (%dd)", len(hist), days)
                return hist
        except Exception as e:
            log.warning("Failed to load extended GP history (%dd): %s", days, e)
        return self._historical_data

    @staticmethod
    def _is_active_mined_factor(factor: dict) -> bool:
        """True only for tradeable mined expressions.

        F-family failure markers are persisted intentionally so an account with
        no admissible factor is not re-mined every hour. They are audit state,
        not alpha expressions, and must be ignored by compute/trading paths.
        """
        return bool(factor.get("expression")) and factor.get("active", True) is not False

    @classmethod
    def _active_mined_count(cls, factors: list[dict] | None) -> int:
        return sum(1 for f in (factors or []) if cls._is_active_mined_factor(f))

    @staticmethod
    def _factor_miner_failure_marker(gp_strat, reason: str, retry_after_hours: int = 24) -> dict:
        now = datetime.now(timezone.utc)
        return {
            "name": f"fm_{gp_strat.id}_NO_ADMISSIBLE_FACTOR",
            "active": False,
            "status": "mining_failed",
            "reason": reason,
            "backend": "factor_miner_gp",
            "family": "F",
            "source_account": gp_strat.id,
            "feature_cols": list(getattr(gp_strat, "gp_feature_subset", None) or []),
            "y_target": getattr(gp_strat, "gp_y_target", "next_1d_ret"),
            "strict_red_sea_threshold": getattr(gp_strat, "gp_global_corr_threshold", None),
            "created_at": now.isoformat(),
            "retry_after": (now + timedelta(hours=retry_after_hours)).isoformat(),
        }

    @staticmethod
    def _factor_miner_retry_due(factors: list[dict] | None) -> bool:
        markers = [f for f in (factors or []) if f.get("status") == "mining_failed"]
        if not markers:
            return False
        retry_after = max((m.get("retry_after") or "") for m in markers)
        if not retry_after:
            return False
        try:
            return datetime.now(timezone.utc) >= datetime.fromisoformat(retry_after)
        except Exception:
            return False

    def _missing_gp_strategies(self):
        """Return accounts that need a mining attempt.

        For FactorMiner F/CF accounts, a present key with an empty list or
        failure marker means the attempt was completed and logged. This preserves
        the memory/red-sea premise and prevents expensive hourly re-mining of
        known-failed accounts. Legacy B/CB keeps the old behavior: empty factor
        lists are treated as missing and can be retried.
        """
        missing = []
        for g in self.gp_strategies:
            factors = self._per_account_mined.get(g.id)
            if g.id not in self._per_account_mined:
                missing.append(g)
                continue
            if getattr(g, "mining_backend", "gplearn") == "factor_miner_gp":
                if self._factor_miner_retry_due(factors):
                    missing.append(g)
                continue
            if not self._active_mined_count(factors):
                missing.append(g)
        return missing

    def mine_gp_factors(self):
        """Run GP factor mining per account (slow, do once then cache).

        Mines factors only for accounts not yet present in the per-account JSON,
        so newly added strategies (e.g. B11+) get bootstrapped on next tick
        without re-mining the existing ones.
        """
        missing = self._missing_gp_strategies()
        if not missing:
            log.info("GP factors already mined or explicitly failed for all %d accounts, skipping mining.",
                     len(self._per_account_mined))
        else:
            log.info("Mining GP alpha factors for %d new account(s): %s (120-day history)...",
                     len(missing), [g.id for g in missing])
            try:
                raw_df = self.fetcher.get_historical(self.universe, days=120)
                mining_data = {}
                if not raw_df.empty and "ticker" in raw_df.columns:
                    for ticker, group in raw_df.groupby("ticker"):
                        df = group.drop(columns=["ticker"]).set_index("datetime").sort_index()
                        mining_data[ticker] = df
                log.info("Got %d tickers for GP mining (120-day)", len(mining_data))
            except Exception as e:
                log.error("Failed to fetch mining data: %s", e)
                mining_data = self._historical_data

            # Mine only the missing accounts; preserve existing factors in each
            # family store. B uses legacy gplearn; F uses FactorMiner-style GP.
            for gp_strat in missing:
                log.info("Mining GP factors for %s (%s)...", gp_strat.id, gp_strat.name)
                if getattr(gp_strat, "mining_backend", "gplearn") == "factor_miner_gp":
                    miner = FactorMinerGPBackend(
                        global_corr_threshold=getattr(gp_strat, "gp_global_corr_threshold", 0.6),
                        replacement_ic_mult=getattr(gp_strat, "gp_replacement_ic_mult", 1.3),
                    )
                    family_ids = {g.id for g in self.gp_strategies if getattr(g, "family", "B") == "F"}
                    factors = miner.mine_factors(
                        account_id=gp_strat.id,
                        historical_data=mining_data,
                        all_mined={**self._factor_miner_mined, **{k: v for k, v in self._per_account_mined.items() if k in family_ids}},
                        family_account_ids=family_ids,
                        n_factors=gp_strat.gp_n_factors,
                        generations=gp_strat.gp_generations,
                        base_seed=gp_strat.gp_seed,
                        population_size=gp_strat.gp_population,
                        n_runs=gp_strat.gp_n_runs,
                        parsimony_coefficient=gp_strat.gp_parsimony,
                        y_target=getattr(gp_strat, "gp_y_target", "next_1d_ret"),
                        feature_subset=getattr(gp_strat, "gp_feature_subset", None),
                        dedup_threshold=getattr(gp_strat, "gp_dedup_threshold", 0.6),
                    )
                    self._factor_miner_mined[gp_strat.id] = factors
                else:
                    miner = GPAlphaMiner()
                    factors = miner.mine_factors(
                        mining_data,
                        n_factors=gp_strat.gp_n_factors,
                        generations=gp_strat.gp_generations,
                        base_seed=gp_strat.gp_seed,
                        population_size=gp_strat.gp_population,
                        n_runs=gp_strat.gp_n_runs,
                        parsimony_coefficient=gp_strat.gp_parsimony,
                        y_target=getattr(gp_strat, "gp_y_target", "next_1d_ret"),
                        feature_subset=getattr(gp_strat, "gp_feature_subset", None),
                        dedup_threshold=getattr(gp_strat, "gp_dedup_threshold", 0.85),
                    )
                    self._legacy_gp_mined[gp_strat.id] = factors
                if not factors and getattr(gp_strat, "mining_backend", "gplearn") == "factor_miner_gp":
                    factors = [self._factor_miner_failure_marker(gp_strat, "all_candidates_rejected_or_no_valid_candidates")]
                    self._factor_miner_mined[gp_strat.id] = factors
                self._per_account_mined[gp_strat.id] = factors
                log.info("  %s: mined %d active factors", gp_strat.id, self._active_mined_count(factors))

            if self._per_account_mined:
                GPAlphaMiner.save_per_account_factors(self._legacy_gp_mined)
                FactorMinerGPBackend.save_factors(self._factor_miner_mined)
                self._gp_mining_done = True
                log.info("Saved per-account GP factors: legacy=%d factorminer=%d",
                         len(self._legacy_gp_mined), len(self._factor_miner_mined))
            else:
                log.warning("GP mining produced no valid factors")
                return

        # Compute GP factors for current data, per account. FactorMiner-style
        # terminals include nested 20-day windows, so use an application window
        # longer than the normal 30d A/Q refresh without changing mining logic.
        gp_history = self._load_gp_compute_history()
        self._per_account_gp_factors = {}
        for gp_strat in self.gp_strategies:
            mined = [f for f in self._per_account_mined.get(gp_strat.id, []) if self._is_active_mined_factor(f)]
            if not mined:
                self._per_account_gp_factors[gp_strat.id] = {}
                continue
            self._per_account_gp_factors[gp_strat.id] = self.gp_miner.compute_gp_factors(
                gp_history, mined
            )
        log.info("Computed per-account GP factors for %d accounts",
                 sum(1 for v in self._per_account_gp_factors.values() if v))

        # 持久化GP因子值到DB
        for acct_id, gp_factors in self._per_account_gp_factors.items():
            if gp_factors:
                try:
                    acct_cfg = next((g for g in self.gp_strategies if g.id == acct_id), None)
                    group_prefix = "fmgp" if acct_cfg and getattr(acct_cfg, "family", "B") == "F" else "gp"
                    self.store.save_factor_df(gp_factors, group=f"{group_prefix}_{acct_id}")
                except Exception as e:
                    log.warning("Failed to save GP factors for %s: %s", acct_id, e)

    # -- Benchmark buy-and-hold -----------------------------------------------
    def initialize_benchmarks(self):
        """Buy-and-hold: for each benchmark, if no position, buy with all cash."""
        current_prices = self._get_current_prices(
            fetch_missing_benchmarks=not getattr(self, "_fast_live_mode", False)
        )
        for bm in self.benchmarks:
            acct = self.engine.get_account(bm["id"])
            ticker = bm["ticker"]
            price = current_prices.get(ticker)
            if not price and not getattr(self, "_fast_live_mode", False):
                # Last-ditch: ask the fetcher for a fresh realtime quote.
                try:
                    q = self.fetcher.get_realtime_quotes([ticker])
                    if q.get(ticker):
                        price = float(q[ticker])
                except Exception:
                    pass
            if not price or price <= 0:
                log.warning("No price for benchmark %s (%s), skipping", bm["id"], ticker)
                continue
            held = acct.get_positions().get(ticker, 0)
            if held > 0:
                continue  # Already holding
            # Buy as many whole shares as possible (no position limit override)
            shares = int(acct.cash * 0.999 / price)  # leave tiny margin for fees
            if shares <= 0:
                continue
            result = self.engine.execute_trade(
                bm["id"], ticker, "buy", shares, price
            )
            if result:
                log.info("Benchmark %s: bought %d shares of %s @ $%.2f",
                         bm["id"], shares, ticker, price)

    # -- Step 3a: Trade original accounts -----------------------------------
    def run_gp_trading_cycle(self):
        """Trade GP accounts (B01-B10 / CB01-CB10) using per-account GP-mined factors."""
        # Defense-in-depth: every cycle entrypoint MUST gate on market hours.
        # run_once() already gates, but scripts/tests sometimes call cycles
        # directly — without this guard we get phantom off-hours trades.
        if not is_market_hours_for(self.market):
            log.info("[%s] Outside market hours — GP cycle skipped", self.market)
            return
        if not self._per_account_gp_factors:
            log.warning("No GP factors available, skipping GP trading")
            return

        log.info("Running GP trading cycle...")
        current_prices = self._get_current_prices(
            fetch_missing_benchmarks=not getattr(self, "_fast_live_mode", False)
        )
        retired = self._retired_accounts()

        for gp_strat in self.gp_strategies:
            if gp_strat.id in retired:
                continue  # retired account: no trading, no rebalance event
            if getattr(self, "_fast_live_mode", False):
                held = {
                    ticker for ticker, shares in
                    self.engine.get_account(gp_strat.id).get_positions().items()
                    if shares > 0
                }
                missing = sorted(held - set(current_prices))
                if missing:
                    log.warning(
                        "[%s] GP fast cycle skipped: stale/missing held quotes %s",
                        gp_strat.id, ",".join(missing[:8]),
                    )
                    continue
            try:
                if not self._should_rebalance(gp_strat.id, gp_strat.rebalance_hours):
                    continue
                executed = self._trade_gp_account(gp_strat, current_prices)
                if executed is False:
                    # Signal generator returned no actionable candidates — do NOT
                    # mark this cycle as a successful rebalance, so the next tick
                    # will retry instead of waiting another full adaptive window.
                    continue
                self._last_rebalance[gp_strat.id] = datetime.now(timezone.utc)
                emit_event(
                    "rebalance",
                    f"🔄 Rebalance {gp_strat.id} (GP, base {gp_strat.rebalance_hours}h)",
                    account=gp_strat.id,
                    market=self.market,
                )
            except TradePersistenceError:
                raise
            except Exception as e:
                log.error("Error trading GP %s: %s", gp_strat.id, e)
                traceback.print_exc()

    def _trade_gp_account(self, gp_strat, current_prices: dict):
        """Execute GP-based trading for one account."""
        acct = self.engine.get_account(gp_strat.id)
        equity = acct.get_equity(current_prices)

        # Get this account's factors/signals. Fast cycles already computed the
        # persisted ranking; use it directly so scoring cannot drift between the
        # bounded quote preparation and the locked trade phase.
        if getattr(self, "_fast_live_mode", False):
            signals = self._prepared_gp_signals.get(
                gp_strat.id, {"buy": [], "sell": []}
            )
            filtered_factors = self._per_account_gp_factors.get(gp_strat.id, {})
        else:
            account_factors = self._per_account_gp_factors.get(gp_strat.id, {})
            if not account_factors:
                return False
            filtered_factors = self._filter_gp_factors(gp_strat, account_factors)
            signals = self.gp_signal_gen.generate_signals(
                filtered_factors,
                top_n=gp_strat.top_n,
                buy_candidates=max(gp_strat.top_n * 5, 30),
            )

        buy_tickers = set(signals["buy"])
        sell_tickers = set(signals["sell"])

        # Empty-signal guard: if the signal generator could not produce ANY
        # buy candidates this cycle (NaN-filled factor row, insufficient rolling
        # history, data gap, etc.), skip the rebalance entirely instead of
        # liquidating every position. Log coverage so we can distinguish
        # "technical not computable" from "valid strategy chose no trade".
        if not buy_tickers:
            usable = 0
            nonempty = 0
            for fdf in filtered_factors.values():
                if fdf is None or fdf.empty:
                    continue
                if fdf.notna().any().any():
                    nonempty += 1
                for i in range(len(fdf) - 1, max(-1, len(fdf) - 6), -1):
                    if len(fdf.iloc[i].dropna()) > 0:
                        usable += 1
                        break
            active_factor_count = self._active_mined_count(self._per_account_mined.get(gp_strat.id, []))
            log.warning(
                "[%s] GP signals empty — skipping rebalance to preserve positions "
                "(active_factors=%d, factor_frames=%d, nonempty_frames=%d, usable_last5=%d, top_n=%d)",
                gp_strat.id, active_factor_count, len(filtered_factors), nonempty, usable, gp_strat.top_n,
            )
            return False

        # --- Churn controls (P0, 2026-06-01) ---
        cc = CHURN_CONTROLS if CHURN_CONTROLS.get("enabled", True) else {}
        HOLD_MULT = cc.get("hold_band_mult", 1) or 1
        # GP signal_gen.generate_signals returns ordered list under signals["buy"];
        # the leading top_n is the BUY band, the leading top_n*HOLD_MULT is the HOLD band.
        hold_tickers = set(signals["buy"][:gp_strat.top_n * HOLD_MULT])
        cooldown_set = get_stop_cooldown_set(
            self.db_path, gp_strat.id, self.market, cc.get("stop_cooldown_hours", 0),
        )

        # --- SELL: stop loss + not in buy list ---
        positions = acct.get_positions()
        held_set = {t for t, s in positions.items() if s > 0}
        min_hold_lock = get_min_hold_set(
            self.db_path, gp_strat.id, self.market,
            cc.get("min_hold_days", 0), held_tickers=held_set,
        )
        for ticker, shares in positions.items():
            if shares <= 0:
                continue
            price = current_prices.get(ticker)
            if not price:
                continue

            pos_data = acct._positions.get(ticker)
            if pos_data and pos_data.avg_cost > 0:
                pnl_pct = (price - pos_data.avg_cost) / pos_data.avg_cost
                if pnl_pct <= -gp_strat.stop_loss:
                    log.info("[%s] GP Stop loss %s: %.1f%%", gp_strat.id, ticker, pnl_pct * 100)
                    try:
                        self._execute_live_signal(gp_strat.id, ticker, "sell", shares, price, current_prices, reason="stop_loss")
                    except TradePersistenceError:
                        raise
                    except Exception as e:
                        log.warning("[%s] GP Sell failed %s: %s", gp_strat.id, ticker, e)
                    continue

            if ticker not in hold_tickers or ticker in sell_tickers:
                if ticker in min_hold_lock:
                    log.info("[%s] GP min_hold skip sell %s (held < %dd)",
                             gp_strat.id, ticker, cc.get("min_hold_days", 0))
                    continue
                try:
                    self._execute_live_signal(gp_strat.id, ticker, "sell", shares, price, current_prices)
                    log.info("[%s] GP Sold %s x%.0f @ $%.2f", gp_strat.id, ticker, shares, price)
                except TradePersistenceError:
                    raise
                except Exception as e:
                    log.warning("[%s] GP Sell failed %s: %s", gp_strat.id, ticker, e)

        # --- BUY ---
        equity = acct.get_equity(current_prices)
        target_per_position = equity * gp_strat.max_position_pct
        active_count = sum(1 for s in acct.get_positions().values() if s > 0)
        # Iterate the full ranked list, not just top_n. With CN 100-share lots,
        # a high-ranked expensive stock may be unbuyable under the position cap;
        # skip it and let the next executable signal fill the target slot.
        for rank, ticker in enumerate(signals["buy"], start=1):
            held = acct.get_positions().get(ticker, 0)
            if active_count >= gp_strat.top_n and held <= 0:
                continue
            if ticker in cooldown_set:
                log.info("[%s] GP cooldown skip %s (stop_loss within %dh)",
                         gp_strat.id, ticker, cc.get("stop_cooldown_hours", 0))
                continue
            price = current_prices.get(ticker)
            if not price or price <= 0:
                continue
            held_value = held * price
            if held_value >= target_per_position * 0.9:
                continue
            budget = min(target_per_position - held_value, acct.cash * 0.95)
            if budget < 5:
                continue
            shares = self.engine.buy_shares_for_budget(budget, price)
            if shares <= 0:
                if self.market == "CN":
                    lot_size = getattr(self.costs, "lot_size", 1)
                    log.info("[%s] GP lot skip %s: budget=¥%.2f price=¥%.2f lot=%s",
                             gp_strat.id, ticker, budget, price, lot_size)
                    self._record_lot_skip(
                        gp_strat.id, ticker, strategy_kind="GP", budget=budget,
                        price=price, target_per_position=target_per_position,
                        held_value=held_value, rank=rank,
                    )
                continue
            try:
                result = self._execute_live_signal(gp_strat.id, ticker, "buy", shares, price, current_prices)
                if result:
                    if held <= 0:
                        active_count += 1
                    log.info("[%s] GP Bought %s x%d @ $%.2f", gp_strat.id, ticker, result["shares"], price)
            except TradePersistenceError:
                raise
            except Exception as e:
                log.warning("[%s] GP Buy failed %s: %s", gp_strat.id, ticker, e)

        return True

    def _filter_gp_factors(self, gp_strat, gp_factors_dict: dict) -> dict:
        """Filter GP factors based on strategy's factor_selection and scoring_method."""
        mined_factors = self._per_account_mined.get(gp_strat.id, [])
        if not mined_factors or not gp_factors_dict:
            return gp_factors_dict
        mined_factors = [f for f in mined_factors if self._is_active_mined_factor(f)]
        if not mined_factors:
            return {}

        # Determine which factor columns to keep
        sorted_by_ic = sorted(mined_factors, key=lambda f: abs(f["ic"]), reverse=True)
        all_names = [f["name"] for f in sorted_by_ic]

        if gp_strat.factor_selection == "top5":
            keep = set(all_names[:5])
        elif gp_strat.factor_selection == "top10":
            keep = set(all_names[:10])
        elif gp_strat.factor_selection == "bottom5":
            keep = set(all_names[-5:]) if len(all_names) >= 5 else set(all_names)
        else:  # "all"
            keep = set(all_names)

        # Apply scoring method filter
        if gp_strat.scoring_method == "top3_only":
            keep = keep & set(all_names[:3]) if len(all_names) >= 3 else keep

        # Filter DataFrames. If none of this account's active factor columns are
        # present, return an empty DataFrame for that ticker instead of the
        # original unfiltered frame; otherwise a missing/failed account can trade
        # stale or wrong columns and appear healthy.
        filtered = {}
        for ticker, fdf in gp_factors_dict.items():
            if fdf is None or fdf.empty:
                filtered[ticker] = fdf
                continue
            cols = [c for c in fdf.columns if c in keep]
            filtered[ticker] = fdf[cols] if cols else fdf.iloc[:, 0:0]
        return filtered

    # -- Step 3c: Trade Qlib Q-accounts ---------------------------------------
    def run_qlib_trading_cycle(self):
        """Trade Qlib Q-accounts (Q01-Q10) using daily-retrained model scores
        stored in factor_values (factor_name='qlib_QXX_score').

        Each model gets its own latest available date's scores; we rank tickers
        and apply the same SELL-stop-loss / BUY-top-N pattern as A/B accounts.

        US-only RTH guard: Qlib scores are trained at 23:00 UTC against the
        previous trading day's close. Entering at pre-market (04:00-09:30 ET)
        crosses thin liquidity / wide spreads / overnight gaps that erode
        5-15bps of alpha per trade. We force Q-accounts to wait until the
        09:30 ET open where fills are tighter. CN keeps the existing CN RTH
        guard (`is_market_hours_cn()` already excludes pre-market for CN).
        """
        if not self.qlib_strategies:
            return  # CN or other markets where Q-accounts are disabled
        # Defense-in-depth market-hours guard (mirrors A/B cycles).
        if not is_market_hours_for(self.market):
            log.info("[%s] Outside market hours — Qlib cycle skipped", self.market)
            return
        if self.market == "US" and not is_us_rth():
            log.info("Outside US RTH (09:30-16:00 ET) — Qlib cycle skipped (avoids pre/post-market alpha leakage)")
            return
        log.info("Running Qlib trading cycle...")
        current_prices = self._get_current_prices(
            fetch_missing_benchmarks=not getattr(self, "_fast_live_mode", False)
        )
        retired = self._retired_accounts()

        for q_strat in self.qlib_strategies:
            if q_strat.id in retired:
                continue
            if getattr(self, "_fast_live_mode", False):
                held = {
                    ticker for ticker, shares in
                    self.engine.get_account(q_strat.id).get_positions().items()
                    if shares > 0
                }
                missing = sorted(held - set(current_prices))
                if missing:
                    log.warning(
                        "[%s] Qlib fast cycle skipped: stale/missing held quotes %s",
                        q_strat.id, ",".join(missing[:8]),
                    )
                    continue
            try:
                if not self._should_rebalance(q_strat.id, q_strat.rebalance_hours):
                    continue
                executed = self._trade_qlib_account(q_strat, current_prices)
                if executed is False:
                    continue
                self._last_rebalance[q_strat.id] = datetime.now(timezone.utc)
                emit_event(
                    "rebalance",
                    f"🔄 Rebalance {q_strat.id} (qlib/{q_strat.model_class}, base {q_strat.rebalance_hours}h)",
                    account=q_strat.id,
                    market=self.market,
                )
            except TradePersistenceError:
                raise
            except Exception as e:
                log.error("Error trading Qlib %s: %s", q_strat.id, e)
                traceback.print_exc()

    def _load_qlib_scores(self, q_id: str) -> list[tuple[str, float]]:
        """Read latest-date qlib scores for one Q-account from factor_values.
        Returns [(ticker, score)] sorted desc by score. Empty if no rows.

        Note: factor_name uses the *base* model id (Q01-Q10) — CN accounts
        (CQ01-CQ10) share the same model architectures but are trained on
        CN qlib data, so we strip the market prefix to find the row.
        """
        import sqlite3
        prefix = ACCOUNT_PREFIX.get(self.market, "")
        base_id = q_id[len(prefix):] if prefix and q_id.startswith(prefix) else q_id
        factor_name = f"qlib_{base_id}_score"
        # IMPORTANT: factor_name is shared across markets (e.g. qlib_Q01_score
        # is written by both US and CN retrains). To avoid CN writes shadowing
        # US's MAX(date) (or vice versa), restrict the date lookup to tickers
        # in the *current market's* universe before picking the latest date.
        try:
            conn = sqlite3.connect(self.db_path)
            uni_list = list(self.universe)
            if not uni_list:
                conn.close()
                return []
            placeholders = ",".join("?" * len(uni_list))
            row = conn.execute(
                f"SELECT MAX(date) FROM factor_values "
                f"WHERE factor_name=? AND ticker IN ({placeholders})",
                (factor_name, *uni_list),
            ).fetchone()
            latest = row[0] if row else None
            if not latest:
                conn.close()
                return []

            # Safety gate: never trade Q accounts on stale ML scores.  The date
            # check is relative to the latest available 1d price date for this
            # market (not wall-clock today) so weekends/holidays do not create
            # false positives.  If retrain/export stops advancing while prices
            # keep advancing, the account no-ops instead of silently reusing old
            # weights.  US gets a tighter window; CN allows an extra day for
            # exchange-calendar / data-provider lag.
            price_row = conn.execute(
                f"SELECT MAX(substr(datetime,1,10)) FROM prices "
                f"WHERE interval='1d' AND ticker IN ({placeholders})",
                tuple(uni_list),
            ).fetchone()
            latest_price_date = price_row[0] if price_row else None
            if latest_price_date:
                from datetime import date as _date
                score_date = _date.fromisoformat(str(latest)[:10])
                px_date = _date.fromisoformat(str(latest_price_date)[:10])
                lag_days = (px_date - score_date).days
                max_lag = 3 if self.market == "CN" else 2
                if lag_days > max_lag:
                    conn.close()
                    msg = (
                        f"[{q_id}] stale qlib scores: factor_name={factor_name} "
                        f"score_date={latest} latest_price_date={latest_price_date} "
                        f"lag_days={lag_days} max_lag={max_lag}; skipping Qlib trading"
                    )
                    log.error(msg)
                    emit_event(
                        "factor",
                        f"⚠️ Stale Qlib score skipped: {q_id}",
                        severity="error",
                        account=q_id,
                        detail={
                            "factor_name": factor_name,
                            "score_date": latest,
                            "latest_price_date": latest_price_date,
                            "lag_days": lag_days,
                            "max_lag": max_lag,
                        },
                        market=self.market,
                    )
                    return []

            rows = conn.execute(
                f"SELECT ticker, value FROM factor_values "
                f"WHERE factor_group='qlib' AND factor_name=? AND date=? AND value IS NOT NULL "
                f"AND ticker IN ({placeholders})",
                (factor_name, latest, *uni_list),
            ).fetchall()
            coverage = len({ticker for ticker, _value in rows}) / len(uni_list)
            if coverage < 0.80:
                conn.close()
                self._emit_persisted_signal_gate(
                    account_id=q_id,
                    factor_group="qlib",
                    reason="coverage",
                    factor_date=latest,
                    latest_price_date=latest_price_date,
                    covered_tickers=len({ticker for ticker, _value in rows}),
                    universe_size=len(uni_list),
                    coverage=coverage,
                    required_coverage=0.80,
                    lag_days=(
                        (
                            datetime.fromisoformat(str(latest_price_date)[:10]).date()
                            - datetime.fromisoformat(str(latest)[:10]).date()
                        ).days
                        if latest_price_date else None
                    ),
                    max_lag_days=3 if self.market == "CN" else 2,
                )
                return []
            conn.close()
        except Exception as e:
            log.warning("[%s] Failed to load qlib scores: %s", q_id, e)
            return []
        # Restrict to current universe and sort desc
        uni = set(self.universe)
        scored = [(t, float(v)) for t, v in rows if t in uni]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _trade_qlib_account(self, q_strat, current_prices: dict):
        """Execute one Q-account's trading using model scores from DB."""
        acct = self.engine.get_account(q_strat.id)
        if getattr(self, "_fast_live_mode", False):
            scored = self._prepared_qlib_scores.get(q_strat.id, [])
        else:
            scored = self._load_qlib_scores(q_strat.id)
        if not scored:
            log.warning("[%s] no qlib scores found in factor_values, skipping", q_strat.id)
            return False

        buy_list = [t for t, _ in scored[:q_strat.top_n]]
        buy_set = set(buy_list)

        # --- Churn controls (P0, 2026-06-01) ---
        cc = CHURN_CONTROLS if CHURN_CONTROLS.get("enabled", True) else {}
        HOLD_MULT = cc.get("hold_band_mult", 1) or 1
        hold_set = {t for t, _ in scored[:q_strat.top_n * HOLD_MULT]}
        cooldown_set = get_stop_cooldown_set(
            self.db_path, q_strat.id, self.market, cc.get("stop_cooldown_hours", 0),
        )

        # --- SELL: stop loss + not in target portfolio ---
        positions = acct.get_positions()
        held_set = {t for t, s in positions.items() if s > 0}
        min_hold_lock = get_min_hold_set(
            self.db_path, q_strat.id, self.market,
            cc.get("min_hold_days", 0), held_tickers=held_set,
        )
        for ticker, shares in positions.items():
            if shares <= 0:
                continue
            price = current_prices.get(ticker)
            if not price:
                continue
            pos_data = acct._positions.get(ticker)
            if pos_data and pos_data.avg_cost > 0:
                pnl_pct = (price - pos_data.avg_cost) / pos_data.avg_cost
                if pnl_pct <= -q_strat.stop_loss:
                    log.info("[%s] Qlib stop loss %s: %.1f%%", q_strat.id, ticker, pnl_pct * 100)
                    try:
                        self._execute_live_signal(
                            q_strat.id, ticker, "sell", shares, price, current_prices,
                            reason="stop_loss",
                        )
                    except TradePersistenceError:
                        raise
                    except Exception as e:
                        log.warning("[%s] Qlib sell failed %s: %s", q_strat.id, ticker, e)
                    continue
            if ticker not in hold_set:
                if ticker in min_hold_lock:
                    log.info("[%s] Qlib min_hold skip sell %s (held < %dd)",
                             q_strat.id, ticker, cc.get("min_hold_days", 0))
                    continue
                try:
                    self._execute_live_signal(
                        q_strat.id, ticker, "sell", shares, price, current_prices,
                    )
                    log.info("[%s] Qlib sold %s x%.0f @ $%.2f", q_strat.id, ticker, shares, price)
                except TradePersistenceError:
                    raise
                except Exception as e:
                    log.warning("[%s] Qlib sell failed %s: %s", q_strat.id, ticker, e)

        # --- BUY ---
        equity = acct.get_equity(current_prices)
        target_per_position = equity * q_strat.max_position_pct
        active_count = sum(1 for s in acct.get_positions().values() if s > 0)
        # Iterate full score ranking so expensive/unbuyable CN board lots do not
        # leave target slots empty; they are skipped and lower-ranked executable
        # candidates can fill the portfolio.
        for rank, (ticker, _score) in enumerate(scored, start=1):
            held = acct.get_positions().get(ticker, 0)
            if active_count >= q_strat.top_n and held <= 0:
                continue
            if ticker in cooldown_set:
                log.info("[%s] Qlib cooldown skip %s (stop_loss within %dh)",
                         q_strat.id, ticker, cc.get("stop_cooldown_hours", 0))
                continue
            price = current_prices.get(ticker)
            if not price or price <= 0:
                continue
            held_value = held * price
            if held_value >= target_per_position * 0.9:
                continue
            budget = min(target_per_position - held_value, acct.cash * 0.95)
            if budget < 5:
                continue
            shares = self.engine.buy_shares_for_budget(budget, price)
            if shares <= 0:
                if self.market == "CN":
                    lot_size = getattr(self.costs, "lot_size", 1)
                    log.info("[%s] Qlib lot skip %s: budget=¥%.2f price=¥%.2f lot=%s",
                             q_strat.id, ticker, budget, price, lot_size)
                    self._record_lot_skip(
                        q_strat.id, ticker, strategy_kind="Qlib", budget=budget,
                        price=price, target_per_position=target_per_position,
                        held_value=held_value, rank=rank, score=_score,
                    )
                continue
            try:
                result = self._execute_live_signal(
                    q_strat.id, ticker, "buy", shares, price, current_prices,
                )
                if result:
                    if held <= 0:
                        active_count += 1
                    log.info("[%s] Qlib bought %s x%d @ $%.2f", q_strat.id, ticker, result["shares"], price)
            except TradePersistenceError:
                raise
            except Exception as e:
                log.warning("[%s] Qlib buy failed %s: %s", q_strat.id, ticker, e)
        return True

    # -- Adaptive rebalance helpers -----------------------------------------
    def _compute_adaptive_hours(self, base_hours: float) -> float:
        """根据近期市场波动率计算自适应换仓周期（小时）。"""
        cfg = ADAPTIVE_REBALANCE
        vol_window = cfg.get("vol_window", 5)
        min_h = cfg.get("min_hours", 12)
        max_h = cfg.get("max_hours", 96)
        silence_thresh = cfg.get("silence_threshold", 0.3)

        if len(self._market_returns) < max(2, vol_window):
            return base_hours

        recent = self._market_returns[-vol_window:]
        sigma_t = float(np.std(recent)) if len(recent) >= 2 else self._sigma_ref

        if sigma_t < 1e-10:
            return float("inf")
        if sigma_t < self._sigma_ref * silence_thresh:
            return float("inf")  # 静默区

        ratio = self._sigma_ref / sigma_t
        hours = base_hours * ratio
        return max(min_h, min(max_h, hours))

    def _should_rebalance(self, account_id: str, base_hours: float) -> bool:
        """判断该账户是否应该换仓。非自适应模式直接返回True。"""
        if not self._adaptive_enabled:
            return True

        now = datetime.now(timezone.utc)
        last = self._last_rebalance.get(account_id)
        if last is None:
            return True  # 首次交易

        adaptive_hours = self._compute_adaptive_hours(base_hours)
        self._rebalance_hours_cache[account_id] = adaptive_hours

        if adaptive_hours == float("inf"):
            log.debug("[%s] Adaptive: silence zone, skip rebalance", account_id)
            return False

        elapsed = (now - last).total_seconds() / 3600.0
        if elapsed >= adaptive_hours:
            log.info("[%s] Adaptive rebalance: %.1fh elapsed >= %.1fh threshold",
                     account_id, elapsed, adaptive_hours)
            return True
        return False

    def _update_market_returns(self):
        """从历史数据计算市场平均日收益率，更新sigma_ref。"""
        all_returns = []
        for ticker, df in self._historical_data.items():
            if df is not None and len(df) >= 2:
                rets = df["close"].pct_change().dropna().tolist()
                all_returns.append(rets)

        if not all_returns:
            return

        # 每天取所有股票的平均收益
        max_len = max(len(r) for r in all_returns)
        daily_avg = []
        for i in range(max_len):
            vals = [r[i] for r in all_returns if i < len(r)]
            if vals:
                daily_avg.append(float(np.mean(vals)))

        self._market_returns = daily_avg

        # sigma_ref = 60日中位数波动率（滚动5日窗口标准差的中位数）
        if len(daily_avg) >= 10:
            vol_window = ADAPTIVE_REBALANCE.get("vol_window", 5)
            rolling_vols = []
            for i in range(vol_window, len(daily_avg)):
                window = daily_avg[i - vol_window:i]
                rolling_vols.append(float(np.std(window)))
            if rolling_vols:
                self._sigma_ref = float(np.median(rolling_vols))
                log.info("Adaptive: sigma_ref=%.6f, %d market returns",
                         self._sigma_ref, len(daily_avg))

        # 持久化市场收益率
        if daily_avg and self._historical_data:
            try:
                # 从任一ticker获取日期索引
                sample_ticker = next(iter(self._historical_data))
                sample_df = self._historical_data[sample_ticker]
                if sample_df is not None and len(sample_df) >= 2:
                    dates = sample_df.index[-len(daily_avg):]
                    pairs = [(str(d)[:10], r) for d, r in zip(dates, daily_avg[-len(dates):])]
                    self.store.save_market_returns(pairs)
            except Exception as e:
                log.warning("Failed to save market returns: %s", e)

    # -- Step 3: Generate signals and trade ---------------------------------
    def run_trading_cycle(self):
        """For each account/strategy, generate signals and execute trades."""
        # Defense-in-depth market-hours guard (see run_gp_trading_cycle).
        if not is_market_hours_for(self.market):
            log.info("[%s] Outside market hours — A-cycle skipped", self.market)
            return
        log.info("Running trading cycle...")

        current_prices = self._get_current_prices(
            fetch_missing_benchmarks=not getattr(self, "_fast_live_mode", False)
        )
        retired = self._retired_accounts()

        for strat in self.strategies:
            if strat.id in retired:
                continue
            if getattr(self, "_fast_live_mode", False):
                held = {
                    ticker for ticker, shares in
                    self.engine.get_account(strat.id).get_positions().items()
                    if shares > 0
                }
                missing = sorted(held - set(current_prices))
                if missing:
                    log.warning(
                        "[%s] Alpha fast cycle skipped: stale/missing held quotes %s",
                        strat.id, ",".join(missing[:8]),
                    )
                    continue
            try:
                if not self._should_rebalance(strat.id, strat.rebalance_hours):
                    continue
                executed = self._trade_account(strat, current_prices)
                if executed is False:
                    continue
                self._last_rebalance[strat.id] = datetime.now(timezone.utc)
                emit_event(
                    "rebalance",
                    f"🔄 Rebalance {strat.id} ({strat.strategy_type}, base {strat.rebalance_hours}h)",
                    account=strat.id,
                    market=self.market,
                )
            except TradePersistenceError:
                raise
            except Exception as e:
                log.error("Error trading %s: %s", strat.id, e)
                traceback.print_exc()

    def _trade_account(self, strat, current_prices: dict):
        """Execute trading logic for one account."""
        acct = self.engine.get_account(strat.id)
        equity = acct.get_equity(current_prices)

        # Generate signals.  Fast cycles prepare account-specific persisted
        # scores up front; an empty prepared list is a technical no-op.
        prepared = getattr(self, "_prepared_alpha_signals", {})
        if getattr(self, "_fast_live_mode", False):
            signals = prepared.get(strat.id, {"buy": [], "sell": []})
        else:
            signals = self.signal_gen.generate_signals(
                self._factors_dict, strat.strategy_type
            )
        if not signals.get("buy"):
            log.warning(
                "[%s] Alpha signals empty — skipping rebalance to preserve positions",
                strat.id,
            )
            return False

        # --- Churn controls (P0, 2026-06-01): hysteresis + cooldown + min_hold ---
        # See trading/churn_controls.py and CHURN_CONTROLS in config/settings.py.
        # Defaults: hold_band_mult=3, stop_cooldown_hours=24, min_hold_days=2.
        # All three default-on; each individually disable-able via settings.
        cc = CHURN_CONTROLS if CHURN_CONTROLS.get("enabled", True) else {}
        HOLD_MULT = cc.get("hold_band_mult", 1) or 1
        buy_tickers = {t for t, _ in signals["buy"][:strat.top_n]}
        hold_tickers = {t for t, _ in signals["buy"][:strat.top_n * HOLD_MULT]}

        # --- SELL: close positions not in hold band ---
        positions = acct.get_positions()
        held_set = {t for t, s in positions.items() if s > 0}
        min_hold_lock = get_min_hold_set(
            self.db_path, strat.id, self.market,
            cc.get("min_hold_days", 0), held_tickers=held_set,
        )
        for ticker, shares in positions.items():
            if shares <= 0:
                continue
            price = current_prices.get(ticker)
            if not price:
                continue

            # Stop loss check (always runs first, never suppressed by min_hold)
            pos_data = acct._positions.get(ticker)
            if pos_data and pos_data.avg_cost > 0:
                pnl_pct = (price - pos_data.avg_cost) / pos_data.avg_cost
                if pnl_pct <= strat.stop_loss * -1:
                    log.info("[%s] Stop loss %s: %.1f%%", strat.id, ticker, pnl_pct * 100)
                    try:
                        self._execute_live_signal(
                            strat.id, ticker, "sell", shares, price, current_prices,
                            reason="stop_loss",
                        )
                    except TradePersistenceError:
                        raise
                    except Exception as e:
                        log.warning("[%s] Sell failed %s: %s", strat.id, ticker, e)
                    continue

            # Not in target portfolio (hold band) -> sell, unless still in min-hold lock
            if ticker not in hold_tickers:
                if ticker in min_hold_lock:
                    log.info("[%s] min_hold skip sell %s (held < %dd)",
                             strat.id, ticker, cc.get("min_hold_days", 0))
                    continue
                try:
                    self._execute_live_signal(
                        strat.id, ticker, "sell", shares, price, current_prices
                    )
                    log.info("[%s] Sold %s x%.0f @ $%.2f", strat.id, ticker, shares, price)
                except TradePersistenceError:
                    raise
                except Exception as e:
                    log.warning("[%s] Sell failed %s: %s", strat.id, ticker, e)

        # --- BUY: ranked signals, with executable-candidate fill ---
        equity = acct.get_equity(current_prices)
        target_per_position = equity * strat.max_position_pct

        cooldown_set = get_stop_cooldown_set(
            self.db_path, strat.id, self.market, cc.get("stop_cooldown_hours", 0),
        )
        STOP_COOLDOWN_HOURS = cc.get("stop_cooldown_hours", 0)
        active_count = sum(1 for s in acct.get_positions().values() if s > 0)

        # Iterate the full ranked signal list. For CN, a top-ranked stock can be
        # too expensive to buy one 100-share lot within the position cap; skip it
        # and let lower-ranked executable candidates fill up to top_n holdings.
        for rank, (ticker, score) in enumerate(signals["buy"], start=1):
            held = acct.get_positions().get(ticker, 0)
            if active_count >= strat.top_n and held <= 0:
                continue
            if ticker in cooldown_set:
                log.info("[%s] cooldown skip %s (stop_loss within %dh)",
                         strat.id, ticker, STOP_COOLDOWN_HOURS)
                continue
            price = current_prices.get(ticker)
            if not price or price <= 0:
                continue

            # Skip if already holding enough
            held_value = held * price
            if held_value >= target_per_position * 0.9:
                continue

            # Calculate executable shares to buy
            budget = min(target_per_position - held_value, acct.cash * 0.95)
            if budget < 5:  # minimum $5 trade
                continue
            shares = self.engine.buy_shares_for_budget(budget, price)
            if shares <= 0:
                if self.market == "CN":
                    lot_size = getattr(self.costs, "lot_size", 1)
                    log.info("[%s] lot skip %s: budget=¥%.2f price=¥%.2f lot=%s",
                             strat.id, ticker, budget, price, lot_size)
                    self._record_lot_skip(
                        strat.id, ticker, strategy_kind="Alpha158", budget=budget,
                        price=price, target_per_position=target_per_position,
                        held_value=held_value, rank=rank, score=score,
                    )
                continue

            try:
                result = self._execute_live_signal(
                    strat.id, ticker, "buy", shares, price, current_prices
                )
                if result:
                    if held <= 0:
                        active_count += 1
                    log.info("[%s] Bought %s x%d @ $%.2f (score=%.3f)",
                             strat.id, ticker, result["shares"], price, score)
            except TradePersistenceError:
                raise
            except Exception as e:
                log.warning("[%s] Buy failed %s: %s", strat.id, ticker, e)

    # -- Step 4: Generate and send report -----------------------------------
    def send_hourly_report(self):
        """Build and send the hourly Telegram report."""
        log.info("Generating hourly report...")

        current_prices = self._get_current_prices()

        # Guard: if no historical/realtime price data was loaded this run,
        # equity would fall back to avg_cost and snap to ~initial_cash,
        # corrupting the equity curve. Refuse to snapshot in that case.
        if not current_prices or not self._historical_data:
            log.error(
                "send_hourly_report aborted: no price data available "
                "(historical=%d, realtime=%d). Run fetch_data() first.",
                len(self._historical_data), len(self._realtime_prices),
            )
            return

        accounts_data = []
        retired = self._retired_accounts()
        # Strategy accounts (A01-A10 / CA01-CA10)
        for strat in self.strategies:
            if strat.id in retired:
                continue
            acct = self.engine.get_account(strat.id)
            equity = acct.get_equity(current_prices)
            pnl = equity - acct.initial_cash
            pnl_pct = (pnl / acct.initial_cash) * 100

            positions_list = []
            for ticker, shares in acct.get_positions().items():
                if shares <= 0:
                    continue
                price = current_prices.get(ticker, 0)
                pos_data = acct._positions.get(ticker)
                cost = pos_data.avg_cost if pos_data else price
                pos_pnl = (price - cost) * shares
                allocation = (shares * price / equity * 100) if equity > 0 else 0
                positions_list.append({
                    "ticker": ticker, "shares": shares, "price": price,
                    "avg_cost": cost, "pnl": pos_pnl, "allocation_pct": allocation,
                })

            accounts_data.append({
                "account_id": strat.id, "strategy": strat.name,
                "factors": strat.factor_names,
                "cash": acct.cash, "equity": equity,
                "pnl": pnl, "pnl_pct": pnl_pct,
                "positions": positions_list,
                "num_trades": len(acct.trade_log),
                "group": "Alpha158",
            })

        # GP accounts (B01-B10 / CB01-CB10)
        for gp_strat in self.gp_strategies:
            if gp_strat.id in retired:
                continue
            acct = self.engine.get_account(gp_strat.id)
            equity = acct.get_equity(current_prices)
            pnl = equity - acct.initial_cash
            pnl_pct = (pnl / acct.initial_cash) * 100

            positions_list = []
            for ticker, shares in acct.get_positions().items():
                if shares <= 0:
                    continue
                price = current_prices.get(ticker, 0)
                pos_data = acct._positions.get(ticker)
                cost = pos_data.avg_cost if pos_data else price
                pos_pnl = (price - cost) * shares
                allocation = (shares * price / equity * 100) if equity > 0 else 0
                positions_list.append({
                    "ticker": ticker, "shares": shares, "price": price,
                    "avg_cost": cost, "pnl": pos_pnl, "allocation_pct": allocation,
                })

            accounts_data.append({
                "account_id": gp_strat.id, "strategy": gp_strat.name,
                "factors": [gp_strat.factor_selection, gp_strat.scoring_method],
                "cash": acct.cash, "equity": equity,
                "pnl": pnl, "pnl_pct": pnl_pct,
                "positions": positions_list,
                "num_trades": len(acct.trade_log),
                "group": "FactorMiner GP" if getattr(gp_strat, "family", "B") == "F" else "GP进化",
            })

        # Qlib Q-accounts (Q01-Q10 / CQ01-CQ10)
        for q_strat in self.qlib_strategies:
            acct = self.engine.get_account(q_strat.id)
            equity = acct.get_equity(current_prices)
            pnl = equity - acct.initial_cash
            pnl_pct = (pnl / acct.initial_cash) * 100

            positions_list = []
            for ticker, shares in acct.get_positions().items():
                if shares <= 0:
                    continue
                price = current_prices.get(ticker, 0)
                pos_data = acct._positions.get(ticker)
                cost = pos_data.avg_cost if pos_data else price
                pos_pnl = (price - cost) * shares
                allocation = (shares * price / equity * 100) if equity > 0 else 0
                positions_list.append({
                    "ticker": ticker, "shares": shares, "price": price,
                    "avg_cost": cost, "pnl": pos_pnl, "allocation_pct": allocation,
                })

            accounts_data.append({
                "account_id": q_strat.id, "strategy": q_strat.name,
                "factors": [q_strat.model_class],
                "cash": acct.cash, "equity": equity,
                "pnl": pnl, "pnl_pct": pnl_pct,
                "positions": positions_list,
                "num_trades": len(acct.trade_log),
                "group": "Qlib模型",
            })

        # Benchmark IDX accounts
        for bm in self.benchmarks:
            acct = self.engine.get_account(bm["id"])
            equity = acct.get_equity(current_prices)
            pnl = equity - acct.initial_cash
            pnl_pct = (pnl / acct.initial_cash) * 100

            positions_list = []
            for ticker, shares in acct.get_positions().items():
                if shares <= 0:
                    continue
                price = current_prices.get(ticker, 0)
                pos_data = acct._positions.get(ticker)
                cost = pos_data.avg_cost if pos_data else price
                pos_pnl = (price - cost) * shares
                allocation = (shares * price / equity * 100) if equity > 0 else 0
                positions_list.append({
                    "ticker": ticker, "shares": shares, "price": price,
                    "avg_cost": cost, "pnl": pos_pnl, "allocation_pct": allocation,
                })

            accounts_data.append({
                "account_id": bm["id"], "strategy": bm["name"],
                "factors": [bm["ticker"]],
                "cash": acct.cash, "equity": equity,
                "pnl": pnl, "pnl_pct": pnl_pct,
                "positions": positions_list,
                "num_trades": len(acct.trade_log),
                "group": "基准指数",
            })

        # Save equity snapshots for all accounts — verify each one first.
        # If any held ticker has no price, skip that account's snapshot to
        # avoid corrupting the equity curve with an avg_cost fallback.
        for ad in accounts_data:
            acct = self.engine.get_account(ad["account_id"])
            try:
                # Strict recomputation catches silent avg_cost fallbacks.
                verified_equity = acct.get_equity(current_prices, strict=True)
            except ValueError as e:
                log.error(
                    "Skipping snapshot for %s: %s",
                    ad["account_id"], e,
                )
                continue
            self.store.save_snapshot(ad["account_id"], ad["cash"], verified_equity,
                                      market=self.market)

        # Sort by PnL descending
        accounts_data.sort(key=lambda a: a["pnl"], reverse=True)

        # Build report text
        now = datetime.now(timezone.utc)
        # Overall distribution header — independent $10k accounts → show dispersion, not sum.
        all_pcts = sorted(a["pnl_pct"] for a in accounts_data)
        def _q_all(xs, q):
            if not xs: return 0.0
            k = (len(xs) - 1) * q
            f = int(k); c = min(f + 1, len(xs) - 1)
            return xs[f] if f == c else xs[f] + (xs[c] - xs[f]) * (k - f)
        total_n = len(all_pcts)
        if total_n:
            med = _q_all(all_pcts, 0.5)
            q1 = _q_all(all_pcts, 0.25)
            q3 = _q_all(all_pcts, 0.75)
            win_n = sum(1 for p in all_pcts if p > 0)
            best_acc = max(accounts_data, key=lambda a: a["pnl_pct"])
            worst_acc = min(accounts_data, key=lambda a: a["pnl_pct"])
        lines = [
            f"📊 量化系统报告 {now.strftime('%Y-%m-%d %H:%M UTC')}",
            "=" * 40,
        ]
        # Risk regime line — ALWAYS shown so a forgotten manual override or
        # auto-armed trailing stop never hides from view.
        try:
            from trading.risk_regime import status_line as _risk_status
            lines.append(_risk_status(market=self.market, db_path=self.db_path))
        except Exception as _e:
            log.warning("risk_regime status_line unavailable: %s", _e)
        if total_n:
            lines.append(
                f"账户收益分布 (n={total_n}): 中位{med:+.2f}%  "
                f"IQR[{q1:+.2f}%, {q3:+.2f}%]  胜率{win_n}/{total_n}"
            )
            lines.append(
                f"最佳 {best_acc['account_id']} {best_acc['pnl_pct']:+.2f}%  |  "
                f"最差 {worst_acc['account_id']} {worst_acc['pnl_pct']:+.2f}%"
            )

        # Group by type
        for group_name in ["Alpha158", "GP进化", "FactorMiner GP", "Qlib模型", "基准指数"]:
            group = [a for a in accounts_data if a.get("group") == group_name]
            if not group:
                continue
            group.sort(key=lambda a: a["pnl"], reverse=True)
            # Distribution stats instead of meaningless aggregate PnL sum
            # (accounts are independent $10k starts; summing them hides dispersion).
            pcts = sorted(a["pnl_pct"] for a in group)
            n = len(pcts)
            def _q(xs, q):
                if not xs: return 0.0
                k = (len(xs) - 1) * q
                f = int(k); c = min(f + 1, len(xs) - 1)
                return xs[f] if f == c else xs[f] + (xs[c] - xs[f]) * (k - f)
            median_pct = _q(pcts, 0.5)
            best = max(group, key=lambda a: a["pnl_pct"])
            worst = min(group, key=lambda a: a["pnl_pct"])
            win_n = sum(1 for p in pcts if p > 0)
            g_emoji = "🧬" if group_name == "GP进化" else "🧪" if group_name == "FactorMiner GP" else "📐" if group_name == "Alpha158" else "🤖" if group_name == "Qlib模型" else "📊"
            lines.append(
                f"\n{g_emoji} {group_name} 组 (n={n}, 中位{median_pct:+.2f}%, "
                f"胜率{win_n}/{n}, 最佳{best['account_id']}{best['pnl_pct']:+.2f}% / "
                f"最差{worst['account_id']}{worst['pnl_pct']:+.2f}%)"
            )
            lines.append("-" * 38)

            # Show GP mined factor expressions under GP group headers
            if group_name in {"GP进化", "FactorMiner GP"}:
                family = "F" if group_name == "FactorMiner GP" else "B"
                total_factors = sum(
                    len(self._per_account_mined.get(gs.id, []))
                    for gs in self.gp_strategies
                    if getattr(gs, "family", "B") == family
                )
                label = "FactorMiner记忆GP" if family == "F" else "每账户独立GP进化"
                lines.append(f"🔬 独立挖掘因子({label}, 共{total_factors}个)")
                lines.append("")

            for i, acct_d in enumerate(group, 1):
                emoji = "🟢" if acct_d["pnl"] >= 0 else "🔴"
                lines.append(f"{emoji} {acct_d['strategy']} ({acct_d['account_id']})")
                # Show factors used by this account with formulas
                if group_name == "Alpha158":
                    factor_list = acct_d["factors"] if acct_d["factors"] else list(FACTOR_FORMULAS.keys())[:5]
                    for fn in factor_list:
                        formula = FACTOR_FORMULAS.get(fn, "?")
                        lines.append(f"  📈{fn} = {formula}")
                elif group_name in {"GP进化", "FactorMiner GP"}:
                    sel = acct_d["factors"][0]  # factor_selection
                    scr = acct_d["factors"][1]  # scoring_method
                    sel_desc = {"all": "全部因子", "top5": "IC前5因子", "top10": "IC前10因子", "bottom5": "IC后5因子(反向)"}.get(sel, sel)
                    scr_desc = {"equal_weight": "等权评分", "ic_weighted": "IC加权评分", "top3_only": "仅取IC前3因子评分"}.get(scr, scr)
                    lines.append(f"  📈策略: {sel_desc}, {scr_desc}")
                    # Show this account's own mined factors
                    acct_mined = self._per_account_mined.get(acct_d["account_id"], [])
                    if acct_mined:
                        lines.append(f"  🧬 挖掘因子({len(acct_mined)}个):")
                        for mf in acct_mined[:5]:  # show top 5
                            math_expr = gp_expr_to_math(mf["expression"])
                            lines.append(f"    {mf['name']}(IC={mf['ic']:.4f}): {math_expr}")
                        if len(acct_mined) > 5:
                            lines.append(f"    ...及{len(acct_mined)-5}个更多因子")
                elif group_name == "基准指数":
                    lines.append(f"  📈 Buy & Hold {acct_d['factors'][0]}")
                lines.append(f"  💰${acct_d['equity']:.2f} 盈亏{acct_d['pnl']:+.2f}({acct_d['pnl_pct']:+.1f}%) 交易{acct_d['num_trades']}")
                if acct_d["positions"]:
                    for pos in sorted(acct_d["positions"], key=lambda p: abs(p["pnl"]), reverse=True)[:3]:
                        pe = "📗" if pos["pnl"] >= 0 else "📕"
                        lines.append(f"  {pe}{pos['ticker']} {pos['shares']:.0f}股 盈亏{pos['pnl']:+.2f}")
                else:
                    lines.append("  空仓")

        report_text = "\n".join(lines)

        # Generate chart
        try:
            chart_path = self.report_gen.generate_hourly_report(accounts_data)[1]
        except Exception as e:
            log.warning("Chart generation failed: %s", e)
            chart_path = None

        # Send via Telegram
        try:
            if self.reporter:
                self.reporter.send_report(report_text, chart_path, force=True)
            log.info("Report sent to Telegram")
        except Exception as e:
            log.error("Failed to send Telegram report: %s", e)

        self._last_report = datetime.now(timezone.utc)

        # 持久化所有账户状态
        try:
            self._save_all_state()
        except Exception as e:
            log.warning("Failed to save state: %s", e)
        return report_text

    # -- Main loop ----------------------------------------------------------
    def run_once(self, send_report: bool = True):
        """Run a single cycle: fetch -> factors -> trade -> (optional) report.

        When send_report=False, skips the Telegram report + chart generation.
        Used by the every-15-minute cron slot so we get more frequent rebalance
        checks without spamming the chat. The top-of-hour slot still sends.

        Market-hours guard: simulation must mirror real trading. Outside US
        extended hours (04:00-20:00 ET, Mon-Fri) we refuse to trade. We still
        refresh data + emit a report (so morning reports show overnight state),
        but no rebalance / position changes occur.
        """
        in_market = is_market_hours_for(self.market)
        if not in_market:
            log.info("Outside %s market window — skipping trading cycle (report-only mode)", self.market)
            # Refresh quotes so the report reflects the latest available close,
            # but DO NOT touch positions / cash / rebalance.
            try:
                self.fetch_data()
            except Exception as e:
                log.warning("fetch_data failed in report-only mode: %s", e)
            try:
                self._realtime_prices = self._fetch_realtime_prices()
            except Exception as e:
                log.warning("realtime price fetch failed in report-only mode: %s", e)
            if send_report:
                return self.send_hourly_report()
            return

        self.fetch_data()
        self.compute_factors()
        self.mine_gp_factors()
        # Metadata is mandatory for every price that can reach execution. This
        # also prevents _get_current_prices() from silently trading a historical
        # close when a live quote is missing.
        self._realtime_prices = self._fetch_realtime_prices(strict_execution=True)
        self.initialize_benchmarks()
        self.run_trading_cycle()
        self.run_gp_trading_cycle()
        self.run_qlib_trading_cycle()
        if send_report:
            return self.send_hourly_report()
        # Still persist state even when not reporting
        try:
            self._save_all_state()
        except Exception as e:
            log.warning("Failed to save state: %s", e)
        return None

    def run_loop(self):
        """Run continuously during market hours, reporting every hour."""
        log.info("Starting main loop...")
        self.fetch_data()
        self.compute_factors()
        self.mine_gp_factors()

        while True:
            try:
                if is_market_hours_for(self.market):
                    # Refresh data every 5 minutes
                    if (self._last_data_fetch is None or
                        (datetime.now(timezone.utc) - self._last_data_fetch).seconds > 300):
                        self.fetch_data()
                        self.compute_factors()
                        self.mine_gp_factors()

                    # Every loop execution uses the same strict metadata gate as
                    # run_once; scalar or historical fallbacks are display-only.
                    self._realtime_prices = self._fetch_realtime_prices(
                        strict_execution=True
                    )

                    # 更新持仓市价 + 写入positions_history快照
                    self._save_all_state()

                    # Trade
                    self.initialize_benchmarks()
                    self.run_trading_cycle()
                    self.run_gp_trading_cycle()
                    self.run_qlib_trading_cycle()

                    # Report every hour
                    if (self._last_report is None or
                        (datetime.now(timezone.utc) - self._last_report).seconds > 3600):
                        self.send_hourly_report()

                    time.sleep(300)  # check every 5 minutes
                else:
                    log.info("Outside market hours, sleeping 30 minutes...")
                    time.sleep(1800)

            except KeyboardInterrupt:
                log.info("Shutting down...")
                break
            except Exception as e:
                log.error("Error in main loop: %s", e)
                traceback.print_exc()
                time.sleep(60)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def _run_for_market(market: str, mode: str, *, dry_run: bool = False):
    """Instantiate a QuantSystem for `market` and dispatch to the requested mode."""
    log.info("===== Running market: %s (%s) =====", market, mode)
    system = QuantSystem(market=market)
    send_report = (market == "US")
    if mode == "once":
        return system.run_once(send_report=send_report)
    if mode == "cycle_no_report":
        return system.run_once(send_report=False)
    if mode == "fast_cycle":
        return system.run_fast_live_cycle(
            prepared_path=os.environ.get("QUANT_FAST_PREPARED_PATH"),
            dry_run=dry_run,
        )
    if mode == "loop":
        # In loop mode each call runs its own forever-loop; we only support
        # single-market loop via this helper. main() handles multi-market.
        return system.run_loop()
    if mode == "test":
        log.info("=== SMOKE TEST [%s] ===", market)
        from config import settings as _settings
        if market == "US":
            original = _settings.STOCK_UNIVERSE
            _settings.STOCK_UNIVERSE = ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA"]
            try:
                report = system.run_once()
            finally:
                _settings.STOCK_UNIVERSE = original
            if report:
                print("\n" + report)
        else:
            system.run_once(send_report=False)
        return None
    raise ValueError(f"Unknown mode: {mode}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Market Quant Trading System")
    parser.add_argument("--once", action="store_true", help="Run single cycle then exit")
    parser.add_argument("--cycle-no-report", action="store_true",
                        help="Run legacy full trading cycle without Telegram report")
    parser.add_argument("--fast-cycle", action="store_true",
                        help="Run bounded live cycle from persisted signals (cron default)")
    parser.add_argument("--dry-run", action="store_true",
                        help="For --fast-cycle: fetch/validate bounded quotes but do not trade/write")
    parser.add_argument("--loop", action="store_true", help="Run continuous loop (US only)")
    parser.add_argument("--test", action="store_true", help="Quick smoke test")
    parser.add_argument("--market", choices=MARKETS + ["ALL"], default="ALL",
                        help="Limit to one market (default: ALL = iterate every market in MARKETS)")
    args = parser.parse_args()

    if args.once:
        mode = "once"
    elif args.fast_cycle:
        mode = "fast_cycle"
    elif args.cycle_no_report:
        mode = "cycle_no_report"
    elif args.loop:
        mode = "loop"
    elif args.test:
        mode = "test"
    else:
        parser.print_help()
        return

    markets = MARKETS if args.market == "ALL" else [args.market]

    if mode == "loop":
        # run_loop is blocking — only meaningful for one market.
        market = markets[0]
        if len(markets) > 1:
            log.warning("--loop only supports one market at a time; using %s", market)
        _run_for_market(market, mode, dry_run=args.dry_run)
        return

    first_failure = None
    for market in markets:
        try:
            try:
                _run_for_market(market, mode, dry_run=args.dry_run)
            except TypeError as exc:
                # Preserve monkeypatchability for legacy two-argument test
                # doubles while production uses the explicit dry-run keyword.
                if "dry_run" not in str(exc):
                    raise
                _run_for_market(market, mode)
        except Exception as e:
            log.exception("Market %s failed: %s", market, e)
            if first_failure is None:
                first_failure = e
            continue
    if first_failure is not None:
        raise first_failure


if __name__ == "__main__":
    main()
