"""
回测引擎 — 回放过去30天历史数据，模拟20个账户的交易
使用现有的因子引擎、信号生成器和交易引擎，逐日回测。
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone
from dataclasses import dataclass

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.expanduser("~/quant-trading")
sys.path.insert(0, PROJECT_ROOT)

from config.settings import STOCK_UNIVERSE, FEES
from data.fetcher import DataFetcher
from factors.alpha_factors import FactorEngine
from factors.signal import SignalGenerator
from factors.gp_signal import GPSignalGenerator
from factors.gp_miner import GPAlphaMiner
from trading.engine import TradingEngine
from trading.costs import MoomooAUCosts
from accounts.strategies import STRATEGIES
from accounts.gp_strategies import active_gp_strategies_for_market
from config.settings import ADAPTIVE_REBALANCE

log = logging.getLogger("backtest")


def _compute_adaptive_rebalance_hours(
    market_returns: list[float],
    sigma_ref: float,
    base_hours: float,
    config: dict = None,
) -> float:
    """根据近期市场波动率计算自适应换仓周期。

    Args:
        market_returns: 最近N天的市场平均日收益率
        sigma_ref: 参考波动率（长期中位数）
        base_hours: 基准换仓周期（小时）
        config: ADAPTIVE_REBALANCE 配置字典

    Returns:
        换仓周期（小时），float('inf') 表示静默（不换仓）
    """
    cfg = config or ADAPTIVE_REBALANCE
    vol_window = cfg.get("vol_window", 5)
    min_h = cfg.get("min_hours", 12)
    max_h = cfg.get("max_hours", 96)
    silence_thresh = cfg.get("silence_threshold", 0.3)

    if len(market_returns) < max(2, vol_window):
        return base_hours  # 数据不足，用默认

    recent = market_returns[-vol_window:]
    sigma_t = float(np.std(recent)) if len(recent) >= 2 else sigma_ref

    if sigma_t < 1e-10:
        return float("inf")
    if sigma_t < sigma_ref * silence_thresh:
        return float("inf")  # 静默区

    ratio = sigma_ref / sigma_t
    hours = base_hours * ratio
    return max(min_h, min(max_h, hours))


@dataclass
class BacktestResult:
    account_id: str
    strategy_name: str
    group: str
    initial_cash: float
    final_equity: float
    pnl: float
    pnl_pct: float
    max_drawdown: float
    sharpe_ratio: float
    total_trades: int
    equity_curve: list  # [(date_str, equity), ...]
    daily_returns: list
    alpha_vs_qqq: float = 0.0  # 绝对超额收益 (pct points)
    alpha_vs_spy: float = 0.0
    beta_vs_qqq: float = 0.0
    beta_vs_spy: float = 0.0
    info_ratio_qqq: float = 0.0  # 年化信息比率
    info_ratio_spy: float = 0.0


@dataclass
class BenchmarkResult:
    ticker: str
    initial_cash: float
    final_equity: float
    pnl_pct: float
    max_drawdown: float
    sharpe_ratio: float
    equity_curve: list
    daily_returns: list


class BacktestEngine:
    """逐日回测引擎，模拟20个账户过去N天的表现。"""

    def __init__(self, days: int = 30, initial_cash: float = 10000.0,
                 adaptive_rebalance: bool | None = None):
        """
        Args:
            days: 回测天数
            initial_cash: 初始资金
            adaptive_rebalance: 是否启用自适应换仓。
                None = 读取 config.settings.ADAPTIVE_REBALANCE["enabled"]
                True/False = 强制开关（覆盖配置）
        """
        self.days = days
        self.initial_cash = initial_cash
        if adaptive_rebalance is None:
            self.adaptive = ADAPTIVE_REBALANCE.get("enabled", False)
        else:
            self.adaptive = adaptive_rebalance
        self.fetcher = DataFetcher()
        self.factor_engine = FactorEngine()
        self.signal_gen = SignalGenerator(buy_top=10, sell_top=10)
        self.gp_signal_gen = GPSignalGenerator()
        self.gp_miner = GPAlphaMiner()
        self.per_account_mined = GPAlphaMiner.load_per_account_factors()
        self.sigma_ref: float = 0.01  # 初始值，run()中计算

    def run(self) -> tuple[list[BacktestResult], list[BenchmarkResult]]:
        """运行完整回测，返回 (账户结果, 基准结果)。"""
        log.info("开始回测: 获取 %d 天历史数据...", self.days)

        # Fetch extra days for factor warm-up (need ~30 extra for rolling windows)
        raw_df = self.fetcher.get_historical(STOCK_UNIVERSE, days=self.days + 40)
        if raw_df.empty:
            log.error("无法获取历史数据")
            return [], []

        # Split into per-ticker DataFrames
        all_data = {}
        for ticker, group in raw_df.groupby("ticker"):
            df = group.drop(columns=["ticker"]).set_index("datetime").sort_index()
            all_data[ticker] = df

        if not all_data:
            log.error("没有有效的股票数据")
            return [], []

        # Get common trading dates (last N days)
        all_dates = set()
        for df in all_data.values():
            all_dates.update(df.index.tolist())
        sorted_dates = sorted(all_dates)

        # Use the last `self.days` trading days for simulation
        sim_dates = sorted_dates[-self.days:] if len(sorted_dates) > self.days else sorted_dates
        log.info("回测区间: %s 至 %s (%d 交易日)",
                 sim_dates[0], sim_dates[-1], len(sim_dates))

        # Create trading engines for each account
        costs = MoomooAUCosts()
        results = []

        # 计算参考波动率（用于自适应换仓）
        if self.adaptive:
            all_returns = []
            for df in all_data.values():
                rets = df["close"].pct_change().dropna().values
                all_returns.extend(rets.tolist())
            vol_window = ADAPTIVE_REBALANCE.get("vol_window", 5)
            chunks = [all_returns[i:i+vol_window] for i in range(0, len(all_returns)-vol_window, vol_window)]
            if chunks:
                self.sigma_ref = float(np.median([np.std(c) for c in chunks]))
            if self.sigma_ref < 1e-10:
                self.sigma_ref = 0.01
            log.info("自适应换仓已启用，σ_ref=%.4f", self.sigma_ref)

        # Run A accounts
        for strat in STRATEGIES:
            result = self._backtest_alpha_account(strat, all_data, sim_dates, costs)
            results.append(result)

        # Run B accounts
        for gp_strat in active_gp_strategies_for_market("US"):
            result = self._backtest_gp_account(gp_strat, all_data, sim_dates, costs)
            results.append(result)

        # Fetch benchmarks (QQQ, SPY) and compute alpha/beta vs each account
        benchmarks = self._build_benchmarks(sim_dates)
        if benchmarks:
            self._attach_alpha_metrics(results, benchmarks)

        return results, benchmarks

    def _build_benchmarks(self, sim_dates) -> list[BenchmarkResult]:
        """拉取 QQQ/SPY 并归一化到 initial_cash 得到基准权益曲线。"""
        out = []
        try:
            bench_df = self.fetcher.get_historical(["QQQ", "SPY"], days=self.days + 40)
        except Exception as e:
            log.warning("基准数据拉取失败: %s", e)
            return out
        if bench_df is None or bench_df.empty:
            return out

        sim_date_set = set(sim_dates)
        for ticker in ["QQQ", "SPY"]:
            sub = bench_df[bench_df["ticker"] == ticker].copy()
            if sub.empty:
                continue
            sub = sub.set_index("datetime").sort_index()
            # 对齐 sim_dates：只取交易日在 sim_dates 内的收盘价
            closes = []
            curve = []
            last_close = None
            for d in sim_dates:
                if d in sub.index:
                    last_close = float(sub.loc[d, "close"])
                    # 若同日重复索引取第一个
                    if hasattr(last_close, "__len__"):
                        last_close = float(sub.loc[d, "close"].iloc[0])
                if last_close is None:
                    # 回退：取 sim_dates[0] 之前的最近一根
                    prev = sub.loc[sub.index <= d]
                    if not prev.empty:
                        last_close = float(prev["close"].iloc[-1])
                if last_close is not None:
                    closes.append(last_close)
                    curve.append((str(d), last_close))  # 价格占位，稍后归一化

            if len(closes) < 2:
                continue

            base_price = closes[0]
            equity_curve = [(d, self.initial_cash * (p / base_price)) for d, p in zip(
                [c[0] for c in curve], closes)]
            equities = [e for _, e in equity_curve]
            daily_returns = []
            for i in range(1, len(equities)):
                if equities[i-1] > 0:
                    daily_returns.append((equities[i] - equities[i-1]) / equities[i-1])

            # Drawdown
            max_dd = 0.0
            peak = equities[0]
            for eq in equities:
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd
            # Sharpe
            sharpe = 0.0
            if daily_returns:
                arr = np.array(daily_returns)
                if arr.std() > 0:
                    sharpe = float((arr.mean() / arr.std()) * np.sqrt(252))

            final_eq = equities[-1]
            pnl_pct = (final_eq - self.initial_cash) / self.initial_cash * 100
            out.append(BenchmarkResult(
                ticker=ticker,
                initial_cash=self.initial_cash,
                final_equity=round(final_eq, 2),
                pnl_pct=round(pnl_pct, 2),
                max_drawdown=round(max_dd * 100, 2),
                sharpe_ratio=round(sharpe, 2),
                equity_curve=equity_curve,
                daily_returns=daily_returns,
            ))
            log.info("基准 %s: 收益 %.2f%% Sharpe %.2f", ticker, pnl_pct, sharpe)
        return out

    def _attach_alpha_metrics(self, results: list[BacktestResult],
                               benchmarks: list[BenchmarkResult]):
        """计算每个账户相对 QQQ/SPY 的超额收益、Beta、信息比率。"""
        bench_map = {b.ticker: b for b in benchmarks}
        qqq = bench_map.get("QQQ")
        spy = bench_map.get("SPY")

        for r in results:
            acct_rets = np.array(r.daily_returns) if r.daily_returns else np.array([])
            for bench, alpha_field, beta_field, ir_field in [
                (qqq, "alpha_vs_qqq", "beta_vs_qqq", "info_ratio_qqq"),
                (spy, "alpha_vs_spy", "beta_vs_spy", "info_ratio_spy"),
            ]:
                if bench is None:
                    continue
                # 超额累计收益（pct points）
                setattr(r, alpha_field, round(r.pnl_pct - bench.pnl_pct, 2))
                b_rets = np.array(bench.daily_returns) if bench.daily_returns else np.array([])
                n = min(len(acct_rets), len(b_rets))
                if n >= 5:
                    a = acct_rets[-n:]
                    b = b_rets[-n:]
                    # Beta = cov(a,b)/var(b)
                    var_b = float(np.var(b))
                    if var_b > 1e-12:
                        beta = float(np.cov(a, b, ddof=0)[0, 1] / var_b)
                        setattr(r, beta_field, round(beta, 3))
                    # Information ratio = mean(a-b)/std(a-b) * sqrt(252)
                    diff = a - b
                    if diff.std() > 1e-12:
                        ir = float(diff.mean() / diff.std() * np.sqrt(252))
                        setattr(r, ir_field, round(ir, 2))

    def _backtest_alpha_account(self, strat, all_data, sim_dates, costs) -> BacktestResult:
        """回测单个Alpha158账户。"""
        engine = TradingEngine(
            max_position_pct=strat.max_position_pct,
            stop_loss_pct=-strat.stop_loss,
            costs=costs,
        )
        acct = engine.create_account(strat.id, initial_cash=self.initial_cash)
        equity_curve = []
        rebalance_counter = 0
        market_returns = []  # 用于自适应换仓

        for i, date in enumerate(sim_dates):
            # Get data up to this date for factor computation
            data_to_date = {}
            current_prices = {}
            for ticker, df in all_data.items():
                mask = df.index <= date
                sub = df.loc[mask]
                if len(sub) >= 20:  # need enough data for factors
                    data_to_date[ticker] = sub
                    current_prices[ticker] = float(sub["close"].iloc[-1])

            if not data_to_date:
                equity_curve.append((str(date), self.initial_cash))
                continue

            # 计算当日市场平均收益率（用于自适应换仓）
            if self.adaptive:
                day_rets = []
                for ticker, df in data_to_date.items():
                    if len(df) >= 2:
                        r = (df["close"].iloc[-1] - df["close"].iloc[-2]) / df["close"].iloc[-2]
                        day_rets.append(r)
                if day_rets:
                    market_returns.append(float(np.mean(day_rets)))

            # 决定换仓周期
            if self.adaptive:
                current_rebalance_hours = _compute_adaptive_rebalance_hours(
                    market_returns, self.sigma_ref, strat.rebalance_hours
                )
            else:
                current_rebalance_hours = strat.rebalance_hours

            # Rebalance check (based on hours, approximate: 1 day = ~7 trading hours)
            rebalance_counter += 7
            should_rebalance = (
                rebalance_counter >= current_rebalance_hours
                and current_rebalance_hours != float("inf")
            )

            if should_rebalance:
                rebalance_counter = 0

                # Compute factors and signals
                factors_dict = self.factor_engine.compute_multi(data_to_date)
                signals = self.signal_gen.generate_signals(factors_dict, strat.strategy_type)

                # Execute trades
                self._execute_alpha_trades(engine, strat, acct, signals, current_prices)

            equity = acct.get_equity(current_prices)
            equity_curve.append((str(date), equity))

        return self._build_result(strat.id, strat.name, "Alpha158", acct, equity_curve, sim_dates, all_data)

    def _backtest_gp_account(self, gp_strat, all_data, sim_dates, costs) -> BacktestResult:
        """回测单个GP进化账户。"""
        engine = TradingEngine(
            max_position_pct=gp_strat.max_position_pct,
            stop_loss_pct=-gp_strat.stop_loss,
            costs=costs,
        )
        acct = engine.create_account(gp_strat.id, initial_cash=self.initial_cash)
        equity_curve = []
        rebalance_counter = 0
        market_returns = []  # 用于自适应换仓

        mined_factors = self.per_account_mined.get(gp_strat.id, [])
        if not mined_factors:
            # No mined factors — return flat equity
            for date in sim_dates:
                equity_curve.append((str(date), self.initial_cash))
            return BacktestResult(
                account_id=gp_strat.id, strategy_name=gp_strat.name, group="GP进化",
                initial_cash=self.initial_cash, final_equity=self.initial_cash,
                pnl=0.0, pnl_pct=0.0, max_drawdown=0.0, sharpe_ratio=0.0,
                total_trades=0, equity_curve=equity_curve, daily_returns=[],
            )

        for i, date in enumerate(sim_dates):
            data_to_date = {}
            current_prices = {}
            for ticker, df in all_data.items():
                mask = df.index <= date
                sub = df.loc[mask]
                if len(sub) >= 20:
                    data_to_date[ticker] = sub
                    current_prices[ticker] = float(sub["close"].iloc[-1])

            if not data_to_date:
                equity_curve.append((str(date), self.initial_cash))
                continue

            # 计算当日市场平均收益率（用于自适应换仓）
            if self.adaptive:
                day_rets = []
                for ticker, df in data_to_date.items():
                    if len(df) >= 2:
                        r = (df["close"].iloc[-1] - df["close"].iloc[-2]) / df["close"].iloc[-2]
                        day_rets.append(r)
                if day_rets:
                    market_returns.append(float(np.mean(day_rets)))

            # 决定换仓周期
            if self.adaptive:
                current_rebalance_hours = _compute_adaptive_rebalance_hours(
                    market_returns, self.sigma_ref, gp_strat.rebalance_hours
                )
            else:
                current_rebalance_hours = gp_strat.rebalance_hours

            rebalance_counter += 7
            should_rebalance = (
                rebalance_counter >= current_rebalance_hours
                and current_rebalance_hours != float("inf")
            )

            if should_rebalance:
                rebalance_counter = 0

                # Compute GP factors
                gp_factors_dict = self.gp_miner.compute_gp_factors(data_to_date, mined_factors)

                # Filter factors by strategy config
                filtered = self._filter_gp_factors(gp_strat, gp_factors_dict, mined_factors)

                # Generate signals
                signals = self.gp_signal_gen.generate_signals(filtered, top_n=gp_strat.top_n)

                # Execute trades
                self._execute_gp_trades(engine, gp_strat, acct, signals, current_prices)

            equity = acct.get_equity(current_prices)
            equity_curve.append((str(date), equity))

        return self._build_result(gp_strat.id, gp_strat.name, "GP进化", acct, equity_curve, sim_dates, all_data)

    def _execute_alpha_trades(self, engine, strat, acct, signals, current_prices):
        """执行Alpha账户交易。"""
        buy_tickers = {t for t, _ in signals["buy"][:strat.top_n]}

        # Sell positions not in buy list + stop loss
        positions = acct.get_positions()
        for ticker, shares in list(positions.items()):
            if shares <= 0:
                continue
            price = current_prices.get(ticker)
            if not price:
                continue
            # Stop loss
            pos_data = acct._positions.get(ticker)
            if pos_data and pos_data.avg_cost > 0:
                pnl_pct = (price - pos_data.avg_cost) / pos_data.avg_cost
                if pnl_pct <= -strat.stop_loss:
                    try:
                        engine.execute_signal(strat.id, ticker, "sell", shares, price, current_prices)
                    except:
                        pass
                    continue
            if ticker not in buy_tickers:
                try:
                    engine.execute_signal(strat.id, ticker, "sell", shares, price, current_prices)
                except:
                    pass

        # Buy
        equity = acct.get_equity(current_prices)
        target = equity * strat.max_position_pct
        for ticker, score in signals["buy"][:strat.top_n]:
            price = current_prices.get(ticker)
            if not price or price <= 0:
                continue
            held = acct.get_positions().get(ticker, 0)
            held_val = held * price
            if held_val >= target * 0.9:
                continue
            budget = min(target - held_val, acct.cash * 0.95)
            if budget < 5:
                continue
            shares = int(budget / price)
            if shares <= 0:
                continue
            try:
                engine.execute_signal(strat.id, ticker, "buy", shares, price, current_prices)
            except:
                pass

    def _execute_gp_trades(self, engine, gp_strat, acct, signals, current_prices):
        """执行GP账户交易。"""
        buy_tickers = set(signals["buy"])
        sell_tickers = set(signals["sell"])

        positions = acct.get_positions()
        for ticker, shares in list(positions.items()):
            if shares <= 0:
                continue
            price = current_prices.get(ticker)
            if not price:
                continue
            pos_data = acct._positions.get(ticker)
            if pos_data and pos_data.avg_cost > 0:
                pnl_pct = (price - pos_data.avg_cost) / pos_data.avg_cost
                if pnl_pct <= -gp_strat.stop_loss:
                    try:
                        engine.execute_signal(gp_strat.id, ticker, "sell", shares, price, current_prices)
                    except:
                        pass
                    continue
            if ticker not in buy_tickers or ticker in sell_tickers:
                try:
                    engine.execute_signal(gp_strat.id, ticker, "sell", shares, price, current_prices)
                except:
                    pass

        equity = acct.get_equity(current_prices)
        target = equity * gp_strat.max_position_pct
        for ticker in signals["buy"][:gp_strat.top_n]:
            price = current_prices.get(ticker)
            if not price or price <= 0:
                continue
            held = acct.get_positions().get(ticker, 0)
            held_val = held * price
            if held_val >= target * 0.9:
                continue
            budget = min(target - held_val, acct.cash * 0.95)
            if budget < 5:
                continue
            shares = int(budget / price)
            if shares <= 0:
                continue
            try:
                engine.execute_signal(gp_strat.id, ticker, "buy", shares, price, current_prices)
            except:
                pass

    def _filter_gp_factors(self, gp_strat, gp_factors_dict, mined_factors):
        """根据策略配置过滤GP因子。"""
        if not mined_factors or not gp_factors_dict:
            return gp_factors_dict

        sorted_by_ic = sorted(mined_factors, key=lambda f: abs(f["ic"]), reverse=True)
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
        for ticker, fdf in gp_factors_dict.items():
            if fdf is None or fdf.empty:
                filtered[ticker] = fdf
                continue
            cols = [c for c in fdf.columns if c in keep]
            filtered[ticker] = fdf[cols] if cols else fdf
        return filtered

    def _build_result(self, account_id, name, group, acct, equity_curve, sim_dates, all_data) -> BacktestResult:
        """构建回测结果。"""
        # Get final equity
        current_prices = {}
        last_date = sim_dates[-1]
        for ticker, df in all_data.items():
            mask = df.index <= last_date
            sub = df.loc[mask]
            if not sub.empty:
                current_prices[ticker] = float(sub["close"].iloc[-1])

        final_equity = acct.get_equity(current_prices)
        pnl = final_equity - self.initial_cash
        pnl_pct = (pnl / self.initial_cash) * 100

        # Daily returns
        equities = [e for _, e in equity_curve]
        daily_returns = []
        for i in range(1, len(equities)):
            if equities[i - 1] > 0:
                daily_returns.append((equities[i] - equities[i - 1]) / equities[i - 1])

        # Max drawdown
        max_dd = 0.0
        peak = equities[0] if equities else self.initial_cash
        for eq in equities:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        # Sharpe ratio (annualized, assuming 252 trading days)
        sharpe = 0.0
        if daily_returns:
            arr = np.array(daily_returns)
            if arr.std() > 0:
                sharpe = (arr.mean() / arr.std()) * np.sqrt(252)

        return BacktestResult(
            account_id=account_id,
            strategy_name=name,
            group=group,
            initial_cash=self.initial_cash,
            final_equity=round(final_equity, 2),
            pnl=round(pnl, 2),
            pnl_pct=round(pnl_pct, 2),
            max_drawdown=round(max_dd * 100, 2),
            sharpe_ratio=round(sharpe, 2),
            total_trades=len(acct.trade_log),
            equity_curve=equity_curve,
            daily_returns=daily_returns,
        )
