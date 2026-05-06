"""
B08 自适应换仓 vs 固定换仓 对比回测
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config.settings import STOCK_UNIVERSE
from data.fetcher import DataFetcher
from factors.gp_miner import GPAlphaMiner
from factors.gp_signal import GPSignalGenerator
from trading.engine import TradingEngine
from trading.costs import MoomooAUCosts
from accounts.gp_strategies import GP_STRATEGIES


def get_adaptive_rebalance_hours(returns_5d, sigma_ref, base_hours=36, min_h=12, max_h=96):
    """自适应换仓周期"""
    sigma_t = np.std(returns_5d) if len(returns_5d) >= 2 else sigma_ref
    if sigma_t < 1e-10:
        return float('inf')  # 静默
    if sigma_t < sigma_ref * 0.3:
        return float('inf')  # 静默区
    ratio = sigma_ref / sigma_t
    hours = base_hours * ratio
    return max(min_h, min(max_h, hours))


def run_b08_backtest(adaptive=False, days=30):
    """Run B08 backtest with fixed or adaptive rebalancing."""
    gp_strat = [s for s in GP_STRATEGIES if s.id == 'B08'][0]
    costs = MoomooAUCosts()

    # Fetch data
    fetcher = DataFetcher()
    raw_df = fetcher.get_historical(STOCK_UNIVERSE, days=days + 40)
    all_data = {}
    for ticker, group in raw_df.groupby('ticker'):
        df = group.drop(columns=['ticker']).set_index('datetime').sort_index()
        all_data[ticker] = df

    sorted_dates = sorted(set(d for df in all_data.values() for d in df.index.tolist()))
    sim_dates = sorted_dates[-days:]

    # 计算参考波动率 (用sim前60天数据)
    # 用SPY近似市场，或用全股票平均收益率
    all_returns = []
    for ticker, df in all_data.items():
        rets = df['close'].pct_change().dropna().values
        all_returns.extend(rets.tolist())
    sigma_ref = np.median([np.std(all_returns[i:i+5]) for i in range(0, len(all_returns)-5, 5)])
    if sigma_ref < 1e-10:
        sigma_ref = 0.01

    # Setup
    engine = TradingEngine(
        max_position_pct=gp_strat.max_position_pct,
        stop_loss_pct=-gp_strat.stop_loss,
        costs=costs,
    )
    acct = engine.create_account('B08', initial_cash=10000.0)
    equity_curve = []
    rebalance_counter = 0
    trade_count = 0
    rebalance_log = []  # (date, hours_used)

    miner = GPAlphaMiner()
    per_account_mined = GPAlphaMiner.load_per_account_factors()
    mined_factors = per_account_mined.get('B08', [])
    gp_signal_gen = GPSignalGenerator()

    # recent market returns for adaptive calc
    market_returns = []

    for i, date in enumerate(sim_dates):
        data_to_date = {}
        current_prices = {}
        for ticker, df in all_data.items():
            sub = df.loc[df.index <= date]
            if len(sub) >= 20:
                data_to_date[ticker] = sub
                current_prices[ticker] = float(sub['close'].iloc[-1])

        if not data_to_date:
            equity_curve.append((str(date), 10000.0))
            continue

        # 计算当日市场平均收益率
        day_returns = []
        for ticker, df in data_to_date.items():
            if len(df) >= 2:
                r = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
                day_returns.append(r)
        if day_returns:
            market_returns.append(np.mean(day_returns))

        # 决定换仓周期
        if adaptive and len(market_returns) >= 5:
            current_rebalance_hours = get_adaptive_rebalance_hours(
                market_returns[-5:], sigma_ref, base_hours=gp_strat.rebalance_hours
            )
        else:
            current_rebalance_hours = gp_strat.rebalance_hours

        rebalance_counter += 7  # 每天+7小时(交易日)
        should_rebalance = rebalance_counter >= current_rebalance_hours

        if should_rebalance and current_rebalance_hours != float('inf'):
            rebalance_counter = 0
            rebalance_log.append((str(date)[:10], current_rebalance_hours))

            gp_factors_dict = miner.compute_gp_factors(data_to_date, mined_factors)

            # Filter: top3_only
            sorted_by_ic = sorted(mined_factors, key=lambda f: abs(f['ic']), reverse=True)
            keep = set(f['name'] for f in sorted_by_ic[:3])
            filtered = {}
            for t, fdf in gp_factors_dict.items():
                if fdf is None or fdf.empty:
                    filtered[t] = fdf
                    continue
                cols = [c for c in fdf.columns if c in keep]
                filtered[t] = fdf[cols] if cols else fdf

            signals = gp_signal_gen.generate_signals(filtered, top_n=gp_strat.top_n)

            # Execute trades
            buy_set = set(signals['buy'])
            sell_set = set(signals['sell'])
            for tk, sh in list(acct.get_positions().items()):
                if sh <= 0:
                    continue
                p = current_prices.get(tk)
                if not p:
                    continue
                pd2 = acct._positions.get(tk)
                if pd2 and pd2.avg_cost > 0:
                    pnl_pct = (p - pd2.avg_cost) / pd2.avg_cost
                    if pnl_pct <= -gp_strat.stop_loss:
                        try:
                            engine.execute_signal('B08', tk, 'sell', sh, p, current_prices)
                            trade_count += 1
                        except:
                            pass
                        continue
                if tk not in buy_set or tk in sell_set:
                    try:
                        engine.execute_signal('B08', tk, 'sell', sh, p, current_prices)
                        trade_count += 1
                    except:
                        pass

            equity_val = acct.get_equity(current_prices)
            target = equity_val * gp_strat.max_position_pct
            for tk in signals['buy'][:gp_strat.top_n]:
                p = current_prices.get(tk)
                if not p or p <= 0:
                    continue
                held = acct.get_positions().get(tk, 0)
                hv = held * p
                if hv >= target * 0.9:
                    continue
                budget = min(target - hv, acct.cash * 0.95)
                if budget < 5:
                    continue
                shares = int(budget / p)
                if shares <= 0:
                    continue
                try:
                    engine.execute_signal('B08', tk, 'buy', shares, p, current_prices)
                    trade_count += 1
                except:
                    pass

        equity = acct.get_equity(current_prices)
        equity_curve.append((str(date), equity))

    final_equity = equity_curve[-1][1] if equity_curve else 10000.0
    pnl = final_equity - 10000.0

    # Sharpe
    daily_ret = []
    for j in range(1, len(equity_curve)):
        r = (equity_curve[j][1] - equity_curve[j-1][1]) / equity_curve[j-1][1]
        daily_ret.append(r)
    sharpe = (np.mean(daily_ret) / np.std(daily_ret) * np.sqrt(252)) if daily_ret and np.std(daily_ret) > 0 else 0

    # Max drawdown
    peak = 10000.0
    max_dd = 0
    for _, eq in equity_curve:
        peak = max(peak, eq)
        dd = (peak - eq) / peak * 100
        max_dd = max(max_dd, dd)

    return {
        'equity_curve': equity_curve,
        'final_equity': final_equity,
        'pnl': pnl,
        'pnl_pct': pnl / 100,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'trades': trade_count,
        'rebalance_log': rebalance_log,
        'sigma_ref': sigma_ref,
    }


def main():
    print("=" * 60)
    print("B08 自适应换仓 vs 固定换仓 对比回测")
    print("=" * 60)

    print("\n⏳ 运行固定换仓 (36h)...")
    fixed = run_b08_backtest(adaptive=False, days=30)

    print("⏳ 运行自适应换仓...")
    adaptive = run_b08_backtest(adaptive=True, days=30)

    print(f"\n参考波动率 σ_ref = {adaptive['sigma_ref']:.4f}")

    print(f"\n{'指标':<20} {'固定36h':>12} {'自适应':>12} {'差异':>12}")
    print("-" * 58)
    print(f"{'最终权益':<20} ${fixed['final_equity']:>10.2f}  ${adaptive['final_equity']:>10.2f}  ${adaptive['final_equity']-fixed['final_equity']:>+10.2f}")
    print(f"{'盈亏':<20} ${fixed['pnl']:>10.2f}  ${adaptive['pnl']:>10.2f}  ${adaptive['pnl']-fixed['pnl']:>+10.2f}")
    print(f"{'收益率':<20} {fixed['pnl']/100:>11.1f}%  {adaptive['pnl']/100:>11.1f}%  {(adaptive['pnl']-fixed['pnl'])/100:>+11.1f}%")
    print(f"{'Sharpe':<20} {fixed['sharpe']:>12.2f}  {adaptive['sharpe']:>12.2f}  {adaptive['sharpe']-fixed['sharpe']:>+12.2f}")
    print(f"{'最大回撤':<20} {fixed['max_dd']:>11.1f}%  {adaptive['max_dd']:>11.1f}%  {adaptive['max_dd']-fixed['max_dd']:>+11.1f}%")
    print(f"{'总交易笔数':<20} {fixed['trades']:>12}  {adaptive['trades']:>12}  {adaptive['trades']-fixed['trades']:>+12}")
    print(f"{'换仓次数':<20} {len(fixed['rebalance_log']):>12}  {len(adaptive['rebalance_log']):>12}  {len(adaptive['rebalance_log'])-len(fixed['rebalance_log']):>+12}")

    if adaptive['rebalance_log']:
        print(f"\n📅 自适应换仓详情：")
        for date, hours in adaptive['rebalance_log']:
            bar = "█" * int(hours / 4)
            print(f"  {date}  周期={hours:5.1f}h  {bar}")

    # 画图
    plt.rcParams['font.family'] = 'WenQuanYi Zen Hei'
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

    # 权益曲线
    ax1 = axes[0]
    dates_f = [d[:10] for d, _ in fixed['equity_curve']]
    eq_f = [e for _, e in fixed['equity_curve']]
    eq_a = [e for _, e in adaptive['equity_curve']]

    ax1.plot(range(len(eq_f)), eq_f, 'b-', linewidth=2, label=f"固定36h (收益{fixed['pnl']:+.0f}, Sharpe={fixed['sharpe']:.2f})")
    ax1.plot(range(len(eq_a)), eq_a, 'r-', linewidth=2, label=f"自适应 (收益{adaptive['pnl']:+.0f}, Sharpe={adaptive['sharpe']:.2f})")
    ax1.fill_between(range(len(eq_f)), eq_f, eq_a, alpha=0.15, color='green' if adaptive['pnl'] > fixed['pnl'] else 'red')
    ax1.set_ylabel('权益 ($)')
    ax1.set_title('B08 染色体交叉 — 固定换仓 vs 自适应换仓（30天回测）', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=10000, color='gray', linestyle='--', alpha=0.5)

    # Set x ticks
    tick_idx = list(range(0, len(dates_f), max(1, len(dates_f)//10)))
    ax1.set_xticks(tick_idx)
    ax1.set_xticklabels([dates_f[i] for i in tick_idx], rotation=45, ha='right')

    # 换仓周期图
    ax2 = axes[1]
    if adaptive['rebalance_log']:
        rb_dates = [d for d, _ in adaptive['rebalance_log']]
        rb_hours = [h for _, h in adaptive['rebalance_log']]
        ax2.bar(range(len(rb_dates)), rb_hours, color='coral', alpha=0.7, label='自适应周期')
        ax2.axhline(y=36, color='blue', linestyle='--', linewidth=1.5, label='固定36h基准')
        ax2.set_ylabel('换仓周期 (小时)')
        ax2.set_xticks(range(len(rb_dates)))
        ax2.set_xticklabels(rb_dates, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('日期')

    plt.tight_layout()
    out_path = 'reports/backtest/b08_adaptive_vs_fixed.png'
    import os
    os.makedirs('reports/backtest', exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📊 对比图已保存: {out_path}")


if __name__ == '__main__':
    main()
