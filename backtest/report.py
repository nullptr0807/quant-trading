"""
回测报告生成器 — 生成文字报告 + 权益曲线图表
支持与 QQQ / SPY 基准对比，计算超额 Alpha。
"""

import os
import logging
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import numpy as np

# 中文字体配置
_CN_FONT = None
for _fname in ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "SimHei", "Microsoft YaHei"]:
    if any(_fname.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        _CN_FONT = _fname
        break
if _CN_FONT:
    plt.rcParams["font.sans-serif"] = [_CN_FONT, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

log = logging.getLogger("backtest")

PROJECT_ROOT = os.path.expanduser("~/quant-trading")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports", "backtest")
os.makedirs(REPORT_DIR, exist_ok=True)


def generate_report(results, days: int = 30, benchmarks=None) -> tuple[str, str | None]:
    """生成回测报告文本和权益曲线图表。

    Args:
        results: list[BacktestResult]
        days: 回测天数
        benchmarks: list[BenchmarkResult] (QQQ / SPY)

    Returns:
        (report_text, chart_path)
    """
    if not results:
        return "回测失败: 无结果", None

    benchmarks = benchmarks or []
    bench_map = {b.ticker: b for b in benchmarks}
    qqq = bench_map.get("QQQ")
    spy = bench_map.get("SPY")

    now = datetime.now(timezone.utc)
    lines = [
        f"📈 回测报告 (过去{days}个交易日)",
        f"生成时间: {now.strftime('%Y-%m-%d %H:%M UTC')}",
        "=" * 40,
    ]

    # 基准概览
    if benchmarks:
        lines.append("\n🎯 基准表现")
        for b in benchmarks:
            lines.append(
                f"  {b.ticker}: {b.pnl_pct:+.2f}% | 回撤{b.max_drawdown:.1f}% | Sharpe={b.sharpe_ratio:.2f}"
            )

    # Split by group
    for group_name, emoji in [("Alpha158", "📐"), ("GP进化", "🧬")]:
        group = [r for r in results if r.group == group_name]
        if not group:
            continue

        group.sort(key=lambda r: r.pnl, reverse=True)
        total_pnl = sum(r.pnl for r in group)
        avg_sharpe = np.mean([r.sharpe_ratio for r in group]) if group else 0
        avg_alpha_qqq = np.mean([r.alpha_vs_qqq for r in group]) if qqq else 0.0
        avg_alpha_spy = np.mean([r.alpha_vs_spy for r in group]) if spy else 0.0

        lines.append(f"\n{emoji} {group_name} 组")
        lines.append(f"总盈亏: ${total_pnl:+.2f} | 平均Sharpe: {avg_sharpe:.2f}")
        if benchmarks:
            lines.append(
                f"平均超额: vs QQQ {avg_alpha_qqq:+.2f}pp | vs SPY {avg_alpha_spy:+.2f}pp"
            )
        lines.append("-" * 38)

        for r in group:
            icon = "🟢" if r.pnl >= 0 else "🔴"
            base = (
                f"{icon} {r.strategy_name} ({r.account_id})\n"
                f"  💰 ${r.final_equity:.0f} 盈亏{r.pnl:+.2f}({r.pnl_pct:+.1f}%)\n"
                f"  📉 最大回撤{r.max_drawdown:.1f}% Sharpe={r.sharpe_ratio:.2f}\n"
                f"  📊 交易{r.total_trades}笔"
            )
            if benchmarks:
                alpha_line = (
                    f"  🆚 αQQQ {r.alpha_vs_qqq:+.2f}pp (β={r.beta_vs_qqq:.2f}, IR={r.info_ratio_qqq:.2f})"
                    f" | αSPY {r.alpha_vs_spy:+.2f}pp (β={r.beta_vs_spy:.2f}, IR={r.info_ratio_spy:.2f})"
                )
                base += "\n" + alpha_line
            lines.append(base)

    # Best/worst (by alpha_vs_spy if benchmarks exist, else by pnl_pct)
    if spy:
        best = max(results, key=lambda r: r.alpha_vs_spy)
        worst = min(results, key=lambda r: r.alpha_vs_spy)
        lines.append(
            f"\n🏆 最高超额α(vs SPY): {best.strategy_name}({best.account_id}) "
            f"{best.alpha_vs_spy:+.2f}pp (绝对{best.pnl_pct:+.1f}%)"
        )
        lines.append(
            f"💀 最低超额α(vs SPY): {worst.strategy_name}({worst.account_id}) "
            f"{worst.alpha_vs_spy:+.2f}pp (绝对{worst.pnl_pct:+.1f}%)"
        )
    else:
        best = max(results, key=lambda r: r.pnl_pct)
        worst = min(results, key=lambda r: r.pnl_pct)
        lines.append(f"\n🏆 最佳: {best.strategy_name}({best.account_id}) {best.pnl_pct:+.1f}%")
        lines.append(f"💀 最差: {worst.strategy_name}({worst.account_id}) {worst.pnl_pct:+.1f}%")

    report_text = "\n".join(lines)

    # Generate chart
    chart_path = _generate_chart(results, benchmarks, days=days)

    return report_text, chart_path


def _generate_chart(results, benchmarks, days: int = 30) -> str | None:
    """生成权益曲线图表（叠加基准）。"""
    try:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), dpi=100)
        fig.suptitle(f"回测权益曲线 (过去{days}个交易日)", fontsize=14, fontweight="bold")

        bench_styles = {
            "QQQ": {"color": "#1f77b4", "linestyle": "-", "linewidth": 2.6, "label": "QQQ基准"},
            "SPY": {"color": "#000000", "linestyle": "--", "linewidth": 2.6, "label": "SPY基准"},
        }

        for ax, (group_name, title) in zip(axes, [("Alpha158", "Alpha158 账户 (A01-A10)"),
                                                    ("GP进化", "GP进化 账户 (B01-B10)")]):
            group = [r for r in results if r.group == group_name]
            if not group:
                continue

            for r in sorted(group, key=lambda r: r.pnl, reverse=True):
                if not r.equity_curve:
                    continue
                dates = list(range(len(r.equity_curve)))
                values = [e for _, e in r.equity_curve]
                alpha = 0.8 if abs(r.pnl_pct) > 1 else 0.4
                ax.plot(dates, values, label=f"{r.account_id} {r.strategy_name}",
                        alpha=alpha, linewidth=1.0)

            # 叠加基准
            for b in benchmarks:
                if not b.equity_curve:
                    continue
                bx = list(range(len(b.equity_curve)))
                by = [e for _, e in b.equity_curve]
                style = bench_styles.get(b.ticker, {})
                ax.plot(bx, by, **style, alpha=0.95, zorder=10)

            ax.axhline(y=10000, color="gray", linestyle=":", alpha=0.5, label="初始资金")
            ax.set_title(title)
            ax.set_xlabel("交易日")
            ax.set_ylabel("权益 ($)")
            ax.legend(fontsize=7, loc="upper left", ncol=2)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        chart_path = os.path.join(REPORT_DIR, f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(chart_path, bbox_inches="tight")
        plt.close()
        log.info("图表已保存: %s", chart_path)
        return chart_path
    except Exception as e:
        log.warning("图表生成失败: %s", e)
        return None
