"""
账户分析器 — 按账户ID查询所有交易数据并生成综合分析图表。

用法:
    from analysis.account_analyzer import AccountAnalyzer
    analyzer = AccountAnalyzer()
    report, chart_path = analyzer.analyze("B08")
"""

import os
import sqlite3
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

# 中文字体
_CN_FONT = None
for _fname in ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "SimHei", "Microsoft YaHei"]:
    if any(_fname.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        _CN_FONT = _fname
        break
if _CN_FONT:
    plt.rcParams["font.sans-serif"] = [_CN_FONT, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

log = logging.getLogger("analysis")

DB_PATH = os.path.expanduser("~/quant-trading/data/trading.db")
REPORT_DIR = os.path.expanduser("~/quant-trading/reports/accounts")
os.makedirs(REPORT_DIR, exist_ok=True)


class AccountAnalyzer:
    """按账户ID查询完整交易数据，计算指标，生成分析图。"""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or DB_PATH

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ── 数据查询 ────────────────────────────────────────────────────────────

    def get_account_meta(self, account_id: str) -> dict | None:
        """获取账户元数据（来自 account_meta 表或策略配置）。"""
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM account_meta WHERE account_id=?", (account_id,)
        ).fetchone()
        conn.close()
        if row:
            return dict(row)
        return None

    def get_equity_curve(self, account_id: str) -> pd.DataFrame:
        """获取权益曲线（accounts 表快照）。"""
        conn = self._conn()
        df = pd.read_sql_query(
            "SELECT timestamp, cash, equity FROM accounts WHERE name=? ORDER BY timestamp",
            conn, params=(account_id,),
        )
        conn.close()
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def get_trades(self, account_id: str) -> pd.DataFrame:
        """获取全部交易记录。"""
        conn = self._conn()
        df = pd.read_sql_query(
            "SELECT * FROM trades WHERE account=? ORDER BY timestamp",
            conn, params=(account_id,),
        )
        conn.close()
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def get_positions_history(self, account_id: str) -> pd.DataFrame:
        """获取持仓变化历史。"""
        conn = self._conn()
        df = pd.read_sql_query(
            "SELECT * FROM positions_history WHERE account=? ORDER BY timestamp",
            conn, params=(account_id,),
        )
        conn.close()
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def get_current_positions(self, account_id: str) -> pd.DataFrame:
        """获取当前持仓。"""
        conn = self._conn()
        df = pd.read_sql_query(
            "SELECT * FROM positions WHERE account=?",
            conn, params=(account_id,),
        )
        conn.close()
        return df

    def get_factors(self, account_id: str) -> list[str]:
        """获取账户使用的因子名称列表。"""
        meta = self.get_account_meta(account_id)
        if meta and meta.get("factors"):
            raw = meta["factors"]
            # GP factors stored as single descriptor like "GP(seed=888,pop=450,gen=20)"
            if raw.startswith("GP("):
                return [raw]
            return raw.split(",")
        # Fallback: 从策略配置获取
        try:
            if account_id.startswith("A"):
                from accounts.strategies import STRATEGIES
                for s in STRATEGIES:
                    if s.id == account_id:
                        return list(s.factor_names)
            else:
                from accounts.gp_strategies import GP_STRATEGIES
                for g in GP_STRATEGIES:
                    if g.id == account_id:
                        # GP factors are mined, load from file
                        from factors.gp_alpha_miner import GPAlphaMiner
                        per_acct = GPAlphaMiner.load_per_account_factors()
                        if account_id in per_acct:
                            return [f.get("name", f"GP_{i}") if isinstance(f, dict) else str(f)
                                    for i, f in enumerate(per_acct[account_id])]
                        return [f"GP因子 (种子={g.gp_seed}, 种群={g.gp_population})"]
        except Exception:
            pass
        return []

    def get_adaptive_state(self, account_id: str) -> dict | None:
        """获取自适应调仓状态。"""
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM adaptive_state WHERE account=?", (account_id,)
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    # ── 指标计算 ────────────────────────────────────────────────────────────

    def compute_metrics(self, equity_df: pd.DataFrame, trades_df: pd.DataFrame,
                        initial_cash: float = 10000.0) -> dict:
        """从权益曲线和交易记录计算核心指标。"""
        metrics = {
            "initial_cash": initial_cash,
            "current_equity": initial_cash,
            "total_return_pct": 0.0,
            "total_pnl": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "max_profit_pct": 0.0,
            "total_trades": 0,
            "win_trades": 0,
            "loss_trades": 0,
            "win_rate": 0.0,
            "avg_trade_pnl": 0.0,
            "total_fees": 0.0,
            "total_slippage": 0.0,
            "trading_days": 0,
            "first_trade": None,
            "last_trade": None,
        }

        # 权益曲线指标
        if not equity_df.empty:
            equities = equity_df["equity"].values
            metrics["current_equity"] = float(equities[-1])
            metrics["total_pnl"] = float(equities[-1] - initial_cash)
            metrics["total_return_pct"] = float((equities[-1] / initial_cash - 1) * 100)

            # 日收益率 → Sharpe
            if len(equities) > 1:
                returns = np.diff(equities) / equities[:-1]
                if np.std(returns) > 0:
                    # 年化 Sharpe（假设每小时一个快照，252天×6.5小时）
                    periods_per_year = 252 * 6.5
                    metrics["sharpe_ratio"] = float(
                        np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year)
                    )

            # 最大回撤
            peak = np.maximum.accumulate(equities)
            drawdown = (peak - equities) / peak * 100
            metrics["max_drawdown_pct"] = float(np.max(drawdown))

            # 最大浮盈
            profit = (equities - initial_cash) / initial_cash * 100
            metrics["max_profit_pct"] = float(np.max(profit))

            # 交易天数
            timestamps = equity_df["timestamp"]
            if len(timestamps) >= 2:
                delta = timestamps.iloc[-1] - timestamps.iloc[0]
                metrics["trading_days"] = max(1, delta.days)

        # 交易记录指标
        if not trades_df.empty:
            metrics["total_trades"] = len(trades_df)
            metrics["total_fees"] = float(trades_df["cost"].sum())
            metrics["total_slippage"] = float(trades_df["slippage"].sum())
            metrics["first_trade"] = str(trades_df["timestamp"].iloc[0])
            metrics["last_trade"] = str(trades_df["timestamp"].iloc[-1])

            # 估算交易盈亏（按 ticker 配对 buy/sell）
            win, loss = self._calc_trade_winloss(trades_df)
            metrics["win_trades"] = win
            metrics["loss_trades"] = loss
            metrics["win_rate"] = float(win / (win + loss) * 100) if (win + loss) > 0 else 0.0

        return metrics

    def _calc_trade_winloss(self, trades_df: pd.DataFrame) -> tuple[int, int]:
        """配对 buy/sell 计算胜负次数。"""
        wins, losses = 0, 0
        # Group by ticker, pair buy→sell
        for ticker, grp in trades_df.groupby("ticker"):
            buys = grp[grp["side"] == "buy"].sort_values("timestamp")
            sells = grp[grp["side"] == "sell"].sort_values("timestamp")
            for _, sell_row in sells.iterrows():
                # 找最近的 buy
                prior_buys = buys[buys["timestamp"] < sell_row["timestamp"]]
                if prior_buys.empty:
                    continue
                avg_buy_price = prior_buys["price"].mean()
                if sell_row["price"] > avg_buy_price:
                    wins += 1
                else:
                    losses += 1
        return wins, losses

    # ── 文字报告 ────────────────────────────────────────────────────────────

    def generate_text_report(self, account_id: str, metrics: dict,
                             factors: list[str], adaptive: dict | None,
                             meta: dict | None) -> str:
        """生成文字报告。"""
        lines = [f"📊 账户分析报告: {account_id}"]
        lines.append(f"生成时间: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        lines.append("=" * 40)

        # 账户信息
        if meta:
            lines.append(f"📋 策略: {meta.get('strategy_name', 'N/A')}")
            lines.append(f"📝 描述: {meta.get('description', 'N/A')}")
            lines.append(f"📅 创建时间: {meta.get('created_at', 'N/A')}")
            group = meta.get("group", "A" if account_id.startswith("A") else "B")
            lines.append(f"📂 分组: {'Alpha158' if group == 'A' else 'GP进化'}")
        else:
            group = "Alpha158" if account_id.startswith("A") else "GP进化"
            lines.append(f"📂 分组: {group}")

        lines.append("")

        # 核心指标
        lines.append("💰 核心指标:")
        lines.append(f"  初始资金: ${metrics['initial_cash']:,.2f}")
        lines.append(f"  当前权益: ${metrics['current_equity']:,.2f}")
        lines.append(f"  总收益: {metrics['total_return_pct']:+.2f}% (${metrics['total_pnl']:+,.2f})")
        lines.append(f"  Sharpe比率: {metrics['sharpe_ratio']:.2f}")
        lines.append(f"  最大回撤: {metrics['max_drawdown_pct']:.2f}%")
        lines.append(f"  最大浮盈: {metrics['max_profit_pct']:.2f}%")
        lines.append(f"  交易天数: {metrics['trading_days']}")
        lines.append("")

        # 交易统计
        lines.append("📈 交易统计:")
        lines.append(f"  总交易次数: {metrics['total_trades']}")
        lines.append(f"  胜率: {metrics['win_rate']:.1f}% ({metrics['win_trades']}胜/{metrics['loss_trades']}负)")
        lines.append(f"  总手续费: ${metrics['total_fees']:.2f}")
        lines.append(f"  总滑点成本: ${metrics['total_slippage']:.2f}")
        if metrics["first_trade"]:
            lines.append(f"  首次交易: {metrics['first_trade'][:19]}")
            lines.append(f"  最近交易: {metrics['last_trade'][:19]}")
        lines.append("")

        # 因子
        if factors:
            lines.append(f"🧬 使用因子 ({len(factors)}个):")
            for i, f in enumerate(factors, 1):
                lines.append(f"  {i}. {f}")
            lines.append("")

        # 自适应状态
        if adaptive:
            lines.append("⚙️ 自适应调仓:")
            lines.append(f"  当前周期: {adaptive.get('rebalance_hours', 'N/A')}h")
            lines.append(f"  参考σ: {adaptive.get('sigma_ref', 'N/A')}")
            lines.append(f"  上次调仓: {adaptive.get('last_rebalance', 'N/A')}")

        return "\n".join(lines)

    # ── 图表 ────────────────────────────────────────────────────────────────

    def generate_chart(self, account_id: str, equity_df: pd.DataFrame,
                       trades_df: pd.DataFrame, positions_hist: pd.DataFrame,
                       metrics: dict) -> str | None:
        """生成综合分析图表（4子图）。"""
        if equity_df.empty:
            return None

        try:
            fig = plt.figure(figsize=(16, 14), dpi=120)
            gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3)

            # ── 1. 权益曲线 + 回撤 (占顶部整行) ──
            ax1 = fig.add_subplot(gs[0, :])
            self._plot_equity_and_drawdown(ax1, equity_df, metrics)

            # ── 2. 交易分布 ──
            ax2 = fig.add_subplot(gs[1, 0])
            self._plot_trade_distribution(ax2, trades_df)

            # ── 3. 持仓变化 ──
            ax3 = fig.add_subplot(gs[1, 1])
            self._plot_position_changes(ax3, positions_hist, trades_df=trades_df)

            # ── 4. 滚动Sharpe ──
            ax4 = fig.add_subplot(gs[2, 0])
            self._plot_rolling_sharpe(ax4, equity_df)

            # ── 5. 交易时间线 ──
            ax5 = fig.add_subplot(gs[2, 1])
            self._plot_trade_timeline(ax5, trades_df)

            group = "Alpha158" if account_id.startswith("A") else "GP进化"
            fig.suptitle(
                f"账户 {account_id} 综合分析 ({group})\n"
                f"收益: {metrics['total_return_pct']:+.2f}%  |  "
                f"Sharpe: {metrics['sharpe_ratio']:.2f}  |  "
                f"最大回撤: {metrics['max_drawdown_pct']:.2f}%",
                fontsize=13, fontweight="bold", y=0.98,
            )

            chart_path = os.path.join(
                REPORT_DIR,
                f"{account_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            )
            plt.savefig(chart_path, bbox_inches="tight", facecolor="white")
            plt.close()
            log.info("账户分析图已保存: %s", chart_path)
            return chart_path

        except Exception as e:
            log.warning("图表生成失败: %s", e, exc_info=True)
            plt.close("all")
            return None

    def _plot_equity_and_drawdown(self, ax, equity_df, metrics):
        """权益曲线 + 回撤阴影。"""
        ts = equity_df["timestamp"]
        eq = equity_df["equity"].values
        initial = metrics["initial_cash"]

        # 权益线
        color = "#2ecc71" if eq[-1] >= initial else "#e74c3c"
        ax.plot(ts, eq, color=color, linewidth=1.5, label="权益")
        ax.axhline(y=initial, color="gray", linestyle="--", alpha=0.5, label=f"初始 ${initial:,.0f}")

        # 回撤阴影
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / peak
        ax2 = ax.twinx()
        ax2.fill_between(ts, dd * 100, alpha=0.15, color="red", label="回撤")
        ax2.set_ylabel("回撤 %", color="red", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="red", labelsize=8)
        ax2.set_ylim(bottom=0)
        ax2.invert_yaxis()

        ax.set_title("权益曲线 & 回撤", fontsize=11)
        ax.set_ylabel("权益 ($)", fontsize=9)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    def _plot_trade_distribution(self, ax, trades_df):
        """交易盈亏分布直方图。"""
        if trades_df.empty:
            ax.text(0.5, 0.5, "暂无交易数据", ha="center", va="center", fontsize=11)
            ax.set_title("交易分布", fontsize=11)
            return

        # 按 ticker 配对估算每笔交易盈亏
        pnls = []
        for ticker, grp in trades_df.groupby("ticker"):
            buys = grp[grp["side"] == "buy"].sort_values("timestamp")
            sells = grp[grp["side"] == "sell"].sort_values("timestamp")
            for _, sell_row in sells.iterrows():
                prior = buys[buys["timestamp"] < sell_row["timestamp"]]
                if prior.empty:
                    continue
                avg_buy = prior["price"].mean()
                pnl_pct = (sell_row["price"] - avg_buy) / avg_buy * 100
                pnls.append(pnl_pct)

        if pnls:
            colors = ["#2ecc71" if p >= 0 else "#e74c3c" for p in pnls]
            ax.bar(range(len(pnls)), pnls, color=colors, alpha=0.7, width=0.8)
            ax.axhline(y=0, color="gray", linewidth=0.5)
            ax.set_xlabel("交易序号", fontsize=9)
            ax.set_ylabel("盈亏 %", fontsize=9)
        else:
            ax.text(0.5, 0.5, "暂无完整买卖配对", ha="center", va="center", fontsize=11)
        ax.set_title("交易盈亏分布", fontsize=11)
        ax.tick_params(labelsize=8)

    def _plot_position_changes(self, ax, positions_hist, trades_df=None):
        """持仓数量随时间变化。优先用positions_history，否则从trades推算。"""
        if positions_hist.empty and trades_df is not None and not trades_df.empty:
            # 从trades推算每个时间点的累计持仓ticker数
            trades_sorted = trades_df.sort_values("timestamp")
            holdings = {}  # ticker -> shares
            times, counts = [], []
            for _, row in trades_sorted.iterrows():
                tk = row["ticker"]
                sh = row.get("shares", 0)
                if row.get("side") == "buy":
                    holdings[tk] = holdings.get(tk, 0) + sh
                else:
                    holdings[tk] = holdings.get(tk, 0) - sh
                    if holdings.get(tk, 0) <= 0:
                        holdings.pop(tk, None)
                times.append(row["timestamp"])
                counts.append(len(holdings))
            if times:
                ax.step(times, counts, where="post", color="#3498db", linewidth=1.2)
                ax.fill_between(times, counts, alpha=0.1, color="#3498db", step="post")
                ax.set_title("持仓数量变化", fontsize=11)
                ax.set_ylabel("持仓数", fontsize=9)
                ax.tick_params(labelsize=8)
                ax.grid(True, alpha=0.3)
                return

        if positions_hist.empty:
            ax.text(0.5, 0.5, "暂无持仓历史", ha="center", va="center", fontsize=11)
            ax.set_title("持仓变化", fontsize=11)
            return

        # 每个时间点的总持仓数
        pos_count = positions_hist.groupby("timestamp")["ticker"].nunique()
        ax.step(pos_count.index, pos_count.values, where="post", color="#3498db", linewidth=1.2)
        ax.fill_between(pos_count.index, pos_count.values, alpha=0.1, color="#3498db", step="post")
        ax.set_title("持仓数量变化", fontsize=11)
        ax.set_ylabel("持仓数", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_rolling_sharpe(self, ax, equity_df):
        """滚动Sharpe比率。"""
        if len(equity_df) < 10:
            ax.text(0.5, 0.5, "数据不足", ha="center", va="center", fontsize=11)
            ax.set_title("滚动Sharpe", fontsize=11)
            return

        eq = equity_df["equity"].values
        returns = np.diff(eq) / eq[:-1]
        window = min(20, len(returns) // 2)
        if window < 3:
            ax.text(0.5, 0.5, "数据不足", ha="center", va="center", fontsize=11)
            ax.set_title("滚动Sharpe", fontsize=11)
            return

        rolling_mean = pd.Series(returns).rolling(window).mean()
        rolling_std = pd.Series(returns).rolling(window).std()
        rolling_sharpe = (rolling_mean / rolling_std * np.sqrt(252 * 6.5)).dropna()

        ts = equity_df["timestamp"].iloc[window:]
        if len(ts) == len(rolling_sharpe):
            ax.plot(ts, rolling_sharpe.values, color="#9b59b6", linewidth=1.2)
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax.fill_between(ts, rolling_sharpe.values, alpha=0.1, color="#9b59b6")
        ax.set_title(f"滚动Sharpe (窗口={window})", fontsize=11)
        ax.set_ylabel("Sharpe", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_trade_timeline(self, ax, trades_df):
        """交易时间线：买卖点标注。"""
        if trades_df.empty:
            ax.text(0.5, 0.5, "暂无交易数据", ha="center", va="center", fontsize=11)
            ax.set_title("交易时间线", fontsize=11)
            return

        buys = trades_df[trades_df["side"] == "buy"]
        sells = trades_df[trades_df["side"] == "sell"]

        if not buys.empty:
            ax.scatter(buys["timestamp"], buys["price"] * buys["shares"],
                       marker="^", color="#2ecc71", s=30, alpha=0.7, label=f"买入 ({len(buys)})")
        if not sells.empty:
            ax.scatter(sells["timestamp"], sells["price"] * sells["shares"],
                       marker="v", color="#e74c3c", s=30, alpha=0.7, label=f"卖出 ({len(sells)})")

        ax.set_title("交易金额时间线", fontsize=11)
        ax.set_ylabel("交易金额 ($)", fontsize=9)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

    # ── 主入口 ──────────────────────────────────────────────────────────────

    def analyze(self, account_id: str) -> tuple[str, str | None]:
        """
        完整分析一个账户。

        Returns:
            (text_report, chart_path)  chart_path 可能为 None
        """
        account_id = account_id.upper().strip()

        # 检查账户是否存在
        conn = self._conn()
        exists = conn.execute(
            "SELECT COUNT(*) FROM accounts WHERE name=?", (account_id,)
        ).fetchone()[0]
        conn.close()

        if not exists:
            return f"❌ 账户 {account_id} 不存在或无数据", None

        # 获取数据
        meta = self.get_account_meta(account_id)
        equity_df = self.get_equity_curve(account_id)
        trades_df = self.get_trades(account_id)
        positions_hist = self.get_positions_history(account_id)
        factors = self.get_factors(account_id)
        adaptive = self.get_adaptive_state(account_id)

        # 初始资金
        initial_cash = 10000.0
        if meta and meta.get("initial_cash"):
            initial_cash = float(meta["initial_cash"])

        # 计算指标
        metrics = self.compute_metrics(equity_df, trades_df, initial_cash)

        # 生成报告
        text = self.generate_text_report(account_id, metrics, factors, adaptive, meta)

        # 生成图表
        chart = self.generate_chart(account_id, equity_df, trades_df, positions_hist, metrics)

        return text, chart

    # ── 账户管理 ────────────────────────────────────────────────────────────

    def list_accounts(self) -> list[dict]:
        """列出所有账户及其最新状态。"""
        conn = self._conn()
        rows = conn.execute("""
            SELECT a.name,
                   MAX(a.timestamp) as last_update,
                   (SELECT equity FROM accounts WHERE name=a.name ORDER BY timestamp DESC LIMIT 1) as equity,
                   (SELECT COUNT(*) FROM trades WHERE account=a.name) as trade_count
            FROM accounts a
            GROUP BY a.name
            ORDER BY a.name
        """).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def delete_account(self, account_id: str, archive: bool = True) -> str:
        """
        删除账户。archive=True 时保留历史数据到 archive_ 表中。
        """
        account_id = account_id.upper().strip()
        conn = sqlite3.connect(self.db_path)

        # 检查存在
        cnt = conn.execute("SELECT COUNT(*) FROM accounts WHERE name=?", (account_id,)).fetchone()[0]
        if cnt == 0:
            conn.close()
            return f"❌ 账户 {account_id} 不存在"

        if archive:
            # 创建归档表（如果不存在）
            conn.execute("""CREATE TABLE IF NOT EXISTS archive_accounts (
                id INTEGER, name TEXT, cash REAL, equity REAL, timestamp TEXT,
                archived_at TEXT)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS archive_trades (
                id INTEGER, account TEXT, ticker TEXT, side TEXT, shares REAL,
                price REAL, cost REAL, slippage REAL, timestamp TEXT, archived_at TEXT)""")

            now = datetime.now(timezone.utc).isoformat()
            conn.execute(f"""INSERT INTO archive_accounts
                SELECT id, name, cash, equity, timestamp, '{now}'
                FROM accounts WHERE name=?""", (account_id,))
            conn.execute(f"""INSERT INTO archive_trades
                SELECT id, account, ticker, side, shares, price, cost, slippage, timestamp, '{now}'
                FROM trades WHERE account=?""", (account_id,))

        # 删除所有表中该账户的数据
        for table, col in [
            ("accounts", "name"), ("trades", "account"), ("positions", "account"),
            ("positions_history", "account"), ("account_state", "account"),
            ("adaptive_state", "account"), ("account_meta", "account_id"),
        ]:
            try:
                conn.execute(f"DELETE FROM {table} WHERE {col}=?", (account_id,))
            except sqlite3.OperationalError:
                pass  # 表可能不存在

        conn.commit()
        conn.close()

        action = "归档并删除" if archive else "永久删除"
        return f"✅ 账户 {account_id} 已{action} ({cnt}条权益快照)"

    def add_account(self, account_id: str, strategy_name: str = "",
                    description: str = "", group: str = "",
                    factors: str = "", initial_cash: float = 10000.0) -> str:
        """注册新账户到 account_meta 表。"""
        account_id = account_id.upper().strip()
        if not group:
            group = "A" if account_id.startswith("A") else "B"

        conn = sqlite3.connect(self.db_path)
        # Ensure account_meta table exists
        _ensure_account_meta(conn)

        exists = conn.execute(
            "SELECT COUNT(*) FROM account_meta WHERE account_id=?", (account_id,)
        ).fetchone()[0]
        if exists:
            conn.close()
            return f"❌ 账户 {account_id} 已存在"

        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO account_meta (account_id, strategy_name, description, \"group\", factors, initial_cash, created_at, status) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (account_id, strategy_name, description, group, factors, initial_cash, now, "active"),
        )
        conn.commit()
        conn.close()
        return f"✅ 账户 {account_id} 已创建 (初始资金 ${initial_cash:,.2f})"


def _ensure_account_meta(conn: sqlite3.Connection):
    """确保 account_meta 表存在。"""
    conn.execute("""CREATE TABLE IF NOT EXISTS account_meta (
        account_id TEXT PRIMARY KEY,
        strategy_name TEXT DEFAULT '',
        description TEXT DEFAULT '',
        "group" TEXT DEFAULT '',
        factors TEXT DEFAULT '',
        initial_cash REAL DEFAULT 10000,
        created_at TEXT,
        status TEXT DEFAULT 'active'
    )""")


def ensure_account_meta_table(db_path: str | None = None):
    """在系统启动时调用，确保 account_meta 表存在。"""
    conn = sqlite3.connect(db_path or DB_PATH)
    _ensure_account_meta(conn)
    conn.commit()
    conn.close()
