"""Generate hourly trading reports with text summaries and equity curve charts."""

import os
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import sys
sys.path.insert(0, os.path.expanduser("~/quant-trading"))
from data.store import DataStore


class ReportGenerator:
    """Creates hourly text + chart reports from account data."""

    def __init__(self, db_path: str | None = None):
        self.store = DataStore(db_path)

    def generate_hourly_report(self, accounts_data: list[dict]) -> tuple[str, str]:
        """Generate an hourly report.

        Parameters
        ----------
        accounts_data : list[dict]
            Each dict has keys: account_id, strategy, factors (list[str]),
            positions (list[dict] with ticker, allocation_pct, pnl).

        Returns
        -------
        tuple[str, str]
            (text_summary, chart_path)
        """
        now = datetime.utcnow()

        # -- Compute total PnL per account and sort descending --
        for acct in accounts_data:
            acct["total_pnl"] = sum(p.get("pnl", 0.0) for p in acct.get("positions", []))
        sorted_accounts = sorted(accounts_data, key=lambda a: a["total_pnl"], reverse=True)

        # -- Build text summary --
        lines = [
            f"\U0001F4CA  Hourly Report  {now.strftime('%Y-%m-%d %H:%M UTC')}",
            "=" * 42,
        ]
        for i, acct in enumerate(sorted_accounts, 1):
            pnl = acct["total_pnl"]
            emoji = "\U0001F7E2" if pnl >= 0 else "\U0001F534"
            lines.append(f"\n{emoji} #{i}  {acct.get('strategy', acct['account_id'])}")
            lines.append(f"   Factors: {', '.join(acct.get('factors', ['-']))}")
            lines.append(f"   PnL: {pnl:+,.2f}")
            positions = acct.get("positions", [])
            if positions:
                lines.append("   Positions:")
                for pos in positions:
                    lines.append(
                        f"     {pos['ticker']:>6s}  "
                        f"alloc {pos.get('allocation_pct', 0):5.1f}%  "
                        f"pnl {pos.get('pnl', 0):+,.2f}"
                    )
        lines.append("\n" + "=" * 42)
        text = "\n".join(lines)

        # -- Generate equity curve chart --
        chart_path = self._generate_chart(accounts_data, now)

        return text, chart_path

    def _generate_chart(self, accounts_data: list[dict], now: datetime) -> str:
        fig, ax = plt.subplots(figsize=(10, 5))

        has_data = False
        for acct in accounts_data:
            aid = acct.get("account_id", acct.get("strategy", "unknown"))
            df = self.store.get_account_history(aid)
            if df.empty:
                continue
            has_data = True
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
            ax.plot(df["timestamp"], df["equity"], label=aid, linewidth=1.4)

        if not has_data:
            ax.text(0.5, 0.5, "No equity data yet", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14)

        ax.set_title("Equity Curves")
        ax.set_xlabel("Time")
        ax.set_ylabel("Equity ($)")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()

        chart_path = f"/tmp/equity_report_{now.strftime('%Y%m%d_%H%M%S')}.png"
        fig.savefig(chart_path, dpi=120)
        plt.close(fig)
        return chart_path


# Need pandas for chart generation
import pandas as pd
