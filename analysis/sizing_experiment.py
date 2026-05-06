"""
Position-Sizing Experiment
==========================
Same alpha (20-day momentum), same universe, same window, same long-top-N
construction — only the sizing rule changes. Compare four schemes:

  EW          Equal-weight (current system baseline)
  IV          1/sigma weighted (risk parity per name)
  EW+VT       Equal-weight + portfolio-level volatility targeting (12% ann)
  IV+VT       Inverse-vol + volatility targeting

Output: per-scheme equity curve, return, vol, Sharpe, max DD, Calmar, turnover.
Saves a PNG comparison chart and a CSV summary.
"""
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DB = Path("~/quant-trading/data/trading.db").expanduser()
OUT_DIR = Path("/tmp/hermes/sizing_exp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- config ----------
TOP_N = 20
LOOKBACK_MOM = 20            # 20-day momentum signal
VOL_WINDOW = 20              # 20-day realized vol
TARGET_VOL_ANN = 0.12        # 12% annualised portfolio vol
MAX_LEVERAGE = 1.5           # cap when vol-targeting scales up
TC_BPS = 5                   # 5 bps one-way transaction cost
START = "2021-01-01"
END   = "2026-05-01"
INITIAL_CASH = 10_000.0
MIN_DOLLAR_VOL = 5e6         # avg daily $ volume filter for liquidity

# ---------- load data ----------
print("Loading daily prices...")
con = sqlite3.connect(DB)
df = pd.read_sql(
    "SELECT ticker, datetime, close, volume FROM prices "
    "WHERE interval='1d' AND datetime>=? AND datetime<=?",
    con, params=(START, END)
)
con.close()
df["datetime"] = pd.to_datetime(df["datetime"])
print(f"  rows={len(df):,} tickers={df.ticker.nunique()}")

# Pivot to wide
close = df.pivot(index="datetime", columns="ticker", values="close").sort_index()
vol_d = df.pivot(index="datetime", columns="ticker", values="volume").sort_index()

# Liquidity filter: average $ daily volume over full window
dollar_vol = (close * vol_d).mean(axis=0)
liquid = dollar_vol[dollar_vol > MIN_DOLLAR_VOL].index
close = close[liquid]
print(f"  after liquidity filter: {close.shape[1]} tickers, {len(close)} days")

# Returns
rets = close.pct_change()

# ---------- alpha signal: 20-day momentum ----------
mom = close.pct_change(LOOKBACK_MOM)
# realised vol per name (used by IV scheme + signal denoising)
sigma = rets.rolling(VOL_WINDOW).std()

# ---------- backtest loop ----------
def backtest(scheme: str):
    """
    scheme in {EW, IV, EW+VT, IV+VT}.
    Daily rebalance on close, return = next-day open->open held position.
    For simplicity we use close-to-close on next bar and assume signals
    formed at t are held during t+1.
    """
    dates = close.index
    n_days = len(dates)
    weights_yest = pd.Series(0.0, index=close.columns)
    pnl_path = []
    turnover_path = []
    leverage_path = []
    rolling_port_var = 0.0   # exponentially-weighted, for VT
    ewma_lambda = 0.94

    for i in range(LOOKBACK_MOM + VOL_WINDOW + 5, n_days - 1):
        date = dates[i]
        # Rank by momentum (cross-sectional)
        m = mom.iloc[i].dropna()
        s = sigma.iloc[i].dropna()
        valid = m.index.intersection(s.index)
        m = m.loc[valid]
        s = s.loc[valid]
        if len(m) < TOP_N * 2:
            pnl_path.append((date, 0.0))
            continue
        # Top-N by momentum
        top = m.nlargest(TOP_N).index

        # Raw weights per scheme
        if scheme.startswith("EW"):
            raw = pd.Series(1.0 / TOP_N, index=top)
        else:  # IV
            inv = 1.0 / s.loc[top].clip(lower=1e-4)
            raw = inv / inv.sum()

        # Apply vol-targeting at portfolio level
        if scheme.endswith("VT"):
            # Estimate portfolio vol from raw weights * recent name vol
            # (assume zero cross-correlation as crude approx; fine for sizing knob)
            est_port_vol_ann = float(np.sqrt(((raw ** 2) * (s.loc[top] ** 2)).sum()) * np.sqrt(252))
            if est_port_vol_ann > 1e-6:
                lev = TARGET_VOL_ANN / est_port_vol_ann
            else:
                lev = 1.0
            lev = min(lev, MAX_LEVERAGE)
            w = raw * lev
        else:
            lev = 1.0
            w = raw

        # Reindex to full universe
        w_full = pd.Series(0.0, index=close.columns)
        w_full.loc[w.index] = w.values

        # Turnover (sum of absolute weight changes)
        turn = float((w_full - weights_yest).abs().sum())
        # Cost in return space: tc_bps * turnover (one-way per side)
        cost = TC_BPS * 1e-4 * turn

        # PnL = sum(w_t * r_{t+1}) - cost
        r_next = rets.iloc[i + 1]
        pnl = float((w_full * r_next).sum(skipna=True)) - cost

        pnl_path.append((dates[i + 1], pnl))
        turnover_path.append(turn)
        leverage_path.append(lev)
        weights_yest = w_full

    pnl_df = pd.DataFrame(pnl_path, columns=["date", "ret"]).set_index("date")
    pnl_df["equity"] = INITIAL_CASH * (1 + pnl_df["ret"]).cumprod()
    return pnl_df, np.mean(turnover_path), np.mean(leverage_path)

schemes = ["EW", "IV", "EW+VT", "IV+VT"]
results = {}
for sc in schemes:
    print(f"Running {sc}...")
    eq, turn, lev = backtest(sc)
    results[sc] = {"eq": eq, "turn": turn, "lev": lev}

# ---------- metrics ----------
def metrics(eq_df, scheme):
    r = eq_df["ret"].dropna()
    eq = eq_df["equity"]
    n = len(r)
    ann_ret = float((eq.iloc[-1] / INITIAL_CASH) ** (252 / n) - 1) if n > 0 else 0.0
    ann_vol = float(r.std() * np.sqrt(252))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = float(dd.min())
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else float("inf")
    return {
        "scheme": scheme,
        "total_ret_%": round(float(eq.iloc[-1] / INITIAL_CASH - 1) * 100, 2),
        "ann_ret_%": round(ann_ret * 100, 2),
        "ann_vol_%": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_dd_%": round(max_dd * 100, 2),
        "calmar": round(calmar, 2),
        "avg_turnover": round(results[scheme]["turn"], 3),
        "avg_leverage": round(results[scheme]["lev"], 3),
    }

summary = pd.DataFrame([metrics(results[s]["eq"], s) for s in schemes])
print("\n=== Summary ===")
print(summary.to_string(index=False))
summary.to_csv(OUT_DIR / "summary.csv", index=False)

# ---------- plot ----------
fig, axes = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [3, 1]})

ax = axes[0]
for sc in schemes:
    eq = results[sc]["eq"]["equity"]
    ax.plot(eq.index, eq.values, label=sc, linewidth=1.5)
ax.set_title(f"Position-Sizing Comparison  (top-{TOP_N} momentum, {START} → {END}, {TC_BPS}bps cost)")
ax.set_ylabel("Equity ($, start = $10,000)")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)

ax = axes[1]
for sc in schemes:
    eq = results[sc]["eq"]["equity"]
    peak = eq.cummax()
    dd = (eq - peak) / peak * 100
    ax.plot(dd.index, dd.values, label=sc, linewidth=1.0)
ax.set_ylabel("Drawdown (%)")
ax.set_xlabel("Date")
ax.legend(loc="lower left", ncol=4, fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plot_path = OUT_DIR / "sizing_comparison.png"
plt.savefig(plot_path, dpi=120)
print(f"\nChart saved: {plot_path}")
print(f"Summary CSV: {OUT_DIR / 'summary.csv'}")
