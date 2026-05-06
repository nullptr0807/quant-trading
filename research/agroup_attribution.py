"""
A-Group Attribution Backtest (US + CN, 2y + YTD)
==================================================
Replays each A01..A10 strategy on historical price data using the V1 composite
(per-factor cs-rank → equal-weight sum) — fully deterministic, zero look-ahead.

Then for each account × (market, window) computes:
  CAPM:  r_a = α + β·r_mkt + ε
  FF3:   r_a = α + β_mkt·r_mkt + β_smb·r_smb + β_mom·r_mom + ε

Where for each market:
  r_mkt = SPY (US) / 000300.SH (CN), excess over 0 (no rf adjustment for sim simplicity)
  r_smb = small-cap-decile return - large-cap-decile return (size proxy = avg dollar volume)
  r_mom = top-decile-12m-momentum return - bottom-decile

Outputs:
  /tmp/hermes/agroup_attribution/
    summary_capm.csv           — α, β, R², t(α), tracking_err per account×market×window
    summary_ff3.csv            — multi-factor version
    scatter_<market>_<window>.png — α-β scatter, point per account
    equity_<market>_<window>.png  — equity curves of all accounts in that panel
    smb_mom_<market>.png       — sanity check: SMB and MOM time series
"""
import sqlite3, json
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sstats

ROOT = Path("/home/gexin/quant-trading")
DB = ROOT / "data/trading.db"
OUT = Path("/tmp/hermes/agroup_attribution")
OUT.mkdir(parents=True, exist_ok=True)

START = "2024-05-01"
END   = "2026-05-01"
YTD_START = "2026-01-01"
INITIAL = 10_000.0
TC_BPS = 5
MIN_DOLLAR_VOL_US = 5e6
MIN_DOLLAR_VOL_CN = 5e7   # CNY scale

# A-group strategy spec — mirrors accounts/strategies.py
STRATS = [
    ("A01", "动量Alpha",   "momentum",       ["ROC_5","ROC_10","ROC_20","MA_RATIO_5","MA_RATIO_10"], 5),
    ("A02", "均值回归",    "mean_reversion", ["RSV","RSI_14","BBPOS_5","BBPOS_10","BBPOS_20"], 5),
    ("A03", "量价策略",    "momentum",       ["VMOM_5","VMOM_10","VSTD_5","ROC_5","MA_RATIO_5"], 5),
    ("A04", "趋势跟踪",    "momentum",       ["BETA_5","BETA_10","BETA_20","MA_RATIO_10","MA_RATIO_20"], 3),
    ("A05", "波动率突破",  "volatility",     ["STD_5","STD_10","STD_20","BBPOS_5","BBPOS_10"], 5),
    ("A06", "综合多因子",  "composite",      None, 5),
    ("A07", "短期动量",    "momentum",       ["ROC_5","MA_RATIO_5","VMOM_5","KMID"], 8),
    ("A08", "价值+动量",   "momentum",       ["KMID","KLEN","KSFT","ROC_10","ROC_20"], 4),
    ("A09", "反转策略",    "mean_reversion", ["RSI_14","RSV","KSFT","KSFT2","BBPOS_20"], 5),
    ("A10", "自适应策略",  "composite",      ["ROC_10","STD_20","RSI_14","BETA_10","VMOM_10"], 5),
]

WINDOWS = [5, 10, 20]
ALL_FACTORS = (
    ["KMID","KLEN","KMID2","KUP","KUP2","KLOW","KLOW2","KSFT","KSFT2"]
    + [f"ROC_{w}" for w in WINDOWS]
    + [f"MA_RATIO_{w}" for w in WINDOWS]
    + [f"VMOM_{w}" for w in WINDOWS]
    + [f"VSTD_{w}" for w in WINDOWS]
    + [f"STD_{w}" for w in WINDOWS]
    + [f"BBPOS_{w}" for w in WINDOWS]
    + ["RSV","RSI_14"]
    + [f"BETA_{w}" for w in WINDOWS]
)

# ---------- universe loaders ----------
def load_universe(market: str):
    import sys; sys.path.insert(0, str(ROOT))
    from config import settings
    if market == "US":
        return list(settings.STOCK_UNIVERSE), "SPY"
    else:
        return list(settings.CN_UNIVERSE), "000300.SH"

# ---------- factor computation ----------
def compute_factors(close, openp, high, low, volume):
    F = {}
    F["KMID"]  = (close - openp) / openp
    F["KLEN"]  = (high - low) / openp
    F["KMID2"] = (close - openp) / (high - low + 1e-12)
    max_oc = np.maximum(openp, close); min_oc = np.minimum(openp, close)
    F["KUP"]   = (high - max_oc) / openp
    F["KUP2"]  = (high - max_oc) / (high - low + 1e-12)
    F["KLOW"]  = (min_oc - low) / openp
    F["KLOW2"] = (min_oc - low) / (high - low + 1e-12)
    F["KSFT"]  = (2*close - high - low) / openp
    F["KSFT2"] = (2*close - high - low) / (high - low + 1e-12)
    for w in WINDOWS:
        F[f"ROC_{w}"]      = close.pct_change(w, fill_method=None)
        F[f"MA_RATIO_{w}"] = close / close.rolling(w).mean()
        F[f"VMOM_{w}"]     = volume / volume.rolling(w).mean()
        F[f"VSTD_{w}"]     = volume.rolling(w).std() / (volume.rolling(w).mean() + 1e-12)
        F[f"STD_{w}"]      = close.rolling(w).std() / close
        ma = close.rolling(w).mean(); sd = close.rolling(w).std()
        F[f"BBPOS_{w}"]    = (close - ma) / (2*sd + 1e-12)
    F["RSV"] = (close - low.rolling(9).min()) / (high.rolling(9).max() - low.rolling(9).min() + 1e-12)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    F["RSI_14"] = 100 - 100/(1 + gain/(loss+1e-12))
    # BETA — rolling slope
    for w in WINDOWS:
        x = np.arange(w, dtype=float); x -= x.mean(); denom = (x*x).sum()
        out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
        arr = close.values
        for i in range(w-1, len(close)):
            y = arr[i-w+1:i+1, :]
            ymean = np.nanmean(y, axis=0)
            ycen = y - ymean
            out.iloc[i] = (x[:,None] * ycen).sum(axis=0) / denom / (ymean + 1e-12)
        F[f"BETA_{w}"] = out
    return F

# ---------- V1 composite ----------
STRATEGY_FACTORS_DEFAULT = {
    "momentum":       ["ROC_5","ROC_10","ROC_20","MA_RATIO_5","MA_RATIO_10","MA_RATIO_20","BETA_5","BETA_10","BETA_20"],
    "mean_reversion": ["RSV","RSI_14","BBPOS_5","BBPOS_10","BBPOS_20","KSFT","KSFT2"],
    "volatility":     ["STD_5","STD_10","STD_20","BBPOS_5","BBPOS_10","BBPOS_20","VSTD_5","VSTD_10","VSTD_20"],
    "composite":      None,
}

def composite_v1(F: dict, strategy_type: str, factor_names):
    """V1: per-factor cs-rank → equal-weight average → cs-rank again."""
    if factor_names:
        cols = [c for c in factor_names if c in F]
    elif strategy_type == "composite":
        cols = list(F.keys())
    else:
        cols = STRATEGY_FACTORS_DEFAULT[strategy_type]
        cols = [c for c in cols if c in F]
    if not cols:
        return None
    s = None
    for n in cols:
        r = F[n].rank(axis=1, pct=True)
        s = r if s is None else s.add(r, fill_value=0)
    s = s / len(cols)
    if strategy_type == "mean_reversion":
        s = 1 - s
    return s

# ---------- replay ----------
def replay_strategy(score: pd.DataFrame, rets: pd.DataFrame, top_n: int, max_pos_pct=None):
    """Daily rebalance, equal-weight top_n, return daily-pct + equity curve."""
    daily_r = []
    prev = set()
    weights_cap = (1.0/top_n) if max_pos_pct is None else min(1.0/top_n, max_pos_pct)
    cash_weight = 1 - weights_cap * top_n
    for d in score.index[:-1]:
        s = score.loc[d].dropna()
        if len(s) < top_n*2:
            daily_r.append((d, 0.0)); continue
        top = s.nlargest(top_n).index
        try:
            r_next = rets.loc[score.index[score.index.get_loc(d)+1], top].mean(skipna=True)
        except (IndexError, KeyError):
            daily_r.append((d, 0.0)); continue
        if pd.isna(r_next): r_next = 0.0
        turnover = len(set(top).symmetric_difference(prev)) / max(top_n, 1)
        cost = TC_BPS * 1e-4 * turnover
        port_r = (weights_cap*top_n) * r_next - cost   # cash earns 0
        daily_r.append((d, port_r))
        prev = set(top)
    df = pd.DataFrame(daily_r, columns=["date","ret"]).set_index("date")
    df["equity"] = INITIAL * (1 + df["ret"]).cumprod()
    return df

# ---------- attribution ----------
def capm_attribution(r_acc: pd.Series, r_mkt: pd.Series):
    df = pd.concat([r_acc.rename("a"), r_mkt.rename("m")], axis=1).dropna()
    if len(df) < 30:
        return None
    x = df["m"].values; y = df["a"].values
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    n, k = len(y), 2
    rss = (resid**2).sum(); tss = ((y-y.mean())**2).sum()
    r2 = 1 - rss/tss if tss > 0 else 0
    sigma2 = rss/(n-k)
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t_alpha = beta[0] / se[0] if se[0] > 0 else 0
    t_beta  = beta[1] / se[1] if se[1] > 0 else 0
    te = float(resid.std() * np.sqrt(252))
    return dict(alpha_ann=float(beta[0]*252), beta=float(beta[1]),
                t_alpha=float(t_alpha), t_beta=float(t_beta),
                r2=float(r2), tracking_err=te, n=n)

def ff3_attribution(r_acc, r_mkt, r_smb, r_mom):
    df = pd.concat([r_acc.rename("a"), r_mkt.rename("m"), r_smb.rename("s"), r_mom.rename("o")], axis=1).dropna()
    if len(df) < 30: return None
    X = np.column_stack([np.ones(len(df)), df["m"], df["s"], df["o"]])
    y = df["a"].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta; resid = y - yhat
    n, k = len(y), 4
    rss = (resid**2).sum(); tss = ((y-y.mean())**2).sum()
    r2 = 1 - rss/tss if tss>0 else 0
    sigma2 = rss/(n-k)
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = beta / np.where(se>0, se, 1)
    return dict(alpha_ann=float(beta[0]*252),
                b_mkt=float(beta[1]), b_smb=float(beta[2]), b_mom=float(beta[3]),
                t_alpha=float(t[0]), t_mkt=float(t[1]), t_smb=float(t[2]), t_mom=float(t[3]),
                r2=float(r2), n=n)

# ---------- per-market run ----------
def run_market(market: str):
    print(f"\n{'='*60}\nMarket: {market}\n{'='*60}")
    universe, bench = load_universe(market)
    min_dvol = MIN_DOLLAR_VOL_US if market == "US" else MIN_DOLLAR_VOL_CN

    print(f"  universe size: {len(universe)}, benchmark: {bench}")
    con = sqlite3.connect(DB)
    placeholders = ",".join("?"*len(universe))
    df = pd.read_sql(
        f"SELECT ticker, datetime, open, high, low, close, volume FROM prices "
        f"WHERE interval='1d' AND datetime>=? AND datetime<=? AND ticker IN ({placeholders})",
        con, params=[START, END]+universe
    )
    bench_df = pd.read_sql(
        "SELECT datetime, close FROM prices WHERE ticker=? AND interval='1d' AND datetime>=? AND datetime<=?",
        con, params=[bench, START, END]
    )
    con.close()
    df["datetime"] = pd.to_datetime(df["datetime"])
    bench_df["datetime"] = pd.to_datetime(bench_df["datetime"])
    bench_r = bench_df.set_index("datetime")["close"].pct_change()

    close  = df.pivot(index="datetime", columns="ticker", values="close").sort_index()
    openp  = df.pivot(index="datetime", columns="ticker", values="open").sort_index()
    high   = df.pivot(index="datetime", columns="ticker", values="high").sort_index()
    low    = df.pivot(index="datetime", columns="ticker", values="low").sort_index()
    volume = df.pivot(index="datetime", columns="ticker", values="volume").sort_index()

    dvol = (close*volume).mean(axis=0)
    liquid = dvol[dvol > min_dvol].index
    close, openp, high, low, volume = [x[liquid] for x in (close, openp, high, low, volume)]
    print(f"  after liquidity filter: {close.shape[1]} tickers, {len(close)} days")

    rets = close.pct_change(fill_method=None)
    print("  computing factors...")
    F = compute_factors(close, openp, high, low, volume)

    # --- build SMB and MOM factor returns ---
    # SMB = small (bottom 30% by dollar vol) - large (top 30%)
    dvol_liq = dvol.reindex(close.columns).dropna()
    rk_size = dvol_liq.rank(pct=True)
    small = rk_size[rk_size <= 0.3].index.intersection(rets.columns)
    large = rk_size[rk_size >= 0.7].index.intersection(rets.columns)
    r_smb = rets[small].mean(axis=1) - rets[large].mean(axis=1)

    # MOM = high 12m momentum minus low (rebalanced monthly cs-rank for simplicity, daily rebalance)
    mom_raw = close.pct_change(252, fill_method=None)
    mom_rank = mom_raw.rank(axis=1, pct=True)
    mom_long  = (mom_rank > 0.7)
    mom_short = (mom_rank < 0.3)
    r_mom = (rets * mom_long).sum(axis=1) / mom_long.sum(axis=1).replace(0, np.nan) \
          - (rets * mom_short).sum(axis=1) / mom_short.sum(axis=1).replace(0, np.nan)

    # --- replay each strategy ---
    print("  replaying A01-A10...")
    accounts = {}
    for sid, name, stype, fnames, top_n in STRATS:
        score = composite_v1(F, stype, fnames)
        if score is None:
            print(f"    {sid}: skipped (no factors)"); continue
        eq = replay_strategy(score, rets, top_n, max_pos_pct=0.25)
        accounts[sid] = eq

    return dict(market=market, bench=bench, accounts=accounts,
                bench_r=bench_r, r_smb=r_smb, r_mom=r_mom)

# ---------- main ----------
results = {m: run_market(m) for m in ["US", "CN"]}

print("\nRunning attribution...")
rows_capm, rows_ff3 = [], []
for window_name, w_start in [("2y", START), ("YTD", YTD_START)]:
    w_start_ts = pd.Timestamp(w_start)
    for market, R in results.items():
        for sid, eq in R["accounts"].items():
            r_a = eq["ret"][eq.index >= w_start_ts]
            r_m = R["bench_r"][R["bench_r"].index >= w_start_ts]
            r_s = R["r_smb"][R["r_smb"].index >= w_start_ts]
            r_o = R["r_mom"][R["r_mom"].index >= w_start_ts]
            cap = capm_attribution(r_a, r_m)
            ff3 = ff3_attribution(r_a, r_m, r_s, r_o)
            if cap:
                rows_capm.append({"market":market, "account":sid, "window":window_name, **cap})
            if ff3:
                rows_ff3.append({"market":market, "account":sid, "window":window_name, **ff3})

capm_df = pd.DataFrame(rows_capm)
ff3_df  = pd.DataFrame(rows_ff3)

# Round for display
for c in ["alpha_ann","beta","t_alpha","t_beta","r2","tracking_err"]:
    if c in capm_df: capm_df[c] = capm_df[c].round(4)
for c in ["alpha_ann","b_mkt","b_smb","b_mom","t_alpha","t_mkt","t_smb","t_mom","r2"]:
    if c in ff3_df: ff3_df[c] = ff3_df[c].round(4)

capm_df.to_csv(OUT/"summary_capm.csv", index=False)
ff3_df.to_csv(OUT/"summary_ff3.csv", index=False)

print("\n=== CAPM ===")
print(capm_df.to_string(index=False))
print("\n=== FF3 ===")
print(ff3_df.to_string(index=False))

# ---------- plots ----------
print("\nGenerating plots...")
# α-β scatter, one per (market, window)
for market in ["US","CN"]:
    for window_name in ["2y","YTD"]:
        sub = capm_df[(capm_df.market==market) & (capm_df.window==window_name)]
        if sub.empty: continue
        fig, ax = plt.subplots(figsize=(8,6))
        sig = sub["t_alpha"].abs() > 2
        ax.scatter(sub["beta"], sub["alpha_ann"]*100, c=sig.map({True:"red",False:"steelblue"}), s=80, alpha=0.7, edgecolors="black")
        for _, r in sub.iterrows():
            ax.annotate(r["account"], (r["beta"], r["alpha_ann"]*100), xytext=(5,5), textcoords="offset points", fontsize=9)
        ax.axhline(0, color="gray", lw=0.5); ax.axvline(1, color="gray", lw=0.5, linestyle="--")
        ax.set_xlabel("Beta"); ax.set_ylabel("Alpha (annualized %)")
        bench = "SPY" if market=="US" else "CSI300"
        ax.set_title(f"{market} A-Group α-β  ({window_name}, vs {bench})\nred = |t(α)|>2 (significant)")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT/f"scatter_{market}_{window_name}.png", dpi=120); plt.close()

# Equity curves per market×window
for market in ["US","CN"]:
    for window_name, w_start in [("2y",START),("YTD",YTD_START)]:
        fig, ax = plt.subplots(figsize=(11,6))
        w_ts = pd.Timestamp(w_start)
        for sid, eq in results[market]["accounts"].items():
            sub = eq[eq.index >= w_ts]
            if len(sub) < 5: continue
            renorm = sub["equity"] / sub["equity"].iloc[0] * INITIAL
            ax.plot(sub.index, renorm, label=sid, linewidth=1.2, alpha=0.8)
        # Add benchmark
        b = results[market]["bench_r"][results[market]["bench_r"].index >= w_ts]
        bench_eq = INITIAL * (1+b).cumprod()
        ax.plot(bench_eq.index, bench_eq.values, label=f"BENCH ({results[market]['bench']})", color="black", linewidth=2.0, linestyle="--")
        bench = "SPY" if market=="US" else "CSI300"
        ax.set_title(f"{market} A-Group Equity Curves ({window_name})")
        ax.set_ylabel("Equity ($/¥)"); ax.legend(loc="best", ncol=2, fontsize=8); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT/f"equity_{market}_{window_name}.png", dpi=120); plt.close()

print(f"\nArtifacts in {OUT}/")
