"""
Factor Composite Method Comparison
===================================
Compare three composite-scoring methods on the same Alpha158 factor set,
same Russell 1000 universe, same backtest window:

  V0  CURRENT     raw mean of factor values per ticker, then cross-section rank
                  (replicates factors/signal.py logic — has RSI_14 量纲 bug)
  V1  RANK-EW     cross-section rank each factor → equal-weight sum → rank
                  (your proposed clean baseline)
  V2  INDUSTRY    MAD winsorize → log-mktcap+sector neutralize → z-score
                  → ICIR-weighted sum → rank
                  (industry-standard pipeline)

Plus a factor-correlation diagnostic to quantify the ROC_5/10/20 redundancy.

Outputs (under /tmp/hermes/factor_compare/):
  - summary.csv          (IC, ICIR, top-decile sharpe, turnover, etc.)
  - equity_curves.png    (top-10% long-only, $10k start)
  - ic_timeseries.png    (rolling 20d IC for each method)
  - factor_corr.png      (factor correlation heatmap)
  - decile_returns.png   (decile spread for V1 vs V2)
"""
import sqlite3, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------- paths -------
ROOT = Path("/home/gexin/quant-trading")
DB = ROOT / "data/trading.db"
OUT = Path("/tmp/hermes/factor_compare_2026")
OUT.mkdir(parents=True, exist_ok=True)

# ------- config -------
START = "2025-09-01"   # need lookback buffer before 2026-01-01
END   = "2026-05-01"
EVAL_START = "2026-01-01"  # only score IC / equity from this date forward
MIN_DOLLAR_VOL = 5e6
TOP_PCT = 0.10            # top decile (long only)
TC_BPS = 5
INITIAL = 10_000.0
ICIR_WIN = 60             # rolling window for ICIR weights

# Factor list — match factors/alpha_factors.py
WINDOWS = [5, 10, 20]
FACTORS = (
    ["KMID","KLEN","KMID2","KUP","KUP2","KLOW","KLOW2","KSFT","KSFT2"]
    + [f"ROC_{w}"      for w in WINDOWS]
    + [f"MA_RATIO_{w}" for w in WINDOWS]
    + [f"VMOM_{w}"     for w in WINDOWS]
    + [f"VSTD_{w}"     for w in WINDOWS]
    + [f"STD_{w}"      for w in WINDOWS]
    + [f"BBPOS_{w}"    for w in WINDOWS]
    + ["RSV","RSI_14"]
    + [f"BETA_{w}" for w in WINDOWS]
)

# ------- load prices -------
print("Loading prices...")
con = sqlite3.connect(DB)
df = pd.read_sql(
    "SELECT ticker, datetime, open, high, low, close, volume FROM prices "
    "WHERE interval='1d' AND datetime>=? AND datetime<=?",
    con, params=(START, END)
)
con.close()
df["datetime"] = pd.to_datetime(df["datetime"])

# Restrict to Russell 1000 + sector mapped
sector_map = json.load(open(OUT / "sector_map.json"))
df = df[df.ticker.isin(sector_map.keys())]

close  = df.pivot(index="datetime", columns="ticker", values="close").sort_index()
openp  = df.pivot(index="datetime", columns="ticker", values="open").sort_index()
high   = df.pivot(index="datetime", columns="ticker", values="high").sort_index()
low    = df.pivot(index="datetime", columns="ticker", values="low").sort_index()
volume = df.pivot(index="datetime", columns="ticker", values="volume").sort_index()

# Liquidity filter
dvol = (close * volume).mean(axis=0)
liquid = dvol[dvol > MIN_DOLLAR_VOL].index
close, openp, high, low, volume = [x[liquid] for x in (close, openp, high, low, volume)]
print(f"  universe: {close.shape[1]} tickers, {len(close)} days")

# Mock market cap proxy = avg dollar volume (we don't have shares outstanding; this is a stand-in for size neutralize)
log_mktcap_proxy = np.log(dvol[liquid])

rets = close.pct_change(fill_method=None)

# ------- compute factors (vectorized panel) -------
print("Computing factors...")
def safe_div(a, b): return a / b.replace(0, np.nan)

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
F["RSV"]    = (close - low.rolling(9).min()) / (high.rolling(9).max() - low.rolling(9).min() + 1e-12)
delta = close.diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = (-delta.clip(upper=0)).rolling(14).mean()
F["RSI_14"] = 100 - 100 / (1 + gain / (loss + 1e-12))
# BETA — rolling slope normalized
def rolling_slope(panel, w):
    x = np.arange(w, dtype=float); x -= x.mean(); denom = (x*x).sum()
    out = pd.DataFrame(index=panel.index, columns=panel.columns, dtype=float)
    arr = panel.values
    for i in range(w-1, len(panel)):
        y = arr[i-w+1:i+1, :]
        ymean = np.nanmean(y, axis=0)
        ycen = y - ymean
        out.iloc[i] = (x[:,None] * ycen).sum(axis=0) / denom / (ymean + 1e-12)
    return out
for w in WINDOWS:
    F[f"BETA_{w}"] = rolling_slope(close, w)

assert set(F.keys()) == set(FACTORS), set(FACTORS) - set(F.keys())

# ------- factor correlation diagnostic (most recent 60 days, time-stacked) -------
print("Diagnostic: factor correlation...")
recent = close.index[-60:]
stacked = {n: F[n].loc[recent].stack(future_stack=True) for n in FACTORS}
fc = pd.DataFrame(stacked).corr()
fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(fc.values, vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xticks(range(len(FACTORS))); ax.set_xticklabels(FACTORS, rotation=90, fontsize=7)
ax.set_yticks(range(len(FACTORS))); ax.set_yticklabels(FACTORS, fontsize=7)
plt.colorbar(im, ax=ax, fraction=0.04)
ax.set_title(f"Factor Correlation (last {len(recent)} days, stacked panel)")
plt.tight_layout(); plt.savefig(OUT/"factor_corr.png", dpi=120); plt.close()

# Highlight: average pairwise |corr| within groups
groups = {
    "ROC*": ["ROC_5","ROC_10","ROC_20"],
    "MA_RATIO*": ["MA_RATIO_5","MA_RATIO_10","MA_RATIO_20"],
    "BETA*": ["BETA_5","BETA_10","BETA_20"],
    "BBPOS*": ["BBPOS_5","BBPOS_10","BBPOS_20"],
    "STD*":   ["STD_5","STD_10","STD_20"],
}
print("Within-group avg |corr|:")
for g, fs in groups.items():
    sub = fc.loc[fs, fs].abs()
    avg = (sub.values.sum() - len(fs)) / (len(fs)*(len(fs)-1))
    print(f"  {g:12s} n={len(fs)} avg|ρ|={avg:.3f}")

# ------- helpers -------
def cs_rank(panel):
    """Cross-sectional percentile rank, 0..1."""
    return panel.rank(axis=1, pct=True)

def cs_zscore(panel):
    mu = panel.mean(axis=1); sd = panel.std(axis=1)
    return panel.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)

def mad_winsorize(panel, n=5):
    med = panel.median(axis=1)
    mad = (panel.sub(med, axis=0)).abs().median(axis=1)
    upper = med + n*1.4826*mad; lower = med - n*1.4826*mad
    return panel.clip(lower=lower, upper=upper, axis=0)

# Pre-compute neutralization design (sector dummies + log mktcap) — STATIC across time (proxy)
sec_series = pd.Series({t: sector_map[t] for t in close.columns if t in sector_map})
sec_dummies = pd.get_dummies(sec_series).astype(float)
size_vec = log_mktcap_proxy.reindex(sec_dummies.index)
X = pd.concat([size_vec.rename("size"), sec_dummies], axis=1).values  # (N, K)
# Project matrix M = X (X'X)^-1 X' for residualization
XtX_inv = np.linalg.pinv(X.T @ X)
M = X @ XtX_inv @ X.T   # N x N
# residual = (I - M) y for any cross-sectional vector y on these tickers
neutralize_tickers = sec_dummies.index.tolist()

def neutralize_panel(panel):
    """Cross-sectional regression residual on size + sector dummies."""
    p = panel.reindex(columns=neutralize_tickers)
    Y = p.values  # T x N
    # residuals: Y - Y @ M.T (since M symmetric)
    # But M is for the neutralize_tickers columns; fillna with col mean for solve
    col_mean = np.nanmean(Y, axis=1, keepdims=True)
    Yf = np.where(np.isnan(Y), col_mean, Y)
    proj = Yf @ M.T
    res = Y - proj  # keep NaN where original was NaN
    out = pd.DataFrame(res, index=p.index, columns=p.columns)
    return out.reindex(columns=panel.columns)

# ------- composite score builders -------
def composite_v0_current():
    """Replicate factors/signal.py 'composite' bug: per-ticker mean of raw values, then xs-rank."""
    # stack all factor panels, then mean over factor axis per (date, ticker)
    arr = np.stack([F[n].values for n in FACTORS], axis=0)   # (F, T, N)
    mean = np.nanmean(arr, axis=0)                            # (T, N)
    score = pd.DataFrame(mean, index=close.index, columns=close.columns)
    return cs_rank(score)

def composite_v1_rank_ew():
    """Cross-section rank each factor → equal-weight sum → rank."""
    s = None
    for n in FACTORS:
        r = cs_rank(F[n])
        s = r if s is None else s.add(r, fill_value=0)
    return cs_rank(s)

def composite_v2_industry():
    """MAD winsorize → neutralize size+sector → z-score → ICIR-weighted sum."""
    # Pre-process each factor
    Z = {}
    for n in FACTORS:
        p = mad_winsorize(F[n], n=5)
        p = neutralize_panel(p)
        Z[n] = cs_zscore(p)
    # Compute rolling IC for each factor vs forward 1d return
    fwd = rets.shift(-1)
    ICs = {}
    for n in FACTORS:
        ic_t = Z[n].corrwith(fwd, axis=1, method="spearman")
        ICs[n] = ic_t
    IC_df = pd.DataFrame(ICs)
    # Rolling ICIR weights (60d)
    mean_ic = IC_df.rolling(ICIR_WIN, min_periods=20).mean()
    std_ic  = IC_df.rolling(ICIR_WIN, min_periods=20).std()
    icir = (mean_ic / (std_ic + 1e-12)).shift(1)  # use yesterday's ICIR — no look-ahead
    # Combine
    score = None
    for n in FACTORS:
        contrib = Z[n].mul(icir[n], axis=0)
        score = contrib if score is None else score.add(contrib, fill_value=0)
    return cs_rank(score)

print("\nBuilding composites...")
methods = {
    "V0_current":  composite_v0_current(),
    "V1_rank_ew":  composite_v1_rank_ew(),
    "V2_industry": composite_v2_industry(),
}

# ------- evaluation: IC and top-decile long-only -------
fwd = rets.shift(-1)
metrics = []
equity_curves = {}
ic_curves = {}
decile_curves = {}

eval_start_ts = pd.Timestamp(EVAL_START)
for name, score in methods.items():
    # IC time series (rank-IC since score already in [0,1])
    aligned = score.dropna(how="all")
    common = aligned.index.intersection(fwd.index)
    common = common[common >= eval_start_ts]
    ic_ts = aligned.loc[common].corrwith(fwd.loc[common], axis=1, method="spearman").dropna()
    rolling_ic = ic_ts.rolling(20).mean()
    ic_curves[name] = rolling_ic

    # Top-decile long-only
    n_top = max(int(score.shape[1] * TOP_PCT), 5)
    eq = INITIAL
    eq_path = []
    prev_holdings = set()
    daily_rets = []
    for d in common[:-1]:
        s = score.loc[d].dropna()
        if len(s) < n_top * 2:
            eq_path.append((d, eq)); continue
        top = set(s.nlargest(n_top).index)
        # Realize next-day return on equally-weighted top
        r = fwd.loc[d, list(top)].mean(skipna=True)
        if pd.isna(r): r = 0.0
        # Turnover cost
        turnover = len(top.symmetric_difference(prev_holdings)) / max(len(top), 1)
        cost = TC_BPS * 1e-4 * turnover
        eq *= (1 + r - cost)
        eq_path.append((d, eq))
        daily_rets.append(r - cost)
        prev_holdings = top
    eq_df = pd.DataFrame(eq_path, columns=["date","equity"]).set_index("date")
    equity_curves[name] = eq_df

    # Decile spread (top - bottom)
    dec_path = []
    for d in common[:-1]:
        s = score.loc[d].dropna()
        if len(s) < 50: continue
        n_dec = len(s) // 10
        top_r = fwd.loc[d, s.nlargest(n_dec).index].mean()
        bot_r = fwd.loc[d, s.nsmallest(n_dec).index].mean()
        dec_path.append((d, top_r, bot_r))
    dec_df = pd.DataFrame(dec_path, columns=["date","top","bot"]).set_index("date")
    decile_curves[name] = dec_df

    # Metrics
    r_arr = np.array(daily_rets)
    ann_ret = float((eq / INITIAL) ** (252/len(r_arr)) - 1) if len(r_arr) > 0 else 0.0
    ann_vol = float(np.std(r_arr) * np.sqrt(252)) if len(r_arr) > 0 else 0.0
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    eq_series = eq_df["equity"]
    peak = eq_series.cummax(); dd = (eq_series-peak)/peak
    metrics.append({
        "method": name,
        "ic_mean":   round(ic_ts.mean(), 4),
        "ic_std":    round(ic_ts.std(), 4),
        "icir":      round(ic_ts.mean()/ic_ts.std(), 3) if ic_ts.std() > 0 else 0,
        "ic_>0_pct": round((ic_ts > 0).mean()*100, 1),
        "ann_ret_%": round(ann_ret*100, 2),
        "ann_vol_%": round(ann_vol*100, 2),
        "sharpe":    round(sharpe, 2),
        "max_dd_%":  round(dd.min()*100, 2),
        "decile_spread_bps": round((dec_df["top"]-dec_df["bot"]).mean()*1e4, 1),
    })

summary = pd.DataFrame(metrics)
print("\n=== Summary ===")
print(summary.to_string(index=False))
summary.to_csv(OUT/"summary.csv", index=False)

# ------- plots -------
fig, ax = plt.subplots(figsize=(11,6))
for name, eq in equity_curves.items():
    ax.plot(eq.index, eq["equity"], label=name, linewidth=1.6)
ax.axhline(INITIAL, color="gray", lw=0.5, alpha=0.5)
ax.set_title(f"Top-{int(TOP_PCT*100)}% Long-Only Equity ({START}→{END}, {TC_BPS}bps)")
ax.set_ylabel("Equity ($)"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUT/"equity_curves.png", dpi=120); plt.close()

fig, ax = plt.subplots(figsize=(11,5))
for name, ic in ic_curves.items():
    ax.plot(ic.index, ic.values, label=name, linewidth=1.2)
ax.axhline(0, color="gray", lw=0.5)
ax.set_title("Rolling 20-day Rank-IC vs forward 1d return")
ax.set_ylabel("IC"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUT/"ic_timeseries.png", dpi=120); plt.close()

fig, ax = plt.subplots(figsize=(11,5))
for name, dc in decile_curves.items():
    spread = (dc["top"] - dc["bot"]).rolling(20).mean()
    ax.plot(spread.index, spread.values*1e4, label=name, linewidth=1.2)
ax.axhline(0, color="gray", lw=0.5)
ax.set_title("Top-Bottom Decile Daily Return Spread (rolling 20d, bps)")
ax.set_ylabel("bps/day"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUT/"decile_returns.png", dpi=120); plt.close()

print(f"\nArtifacts in {OUT}/")
print("  summary.csv, equity_curves.png, ic_timeseries.png, factor_corr.png, decile_returns.png")
