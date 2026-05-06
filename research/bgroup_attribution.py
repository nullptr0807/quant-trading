"""
B-Group Walk-Forward Attribution Backtest
==========================================
Walk-forward GP factor mining + replay + CAPM/FF3 attribution.

Per quarter, for each market:
  1. Train GP miner on previous 252 days → produces 20 factors
  2. Apply factors to next quarter (frozen) → daily composite scores
  3. Three account personalities replay on those scores

Quarters: 2024Q3 → 2026Q2 (8 quarters, 2 years walk-forward).

Outputs to /tmp/hermes/bgroup_attribution/:
  summary_capm.csv, summary_ff3.csv
  scatter_<market>_<window>.png
  equity_<market>_<window>.png
  gp_factor_meta.json   — which factors were mined per quarter, with IC
"""
import sqlite3, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, "/home/gexin/quant-trading")
from factors.gp_miner import GPAlphaMiner, _compute_features, _eval_expression, FEATURE_COLS

ROOT = Path("/home/gexin/quant-trading")
DB = ROOT / "data/trading.db"
OUT = Path("/tmp/hermes/bgroup_attribution")
OUT.mkdir(parents=True, exist_ok=True)

START = "2024-07-01"   # incl. 1y of training history before first quarter
END   = "2026-05-01"
YTD_START = "2026-01-01"
INITIAL = 10_000.0
TC_BPS = 5
MIN_DOLLAR_VOL_US = 5e6
MIN_DOLLAR_VOL_CN = 5e7

# Walk-forward quarter starts (predict period starts)
QUARTERS = [
    ("2024Q3", "2024-07-01", "2024-09-30"),
    ("2024Q4", "2024-10-01", "2024-12-31"),
    ("2025Q1", "2025-01-01", "2025-03-31"),
    ("2025Q2", "2025-04-01", "2025-06-30"),
    ("2025Q3", "2025-07-01", "2025-09-30"),
    ("2025Q4", "2025-10-01", "2025-12-31"),
    ("2026Q1", "2026-01-01", "2026-03-31"),
    ("2026Q2", "2026-04-01", "2026-05-01"),
]

# Account "personalities" sharing the per-quarter GP factor pool
PERSONALITIES = [
    ("B-Top5",       dict(top_n=5,  y_target="next_1d_ret")),
    ("B-Top10-3d",   dict(top_n=10, y_target="next_3d_ret")),
    ("B-Reversal",   dict(top_n=8,  y_target="reversal_2d")),
]

# GP mining params (moderate cost)
MINER_KW = dict(n_factors=20, generations=15, population_size=200, n_runs=3, base_seed=42)


def load_market(market):
    sys.path.insert(0, str(ROOT))
    from config import settings
    if market == "US":
        return list(settings.STOCK_UNIVERSE), "SPY", MIN_DOLLAR_VOL_US
    return list(settings.CN_UNIVERSE), "000300.SH", MIN_DOLLAR_VOL_CN


def mine_quarter(historical_data, y_target):
    """Run GP miner on training window, return list of factor dicts."""
    miner = GPAlphaMiner()
    factors = miner.mine_factors(historical_data, y_target=y_target, **MINER_KW)
    return factors


def apply_factors(historical_data, factors):
    """Compute factor values for each ticker. Returns {factor_name: panel(date×ticker)}."""
    if not factors:
        return {}
    panels = {f["name"]: {} for f in factors}
    for ticker, df in historical_data.items():
        feat = _compute_features(df)
        valid_idx = feat.dropna().index
        if len(valid_idx) < 1:
            continue
        for f_info in factors:
            cols = f_info.get("feature_cols") or FEATURE_COLS
            try:
                X_tick = feat.loc[valid_idx, cols].values
                col = _eval_expression(f_info["expression"], X_tick)
                s = pd.Series(col, index=valid_idx)
                panels[f_info["name"]][ticker] = s
            except Exception:
                continue
    out = {}
    for name, d in panels.items():
        if d:
            out[name] = pd.DataFrame(d)
    return out


def composite_score(factor_panels):
    """V1: per-factor cs-rank → equal-weight average."""
    if not factor_panels: return None
    s = None
    for name, panel in factor_panels.items():
        # IC-direction: not flipping per-IC sign — equal-weight neutral combine.
        r = panel.rank(axis=1, pct=True)
        s = r if s is None else s.add(r, fill_value=0)
    return s / len(factor_panels)


def replay_strategy(score, rets, top_n):
    daily = []
    prev = set()
    for d in score.index[:-1]:
        s = score.loc[d].dropna()
        if len(s) < top_n*2:
            daily.append((d, 0.0)); continue
        top = s.nlargest(top_n).index
        try:
            r_next = rets.loc[score.index[score.index.get_loc(d)+1], top].mean(skipna=True)
        except (IndexError, KeyError):
            daily.append((d, 0.0)); continue
        if pd.isna(r_next): r_next = 0.0
        turnover = len(set(top).symmetric_difference(prev)) / max(top_n, 1)
        cost = TC_BPS*1e-4*turnover
        daily.append((d, r_next - cost))
        prev = set(top)
    df = pd.DataFrame(daily, columns=["date","ret"]).set_index("date")
    df["equity"] = INITIAL * (1 + df["ret"]).cumprod()
    return df


def capm_attribution(r_acc, r_mkt):
    df = pd.concat([r_acc.rename("a"), r_mkt.rename("m")], axis=1).dropna()
    if len(df) < 30: return None
    x = df["m"].values; y = df["a"].values
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta; resid = y - yhat
    n, k = len(y), 2
    rss = (resid**2).sum(); tss = ((y-y.mean())**2).sum()
    r2 = 1 - rss/tss if tss>0 else 0
    sigma2 = rss/(n-k)
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    return dict(alpha_ann=float(beta[0]*252), beta=float(beta[1]),
                t_alpha=float(beta[0]/se[0] if se[0]>0 else 0),
                t_beta=float(beta[1]/se[1] if se[1]>0 else 0),
                r2=float(r2), tracking_err=float(resid.std()*np.sqrt(252)), n=n)


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
    return dict(alpha_ann=float(beta[0]*252), b_mkt=float(beta[1]), b_smb=float(beta[2]), b_mom=float(beta[3]),
                t_alpha=float(t[0]), t_mkt=float(t[1]), t_smb=float(t[2]), t_mom=float(t[3]),
                r2=float(r2), n=n)


def run_market(market):
    print(f"\n{'='*60}\nMarket: {market}\n{'='*60}")
    universe, bench, min_dvol = load_market(market)
    con = sqlite3.connect(DB)
    placeholders = ",".join("?"*len(universe))
    df = pd.read_sql(
        f"SELECT ticker, datetime, open, high, low, close, volume FROM prices "
        f"WHERE interval='1d' AND datetime>=? AND datetime<=? AND ticker IN ({placeholders})",
        con, params=[START, END]+universe,
    )
    bench_df = pd.read_sql(
        "SELECT datetime, close FROM prices WHERE ticker=? AND interval='1d' AND datetime>=? AND datetime<=?",
        con, params=[bench, START, END]
    )
    con.close()
    df["datetime"] = pd.to_datetime(df["datetime"])
    bench_df["datetime"] = pd.to_datetime(bench_df["datetime"])
    bench_r = bench_df.set_index("datetime")["close"].pct_change()

    close = df.pivot(index="datetime", columns="ticker", values="close").sort_index()
    volume = df.pivot(index="datetime", columns="ticker", values="volume").sort_index()
    dvol = (close*volume).mean(axis=0)
    liquid = dvol[dvol > min_dvol].index
    df = df[df.ticker.isin(liquid)]

    close_l = df.pivot(index="datetime", columns="ticker", values="close").sort_index()
    vol_l = df.pivot(index="datetime", columns="ticker", values="volume").sort_index()
    rets = close_l.pct_change(fill_method=None)
    print(f"  liquid universe: {close_l.shape[1]} tickers, {len(close_l)} days")

    # SMB / MOM
    dvol_liq = dvol.reindex(close_l.columns).dropna()
    rk = dvol_liq.rank(pct=True)
    small = rk[rk <= 0.3].index.intersection(rets.columns)
    large = rk[rk >= 0.7].index.intersection(rets.columns)
    r_smb = rets[small].mean(axis=1) - rets[large].mean(axis=1)
    mom_rank = close_l.pct_change(252, fill_method=None).rank(axis=1, pct=True)
    mom_long = mom_rank > 0.7; mom_short = mom_rank < 0.3
    r_mom = ((rets*mom_long).sum(axis=1)/mom_long.sum(axis=1).replace(0, np.nan)) \
          - ((rets*mom_short).sum(axis=1)/mom_short.sum(axis=1).replace(0, np.nan))

    # Per-ticker historical_data dict (reused by GP miner)
    historical_data = {t: g.set_index("datetime")[["open","high","low","close","volume"]]
                       for t, g in df.groupby("ticker")}

    # Per-personality cumulative score, glued across quarters
    score_panels = {p[0]: [] for p in PERSONALITIES}
    factor_meta = {p[0]: [] for p in PERSONALITIES}

    for qname, q_start, q_end in QUARTERS:
        train_end = pd.Timestamp(q_start) - pd.Timedelta(days=1)
        train_start = train_end - pd.Timedelta(days=365)
        print(f"  [{market} {qname}] train {train_start.date()} → {train_end.date()},  predict {q_start} → {q_end}")
        # Build training subset
        train_data = {}
        for tk, d in historical_data.items():
            sub = d.loc[(d.index >= train_start) & (d.index <= train_end)]
            if len(sub) > 100:
                train_data[tk] = sub
        if len(train_data) < 50:
            print("    skipped: insufficient training data"); continue

        # Predict subset (factors applied to predict window — use training window for stable lookback context too)
        predict_start_ts = pd.Timestamp(q_start)
        predict_end_ts = pd.Timestamp(q_end)
        predict_data = {tk: d.loc[d.index <= predict_end_ts] for tk, d in historical_data.items()}

        for pname, pcfg in PERSONALITIES:
            t0 = time.time()
            factors = mine_quarter(train_data, y_target=pcfg["y_target"])
            print(f"    {pname}: mined {len(factors)} factors in {time.time()-t0:.1f}s, top IC={factors[0]['ic']:.3f}" if factors else f"    {pname}: NO factors")
            if not factors: continue
            factor_meta[pname].append({"quarter": qname, "n_factors": len(factors),
                                       "top_ic": float(factors[0]["ic"]),
                                       "expressions": [f["expression"][:120] for f in factors[:5]]})
            panels = apply_factors(predict_data, factors)
            score = composite_score(panels)
            if score is None: continue
            # Slice to predict window only (avoid leak from training period overlap)
            score = score.loc[(score.index >= predict_start_ts) & (score.index <= predict_end_ts)]
            score_panels[pname].append(score)

    # Concatenate all quarters per personality, replay strategy
    accounts = {}
    for pname, pcfg in PERSONALITIES:
        if not score_panels[pname]: continue
        full_score = pd.concat(score_panels[pname]).sort_index()
        full_score = full_score[~full_score.index.duplicated(keep="first")]
        eq = replay_strategy(full_score, rets, pcfg["top_n"])
        accounts[pname] = eq

    # Save factor metadata
    with open(OUT/f"factor_meta_{market}.json", "w") as f:
        json.dump(factor_meta, f, indent=2, default=str)

    return dict(market=market, bench=bench, accounts=accounts,
                bench_r=bench_r, r_smb=r_smb, r_mom=r_mom)


# ---------- main ----------
results = {m: run_market(m) for m in ["US","CN"]}

print("\nRunning attribution...")
rows_capm, rows_ff3 = [], []
for window_name, w_start in [("walkfwd", "2024-07-01"), ("YTD", YTD_START)]:
    w_ts = pd.Timestamp(w_start)
    for market, R in results.items():
        for sid, eq in R["accounts"].items():
            r_a = eq["ret"][eq.index >= w_ts]
            r_m = R["bench_r"][R["bench_r"].index >= w_ts]
            r_s = R["r_smb"][R["r_smb"].index >= w_ts]
            r_o = R["r_mom"][R["r_mom"].index >= w_ts]
            cap = capm_attribution(r_a, r_m)
            ff3 = ff3_attribution(r_a, r_m, r_s, r_o)
            if cap: rows_capm.append({"market": market, "account": sid, "window": window_name, **cap})
            if ff3: rows_ff3.append({"market": market, "account": sid, "window": window_name, **ff3})

capm_df = pd.DataFrame(rows_capm)
ff3_df = pd.DataFrame(rows_ff3)
for c in ["alpha_ann","beta","t_alpha","t_beta","r2","tracking_err"]:
    if c in capm_df: capm_df[c] = capm_df[c].round(4)
for c in ["alpha_ann","b_mkt","b_smb","b_mom","t_alpha","t_mkt","t_smb","t_mom","r2"]:
    if c in ff3_df: ff3_df[c] = ff3_df[c].round(4)

capm_df.to_csv(OUT/"summary_capm.csv", index=False)
ff3_df.to_csv(OUT/"summary_ff3.csv", index=False)

print("\n=== CAPM ==="); print(capm_df.to_string(index=False))
print("\n=== FF3 ==="); print(ff3_df.to_string(index=False))

# Plots
print("\nGenerating plots...")
for market in ["US","CN"]:
    for window_name in ["walkfwd","YTD"]:
        sub = capm_df[(capm_df.market==market) & (capm_df.window==window_name)]
        if sub.empty: continue
        fig, ax = plt.subplots(figsize=(8,6))
        sig = sub["t_alpha"].abs() > 2
        ax.scatter(sub["beta"], sub["alpha_ann"]*100, c=sig.map({True:"red",False:"steelblue"}), s=120, alpha=0.7, edgecolors="black")
        for _, r in sub.iterrows():
            ax.annotate(r["account"], (r["beta"], r["alpha_ann"]*100), xytext=(5,5), textcoords="offset points", fontsize=10)
        ax.axhline(0, color="gray", lw=0.5); ax.axvline(1, color="gray", lw=0.5, linestyle="--")
        ax.set_xlabel("Beta"); ax.set_ylabel("Alpha (annualized %)")
        bench = "SPY" if market=="US" else "CSI300"
        ax.set_title(f"{market} B-Group α-β  ({window_name}, vs {bench})\nred = |t(α)|>2 (significant)")
        ax.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(OUT/f"scatter_{market}_{window_name}.png", dpi=120); plt.close()

for market in ["US","CN"]:
    for window_name, w_start in [("walkfwd","2024-07-01"),("YTD",YTD_START)]:
        fig, ax = plt.subplots(figsize=(11,6))
        w_ts = pd.Timestamp(w_start)
        for sid, eq in results[market]["accounts"].items():
            sub = eq[eq.index >= w_ts]
            if len(sub) < 5: continue
            renorm = sub["equity"] / sub["equity"].iloc[0] * INITIAL
            ax.plot(sub.index, renorm, label=sid, linewidth=1.6, alpha=0.85)
        b = results[market]["bench_r"][results[market]["bench_r"].index >= w_ts]
        bench_eq = INITIAL * (1+b).cumprod()
        ax.plot(bench_eq.index, bench_eq.values, label=f"BENCH ({results[market]['bench']})", color="black", linewidth=2.0, linestyle="--")
        ax.set_title(f"{market} B-Group Walk-Forward Equity ({window_name})")
        ax.set_ylabel("Equity"); ax.legend(loc="best", fontsize=9); ax.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(OUT/f"equity_{market}_{window_name}.png", dpi=120); plt.close()

print(f"\nDone. Artifacts in {OUT}/")
