"""Compare XGBoost objectives: reg:squarederror (RMSE) vs rank:pairwise (ranking).

Same hparams, same Alpha158 features, same walk-forward folds. For each fold:
  - Train both models
  - Compute Rank IC & ICIR on test segment
  - Build top-decile long-only and top-vs-bottom long-short portfolios
    (rebalanced daily, equal-weighted, ~10% of universe per side)
  - Annualize returns and compute Sharpe

This is the experiment that connects model choice → actual PnL.
"""
from __future__ import annotations
import logging, sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/gexin/quant-trading")
logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("tune_obj")
log.setLevel(logging.INFO)

import qlib
from qlib.constant import REG_US
qlib.init(provider_uri="/home/gexin/.qlib/qlib_data/us_data", region=REG_US)
from qlib.data import D
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
import xgboost as xgb

# Same booster hparams for both — only `objective` differs.
COMMON = dict(
    eval_metric="rmse",     # used only for early-stop monitoring
    max_depth=8,
    learning_rate=0.05,
    colsample_bytree=0.9,
)
NUM_BOOST_ROUND = 200
EARLY_STOP = 30

OBJECTIVES = {
    "rmse": dict(objective="reg:squarederror"),
    "rank": dict(objective="rank:pairwise"),
}

# Walk-forward folds
cal = D.calendar(freq="day")
N = len(cal)
TRAIN, VALID, TEST = 500, 60, 40
N_FOLDS = 8
STEP = 60

folds = []
end_idx = N
for f in range(N_FOLDS):
    test_end = end_idx
    test_start = test_end - TEST
    valid_end = test_start
    valid_start = valid_end - VALID
    train_end = valid_start
    train_start = train_end - TRAIN
    if train_start < 0:
        break
    folds.append(dict(
        fold=N_FOLDS - 1 - f,
        train=(cal[train_start].date(), cal[train_end - 1].date()),
        valid=(cal[valid_start].date(), cal[valid_end - 1].date()),
        test=(cal[test_start].date(), cal[test_end - 1].date()),
    ))
    end_idx -= STEP
folds = list(reversed(folds))
log.info("folds: %d", len(folds))


def _daily_group_sizes(idx) -> list[int]:
    """For ranking objectives: list of group sizes per unique day, in index order."""
    dates = idx.get_level_values(0)
    # preserve order, count consecutive same-date runs
    out = []
    last = None
    n = 0
    for d in dates:
        if d == last:
            n += 1
        else:
            if last is not None:
                out.append(n)
            last = d
            n = 1
    if last is not None:
        out.append(n)
    return out


def run_fold(fold, obj_kwargs, label_objective: str):
    ts, te = fold["train"]; vs, ve = fold["valid"]; tts, tte = fold["test"]
    handler = Alpha158(
        start_time=str(ts), end_time=str(tte),
        fit_start_time=str(ts), fit_end_time=str(te),
        instruments="all",
    )
    dataset = DatasetH(handler=handler, segments={
        "train": (str(ts), str(te)),
        "valid": (str(vs), str(ve)),
        "test":  (str(tts), str(tte)),
    })
    df_train, df_valid = dataset.prepare(
        ["train", "valid"], col_set=["feature", "label"],
        data_key=DataHandlerLP.DK_L,
    )
    x_tr, y_tr = df_train["feature"], df_train["label"]
    x_va, y_va = df_valid["feature"], df_valid["label"]
    # Sort by datetime so groups are contiguous
    idx_tr = x_tr.index; idx_va = x_va.index
    order_tr = idx_tr.sortlevel(0)[1]
    order_va = idx_va.sortlevel(0)[1]
    x_tr = x_tr.iloc[order_tr]; y_tr = y_tr.iloc[order_tr]
    x_va = x_va.iloc[order_va]; y_va = y_va.iloc[order_va]

    dtr = xgb.DMatrix(x_tr.values, label=np.squeeze(y_tr.values))
    dva = xgb.DMatrix(x_va.values, label=np.squeeze(y_va.values))
    if label_objective == "rank":
        dtr.set_group(_daily_group_sizes(x_tr.index))
        dva.set_group(_daily_group_sizes(x_va.index))

    params = dict(COMMON); params.update(obj_kwargs)
    t0 = time.time()
    booster = xgb.train(
        params, dtr,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dtr, "train"), (dva, "valid")],
        early_stopping_rounds=EARLY_STOP,
        verbose_eval=False,
    )
    fit_s = time.time() - t0

    # Predict on test
    df_test = dataset.prepare("test", col_set=["feature", "label"],
                              data_key=DataHandlerLP.DK_L)
    x_te, y_te = df_test["feature"], df_test["label"]
    pred = pd.Series(booster.predict(xgb.DMatrix(x_te.values)),
                     index=x_te.index, name="score")
    # Drop NaN labels
    label_col = y_te.columns[0] if isinstance(y_te, pd.DataFrame) else None
    label_series = y_te[label_col] if label_col is not None else y_te
    df = pd.concat([pred, label_series.rename("label")], axis=1).dropna()

    # Daily Rank IC
    daily_ic = df.groupby(level=0).apply(
        lambda g: g["score"].rank().corr(g["label"].rank()) if len(g) > 5 else np.nan
    ).dropna()

    # Top-decile long-only + long-short PnL (equal-weighted, daily rebal)
    def _bucket_returns(g):
        n = len(g)
        if n < 20:
            return pd.Series({"top": np.nan, "bot": np.nan, "ls": np.nan})
        k = max(1, n // 10)
        top = g.nlargest(k, "score")["label"].mean()
        bot = g.nsmallest(k, "score")["label"].mean()
        return pd.Series({"top": top, "bot": bot, "ls": top - bot})

    daily_ret = df.groupby(level=0).apply(_bucket_returns)

    return dict(
        fit_s=round(fit_s, 1),
        n_test_days=len(daily_ic),
        mean_ic=float(daily_ic.mean()),
        ic_std=float(daily_ic.std()),
        icir=float(daily_ic.mean() / daily_ic.std()) if daily_ic.std() > 0 else np.nan,
        # Top decile (long-only)
        top_mean_daily=float(daily_ret["top"].mean()),
        top_ann=float(daily_ret["top"].mean() * 252),
        top_sharpe=float(daily_ret["top"].mean() / daily_ret["top"].std() * np.sqrt(252))
                   if daily_ret["top"].std() > 0 else np.nan,
        # Long-short
        ls_mean_daily=float(daily_ret["ls"].mean()),
        ls_ann=float(daily_ret["ls"].mean() * 252),
        ls_sharpe=float(daily_ret["ls"].mean() / daily_ret["ls"].std() * np.sqrt(252))
                  if daily_ret["ls"].std() > 0 else np.nan,
    )


results = {name: [] for name in OBJECTIVES}
for fold in folds:
    for name, obj_kwargs in OBJECTIVES.items():
        log.info("[fold %d / %s] running...", fold["fold"], name)
        r = run_fold(fold, obj_kwargs, label_objective=name)
        r["fold"] = fold["fold"]
        results[name].append(r)
        log.info("  → IC=%.4f ICIR=%.3f | top_ann=%.2f%% (Sh=%.2f) | LS_ann=%.2f%% (Sh=%.2f)",
                 r["mean_ic"], r["icir"],
                 r["top_ann"]*100, r["top_sharpe"],
                 r["ls_ann"]*100, r["ls_sharpe"])


def agg(rs, k):
    vs = [r[k] for r in rs if not (isinstance(r[k], float) and np.isnan(r[k]))]
    return float(np.mean(vs)) if vs else float("nan")


print("\n" + "="*88)
print(f"{'fold':>4} | {'RMSE IC':>8} {'ICIR':>7} {'top%':>7} {'LS%':>7} | "
      f"{'rank IC':>8} {'ICIR':>7} {'top%':>7} {'LS%':>7}")
print("-"*88)
for i in range(len(folds)):
    r1 = results["rmse"][i]; r2 = results["rank"][i]
    print(f"{r1['fold']:>4} | "
          f"{r1['mean_ic']:>8.4f} {r1['icir']:>7.3f} "
          f"{r1['top_ann']*100:>7.2f} {r1['ls_ann']*100:>7.2f} | "
          f"{r2['mean_ic']:>8.4f} {r2['icir']:>7.3f} "
          f"{r2['top_ann']*100:>7.2f} {r2['ls_ann']*100:>7.2f}")
print("-"*88)
print(f"{'AVG':>4} | "
      f"{agg(results['rmse'],'mean_ic'):>8.4f} {agg(results['rmse'],'icir'):>7.3f} "
      f"{agg(results['rmse'],'top_ann')*100:>7.2f} {agg(results['rmse'],'ls_ann')*100:>7.2f} | "
      f"{agg(results['rank'],'mean_ic'):>8.4f} {agg(results['rank'],'icir'):>7.3f} "
      f"{agg(results['rank'],'top_ann')*100:>7.2f} {agg(results['rank'],'ls_ann')*100:>7.2f}")
print(f"{'    ':>4} | {'':<32} | top_Sh={agg(results['rank'],'top_sharpe'):.2f}  "
      f"LS_Sh={agg(results['rank'],'ls_sharpe'):.2f}")
print(f"{'    ':>4} | top_Sh={agg(results['rmse'],'top_sharpe'):.2f}  "
      f"LS_Sh={agg(results['rmse'],'ls_sharpe'):.2f}")
print("="*88)
print("Note: top% / LS% are ANNUALIZED returns (×252); "
      "top = top decile long-only, LS = top - bottom decile (long-short)")

out = Path("/tmp/q02_rmse_vs_rank.json")
folds_serial = [{"fold": f["fold"],
                 "train":[str(x) for x in f["train"]],
                 "valid":[str(x) for x in f["valid"]],
                 "test":[str(x) for x in f["test"]]} for f in folds]
out.write_text(json.dumps({"folds": folds_serial, "results": results}, indent=2))
print(f"\nsaved: {out}")
