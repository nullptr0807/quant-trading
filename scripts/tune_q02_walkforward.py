"""Walk-forward comparison: Q02 baseline vs proposed XGBoost hparams.

5 folds, each train=180 / valid=30 / test=20, rolling forward by 20.
Computes daily Rank IC on test segment, then ICIR & mean IC across folds.
"""
from __future__ import annotations
import logging, os, sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = "/home/gexin/quant-trading"
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("tune_q02")
log.setLevel(logging.INFO)

import qlib
from qlib.constant import REG_US
qlib.init(provider_uri="/home/gexin/.qlib/qlib_data/us_data", region=REG_US)
from qlib.data import D
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from factors.qlib_signal import XGBModelFixed as XGBModel

PARAM_SETS = {
    "baseline": dict(
        eval_metric="rmse", colsample_bytree=0.9, max_depth=8,
        learning_rate=0.05, num_boost_round=200, early_stopping_rounds=30,
    ),
    "proposed": dict(
        eval_metric="rmse", max_depth=5, learning_rate=0.03,
        num_boost_round=500, early_stopping_rounds=50,
        colsample_bytree=0.7, subsample=0.8,
        reg_lambda=2.0, min_child_weight=10,
    ),
}

cal = D.calendar(freq="day")
N = len(cal)
TRAIN, VALID, TEST = 500, 60, 40
N_FOLDS = 8
STEP = 60

# Build folds: last fold ends at cal[-1]; roll backwards by STEP.
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
for f in folds:
    log.info("  fold %d: train=%s..%s valid=%s..%s test=%s..%s",
             f["fold"], *f["train"], *f["valid"], *f["test"])


def run_fold(fold, kwargs):
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
    model = XGBModel(**kwargs)
    t0 = time.time()
    model.fit(dataset)
    fit_s = time.time() - t0
    pred = model.predict(dataset)
    if hasattr(pred, "to_frame"):
        pred = pred.to_frame("score")

    # Get test labels
    test_df = dataset.prepare("test", col_set=["label"], data_key="raw")
    # test_df columns are MultiIndex; pick label
    label_col = test_df.columns[0]
    labels = test_df[label_col]
    df = pd.concat([pred["score"], labels.rename("label")], axis=1).dropna()

    # Daily Rank IC
    daily_ic = df.groupby(level=0).apply(
        lambda g: g["score"].rank().corr(g["label"].rank()) if len(g) > 5 else np.nan
    ).dropna()
    return dict(
        fit_s=round(fit_s, 1),
        n_test_days=len(daily_ic),
        mean_ic=float(daily_ic.mean()),
        ic_std=float(daily_ic.std()),
        icir=float(daily_ic.mean() / daily_ic.std()) if daily_ic.std() > 0 else np.nan,
        pos_ic_pct=float((daily_ic > 0).mean()),
    )


results = {name: [] for name in PARAM_SETS}
for fold in folds:
    for name, kwargs in PARAM_SETS.items():
        log.info("[fold %d / %s] running...", fold["fold"], name)
        r = run_fold(fold, kwargs)
        r["fold"] = fold["fold"]
        results[name].append(r)
        log.info("  → mean_ic=%.4f icir=%.3f pos%%=%.1f%% (%.1fs)",
                 r["mean_ic"], r["icir"], r["pos_ic_pct"]*100, r["fit_s"])

print("\n" + "="*72)
print(f"{'fold':>4} | {'baseline IC':>12} {'baseline ICIR':>14} | {'proposed IC':>12} {'proposed ICIR':>14}")
print("-"*72)
for i in range(len(folds)):
    b = results["baseline"][i]; p = results["proposed"][i]
    print(f"{b['fold']:>4} | {b['mean_ic']:>12.4f} {b['icir']:>14.3f} | {p['mean_ic']:>12.4f} {p['icir']:>14.3f}")
print("-"*72)
def agg(rs, k):
    vals = [r[k] for r in rs if not np.isnan(r[k])]
    return float(np.mean(vals)) if vals else float("nan")
print(f"{'AVG':>4} | {agg(results['baseline'],'mean_ic'):>12.4f} {agg(results['baseline'],'icir'):>14.3f} | {agg(results['proposed'],'mean_ic'):>12.4f} {agg(results['proposed'],'icir'):>14.3f}")
print("="*72)
out = Path("/tmp/q02_walkforward.json")
folds_serial = [{"fold": f["fold"], "train":[str(x) for x in f["train"]], "valid":[str(x) for x in f["valid"]], "test":[str(x) for x in f["test"]]} for f in folds]
out.write_text(json.dumps({"folds": folds_serial, "results": results}, indent=2, default=str))
print(f"\nsaved: {out}")
