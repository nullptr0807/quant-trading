"""Qlib-based daily ranking signal: train 10 different models on Alpha158
features, output per-ticker scores for top-N selection by Q01-Q10 accounts.

Architecture (memory-isolated): each model is trained in its own subprocess
via scripts/qlib_retrain.py — this module is *imported* by that subprocess
to do one model's worth of work and exit. The orchestrator never holds more
than one trained model in RAM.

Output: predictions written to `factor_values` table:
    factor_group = "qlib"
    factor_name = f"qlib_{model_id}_score"   # e.g. qlib_Q08_score
    PRIMARY KEY includes (ticker, date, factor_name, factor_group), so Qlib,
    GP, FactorMiner, and Alpha158 rows cannot overwrite each other.
    one row per (ticker, date, model) — typically the latest trading day.

Reads: ~/.qlib/qlib_data/us_data (built by factors/qlib_export.py).
"""
from __future__ import annotations

import argparse
import gc
import logging
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = os.path.expanduser("~/quant-trading")
sys.path.insert(0, PROJECT_ROOT)

DB_PATH = os.path.join(PROJECT_ROOT, "data", "trading.db")
QLIB_US_DIR = os.path.expanduser("~/.qlib/qlib_data/us_data")
QLIB_CN_DIR = os.path.expanduser("~/.qlib/qlib_data/cn_data")

log = logging.getLogger("qlib_signal")


# ─── XGBoost wrapper fix ─────────────────────────────────────────────────────
# qlib's XGBModel pushes ALL __init__ kwargs into self._params and passes
# them to xgb.train(self._params, ...) as booster params. But
# `num_boost_round` and `early_stopping_rounds` are FUNCTION arguments of
# xgb.train, not booster params — xgboost silently ignores them and emits:
#   "WARNING: Parameters: { num_boost_round, early_stopping_rounds } are
#    not used."
# Result: our num_boost_round=200 / early_stopping_rounds=30 in MODEL_SPECS
# were no-ops. xgboost fell back to its own defaults.
#
# Fix: subclass that pops these into fit() kwargs where they belong.
class XGBModelFixed:
    """Drop-in replacement for qlib.contrib.model.xgboost.XGBModel that
    correctly routes num_boost_round / early_stopping_rounds to xgb.train()
    as function args instead of booster params.

    Also supports XGBoost ranking objectives (rank:pairwise, rank:ndcg, etc.)
    via auto-grouping by datetime: each trading day = one ranking group.
    Ranking outperforms RMSE for top-N stock selection (validated on
    Alpha158 walk-forward: ~+27% mean IC, ~+65% top-decile Sharpe).
    """
    _XGB_TRAIN_KWARGS = {"num_boost_round", "early_stopping_rounds",
                         "verbose_eval", "evals_result"}

    def __init__(self, **kwargs):
        # Separate xgb.train function-args from booster params
        self._train_kwargs = {k: kwargs.pop(k) for k in list(kwargs)
                              if k in self._XGB_TRAIN_KWARGS}
        self._params = dict(kwargs)
        self.model = None

    @staticmethod
    def _is_ranking(params: dict) -> bool:
        obj = params.get("objective", "")
        return isinstance(obj, str) and obj.startswith("rank:")

    @staticmethod
    def _daily_group_sizes(idx) -> list:
        """For ranking: list of group sizes per consecutive same-date run."""
        dates = idx.get_level_values(0)
        out, last, n = [], None, 0
        for d in dates:
            if d == last:
                n += 1
            else:
                if last is not None:
                    out.append(n)
                last, n = d, 1
        if last is not None:
            out.append(n)
        return out

    def fit(self, dataset, **fit_kwargs):
        train_kwargs = dict(self._train_kwargs)
        train_kwargs.update({k: fit_kwargs.pop(k) for k in list(fit_kwargs)
                             if k in self._XGB_TRAIN_KWARGS})
        train_kwargs.setdefault("num_boost_round", 1000)
        train_kwargs.setdefault("early_stopping_rounds", 50)
        train_kwargs.setdefault("verbose_eval", 20)

        import numpy as np
        import pandas as pd
        import xgboost as xgb
        from qlib.data.dataset.handler import DataHandlerLP

        df_train, df_valid = dataset.prepare(
            ["train", "valid"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        x_train, y_train = df_train["feature"], df_train["label"]
        x_valid, y_valid = df_valid["feature"], df_valid["label"]

        is_ranking = self._is_ranking(self._params)
        if is_ranking:
            # Sort by datetime so per-day groups are contiguous
            order_tr = x_train.index.sortlevel(0)[1]
            order_va = x_valid.index.sortlevel(0)[1]
            x_train = x_train.iloc[order_tr]; y_train = y_train.iloc[order_tr]
            x_valid = x_valid.iloc[order_va]; y_valid = y_valid.iloc[order_va]

        y_train_1d = np.squeeze(y_train.values)
        y_valid_1d = np.squeeze(y_valid.values)
        dtrain = xgb.DMatrix(x_train.values, label=y_train_1d)
        dvalid = xgb.DMatrix(x_valid.values, label=y_valid_1d)

        # Build daily group sizes for both ranking AND custom IC metric
        # (IC is computed per-day, so we always need the day boundaries)
        groups_tr = self._daily_group_sizes(x_train.index)
        groups_va = self._daily_group_sizes(x_valid.index)
        if is_ranking:
            dtrain.set_group(groups_tr)
            dvalid.set_group(groups_va)

        # Custom Rank IC eval metric — daily Spearman, averaged.
        # Higher = better, so we pass maximize=True so early-stopping uses IC
        # instead of RMSE. Critical for ranking objectives where train-RMSE
        # actually goes UP as the pairwise loss improves the ordering.
        groups_by_id = {id(dtrain): groups_tr, id(dvalid): groups_va}

        def daily_rank_ic(preds: np.ndarray, dmat: xgb.DMatrix):
            labels = dmat.get_label()
            grp = groups_by_id.get(id(dmat))
            if grp is None:
                # Fallback: treat all as one group
                ic = pd.Series(preds).rank().corr(pd.Series(labels).rank())
                return "ic", float(ic) if not np.isnan(ic) else 0.0
            ics = []
            i = 0
            for g in grp:
                if g >= 5:
                    p = pd.Series(preds[i:i+g]).rank()
                    l = pd.Series(labels[i:i+g]).rank()
                    c = p.corr(l)
                    if not np.isnan(c):
                        ics.append(c)
                i += g
            return "ic", float(np.mean(ics)) if ics else 0.0

        evals_result = train_kwargs.pop("evals_result", {})
        self.model = xgb.train(
            self._params,
            dtrain=dtrain,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            evals_result=evals_result,
            custom_metric=daily_rank_ic,
            maximize=True,                # IC: higher is better
            **train_kwargs,
        )

    def predict(self, dataset, segment="test"):
        import xgboost as xgb
        from qlib.data.dataset.handler import DataHandlerLP
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare(segment, col_set="feature",
                                 data_key=DataHandlerLP.DK_I)
        import pandas as pd
        return pd.Series(self.model.predict(xgb.DMatrix(x_test)), index=x_test.index)


# ─── Model specs ─────────────────────────────────────────────────────────────

@dataclass
class ModelSpec:
    """One model's train/predict recipe."""
    id: str                      # account id, e.g. "Q01"
    name: str                    # human label, e.g. "LightGBM"
    model_class: str             # qlib import path
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    needs_processors: bool = False    # True for NN/linear (need Fillna/Standardize)
    feature_set: str = "Alpha158"     # 'Alpha158' (158 features) or 'Alpha360' (6×60 OHLCV)


# Conservative hyperparams tuned for CPU + 4GB RAM + 4GB swap.
# Q01-Q05 use Alpha158 (rich hand-crafted features), Q06-Q10 use Alpha360
# because Qlib's RNN/Transformer/TCN forward pass requires reshape to (T, F)
# and only Alpha360's 6×60 layout matches that contract.
MODEL_SPECS: list[ModelSpec] = [
    ModelSpec(
        id="Q01", name="LightGBM",
        model_class="qlib.contrib.model.gbdt.LGBModel",
        model_kwargs=dict(
            loss="mse", learning_rate=0.05, num_leaves=128,
            feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
            num_boost_round=200, early_stopping_rounds=30,
        ),
    ),
    ModelSpec(
        id="Q02", name="XGBoost",
        model_class="factors.qlib_signal.XGBModelFixed",
        model_kwargs=dict(
            # Switched from reg:squarederror to rank:pairwise (LambdaMART).
            # Walk-forward validated 2026-05: ~+27% mean Rank IC,
            # ~+65% top-decile Sharpe vs RMSE objective. Top-N selection
            # cares about ordering, not absolute return prediction.
            #
            # XGBModelFixed installs a custom daily-Rank-IC eval metric +
            # maximize=True, so early-stopping uses IC (not RMSE, which
            # rises as the pairwise loss improves).
            objective="rank:pairwise",
            # eval_metric set to rmse to suppress XGBoost's auto-pick of
            # NDCG (which requires integer labels — our labels are
            # CSZScoreNorm floats and would crash). Custom daily-Rank-IC
            # is appended AFTER, and xgb early-stopping uses the LAST
            # metric, so IC drives early-stop. RMSE is just a logging
            # noise column here.
            eval_metric="rmse",
            colsample_bytree=0.9, max_depth=8,
            learning_rate=0.05, num_boost_round=200,
            early_stopping_rounds=30,
        ),
    ),
    ModelSpec(
        id="Q03", name="CatBoost",
        model_class="qlib.contrib.model.catboost_model.CatBoostModel",
        model_kwargs=dict(
            loss_function="RMSE", learning_rate=0.05, depth=8,
            iterations=200, early_stopping_rounds=30,
            # NB: do NOT pass verbose / verbose_eval / silent / logging_level —
            # qlib's wrapper sets one internally and CatBoost rejects duplicates.
        ),
    ),
    ModelSpec(
        id="Q04", name="Ridge",
        model_class="qlib.contrib.model.linear.LinearModel",
        model_kwargs=dict(estimator="ridge", alpha=0.05),
        needs_processors=True,
    ),
    ModelSpec(
        id="Q05", name="MLP",
        model_class="qlib.contrib.model.pytorch_nn.DNNModelPytorch",
        model_kwargs=dict(
            lr=1e-3, max_steps=2000, batch_size=2048,
            early_stop_rounds=50, eval_steps=100, optimizer="adam", loss="mse",
            GPU=0,  # qlib uses 0 == CPU when cuda unavailable
            pt_model_kwargs={"input_dim": 158, "layers": (128, 64)},
        ),
        needs_processors=True,
    ),
    ModelSpec(
        id="Q06", name="LSTM",
        model_class="qlib.contrib.model.pytorch_lstm.LSTM",
        model_kwargs=dict(
            d_feat=6, hidden_size=32, num_layers=1, dropout=0.0,
            n_epochs=8, lr=2e-3, early_stop=4, batch_size=2048,
            metric="loss", loss="mse", GPU=0,
        ),
        needs_processors=True,
        feature_set="Alpha360",
    ),
    ModelSpec(
        id="Q07", name="GRU",
        model_class="qlib.contrib.model.pytorch_gru.GRU",
        model_kwargs=dict(
            d_feat=6, hidden_size=32, num_layers=1, dropout=0.0,
            n_epochs=8, lr=2e-3, early_stop=4, batch_size=2048,
            metric="loss", loss="mse", GPU=0,
        ),
        needs_processors=True,
        feature_set="Alpha360",
    ),
    ModelSpec(
        id="Q08", name="Transformer",
        model_class="qlib.contrib.model.pytorch_transformer.TransformerModel",
        model_kwargs=dict(
            d_feat=6, d_model=32, nhead=2, num_layers=1, dropout=0.0,
            n_epochs=6, lr=2e-3, early_stop=3, batch_size=512,
            reg=1e-3,
            metric="loss", loss="mse", GPU=0,
        ),
        needs_processors=True,
        feature_set="Alpha360",
    ),
    ModelSpec(
        id="Q09", name="TCN",
        model_class="qlib.contrib.model.pytorch_tcn.TCN",
        model_kwargs=dict(
            d_feat=6, num_layers=3, n_chans=16, kernel_size=3, dropout=0.0,
            n_epochs=8, lr=2e-3, early_stop=4, batch_size=2048,
            metric="loss", loss="mse", GPU=0,
        ),
        needs_processors=True,
        feature_set="Alpha360",
    ),
    ModelSpec(
        id="Q10", name="ALSTM",
        model_class="qlib.contrib.model.pytorch_alstm.ALSTM",
        model_kwargs=dict(
            d_feat=6, hidden_size=32, num_layers=1, dropout=0.0,
            n_epochs=8, lr=2e-3, early_stop=4, batch_size=2048,
            metric="loss", loss="mse", GPU=0, rnn_type="GRU",
        ),
        needs_processors=True,
        feature_set="Alpha360",
    ),
]


def get_spec(model_id: str) -> ModelSpec:
    for s in MODEL_SPECS:
        if s.id == model_id:
            return s
    raise KeyError(f"unknown model_id: {model_id!r}")


# ─── Train / predict for a single model ──────────────────────────────────────

def _import_class(path: str):
    """qlib.contrib.model.gbdt.LGBModel → LGBModel class."""
    mod_path, cls_name = path.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)


def build_handler(start_time: str, end_time: str,
                  fit_start_time: str, fit_end_time: str,
                  market: str = "US",
                  needs_processors: bool = False,
                  feature_set: str = "Alpha158"):
    """Build a feature handler.

    feature_set:
      - 'Alpha158' (default): 158 hand-crafted factors. Suits GBDT / linear / MLP.
      - 'Alpha360' : 6 OHLCV features × 60 lookback days. Required for the
                     Qlib RNN family (LSTM/GRU/ALSTM/Transformer/TCN), whose
                     forward pass reshapes input to (T, F).

    NN models (incl. MLP) need infer/learn processors so NaN cells are
    filled (GBDT handle NaN natively).
    """
    if feature_set == "Alpha360":
        from qlib.contrib.data.handler import Alpha360
        HCls = Alpha360
    else:
        from qlib.contrib.data.handler import Alpha158
        HCls = Alpha158
    kwargs = dict(
        start_time=start_time, end_time=end_time,
        fit_start_time=fit_start_time, fit_end_time=fit_end_time,
        instruments="all",
    )
    if needs_processors:
        kwargs.update(dict(
            infer_processors=[
                {"class": "RobustZScoreNorm",
                 "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                {"class": "Fillna",
                 "kwargs": {"fields_group": "feature"}},
            ],
            learn_processors=[
                {"class": "DropnaLabel"},
                {"class": "CSRankNorm",
                 "kwargs": {"fields_group": "label"}},
            ],
        ))
    return HCls(**kwargs)


def init_qlib(market: str = "US"):
    import qlib
    from qlib.constant import REG_US, REG_CN
    region = REG_CN if market == "CN" else REG_US
    provider_uri = QLIB_CN_DIR if market == "CN" else QLIB_US_DIR

    # Qlib's default MLflow experiment manager writes under ./mlruns. When daily
    # US/CN retrains or retries overlap, the shared tracking store has produced
    # intermittent `sqlite3.OperationalError: database is locked` failures. The
    # model artifacts we actually depend on are saved separately under
    # data/qlib_checkpoints, so give each subprocess an isolated temp MLflow URI.
    tracking_uri = os.environ.get(
        "QLIB_MLFLOW_URI",
        f"file:/tmp/quant_qlib_mlruns/{market.lower()}_{os.getpid()}",
    )
    exp_manager = {
        "class": "MLflowExpManager",
        "module_path": "qlib.workflow.expm",
        "kwargs": {
            "uri": tracking_uri,
            "default_exp_name": f"QuantQlib-{market}",
        },
    }
    qlib.init(provider_uri=provider_uri, region=region, exp_manager=exp_manager)


def train_and_predict(spec: ModelSpec,
                      market: str = "US",
                      train_days: int = 360,
                      valid_days: int = 60,
                      predict_days: int = 5,
                      return_artifacts: bool = False):
    """Train one model on rolling window, predict latest scores.

    Window layout (T = today, exclusive end):
        train: [T - train_days - valid_days - predict_days, T - valid_days - predict_days)
        valid: [T - valid_days - predict_days,              T - predict_days)
        test:  [T - predict_days,                           T]

    Returns DataFrame indexed by (datetime, instrument), single 'score' column.
    If return_artifacts=True returns (pred, model, dataset, train_window) so
    the caller can save a frozen checkpoint.
    """
    init_qlib(market)
    import pandas as pd
    from datetime import datetime, timedelta

    # Use latest calendar date as T (qlib's calendar = our trading days)
    from qlib.data import D
    cal = D.calendar(freq="day")
    T = cal[-1]  # last available trading day
    test_start = cal[max(0, len(cal) - predict_days)]
    valid_end = cal[max(0, len(cal) - predict_days - 1)]
    valid_start = cal[max(0, len(cal) - predict_days - valid_days)]
    train_end = cal[max(0, len(cal) - predict_days - valid_days - 1)]
    train_start = cal[max(0, len(cal) - predict_days - valid_days - train_days)]

    log.info("[%s/%s] train=[%s..%s] valid=[%s..%s] test=[%s..%s]",
             spec.id, spec.name,
             train_start.date(), train_end.date(),
             valid_start.date(), valid_end.date(),
             test_start.date(), T.date())

    handler = build_handler(
        start_time=str(train_start.date()),
        end_time=str(T.date()),
        fit_start_time=str(train_start.date()),
        fit_end_time=str(train_end.date()),
        market=market,
        needs_processors=spec.needs_processors,
        feature_set=spec.feature_set,
    )

    from qlib.data.dataset import DatasetH
    # All models in this spec use DatasetH (not TSDatasetH). The Qlib
    # RNN/Transformer/TCN family reshape Alpha360's 6×60 layout internally
    # (forward: x.reshape(N, d_feat, -1)) — they expect a flat feature dim.
    dataset = DatasetH(
        handler=handler,
        segments={
            "train": (str(train_start.date()), str(train_end.date())),
            "valid": (str(valid_start.date()), str(valid_end.date())),
            "test":  (str(test_start.date()),  str(T.date())),
        },
    )

    Cls = _import_class(spec.model_class)
    model = Cls(**spec.model_kwargs)
    t0 = time.time()
    # Several qlib PyTorch models call R.get_recorder() inside fit().  Without
    # an explicit experiment context they fail with "No valid experiment has
    # been found" even when qlib.init() configured an exp_manager.  Wrap all
    # fits in R.start() so GBDT/linear models share the same safe path and
    # recorder-hungry GRU/Transformer/ALSTM models always have a context.
    from qlib.workflow import R
    exp_name = f"QuantQlib-{market}"
    rec_name = f"{spec.id}_{int(time.time())}_{os.getpid()}"
    with R.start(experiment_name=exp_name, recorder_name=rec_name):
        model.fit(dataset)
    log.info("[%s] fit: %.1fs", spec.id, time.time() - t0)

    t0 = time.time()
    pred = model.predict(dataset)
    log.info("[%s] predict: %.2fs", spec.id, time.time() - t0)

    # pred is a Series indexed by (datetime, instrument)
    if hasattr(pred, "to_frame"):
        pred = pred.to_frame("score")
    pred.columns = ["score"]
    if return_artifacts:
        train_window = {
            "train": (str(train_start.date()), str(train_end.date())),
            "valid": (str(valid_start.date()), str(valid_end.date())),
            "test":  (str(test_start.date()),  str(T.date())),
        }
        return pred, model, dataset, train_window
    return pred


# ─── Persistence ─────────────────────────────────────────────────────────────

def write_predictions_to_db(spec: ModelSpec, pred, market: str = "US") -> int:
    """Write per-(ticker,date) score rows into factor_values.

    Existing schema (PK = ticker, date, factor_name; factor_group is a
    label-only column), so we put the model id in factor_name to avoid
    PK collisions across models:
        factor_name  = f"qlib_{spec.id}_score"   e.g. 'qlib_Q01_score'
        factor_group = 'qlib'
    """
    if pred is None or pred.empty:
        log.warning("[%s] empty prediction, nothing to write", spec.id)
        return 0
    factor_name = f"qlib_{spec.id}_score"
    factor_group = "qlib"
    rows = []
    # pred index: (datetime, instrument)
    for (dt, tk), row in pred.iterrows():
        score = float(row["score"])
        if score != score:  # NaN
            continue
        date_str = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)[:10]
        rows.append((tk, date_str, factor_name, score, factor_group))

    if not rows:
        log.warning("[%s] all scores NaN", spec.id)
        return 0

    for attempt in range(5):
        conn = sqlite3.connect(DB_PATH, timeout=60)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=60000")
            conn.executemany(
                "INSERT OR REPLACE INTO factor_values "
                "(ticker, date, factor_name, value, factor_group) VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()
            break
        except sqlite3.OperationalError as e:
            conn.rollback()
            if "locked" not in str(e).lower() or attempt == 4:
                raise
            wait = min(2 ** attempt, 10)
            log.warning("[%s] factor_values locked while writing qlib scores; retry %d/5 in %ss",
                        spec.id, attempt + 1, wait)
            time.sleep(wait)
        finally:
            conn.close()
    log.info("[%s] wrote %d rows to factor_values factor_name=%s",
             spec.id, len(rows), factor_name)
    return len(rows)


# ─── Entry point ─────────────────────────────────────────────────────────────

def run_one_model(model_id: str, market: str = "US",
                  train_days: int = 360, valid_days: int = 60,
                  predict_days: int = 5,
                  save_checkpoint: bool = True) -> dict:
    """Top-level: train + predict + persist for ONE model id. Used by
    scripts/qlib_retrain.py via subprocess.

    When save_checkpoint=True (default) we also freeze the model + fitted
    handler/dataset to ~/quant-trading/data/qlib_checkpoints/<market>/<id>/<date>.pkl
    so future backtests can replay without re-training (zero look-ahead).
    """
    spec = get_spec(model_id)
    t0 = time.time()
    pred, model, dataset, train_window = train_and_predict(
        spec, market=market,
        train_days=train_days, valid_days=valid_days,
        predict_days=predict_days,
        return_artifacts=True,
    )
    n = write_predictions_to_db(spec, pred, market=market)
    elapsed = time.time() - t0
    summary = {
        "model_id": spec.id, "name": spec.name,
        "rows": n, "elapsed_s": round(elapsed, 1),
        "market": market,
    }

    # Freeze a point-in-time checkpoint so future backtests can replay
    # this exact (model, processors) without seeing future data.
    if save_checkpoint:
        try:
            from factors.qlib_checkpoint import save_checkpoint as _save_ckpt
            ckpt_meta = _save_ckpt(
                spec=spec, model=model, dataset=dataset, pred=pred,
                market=market,
                train_window=train_window,
                elapsed_s=round(elapsed, 1),
            )
            summary["checkpoint"] = {
                "saved": ckpt_meta.get("saved", True) is not False,
                "pkl_bytes": ckpt_meta.get("pkl_bytes"),
                "date": ckpt_meta.get("date"),
            }
        except Exception as e:
            log.warning("[%s] checkpoint save failed (predictions still wrote): %s",
                        spec.id, e)
            summary["checkpoint"] = {"saved": False, "error": str(e)}

    log.info("[%s] DONE: %s", spec.id, summary)
    # Help RSS drop before subprocess exits
    del pred, model, dataset
    gc.collect()
    return summary


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="model id (Q01..Q10)")
    p.add_argument("--market", default="US", choices=["US", "CN"])
    p.add_argument("--train-days", type=int, default=360)
    p.add_argument("--valid-days", type=int, default=60)
    p.add_argument("--predict-days", type=int, default=5)
    args = p.parse_args()
    run_one_model(args.model, market=args.market,
                  train_days=args.train_days, valid_days=args.valid_days,
                  predict_days=args.predict_days)


if __name__ == "__main__":
    main()
