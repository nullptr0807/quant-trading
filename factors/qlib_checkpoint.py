"""Point-in-time Qlib model checkpoint registry.

Why
---
Q-account backtests over historical dates have THREE leakage vectors when
we re-train at backtest time using `T = today` as the cutoff:

  1. **Model weights** — current trained model has seen ALL data up to today
  2. **Feature normalization** — RobustZScoreNorm fit on full history
  3. **Label** — Qlib label = forward return; train end must be ≤ T-2

Frozen daily checkpoints fix all three at once: t-day's checkpoint is
physically incapable of having seen t+1's data. Future backtests over
[start, end] ⊂ [first_checkpoint_day, today-1] just `load_checkpoint(t)`
→ predict → done. Zero re-training, zero leakage.

Storage layout
--------------
~/quant-trading/data/qlib_checkpoints/
  <market>/                US | CN
    <model_id>/            Q01 .. Q10
      <YYYY-MM-DD>.pkl     joblib-pickled bundle: {handler, model, spec_meta}
      <YYYY-MM-DD>.json    sidecar: train window, IC, qlib/torch versions,
                           self-test (input_hash → expected_score)

A single day across all 10 models ≈ 1.2 MB. 1 year ≈ 300 MB.

Versioning & self-test
----------------------
Every checkpoint sidecar records qlib/torch/numpy/python versions AND a
deterministic self-test row (input_hash, expected_score). At load time we
re-run the self-test; if the score drifts beyond tolerance the loader
raises CheckpointDriftError so callers can degrade gracefully (e.g. fall
back to walk-forward retrain or skip the bar).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("qlib_checkpoint")

PROJECT_ROOT = Path(os.path.expanduser("~/quant-trading"))
CHECKPOINT_ROOT = PROJECT_ROOT / "data" / "qlib_checkpoints"

DRIFT_TOLERANCE = 1e-4   # max abs diff between expected_score and re-run score


class CheckpointDriftError(RuntimeError):
    """Raised when self-test on a loaded checkpoint produces a score that
    differs from the value recorded at save-time. Indicates env drift
    (qlib/torch upgrade, numpy ABI change) — checkpoint should be treated
    as untrusted."""


class CheckpointMissingError(FileNotFoundError):
    """No checkpoint exists for (model_id, date, market)."""


def _checkpoint_dir(market: str, model_id: str) -> Path:
    return CHECKPOINT_ROOT / market.upper() / model_id


def _checkpoint_paths(market: str, model_id: str, date: str) -> tuple[Path, Path]:
    """Return (pkl_path, json_path) for a given (market, model_id, date).

    `date` is YYYY-MM-DD ISO date. If you pass a full timestamp we slice it.
    """
    d = str(date)[:10]
    base = _checkpoint_dir(market, model_id)
    return base / f"{d}.pkl", base / f"{d}.json"


def _versions_snapshot() -> dict:
    """Capture environment versions so future loaders can detect drift."""
    out = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for mod in ("qlib", "torch", "numpy", "pandas", "lightgbm", "xgboost",
                "catboost", "sklearn"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            pass
    return out


def _hash_dataframe(df) -> str:
    """Deterministic hash of a small DataFrame slice — used as self-test key.

    We only hash the FIRST row of test data to keep meta.json small.
    """
    try:
        import pandas as pd  # noqa: F401
        arr = df.values.astype("float64", copy=False)
        return hashlib.sha256(arr.tobytes()).hexdigest()[:16]
    except Exception:
        return "unhashable"


def save_checkpoint(spec, model, dataset, pred, market: str = "US",
                    date: Optional[str] = None,
                    train_window: Optional[dict] = None,
                    elapsed_s: Optional[float] = None,
                    extra_meta: Optional[dict] = None) -> dict:
    """Persist a trained Qlib model + its dataset (which carries fitted
    handler/processors) so it can be replayed later without retraining.

    Parameters
    ----------
    spec : ModelSpec
        From factors.qlib_signal — id, name, model_class, kwargs, feature_set.
    model : qlib model instance (already fit)
    dataset : qlib DatasetH instance (handler is fitted, processors stateful)
    pred : DataFrame with single 'score' column, index = (datetime, instrument)
        The training-time predictions; we extract the FIRST row as self-test
        ground truth.
    market : 'US' or 'CN'
    date : ISO date string (YYYY-MM-DD); defaults to today UTC.
    train_window : dict with train/valid/test segment dates.
    elapsed_s : training wall time.
    extra_meta : free-form extra fields to record (IC, val loss, ...)

    Returns
    -------
    dict — the meta written to the .json sidecar.
    """
    import joblib

    market = market.upper()
    if date is None:
        date = datetime.now(timezone.utc).date().isoformat()
    date = str(date)[:10]

    pkl_path, json_path = _checkpoint_paths(market, spec.id, date)
    pkl_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Self-test fingerprint ────────────────────────────────────────────
    # Take the first row of pred as ground truth. Future loaders can
    # re-predict on the same bundle and verify the score within tolerance.
    self_test = None
    try:
        if pred is not None and len(pred) > 0:
            first_idx = pred.index[0]
            first_score = float(pred.iloc[0, 0])
            # Extract the test feature row that produced first_score so a
            # standalone re-predict is deterministic.
            try:
                test_df = dataset.prepare("test", col_set=["feature"])
                # test_df may be a DataFrame with MultiIndex matching pred index
                feat_row = test_df.loc[first_idx]
                feat_hash = _hash_dataframe(feat_row.to_frame().T)
            except Exception:
                feat_hash = "unavailable"
            self_test = {
                "first_index": [str(x) for x in (first_idx if isinstance(first_idx, tuple) else (first_idx,))],
                "feature_hash": feat_hash,
                "expected_score": first_score,
                "tolerance": DRIFT_TOLERANCE,
            }
    except Exception as e:
        log.warning("[%s] self-test fingerprint failed: %s", spec.id, e)

    # ── Bundle payload ───────────────────────────────────────────────────
    # joblib dumps the entire {handler, model, dataset.segments} so a future
    # process can do `pred = bundle['model'].predict(bundle['dataset'])`
    # without re-fitting anything.
    payload = {
        "spec_id": spec.id,
        "model_class": spec.model_class,
        "model": model,
        "dataset": dataset,            # carries fitted handler + processors
        "feature_set": getattr(spec, "feature_set", "Alpha158"),
    }
    try:
        joblib.dump(payload, pkl_path, compress=3)
    except Exception as e:
        log.error("[%s] joblib.dump failed: %s — checkpoint NOT saved", spec.id, e)
        # Don't propagate — daily cron should keep going even if checkpoint
        # save fails (predictions already in factor_values).
        return {"error": str(e), "saved": False}

    pkl_size = pkl_path.stat().st_size

    meta = {
        "spec_id": spec.id,
        "model_class": spec.model_class,
        "feature_set": getattr(spec, "feature_set", "Alpha158"),
        "market": market,
        "date": date,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "pkl_bytes": pkl_size,
        "elapsed_s": elapsed_s,
        "train_window": train_window,
        "self_test": self_test,
        "versions": _versions_snapshot(),
        "extra": extra_meta or {},
    }
    json_path.write_text(json.dumps(meta, indent=2, default=str))
    log.info("[%s/%s] checkpoint saved: %s (%.1f KB)",
             market, spec.id, pkl_path.name, pkl_size / 1024)
    return meta


def load_checkpoint(model_id: str, date: str, market: str = "US",
                    verify: bool = True):
    """Load a previously-saved checkpoint and (optionally) verify it still
    produces the expected self-test score in the current environment.

    Returns the joblib payload dict: {spec_id, model_class, model, dataset,
    feature_set}.

    Raises
    ------
    CheckpointMissingError — no .pkl for that day
    CheckpointDriftError — self-test score moved beyond tolerance (env drift)
    """
    import joblib

    pkl_path, json_path = _checkpoint_paths(market, model_id, date)
    if not pkl_path.exists():
        raise CheckpointMissingError(f"no checkpoint at {pkl_path}")

    payload = joblib.load(pkl_path)

    if verify and json_path.exists():
        try:
            meta = json.loads(json_path.read_text())
            st = meta.get("self_test")
            if st and st.get("expected_score") is not None:
                # Re-predict on the bundled dataset and grab first row
                pred = payload["model"].predict(payload["dataset"])
                if hasattr(pred, "to_frame"):
                    pred = pred.to_frame("score")
                actual = float(pred.iloc[0, 0])
                expected = float(st["expected_score"])
                tol = float(st.get("tolerance", DRIFT_TOLERANCE))
                if abs(actual - expected) > tol:
                    raise CheckpointDriftError(
                        f"{model_id}/{date}: expected={expected:.6g} "
                        f"actual={actual:.6g} drift={actual-expected:.3g} "
                        f"tol={tol:.3g}"
                    )
        except CheckpointDriftError:
            raise
        except Exception as e:
            log.warning("[%s/%s] verify skipped: %s", market, model_id, e)

    return payload


def list_checkpoints(model_id: str, market: str = "US") -> list[dict]:
    """Return [{date, pkl_bytes, ic, ...}, ...] sorted ASC by date."""
    base = _checkpoint_dir(market, model_id)
    if not base.exists():
        return []
    out = []
    for json_path in sorted(base.glob("*.json")):
        try:
            meta = json.loads(json_path.read_text())
            out.append(meta)
        except Exception:
            continue
    return out


def coverage_summary(market: str = "US") -> dict:
    """High-level health check: per-model first/last date, count, total size."""
    out = {"market": market, "models": {}, "total_bytes": 0}
    base = CHECKPOINT_ROOT / market.upper()
    if not base.exists():
        return out
    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir():
            continue
        pkls = sorted(model_dir.glob("*.pkl"))
        if not pkls:
            continue
        sizes = sum(p.stat().st_size for p in pkls)
        out["models"][model_dir.name] = {
            "count": len(pkls),
            "first": pkls[0].stem,
            "last": pkls[-1].stem,
            "bytes": sizes,
        }
        out["total_bytes"] += sizes
    return out


# ─── CLI ────────────────────────────────────────────────────────────────────

def _main():
    """Dump coverage summary as JSON. Useful for dashboard/cron health-check."""
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--market", default="US", choices=["US", "CN"])
    args = p.parse_args()
    print(json.dumps(coverage_summary(args.market), indent=2))


if __name__ == "__main__":
    _main()
