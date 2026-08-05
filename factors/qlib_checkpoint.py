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


class CheckpointCoverageError(RuntimeError):
    """Checkpoint series is incomplete or lacks point-in-time semantics."""


class FrozenFeatureDataset:
    """Minimal Qlib Dataset interface backed by already-processed PIT features."""
    def __init__(self, features):
        self.features = features
        self.segments = {"test": "test"}

    def prepare(self, segments="test", col_set=None, data_key=None, **kwargs):
        if isinstance(segments, (list, tuple)):
            return [self.features.copy() for _ in segments]
        return self.features.copy()


def _publication_marker_path(market: str, date: str) -> Path:
    return CHECKPOINT_ROOT / market.upper() / f"publication-{str(date)[:10]}.json"


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

    # ── Frozen processed features + self-test fingerprint ────────────────
    self_test = None
    frozen_features = None
    try:
        frozen_features = dataset.prepare("test", col_set=["feature"])
        if pred is not None and len(pred) > 0 and frozen_features is not None:
            first_idx = pred.index[0]
            first_score = float(pred.iloc[0, 0])
            feat_row = frozen_features.loc[first_idx]
            feat_hash = _hash_dataframe(feat_row.to_frame().T)
            self_test = {
                "first_index": [str(x) for x in (first_idx if isinstance(first_idx, tuple) else (first_idx,))],
                "feature_hash": feat_hash,
                "expected_score": first_score,
                "tolerance": DRIFT_TOLERANCE,
            }
    except Exception as e:
        log.warning("[%s] frozen feature/self-test capture failed: %s", spec.id, e)

    if frozen_features is None or self_test is None:
        return {"error": "frozen feature/self-test capture failed", "saved": False}

    # Store fitted/processed PIT feature values rather than a live DatasetH
    # handler. Qlib handlers lose transient `_infer` frames when pickled, while
    # the processed frame is immutable and replayable across processes.
    payload = {
        "spec_id": spec.id,
        "model_class": spec.model_class,
        "model": model,
        "frozen_test_features": frozen_features,
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
        "processor_fit_end": (
            str((train_window or {}).get("train", [None, None])[-1])[:10]
            if (train_window or {}).get("train") else None
        ),
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

    if verify:
        if not json_path.exists():
            raise CheckpointCoverageError(f"missing checkpoint sidecar at {json_path}")
        try:
            meta = json.loads(json_path.read_text())
            st = meta.get("self_test")
            if not st or st.get("expected_score") is None:
                raise CheckpointCoverageError(
                    f"checkpoint {market}/{model_id}/{date} has no score self-test"
                )
            if st and st.get("expected_score") is not None:
                # Re-predict from frozen, already-processed PIT features. Legacy
                # DatasetH payloads are unsupported because their transient
                # `_infer` frame is not reliably pickleable.
                features = payload.get("frozen_test_features")
                if features is None:
                    raise CheckpointCoverageError("checkpoint has no frozen test features")
                pred = payload["model"].predict(FrozenFeatureDataset(features))
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
        except (CheckpointDriftError, CheckpointCoverageError):
            raise
        except Exception as e:
            raise CheckpointCoverageError(
                f"checkpoint verification failed for {market}/{model_id}/{date}: {e}"
            ) from e

    return payload


def checkpoint_ready_for_publication(model_id: str, date: str, market: str,
                                     expected_universe_count: int) -> tuple[bool, str]:
    """Cheap live gate: require payload, self-test sidecar and exact PIT proof."""
    pkl_path, json_path = _checkpoint_paths(market, model_id, date)
    if not pkl_path.exists() or pkl_path.stat().st_size <= 0:
        return False, "checkpoint_payload_missing"
    if not json_path.exists():
        return False, "checkpoint_sidecar_missing"
    try:
        meta = json.loads(json_path.read_text())
    except Exception:
        return False, "checkpoint_sidecar_invalid"
    extra = meta.get("extra") or {}
    if extra.get("point_in_time_complete") is not True:
        return False, "checkpoint_pit_incomplete"
    if int(extra.get("universe_count") or 0) != int(expected_universe_count):
        return False, "checkpoint_universe_count_mismatch"
    if (meta.get("self_test") or {}).get("expected_score") is None:
        return False, "checkpoint_self_test_missing"
    marker_path = _publication_marker_path(market, date)
    try:
        marker = json.loads(marker_path.read_text())
        verified = (marker.get("models") or {})[model_id]
        pkl_stat = pkl_path.stat()
        json_stat = json_path.stat()
        if int(verified.get("pkl_size", -1)) != pkl_stat.st_size:
            return False, "checkpoint_changed_after_verification"
        if int(verified.get("pkl_mtime_ns", -1)) != pkl_stat.st_mtime_ns:
            return False, "checkpoint_changed_after_verification"
        if int(verified.get("json_size", -1)) != json_stat.st_size:
            return False, "checkpoint_changed_after_verification"
        if int(verified.get("json_mtime_ns", -1)) != json_stat.st_mtime_ns:
            return False, "checkpoint_changed_after_verification"
    except Exception:
        return False, "checkpoint_publication_marker_missing"
    return True, "ok"


def _checkpoint_meta(model_id: str, date: str, market: str) -> dict:
    _, json_path = _checkpoint_paths(market, model_id, date)
    if not json_path.exists():
        raise CheckpointCoverageError(f"missing checkpoint sidecar at {json_path}")
    try:
        return json.loads(json_path.read_text())
    except Exception as exc:
        raise CheckpointCoverageError(f"invalid checkpoint sidecar at {json_path}") from exc


def _validate_pit_semantics(meta: dict, payload: dict, market: str) -> None:
    checkpoint_date = str(meta.get("date", ""))[:10]
    train_window = meta.get("train_window") or {}
    train_segment = train_window.get("train") or []
    if len(train_segment) < 2 or str(train_segment[-1])[:10] > checkpoint_date:
        raise CheckpointCoverageError(
            f"checkpoint train window is not bounded by as-of date {checkpoint_date}"
        )
    processor_fit_end = str(meta.get("processor_fit_end") or "")[:10]
    if not processor_fit_end or processor_fit_end > checkpoint_date:
        raise CheckpointCoverageError(
            "checkpoint does not prove processors were fitted by its as-of date"
        )
    if "frozen_test_features" not in payload or "model" not in payload:
        raise CheckpointCoverageError("checkpoint lacks frozen model/processed PIT features")
    # A frozen model/processor is necessary but not sufficient: the checkpoint
    # must also prove that its instrument universe was point-in-time. Current
    # Russell membership cannot be presented as capital-allocation-valid for
    # either market.
    if not (meta.get("extra") or {}).get("point_in_time_complete", False):
        raise CheckpointCoverageError(
            f"{market.upper()} checkpoint lacks point-in-time universe semantics"
        )


def require_checkpoint_coverage(
    model_ids: list[str], signal_dates: list[str], *, market: str = "US"
) -> dict:
    """Require an exact checkpoint for every model/date; never forward-fill."""
    market = market.upper()
    requested = sorted({str(day)[:10] for day in signal_dates})
    if not requested:
        raise CheckpointCoverageError("no checkpoint dates requested")
    available_by_model: dict[str, set[str]] = {}
    missing: list[str] = []
    for model_id in model_ids:
        base = _checkpoint_dir(market, model_id)
        available = {
            path.stem for path in base.glob("*.pkl")
            if (base / f"{path.stem}.json").exists()
        } if base.exists() else set()
        available_by_model[model_id] = available
        missing.extend(
            f"{model_id}/{day}" for day in requested if day not in available
        )
    if missing:
        raise CheckpointCoverageError(
            "missing checkpoint coverage: " + ", ".join(missing[:20])
        )
    common = set.intersection(*available_by_model.values()) if available_by_model else set()
    if not common:
        raise CheckpointCoverageError("models have no common full-coverage date")
    return {
        "market": market,
        "models": list(model_ids),
        "first_full_coverage_date": min(common),
        "last_full_coverage_date": max(common),
        "requested_dates": requested,
        "complete": True,
    }


def predict_checkpoint_scores(
    model_id: str,
    *,
    as_of: str,
    execution_date: str,
    market: str = "US",
) -> tuple[dict[str, float], dict]:
    """Load the exact T checkpoint and score T for execution strictly after T."""
    as_of = str(as_of)[:10]
    execution_date = str(execution_date)[:10]
    if execution_date <= as_of:
        raise CheckpointCoverageError(
            f"execution_date={execution_date} must be after checkpoint as_of={as_of}"
        )
    require_checkpoint_coverage([model_id], [as_of], market=market)
    payload = load_checkpoint(model_id, as_of, market=market, verify=True)
    meta = _checkpoint_meta(model_id, as_of, market)
    _validate_pit_semantics(meta, payload, market)
    features = payload.get("frozen_test_features")
    if features is None:
        raise CheckpointCoverageError("checkpoint has no frozen test features")
    pred = payload["model"].predict(FrozenFeatureDataset(features))
    if hasattr(pred, "to_frame"):
        pred = pred.to_frame("score")
    if not hasattr(pred, "index"):
        raise CheckpointCoverageError("checkpoint prediction has no dated index")
    import pandas as pd
    if isinstance(pred.index, pd.MultiIndex):
        dates = pd.to_datetime(pred.index.get_level_values(0)).strftime("%Y-%m-%d")
        day_pred = pred.loc[dates == as_of]
        instruments = day_pred.index.get_level_values(-1)
    else:
        dates = pd.to_datetime(pred.index).strftime("%Y-%m-%d")
        day_pred = pred.loc[dates == as_of]
        instruments = day_pred.index
    if day_pred.empty:
        raise CheckpointCoverageError(
            f"checkpoint {model_id}/{as_of} contains no scores for its as-of date"
        )
    values = day_pred.iloc[:, 0]
    scores = {str(ticker): float(score) for ticker, score in zip(instruments, values)}
    provenance = {
        "kind": "qlib_daily_checkpoint",
        "model_id": model_id,
        "market": market.upper(),
        "checkpoint_date": as_of,
        "execution_date": execution_date,
        "capital_allocation_valid": True,
    }
    return scores, provenance


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
    """Inspect coverage or replay one exact daily checkpoint without retraining."""
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--market", default="US", choices=["US", "CN"])
    p.add_argument("--model", help="model id for exact checkpoint replay, e.g. Q01")
    p.add_argument("--as-of", help="checkpoint/signal date (YYYY-MM-DD; exact, no fallback)")
    p.add_argument("--execution-date", help="strictly later executable date")
    args = p.parse_args()
    if any((args.model, args.as_of, args.execution_date)):
        if not all((args.model, args.as_of, args.execution_date)):
            p.error("--model, --as-of and --execution-date are required together")
        scores, provenance = predict_checkpoint_scores(
            args.model, as_of=args.as_of, execution_date=args.execution_date,
            market=args.market,
        )
        print(json.dumps({"scores": scores, "provenance": provenance}, indent=2))
    else:
        print(json.dumps(coverage_summary(args.market), indent=2))


if __name__ == "__main__":
    _main()
