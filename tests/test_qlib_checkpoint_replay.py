from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import pandas as pd
import pytest


class FakeModel:
    def predict(self, dataset):
        return dataset.prepare().iloc[:, 0].copy()


class FakeTrainingDataset:
    def __init__(self, features):
        self.features = features
    def prepare(self, segment, col_set=None):
        return self.features.copy()


def _write_checkpoint(root: Path, market: str, model_id: str, day: str, *, score=0.25, complete=True):
    base = root / market / model_id
    base.mkdir(parents=True, exist_ok=True)
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(day), "AAA")], names=["datetime", "instrument"]
    )
    pred = pd.Series([score], index=idx)
    frozen = pred.to_frame('feature_0')
    joblib.dump(
        {"spec_id": model_id, "model": FakeModel(), "frozen_test_features": frozen, "processors_fitted": True},
        base / f"{day}.pkl",
    )
    meta = {
        "spec_id": model_id,
        "market": market,
        "date": day,
        "train_window": {"train": ["2025-01-01", day]},
        "processor_fit_end": day,
        "self_test": {"expected_score": score, "tolerance": 1e-8},
        "extra": {"point_in_time_complete": complete},
    }
    (base / f"{day}.json").write_text(json.dumps(meta))


def test_save_checkpoint_freezes_processed_features_for_cross_process_replay(tmp_path, monkeypatch):
    import factors.qlib_checkpoint as qc
    monkeypatch.setattr(qc, 'CHECKPOINT_ROOT', tmp_path)
    day='2026-07-09'
    idx=pd.MultiIndex.from_tuples([(pd.Timestamp(day),'AAA')],names=['datetime','instrument'])
    features=pd.DataFrame({'feature_0':[.75]},index=idx)
    pred=pd.DataFrame({'score':[.75]},index=idx)
    spec=SimpleNamespace(id='Q01',name='fake',model_class='Fake',feature_set='Alpha158')
    meta=qc.save_checkpoint(
        spec,FakeModel(),FakeTrainingDataset(features),pred,
        market='US',date=day,train_window={'train':['2025-01-01','2026-07-01']},
        extra_meta={'point_in_time_complete':True,'universe_count':1},
    )
    assert meta.get('saved') is not False
    payload=qc.load_checkpoint('Q01',day,market='US',verify=True)
    assert 'frozen_test_features' in payload and 'dataset' not in payload
    scores,_=qc.predict_checkpoint_scores('Q01',as_of=day,execution_date='2026-07-10',market='US')
    assert scores=={'AAA':.75}


def test_daily_checkpoint_replay_scores_exact_asof_without_future_checkpoint(tmp_path, monkeypatch):
    import factors.qlib_checkpoint as qc

    monkeypatch.setattr(qc, "CHECKPOINT_ROOT", tmp_path)
    _write_checkpoint(tmp_path, "US", "Q01", "2026-07-09", score=0.75)
    scores, provenance = qc.predict_checkpoint_scores(
        "Q01", as_of="2026-07-09", execution_date="2026-07-10", market="US"
    )
    assert scores == {"AAA": 0.75}
    assert provenance["checkpoint_date"] == "2026-07-09"
    assert provenance["execution_date"] == "2026-07-10"


def test_checkpoint_coverage_starts_at_first_date_all_models_cover(tmp_path, monkeypatch):
    import factors.qlib_checkpoint as qc

    monkeypatch.setattr(qc, "CHECKPOINT_ROOT", tmp_path)
    _write_checkpoint(tmp_path, "US", "Q01", "2026-07-08")
    _write_checkpoint(tmp_path, "US", "Q01", "2026-07-09")
    _write_checkpoint(tmp_path, "US", "Q02", "2026-07-09")

    coverage = qc.require_checkpoint_coverage(
        ["Q01", "Q02"], ["2026-07-09"], market="US"
    )
    assert coverage["first_full_coverage_date"] == "2026-07-09"
    with pytest.raises(qc.CheckpointCoverageError, match="missing checkpoint"):
        qc.require_checkpoint_coverage(
            ["Q01", "Q02"], ["2026-07-08", "2026-07-09"], market="US"
        )


def test_cn_checkpoint_with_incomplete_semantics_fails_closed(tmp_path, monkeypatch):
    import factors.qlib_checkpoint as qc

    monkeypatch.setattr(qc, "CHECKPOINT_ROOT", tmp_path)
    _write_checkpoint(tmp_path, "CN", "Q01", "2026-07-09", complete=False)
    with pytest.raises(qc.CheckpointCoverageError, match="lacks point-in-time universe semantics"):
        qc.predict_checkpoint_scores(
            "Q01", as_of="2026-07-09", execution_date="2026-07-10", market="CN"
        )


def test_us_checkpoint_without_pit_universe_proof_also_fails_closed(tmp_path, monkeypatch):
    import factors.qlib_checkpoint as qc

    monkeypatch.setattr(qc, "CHECKPOINT_ROOT", tmp_path)
    _write_checkpoint(tmp_path, "US", "Q01", "2026-07-09", complete=False)
    with pytest.raises(qc.CheckpointCoverageError, match="lacks point-in-time universe semantics"):
        qc.predict_checkpoint_scores(
            "Q01", as_of="2026-07-09", execution_date="2026-07-10", market="US"
        )


def test_checkpoint_score_drift_is_not_silently_skipped(tmp_path, monkeypatch):
    import factors.qlib_checkpoint as qc

    monkeypatch.setattr(qc, "CHECKPOINT_ROOT", tmp_path)
    _write_checkpoint(tmp_path, "US", "Q01", "2026-07-09", score=0.5)
    meta_path = tmp_path / "US" / "Q01" / "2026-07-09.json"
    meta = json.loads(meta_path.read_text())
    meta["self_test"]["expected_score"] = 0.9
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(qc.CheckpointDriftError):
        qc.load_checkpoint("Q01", "2026-07-09", market="US")