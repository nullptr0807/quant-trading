from __future__ import annotations

import numpy as np
import pandas as pd


def test_prepare_dataset_returns_globally_date_sorted_rows():
    import factors.gp_miner as gm

    idx = pd.date_range("2025-01-01", periods=45, freq="D")

    def frame(scale):
        close = pd.Series(np.arange(1, len(idx) + 1, dtype=float) * scale, index=idx)
        return pd.DataFrame(
            {
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1_000_000.0,
            },
            index=idx,
        )

    X, y, dates = gm._prepare_dataset(
        {"ZZZ": frame(2.0), "AAA": frame(1.0)},
        feature_cols=["ret_1"],
        y_target="next_1d_ret",
        return_dates=True,
    )

    assert len(X) == len(y) == len(dates)
    assert list(dates) == sorted(dates)
    split = int(len(dates) * 0.8)
    assert max(dates[:split]) <= min(dates[split:])


def test_prepare_dataset_normalizes_intraday_timestamps_to_market_dates():
    import factors.gp_miner as gm

    idx = pd.date_range("2025-01-01 14:00:00", periods=45, freq="D", tz="UTC")
    close = pd.Series(np.arange(1, len(idx) + 1, dtype=float), index=idx)
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000.0,
        },
        index=idx,
    )

    _, _, dates = gm._prepare_dataset(
        {"AAA": frame}, feature_cols=["ret_1"],
        y_target="next_1d_ret", return_dates=True,
    )

    assert all(pd.Timestamp(value).hour == 0 for value in dates)


def test_gp_holdout_never_splits_one_cross_section_date(monkeypatch):
    import factors.gp_miner as gm

    dates = np.array(
        [np.datetime64(f"2025-01-{day:02d}") for day in range(1, 11) for _ in range(5)]
    )
    X = np.arange(len(dates), dtype=float).reshape(-1, 1)
    y = np.arange(len(dates), dtype=float)
    monkeypatch.setattr(gm, "_prepare_dataset", lambda *a, **k: (X, y, dates))

    seen = {}

    class Program:
        fitness_ = 1.0

        def __str__(self):
            return "X0"

    class Dummy:
        def __init__(self, **kwargs):
            self._best_programs = [Program()]

        def fit(self, X_fit, y_fit):
            seen["train_len"] = len(X_fit)
            return self

        def transform(self, values):
            return values[:, :1]

    monkeypatch.setattr(gm, "SymbolicTransformer", Dummy)
    gm.GPAlphaMiner().mine_factors(
        {"X": pd.DataFrame()}, n_factors=1, n_runs=1,
        feature_subset=["x"], dedup_threshold=1.1,
    )

    split = seen["train_len"]
    assert dates[split - 1] < dates[split]


def test_gp_miner_selects_on_global_date_holdout_ic(monkeypatch):
    import factors.gp_miner as gm

    n = 50
    X = np.arange(n, dtype=float).reshape(-1, 1)
    y = np.concatenate([np.arange(40, dtype=float), -np.arange(10, dtype=float)])
    dates = np.array(pd.date_range("2025-01-01", periods=n), dtype="datetime64[ns]")
    monkeypatch.setattr(
        gm,
        "_prepare_dataset",
        lambda *a, **k: (X, y, dates),
    )

    class Program:
        fitness_ = 1.0

        def __str__(self):
            return "X0"

    class Dummy:
        def __init__(self, **kwargs):
            self._best_programs = [Program()]

        def fit(self, X_fit, y_fit):
            assert len(X_fit) == 40
            return self

        def transform(self, values):
            return values[:, :1]

    monkeypatch.setattr(gm, "SymbolicTransformer", Dummy)
    got = gm.GPAlphaMiner().mine_factors(
        {"X": pd.DataFrame()},
        n_factors=1,
        n_runs=1,
        feature_subset=["x"],
        dedup_threshold=1.1,
    )

    assert got[0]["selection_basis"] == "global_date_20pct_holdout_ic"
    assert got[0]["train_ic"] > 0.9
    assert got[0]["oos_ic"] < -0.9
    assert got[0]["ic"] == got[0]["oos_ic"]
