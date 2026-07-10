from __future__ import annotations

import pandas as pd


def test_replay_history_excludes_execution_day_and_uses_execution_open():
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).parents[1] / "scripts" / "replay_us.py"
    spec = importlib.util.spec_from_file_location("hardening_replay_us", module_path)
    assert spec and spec.loader
    replay_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay_module)
    USReplay = replay_module.USReplay

    replay = object.__new__(USReplay)
    idx = pd.to_datetime(["2026-07-08", "2026-07-09", "2026-07-10"])
    replay.daily = {
        "AAA": pd.DataFrame(
            {
                "open": [90.0, 100.0, 130.0],
                "close": [95.0, 120.0, 140.0],
            },
            index=idx,
        )
    }

    history = replay._slice_history(pd.Timestamp("2026-07-10").date())
    prices = replay._current_prices(history, pd.Timestamp("2026-07-10").date())

    assert history["AAA"].index.max() == pd.Timestamp("2026-07-09")
    assert prices == {"AAA": 130.0}
