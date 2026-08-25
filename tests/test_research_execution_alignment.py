from __future__ import annotations

import pandas as pd
import sqlite3
import pytest


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


def _replay_module():
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).parents[1] / "scripts" / "replay_us.py"
    spec = importlib.util.spec_from_file_location("raw_replay_us", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_broker_replay_never_falls_back_to_adjusted_prices():
    module = _replay_module()
    replay = object.__new__(module.USReplay)
    replay.execution_mode = "broker"
    replay.raw_daily = {}
    replay.daily = {
        "AAA": pd.DataFrame(
            {"open": [100.0]}, index=pd.to_datetime(["2026-07-10"])
        )
    }
    with pytest.raises(module.ReplayCoverageError, match="raw execution price"):
        replay._current_prices({}, pd.Timestamp("2026-07-10").date())


def test_broker_replay_loader_excludes_quarantined_raw_bar():
    module = _replay_module()
    con = sqlite3.connect(":memory:")
    con.executescript(
        """
        CREATE TABLE prices_raw (
            ticker TEXT, datetime TEXT, interval TEXT,
            open REAL, high REAL, low REAL, close REAL, volume REAL
        );
        CREATE TABLE price_quality_issues (
            price_mode TEXT, ticker TEXT, datetime TEXT, interval TEXT,
            issue_type TEXT, detected_at TEXT, detail TEXT
        );
        INSERT INTO prices_raw VALUES ('AAA','2026-07-09','1d',10,11,9,10,1);
        INSERT INTO prices_raw VALUES ('AAA','2026-07-10','1d',10,9,8,11,1);
        INSERT INTO price_quality_issues VALUES (
            'raw','AAA','2026-07-10','1d','invalid_ohlc','now','bad'
        );
        """
    )
    replay = object.__new__(module.USReplay)
    replay.conn = con

    loaded = replay._load_daily(table="prices_raw")

    assert list(loaded["AAA"].index.strftime("%Y-%m-%d")) == ["2026-07-09"]


def test_split_and_cash_dividend_adjust_shares_cash_before_next_bar_execution():
    module = _replay_module()
    account = module.USAccount("A01", 1000.0)
    pos = account.positions.setdefault("AAA", module.USPos())
    pos.shares, pos.avg_cost, pos.total_cost = 10, 100, 1000

    module.apply_corporate_actions(
        account,
        [
            {"ticker": "AAA", "action_type": "split", "ratio": 2.0},
            {"ticker": "AAA", "action_type": "cash_dividend", "cash_per_share": 1.5},
        ],
    )
    assert pos.shares == 20
    assert pos.avg_cost == 50
    assert pos.total_cost == 1000
    assert account.cash == 1030


def test_execution_uses_raw_open_and_fee_adjusted_share_bookkeeping():
    module = _replay_module()
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE trades (account,ticker,side,shares,price,cost,slippage,timestamp,market)")
    replay = object.__new__(module.USReplay)
    replay.conn, replay.label, replay.trade_count = con, "test", 0
    replay.churn_stats = {}
    replay.costs = module.MoomooAUCosts(slippage_pct=0.0)
    account = module.USAccount("A01", 1000.0)

    assert replay.execute(account, "AAA", "buy", 5, 100.0, pd.Timestamp("2026-07-10").date())
    trade = con.execute("SELECT shares,price,cost FROM trades").fetchone()
    assert trade == (5.0, 100.0, 1.005)
    assert account.positions["AAA"].shares == 5
    assert account.cash == pytest.approx(498.995)
