import sqlite3
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from data.store import DataStore
from data.quotes import RealtimeQuote
from main import QuantSystem
from trading.account import _Position
from trading.engine import TradingEngine


def _bare_system(tmp_path, universe=("AAA", "BBB", "CCC", "DDD")):
    system = object.__new__(QuantSystem)
    system.market = "US"
    system.universe = list(universe)
    system.db_path = str(tmp_path / "trading.db")
    system.store = DataStore(system.db_path)
    system.benchmarks = [{"id": "IDX_SPY", "ticker": "SPY", "name": "SPY"}]
    system.strategies = []
    system.gp_strategies = []
    system.qlib_strategies = []
    system._per_account_mined = {}
    system._per_account_gp_factors = {}
    system._historical_data = {}
    system._realtime_prices = {}
    system._adaptive_enabled = False
    system._last_rebalance = {}
    system._rebalance_hours_cache = {}
    system._lot_skip_seen = {}
    system.signal_gen = __import__("factors.signal", fromlist=["SignalGenerator"]).SignalGenerator(
        buy_top=30, sell_top=10, decorrelate=False
    )
    system.gp_signal_gen = __import__("factors.gp_signal", fromlist=["GPSignalGenerator"]).GPSignalGenerator()
    system.engine = TradingEngine()
    system.engine.create_account("IDX_SPY", initial_cash=10_000)
    return system


def _insert_prices_and_factors(system, *, price_date="2026-07-10", rows=()):
    with sqlite3.connect(system.db_path) as conn:
        conn.executemany(
            "INSERT INTO prices "
            "(ticker,datetime,interval,open,high,low,close,volume) "
            "VALUES (?,?,'1d',1,1,1,1,1)",
            [(ticker, price_date) for ticker in system.universe],
        )
        conn.executemany(
            "INSERT INTO factor_values "
            "(ticker,date,factor_name,value,factor_group) VALUES (?,?,?,?,?)",
            rows,
        )


def test_persisted_alpha_uses_each_account_factor_names_and_cross_sectional_ranks(
    tmp_path, monkeypatch
):
    system = _bare_system(tmp_path, universe=("AAA", "BBB", "CCC"))
    system.strategies = [
        SimpleNamespace(
            id="A01", factor_names=["F_MOM"], strategy_type="momentum",
            top_n=1, rebalance_hours=24,
        ),
        SimpleNamespace(
            id="A02", factor_names=["F_REV"], strategy_type="momentum",
            top_n=1, rebalance_hours=24,
        ),
    ]
    for strategy in system.strategies:
        system.engine.create_account(strategy.id, initial_cash=10_000)
    _insert_prices_and_factors(
        system,
        rows=[
            ("AAA", "2026-07-09", "F_MOM", 3.0, "alpha158"),
            ("BBB", "2026-07-09", "F_MOM", 2.0, "alpha158"),
            ("CCC", "2026-07-09", "F_MOM", 1.0, "alpha158"),
            ("AAA", "2026-07-09", "F_REV", 1.0, "alpha158"),
            ("BBB", "2026-07-09", "F_REV", 2.0, "alpha158"),
            ("CCC", "2026-07-09", "F_REV", 3.0, "alpha158"),
        ],
    )
    monkeypatch.setattr("main.emit_event", lambda *args, **kwargs: None)

    quote_tickers = system.prepare_fast_live_cycle()

    assert system._prepared_alpha_signals["A01"]["buy"][0][0] == "AAA"
    assert system._prepared_alpha_signals["A02"]["buy"][0][0] == "CCC"
    assert {"AAA", "BBB", "CCC", "SPY"}.issubset(quote_tickers)


def test_persisted_gp_and_f_load_only_their_group_and_active_factor_names(
    tmp_path, monkeypatch
):
    system = _bare_system(tmp_path, universe=("AAA", "BBB", "CCC"))
    system.gp_strategies = [
        SimpleNamespace(
            id="B01", family="B", factor_selection="all",
            scoring_method="equal_weight", top_n=1, rebalance_hours=4,
        ),
        SimpleNamespace(
            id="F11", family="F", factor_selection="all",
            scoring_method="equal_weight", top_n=1, rebalance_hours=4,
        ),
    ]
    for strategy in system.gp_strategies:
        system.engine.create_account(strategy.id, initial_cash=10_000)
    system._per_account_mined = {
        "B01": [
            {"name": "b_active", "expression": "X0", "active": True, "ic": 0.2},
            {"name": "b_inactive", "expression": "X1", "active": False, "ic": 0.3},
        ],
        "F11": [
            {"name": "f_active", "expression": "X0", "active": True, "ic": 0.2},
            {"name": "failure", "active": False, "status": "mining_failed"},
        ],
    }
    rows = []
    for ticker, b_value, f_value in (("AAA", 3, 1), ("BBB", 2, 2), ("CCC", 1, 3)):
        rows.extend(
            [
                (ticker, "2026-07-09", "b_active", b_value, "gp_B01"),
                (ticker, "2026-07-09", "b_inactive", 999, "gp_B01"),
                (ticker, "2026-07-09", "f_active", f_value, "fmgp_F11"),
            ]
        )
    _insert_prices_and_factors(system, rows=rows)
    monkeypatch.setattr("main.emit_event", lambda *args, **kwargs: None)

    system.prepare_fast_live_cycle()

    assert set(system._per_account_gp_factors["B01"]["AAA"].columns) == {"b_active"}
    assert set(system._per_account_gp_factors["F11"]["AAA"].columns) == {"f_active"}
    assert system._prepared_gp_signals["B01"]["buy"][0] == "AAA"
    assert system._prepared_gp_signals["F11"]["buy"][0] == "CCC"


def test_persisted_cycle_skips_explicit_empty_gp_config_without_data_gate_error(
    tmp_path, monkeypatch
):
    system = _bare_system(tmp_path, universe=("AAA",))
    system.gp_strategies = [
        SimpleNamespace(
            id="F12", family="F", factor_selection="all",
            scoring_method="equal_weight", top_n=1, rebalance_hours=4,
        ),
    ]
    system._per_account_mined = {
        "F12": [{"name": "old", "expression": "X0", "active": False}],
    }
    system.engine.create_account("F12", initial_cash=10_000)
    monkeypatch.setattr(
        system, "_load_latest_persisted_factor_frames",
        lambda **kwargs: pytest.fail("empty config must skip persisted data gate"),
    )

    quote_tickers = system.prepare_fast_live_cycle()

    assert system._per_account_gp_factors["F12"] == {}
    assert system._prepared_gp_signals["F12"] == {"buy": [], "sell": []}
    assert quote_tickers == {"SPY"}


@pytest.mark.parametrize(
    ("factor_date", "covered", "reason"),
    [
        ("2026-07-01", ("AAA", "BBB", "CCC", "DDD"), "stale"),
        ("2026-07-09", ("AAA",), "coverage"),
    ],
)
def test_persisted_factor_freshness_and_coverage_gate_noops(
    tmp_path, monkeypatch, factor_date, covered, reason
):
    system = _bare_system(tmp_path)
    _insert_prices_and_factors(
        system,
        rows=[(ticker, factor_date, "F1", i, "alpha158") for i, ticker in enumerate(covered)],
    )
    emitted = []
    monkeypatch.setattr("main.emit_event", lambda *args, **kwargs: emitted.append((args, kwargs)))

    frames = system._load_latest_persisted_factor_frames(
        account_id="A01",
        factor_group="alpha158",
        factor_names=["F1"],
    )

    assert frames == {}
    assert emitted
    assert emitted[0][1]["detail"]["reason"] == reason


def test_empty_technical_alpha_signal_preserves_positions(tmp_path):
    system = _bare_system(tmp_path, universe=("AAA",))
    strategy = SimpleNamespace(
        id="A01", factor_names=["F1"], strategy_type="momentum",
        top_n=1, rebalance_hours=24, stop_loss=0.03, max_position_pct=0.2,
    )
    system.strategies = [strategy]
    account = system.engine.create_account("A01", initial_cash=10_000)
    account.cash = 9_000
    account._positions["HELD"] = _Position(shares=10, avg_cost=100, total_cost=1_000)
    system._prepared_alpha_signals = {"A01": {"buy": [], "sell": []}}
    system._fast_live_mode = True

    executed = system._trade_account(strategy, {"HELD": 100.0})

    assert executed is False
    assert account.get_positions() == {"HELD": 10}


def test_fast_cycle_skips_producers_and_quotes_only_prepared_bounded_tickers(
    tmp_path, monkeypatch
):
    system = _bare_system(tmp_path, universe=tuple(f"U{i:04d}" for i in range(1004)))
    account = system.engine.create_account("A01", initial_cash=10_000)
    account._positions["HELD"] = _Position(shares=1, avg_cost=10, total_cost=10)
    requested = []
    calls = []

    class Fetcher:
        def get_realtime_quote_metadata(self, tickers):
            requested.append(list(tickers))
            now = datetime.now(timezone.utc)
            return {
                ticker: RealtimeQuote(ticker, 10.0, "test", now, now, True)
                for ticker in tickers
            }

        def get_historical(self, *args, **kwargs):
            pytest.fail("fast cycle must not request historical data")

    system.fetcher = Fetcher()
    monkeypatch.setattr("main.is_market_hours_for", lambda market: True)
    monkeypatch.setattr(system, "fetch_data", lambda: pytest.fail("fetch_data must not run"))
    monkeypatch.setattr(system, "compute_factors", lambda: pytest.fail("compute_factors must not run"))
    monkeypatch.setattr(system, "mine_gp_factors", lambda: pytest.fail("mine_gp_factors must not run"))
    monkeypatch.setattr(
        system,
        "prepare_fast_live_cycle",
        lambda: {"HELD", "CANDIDATE", "SPY"},
    )
    monkeypatch.setattr(system, "initialize_benchmarks", lambda: calls.append("benchmarks"))
    monkeypatch.setattr(system, "run_trading_cycle", lambda: calls.append("alpha"))
    monkeypatch.setattr(system, "run_gp_trading_cycle", lambda: calls.append("gp"))
    monkeypatch.setattr(system, "run_qlib_trading_cycle", lambda: calls.append("qlib"))
    monkeypatch.setattr(system, "_save_all_state", lambda: calls.append("save"))

    system.run_fast_live_cycle()

    assert requested == [["CANDIDATE", "HELD", "SPY"]]
    assert len(requested[0]) < len(system.universe)
    assert calls == ["benchmarks", "alpha", "gp", "qlib", "save"]


def test_fast_cycle_can_consume_prepared_artifact_without_reloading_factors(
    tmp_path, monkeypatch
):
    system = _bare_system(tmp_path, universe=("AAA", "BBB"))
    artifact = tmp_path / "prepared.json"
    artifact.write_text(__import__("json").dumps({
        "market": "US",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "tickers": ["AAA", "SPY"],
        "prepared_alpha_signals": {"A01": {"buy": [["AAA", 1.0]], "sell": []}},
        "prepared_gp_signals": {},
        "prepared_qlib_scores": {"Q01": [["AAA", 1.0]]},
    }))
    requested = []
    class Fetcher:
        def get_realtime_quote_metadata(self, tickers):
            requested.append(list(tickers))
            now = datetime.now(timezone.utc)
            return {
                ticker: RealtimeQuote(ticker, 10.0, "test", now, now, True)
                for ticker in tickers
            }
    system.fetcher = Fetcher()
    monkeypatch.setattr("main.is_market_hours_for", lambda market: True)
    monkeypatch.setattr(
        system,
        "prepare_fast_live_cycle",
        lambda: pytest.fail("prepared artifact must avoid DB factor preparation inside lock"),
    )
    monkeypatch.setattr(system, "initialize_benchmarks", lambda: None)
    monkeypatch.setattr(system, "run_trading_cycle", lambda: None)
    monkeypatch.setattr(system, "run_gp_trading_cycle", lambda: None)
    monkeypatch.setattr(system, "run_qlib_trading_cycle", lambda: None)
    monkeypatch.setattr(system, "_save_all_state", lambda: None)

    system.run_fast_live_cycle(str(artifact))

    assert requested == [["AAA", "SPY"]]
    assert system._prepared_alpha_signals["A01"]["buy"][0][0] == "AAA"
    assert system._prepared_qlib_scores["Q01"] == [("AAA", 1.0)]


def test_fast_trade_phase_reuses_prepared_gp_and_qlib_signals(tmp_path, monkeypatch):
    system = _bare_system(tmp_path, universe=("AAA", "BBB"))
    gp = SimpleNamespace(
        id="B01", family="B", factor_selection="all", scoring_method="equal_weight",
        top_n=1, rebalance_hours=4, stop_loss=0.03, max_position_pct=0.2,
    )
    q = SimpleNamespace(
        id="Q01", top_n=1, rebalance_hours=24, stop_loss=0.04,
        max_position_pct=0.2,
    )
    system.gp_strategies = [gp]
    system.qlib_strategies = [q]
    system.engine.create_account("B01", initial_cash=10_000)
    system.engine.create_account("Q01", initial_cash=10_000)
    system._fast_live_mode = True
    system._prepared_gp_signals = {"B01": {"buy": ["AAA"], "sell": ["BBB"]}}
    system._prepared_qlib_scores = {"Q01": [("AAA", 1.0), ("BBB", 0.0)]}
    system._per_account_gp_factors = {"B01": {}}
    monkeypatch.setattr(
        system.gp_signal_gen,
        "generate_signals",
        lambda *a, **k: pytest.fail("fast GP phase must reuse prepared signals"),
    )
    monkeypatch.setattr(
        system,
        "_load_qlib_scores",
        lambda *a, **k: pytest.fail("fast Q phase must reuse prepared scores"),
    )

    assert system._trade_gp_account(gp, {"AAA": 10.0, "BBB": 10.0}) is True
    assert system._trade_qlib_account(q, {"AAA": 10.0, "BBB": 10.0}) is True


def test_qlib_coverage_gate_uses_latest_complete_date_not_sparse_tail(tmp_path, monkeypatch):
    system = _bare_system(tmp_path)
    _insert_prices_and_factors(
        system,
        rows=[
            (ticker, "2026-07-08", "qlib_Q01_score", float(i), "qlib")
            for i, ticker in enumerate(system.universe)
        ] + [("AAA", "2026-07-09", "qlib_Q01_score", 99.0, "qlib")],
    )
    emitted = []
    monkeypatch.setattr("main.emit_event", lambda *args, **kwargs: emitted.append((args, kwargs)))
    monkeypatch.setattr(
        "factors.qlib_checkpoint.checkpoint_ready_for_publication",
        lambda *args, **kwargs: (True, "ok"),
    )

    scored = system._load_qlib_scores("Q01")
    assert len(scored) == len(system.universe)
    assert scored[0][0] == "DDD"
    assert emitted == []


def test_qlib_coverage_gate_rejects_when_no_complete_date_exists(tmp_path, monkeypatch):
    system = _bare_system(tmp_path)
    _insert_prices_and_factors(
        system,
        rows=[("AAA", "2026-07-09", "qlib_Q01_score", 1.0, "qlib")],
    )
    emitted = []
    monkeypatch.setattr("main.emit_event", lambda *args, **kwargs: emitted.append((args, kwargs)))

    assert system._load_qlib_scores("Q01") == []
    assert emitted
    assert emitted[0][1]["detail"]["reason"] == "coverage"


def test_fast_cycle_cli_dispatches_one_explicit_market(monkeypatch):
    import main as trading_main

    calls = []
    monkeypatch.setattr(trading_main, "_run_for_market", lambda market, mode: calls.append((market, mode)))
    monkeypatch.setattr(sys, "argv", ["main.py", "--fast-cycle", "--market", "US"])

    trading_main.main()

    assert calls == [("US", "fast_cycle")]


def test_run_cycle_wrappers_use_fast_cycle_and_explicit_us_market():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name in ("run_cycle.sh", "run_cycle_quiet.sh"):
        text = (root / "scripts" / name).read_text()
        assert "--fast-cycle --market" in text
        assert '"$MARKET"' in text
        assert "scripts.prepare_fast_cycle" in text
        assert "run_scheduled_job.sh" in text
        scheduler = (root / "scripts/run_scheduled_job.sh").read_text()
        assert "/usr/bin/flock" in scheduler
        assert "QUANT_FAST_PREPARED_PATH" in text
        assert "main.py --cycle-no-report" not in text
