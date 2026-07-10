import copy
import sqlite3

import pytest

from data.store import DataStore
from main import QuantSystem
from trading.account import _Position
from trading.engine import TradingEngine


def _account_state(account):
    return {
        "cash": account.cash,
        "positions": copy.deepcopy(account._positions),
        "trade_log": copy.deepcopy(account.trade_log),
    }


def _system_with_account(tmp_path, account_name="A01"):
    system = object.__new__(QuantSystem)
    system.market = "US"
    system.store = DataStore(str(tmp_path / "trading.db"))
    system.engine = TradingEngine(trade_callback=system._on_trade)
    account = system.engine.create_account(account_name, initial_cash=10_000)
    return system, account


def _count_rows(db_path, table):
    with sqlite3.connect(db_path) as conn:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


@pytest.mark.parametrize("side", ["buy", "sell"])
def test_callback_failure_rolls_back_account_memory(side):
    def fail_callback(_account_name, _trade):
        raise RuntimeError("persistence failed")

    engine = TradingEngine(trade_callback=fail_callback)
    account = engine.create_account("A01", initial_cash=10_000)
    if side == "sell":
        account.cash = 8_000
        account._positions["AAPL"] = _Position(
            shares=10, avg_cost=200.0, total_cost=2_000.0
        )
    before = _account_state(account)

    with pytest.raises(RuntimeError, match="persistence failed"):
        engine.execute_signal(
            "A01", "AAPL", side, 5, 100.0, {"AAPL": 100.0}
        )

    assert _account_state(account) == before


def test_successful_main_trade_atomically_persists_ledger_and_state(tmp_path):
    system, account = _system_with_account(tmp_path)

    trade = system.engine.execute_signal(
        "A01", "AAPL", "buy", 5, 100.0, {"AAPL": 100.0}
    )

    assert trade is not None
    with sqlite3.connect(system.store.db_path) as conn:
        persisted_trade = conn.execute(
            "SELECT account,ticker,side,shares,price,market FROM trades"
        ).fetchone()
        persisted_event = conn.execute(
            "SELECT category,account,ticker,market FROM events"
        ).fetchone()
        persisted_state = conn.execute(
            "SELECT cash,initial_cash,market FROM account_state WHERE account='A01'"
        ).fetchone()
        persisted_position = conn.execute(
            "SELECT ticker,shares,avg_cost,total_cost,market "
            "FROM positions WHERE account='A01'"
        ).fetchone()
        counts = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("trades", "events", "account_state", "positions")
        }

    assert persisted_trade == ("A01", "AAPL", "buy", 5.0, 100.0, "US")
    assert persisted_event == ("trade", "A01", "AAPL", "US")
    assert persisted_state[:2] == pytest.approx((account.cash, account.initial_cash))
    assert persisted_state[2] == "US"
    assert persisted_position[:2] == ("AAPL", 5.0)
    assert persisted_position[2:4] == pytest.approx(
        (account._positions["AAPL"].avg_cost, account._positions["AAPL"].total_cost)
    )
    assert persisted_position[4] == "US"
    assert counts == {"trades": 1, "events": 1, "account_state": 1, "positions": 1}


def test_atomic_trade_preserves_current_prices_for_untouched_positions(tmp_path):
    system, account = _system_with_account(tmp_path)
    account._positions["MSFT"] = _Position(
        shares=2, avg_cost=300.0, total_cost=600.0
    )
    with sqlite3.connect(system.store.db_path) as conn:
        conn.execute(
            "INSERT INTO account_state (account,cash,initial_cash,updated_at,market) "
            "VALUES ('A01',9400,10000,'2026-07-09T20:00:00+00:00','US')"
        )
        conn.execute(
            "INSERT INTO positions "
            "(account,ticker,shares,avg_cost,total_cost,current_price,updated_at,market) "
            "VALUES ('A01','MSFT',2,300,600,432.1,'2026-07-09T20:00:00+00:00','US')"
        )

    system.engine.execute_signal(
        "A01", "AAPL", "buy", 5, 100.0, {"AAPL": 100.0, "MSFT": 432.1}
    )

    with sqlite3.connect(system.store.db_path) as conn:
        msft_price = conn.execute(
            "SELECT current_price FROM positions WHERE account='A01' AND ticker='MSFT'"
        ).fetchone()[0]
    assert msft_price == pytest.approx(432.1)


def test_atomic_trade_persists_slippage_amount(tmp_path):
    system, _ = _system_with_account(tmp_path)

    system.engine.execute_signal(
        "A01", "AAPL", "buy", 5, 100.0, {"AAPL": 100.0}
    )

    with sqlite3.connect(system.store.db_path) as conn:
        slippage = conn.execute("SELECT slippage FROM trades").fetchone()[0]
    assert slippage == pytest.approx(5 * (100.05 - 100.0))


def test_injected_atomic_write_failure_leaves_memory_and_database_unchanged(tmp_path):
    system, account = _system_with_account(tmp_path)
    before = _account_state(account)

    with sqlite3.connect(system.store.db_path) as conn:
        conn.execute(
            """
            CREATE TRIGGER fail_trade_event
            BEFORE INSERT ON events
            BEGIN
                SELECT RAISE(ABORT, 'injected event insert failure');
            END
            """
        )

    with pytest.raises(RuntimeError, match="injected event insert failure"):
        system.engine.execute_signal(
            "A01", "AAPL", "buy", 5, 100.0, {"AAPL": 100.0}
        )

    assert _account_state(account) == before
    for table in ("trades", "events", "account_state", "positions"):
        assert _count_rows(system.store.db_path, table) == 0


def test_atomic_trade_refuses_to_overwrite_an_unrestored_db_book(tmp_path):
    system, account = _system_with_account(tmp_path)
    with sqlite3.connect(system.store.db_path) as conn:
        conn.execute(
            "INSERT INTO positions "
            "(account,ticker,shares,avg_cost,total_cost,current_price,updated_at,market) "
            "VALUES ('A01','MSFT',2,300,600,432.1,'2026-07-09T20:00:00+00:00','US')"
        )
    before = _account_state(account)

    with pytest.raises(RuntimeError, match="precondition failed"):
        system.engine.execute_signal(
            "A01", "AAPL", "buy", 5, 100.0, {"AAPL": 100.0}
        )

    assert _account_state(account) == before
    with sqlite3.connect(system.store.db_path) as conn:
        assert conn.execute(
            "SELECT ticker,shares FROM positions WHERE account='A01'"
        ).fetchall() == [("MSFT", 2.0)]
        assert conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 0


def test_benchmark_trade_uses_same_fail_closed_engine_path():
    def fail_callback(_account_name, _trade):
        raise RuntimeError("persistence failed")

    system = object.__new__(QuantSystem)
    system.benchmarks = [{"id": "IDX_SPY", "ticker": "SPY"}]
    system.engine = TradingEngine(trade_callback=fail_callback)
    account = system.engine.create_account("IDX_SPY", initial_cash=10_000)
    system._get_current_prices = lambda *args, **kwargs: {"SPY": 100.0}
    before = _account_state(account)

    with pytest.raises(RuntimeError, match="persistence failed"):
        system.initialize_benchmarks()

    assert _account_state(account) == before
