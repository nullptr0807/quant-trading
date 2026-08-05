import json
import sqlite3
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, cast

import pytest

from data.store import DataStore
from main import QuantSystem
from trading.account import _Position
from trading.costs import CNCosts, MoomooAUCosts
from trading.engine import TradingEngine


def _cn_engine(**kwargs) -> TradingEngine:
    return TradingEngine(costs=cast(Any, CNCosts()), **kwargs)


def _trade_rows(conn, account="CB16", ticker="300418.SZ"):
    return conn.execute(
        "SELECT side,shares FROM trades WHERE account=? AND ticker=? ORDER BY id",
        (account, ticker),
    ).fetchall()


def test_cn_engine_blocks_same_day_sell_but_us_is_unchanged():
    now = datetime(2026, 7, 10, 5, 25, tzinfo=timezone.utc)

    cn = _cn_engine(clock=lambda: now)
    cn_acct = cn.create_account("CB16", initial_cash=100_000)
    bought = cn.execute_signal(
        "CB16", "300418.SZ", "buy", 300, 51.88, {"300418.SZ": 51.88}
    )

    assert bought is not None
    assert cn_acct.get_sellable_shares("300418.SZ", as_of=now) == 0
    assert cn.execute_signal(
        "CB16", "300418.SZ", "sell", 300, 50.28,
        {"300418.SZ": 50.28}, reason="stop_loss",
    ) is None
    assert cn_acct.get_positions() == {"300418.SZ": 300}

    us = TradingEngine(costs=MoomooAUCosts(), clock=lambda: now)
    us_acct = us.create_account("B16", initial_cash=100_000)
    us.execute_signal("B16", "AAPL", "buy", 3, 100.0, {"AAPL": 100.0})
    sold = us.execute_signal(
        "B16", "AAPL", "sell", 3, 99.0, {"AAPL": 99.0}, reason="stop_loss"
    )
    assert sold is not None
    assert us_acct.get_positions() == {}


def test_cn_engine_sells_only_settled_lots_after_add_on_buy():
    now = datetime(2026, 7, 10, 5, 25, tzinfo=timezone.utc)
    engine = _cn_engine(clock=lambda: now)
    acct = engine.create_account("CB16", initial_cash=100_000)
    acct._positions["300418.SZ"] = _Position(
        shares=500, avg_cost=51.0, total_cost=25_500,
        lots=[
            {"shares": 200, "bought_at": "2026-07-09T02:00:00+00:00"},
            {"shares": 300, "bought_at": "2026-07-10T02:08:28+00:00"},
        ],
    )

    trade = engine.execute_signal(
        "CB16", "300418.SZ", "sell", 500, 49.0,
        {"300418.SZ": 49.0}, reason="stop_loss",
    )

    assert trade is not None
    assert trade["shares"] == 200
    assert acct.get_positions() == {"300418.SZ": 300}
    assert acct.get_sellable_shares("300418.SZ", as_of=now) == 0


def test_cn_restore_recovers_lots_from_durable_trades_and_blocks_cb16_regression(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "trading.db"
    store = DataStore(str(db_path))
    store.save_account_state("CB16", 84_000, initial_cash=100_000, market="CN")
    store.save_positions(
        "CB16",
        [{
            "ticker": "300418.SZ", "shares": 300, "avg_cost": 51.9231256667,
            "total_cost": 15_576.9377, "current_price": 50.28,
        }],
        market="CN",
    )
    store.save_trade(
        "CB16", "300418.SZ", "buy", 300, 51.88, 5.0,
        timestamp="2026-07-10T02:08:28.169510+00:00", market="CN",
    )

    system = object.__new__(QuantSystem)
    system.market = "CN"
    system.db_path = str(db_path)
    system.store = store
    system.strategies = []
    system.gp_strategies = [SimpleNamespace(id="CB16")]
    system.qlib_strategies = []
    system.benchmarks = []
    system._adaptive_enabled = False
    now = datetime(2026, 7, 10, 5, 25, tzinfo=timezone.utc)
    system.engine = _cn_engine(clock=lambda: now)
    system.engine.create_account("CB16", initial_cash=100_000)

    system._restore_state()
    acct = system.engine.get_account("CB16")

    assert acct.get_sellable_shares("300418.SZ", as_of=now) == 0
    assert system.engine.execute_signal(
        "CB16", "300418.SZ", "sell", 300, 50.28,
        {"300418.SZ": 50.28}, reason="stop_loss",
    ) is None
    with sqlite3.connect(db_path) as conn:
        assert _trade_rows(conn) == [("buy", 300.0)]
        assert conn.execute(
            "SELECT shares FROM positions WHERE account='CB16' AND ticker='300418.SZ'"
        ).fetchone()[0] == 300


def test_cn_restore_fails_closed_when_trade_ledger_cannot_explain_position(tmp_path):
    db_path = tmp_path / "trading.db"
    store = DataStore(str(db_path))
    store.save_account_state("CB16", 84_000, initial_cash=100_000, market="CN")
    store.save_positions(
        "CB16",
        [{
            "ticker": "300418.SZ", "shares": 300, "avg_cost": 51.9,
            "total_cost": 15_570, "current_price": 50.28,
        }],
        market="CN",
    )

    system = object.__new__(QuantSystem)
    system.market = "CN"
    system.db_path = str(db_path)
    system.store = store
    system.strategies = []
    system.gp_strategies = [SimpleNamespace(id="CB16")]
    system.qlib_strategies = []
    system.benchmarks = []
    system._adaptive_enabled = False
    now = datetime(2026, 7, 10, 5, 25, tzinfo=timezone.utc)
    system.engine = _cn_engine(clock=lambda: now)
    system.engine.create_account("CB16", initial_cash=100_000)

    system._restore_state()
    acct = system.engine.get_account("CB16")

    assert acct.get_sellable_shares("300418.SZ", as_of=now) == 0
    assert system.engine.execute_signal(
        "CB16", "300418.SZ", "sell", 300, 50.28,
        {"300418.SZ": 50.28}, reason="stop_loss",
    ) is None


def _make_stop_db(path):
    store = DataStore(str(path))
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS account_meta ("
            "account_id TEXT PRIMARY KEY, market TEXT, status TEXT DEFAULT 'active')"
        )
        conn.execute(
            "INSERT INTO account_meta (account_id,market,status) VALUES ('CB16','CN','active')"
        )
        conn.execute(
            "INSERT INTO account_state (account,cash,initial_cash,updated_at,market) "
            "VALUES ('CB16',84000,100000,'2026-07-10T02:08:28+00:00','CN')"
        )
        conn.execute(
            "INSERT INTO positions "
            "(account,ticker,shares,avg_cost,total_cost,current_price,updated_at,market) "
            "VALUES ('CB16','300418.SZ',300,51.9231256667,15576.9377,51.88,"
            "'2026-07-10T02:08:28+00:00','CN')"
        )
    return store


def test_stop_loss_reads_trailing_regime_from_selected_database(tmp_path, monkeypatch):
    from scripts import update_prices as updater
    from trading import risk_regime

    db = tmp_path / "selected.db"
    _make_stop_db(db)
    with sqlite3.connect(db) as con:
        con.execute("UPDATE account_meta SET market='US' WHERE account_id='CB16'")
        con.execute("UPDATE account_state SET market='US' WHERE account='CB16'")
        con.execute("UPDATE positions SET market='US',avg_cost=100,total_cost=30000")
        con.execute(
            "INSERT INTO positions_history (account,ticker,shares,avg_cost,market_price,"
            "market_value,unrealized_pnl,timestamp,market) "
            "VALUES ('CB16','300418.SZ',300,100,110,33000,3000,"
            "'2026-07-10T13:00:00+00:00','US')"
        )

    seen = []
    monkeypatch.setattr(updater, "_is_market_hours_now", lambda: True)
    monkeypatch.setattr(updater, "_is_market_open_for", lambda market: True)
    monkeypatch.setattr(
        risk_regime,
        "get_effective_trailing_stop",
        lambda *, market, db_path: seen.append((market, db_path)) or None,
    )

    with sqlite3.connect(db) as con:
        con.row_factory = sqlite3.Row
        updater.check_stop_losses(con, {"300418.SZ": 105.0}, execute=False)

    assert seen == [("US", str(db))]


def test_update_prices_cb16_cf15_same_day_stop_loss_is_guard_not_trade(
    tmp_path, monkeypatch
):
    import scripts.update_prices as updater

    db_path = tmp_path / "trading.db"
    _make_stop_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO account_meta (account_id,market,status) VALUES ('CF15','CN','active')"
        )
        conn.execute(
            "INSERT INTO account_state (account,cash,initial_cash,updated_at,market) "
            "VALUES ('CF15',79000,100000,'2026-07-10T02:08:29+00:00','CN')"
        )
        conn.execute(
            "INSERT INTO positions "
            "(account,ticker,shares,avg_cost,total_cost,current_price,updated_at,market) "
            "VALUES ('CF15','688036.SH',200,66.608941,13321.7882,66.55,"
            "'2026-07-10T02:08:29+00:00','CN')"
        )
        conn.executemany(
            "INSERT INTO trades "
            "(account,ticker,side,shares,price,cost,slippage,timestamp,market) "
            "VALUES (?,?,?,?,?,?,0,?,'CN')",
            [
                ("CB16", "300418.SZ", "buy", 300, 51.88, 5.0,
                 "2026-07-10T02:08:28.169510+00:00"),
                ("CF15", "688036.SH", "buy", 200, 66.55, 5.0,
                 "2026-07-10T02:08:29.227756+00:00"),
            ],
        )

    monkeypatch.setattr(updater, "_is_market_hours_now", lambda: True)
    monkeypatch.setattr(updater, "_is_market_open_for", lambda market: True)
    monkeypatch.setattr(
        updater, "_now_utc", lambda: datetime(2026, 7, 10, 5, 30, tzinfo=timezone.utc)
    )
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    executed = updater.check_stop_losses(
        conn, {"300418.SZ": 50.28, "688036.SH": 64.0}
    )

    assert executed == []
    assert conn.execute(
        "SELECT COUNT(*) FROM trades WHERE side='sell'"
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM positions WHERE account IN ('CB16','CF15')"
    ).fetchone()[0] == 2
    guards = conn.execute(
        "SELECT account,ticker,category,title,detail FROM events "
        "WHERE category='guard' ORDER BY account"
    ).fetchall()
    assert [(r["account"], r["ticker"]) for r in guards] == [
        ("CB16", "300418.SZ"), ("CF15", "688036.SH")
    ]
    assert all(json.loads(r["detail"])["reason"] == "cn_t_plus_one" for r in guards)


def test_update_prices_cn_stop_sells_only_yesterday_quantity(tmp_path, monkeypatch):
    import scripts.update_prices as updater

    db_path = tmp_path / "trading.db"
    _make_stop_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE positions SET shares=500,total_cost=25961.56")
        conn.executemany(
            "INSERT INTO trades "
            "(account,ticker,side,shares,price,cost,slippage,timestamp,market) "
            "VALUES ('CB16','300418.SZ','buy',?,?,5,0,?,'CN')",
            [
                (200, 51.0, "2026-07-09T02:00:00+00:00"),
                (300, 51.88, "2026-07-10T02:08:28+00:00"),
            ],
        )

    monkeypatch.setattr(updater, "_is_market_hours_now", lambda: True)
    monkeypatch.setattr(updater, "_is_market_open_for", lambda market: True)
    monkeypatch.setattr(
        updater, "_now_utc", lambda: datetime(2026, 7, 10, 5, 30, tzinfo=timezone.utc)
    )
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    executed = updater.check_stop_losses(conn, {"300418.SZ": 49.0})

    assert len(executed) == 1
    assert executed[0]["shares"] == 200
    assert conn.execute(
        "SELECT shares FROM positions WHERE account='CB16' AND ticker='300418.SZ'"
    ).fetchone()[0] == 300
    assert conn.execute(
        "SELECT shares FROM trades WHERE account='CB16' AND side='sell'"
    ).fetchone()[0] == 200


def test_update_prices_cn_limit_down_and_unknown_quote_state_fail_closed(
    tmp_path, monkeypatch
):
    import scripts.update_prices as updater

    db_path = tmp_path / "trading.db"
    _make_stop_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO trades "
            "(account,ticker,side,shares,price,cost,slippage,timestamp,market) "
            "VALUES ('CB16','300418.SZ','buy',300,51.88,5,0,"
            "'2026-07-09T02:08:28+00:00','CN')"
        )
        conn.execute(
            "INSERT INTO prices_raw (ticker,datetime,interval,open,high,low,close,volume) "
            "VALUES ('300418.SZ','2026-07-09','1d',55,55,55,55,100000)"
        )

    monkeypatch.setattr(updater, "_is_market_hours_now", lambda: True)
    monkeypatch.setattr(updater, "_is_market_open_for", lambda market: True)
    monkeypatch.setattr(
        updater, "_now_utc", lambda: datetime(2026, 7, 10, 5, 30, tzinfo=timezone.utc)
    )

    for quote, expected_reason in (
        ({"price": 49.5, "prev_close": 55.0, "volume": 12_000}, "cn_limit_down"),
        ({"price": 49.0}, "execution_state_unknown"),
        ({"price": 49.0, "prev_close": 55.0, "volume": 0}, "cn_suspended_or_no_volume"),
    ):
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        executed = updater.check_stop_losses(conn, {"300418.SZ": quote})
        assert executed == []
        assert conn.execute("SELECT COUNT(*) FROM trades WHERE side='sell'").fetchone()[0] == 0
        detail = json.loads(conn.execute(
            "SELECT detail FROM events WHERE category='guard' ORDER BY id DESC LIMIT 1"
        ).fetchone()[0])
        assert detail["reason"] == expected_reason
        conn.close()


def test_main_cn_trade_guard_blocks_limit_and_unknown_state(tmp_path, monkeypatch):
    db_path = tmp_path / "trading.db"
    store = DataStore(str(db_path))
    system = object.__new__(QuantSystem)
    system.market = "CN"
    system.db_path = str(db_path)
    system.store = store
    system.engine = _cn_engine(clock=lambda: datetime(2026, 7, 10, 5, 30, tzinfo=timezone.utc))
    acct = system.engine.create_account("CB16", initial_cash=100_000)
    acct._positions["300418.SZ"] = _Position(
        shares=300, avg_cost=51.9, total_cost=15_570,
        lots=[{"shares": 300, "bought_at": "2026-07-09T02:00:00+00:00"}],
    )
    system._realtime_quote_details = {
        "300418.SZ": {"price": 49.5, "prev_close": 55.0, "volume": 12_000}
    }
    emitted = []
    monkeypatch.setattr("main.emit_event", lambda *a, **k: emitted.append((a, k)))

    assert system._execute_live_signal(
        "CB16", "300418.SZ", "sell", 300, 49.5,
        {"300418.SZ": 49.5}, reason="stop_loss",
    ) is None
    assert acct.get_positions() == {"300418.SZ": 300}
    assert emitted[-1][0][0] == "guard"
    assert emitted[-1][1]["detail"]["reason"] == "cn_limit_down"

    system._realtime_quote_details = {}
    assert system._execute_live_signal(
        "CB16", "300418.SZ", "sell", 300, 49.0,
        {"300418.SZ": 49.0}, reason="stop_loss",
    ) is None
    assert emitted[-1][1]["detail"]["reason"] == "execution_state_unknown"


def test_update_prices_stop_loss_map_covers_a_and_q_accounts_both_markets():
    from accounts.qlib_strategies import QLIB_STRATEGIES
    from accounts.strategies import STRATEGIES
    from scripts.update_prices import STOP_LOSS_BY_ACCT

    for strat in STRATEGIES:
        assert STOP_LOSS_BY_ACCT[strat.id] == strat.stop_loss
        assert STOP_LOSS_BY_ACCT[f"C{strat.id}"] == strat.stop_loss
    for strat in QLIB_STRATEGIES:
        assert STOP_LOSS_BY_ACCT[strat.id] == strat.stop_loss
        assert STOP_LOSS_BY_ACCT[f"C{strat.id}"] == strat.stop_loss
