from typing import cast, Any

from trading.account import VirtualAccount, _Position
from trading.costs import CNCosts, MoomooAUCosts
from trading.engine import TradingEngine


def _cn_engine(**kwargs) -> TradingEngine:
    return TradingEngine(costs=cast(Any, CNCosts()), **kwargs)


def test_cn_buy_rounds_down_to_100_share_lots():
    trades = []
    engine = _cn_engine(trade_callback=lambda acct, trade: trades.append(trade))
    engine.create_account("CA01", initial_cash=100_000)

    trade = engine.execute_signal("CA01", "600519.SH", "buy", 123, 10.0, {"600519.SH": 10.0})

    assert trade is not None
    assert trade["shares"] == 100
    assert engine.get_account("CA01").get_positions()["600519.SH"] == 100
    assert trades[-1]["shares"] == 100


def test_cn_buy_below_one_lot_is_blocked():
    engine = _cn_engine()
    engine.create_account("CA01", initial_cash=100_000)

    trade = engine.execute_signal("CA01", "600519.SH", "buy", 99, 10.0, {"600519.SH": 10.0})

    assert trade is None
    assert engine.get_account("CA01").get_positions() == {}


def test_cn_buy_after_position_limit_rounds_down_to_100_share_lots():
    engine = _cn_engine(max_position_pct=0.30)
    engine.create_account("CA01", initial_cash=100_000)

    trade = engine.execute_signal("CA01", "600519.SH", "buy", 5000, 10.0, {"600519.SH": 10.0})

    assert trade is not None
    assert trade["shares"] == 2900
    assert trade["shares"] % 100 == 0


def test_us_buy_still_allows_single_share_orders():
    engine = TradingEngine(costs=MoomooAUCosts())
    engine.create_account("A01", initial_cash=10_000)

    trade = engine.execute_signal("A01", "AAPL", "buy", 17, 100.0, {"AAPL": 100.0})

    assert trade is not None
    assert trade["shares"] == 17


def test_cn_full_sell_can_clear_legacy_odd_lot_position():
    engine = _cn_engine()
    acct: VirtualAccount = engine.create_account("CA01", initial_cash=100_000)
    # Existing live CN positions already include legacy odd lots. Full exits must
    # remain possible so repair/stop-loss/signal sells can clear them naturally.
    acct._positions["600519.SH"] = _Position(shares=415, avg_cost=10.0, total_cost=4150.0)

    trade = engine.execute_signal("CA01", "600519.SH", "sell", 415, 10.0, {"600519.SH": 10.0})

    assert trade is not None
    assert trade["shares"] == 415
    assert acct.get_positions() == {}
