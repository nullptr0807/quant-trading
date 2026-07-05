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


def test_cn_budget_too_small_for_one_board_lot_returns_zero_buy_shares():
    engine = _cn_engine()

    assert engine.buy_shares_for_budget(20_000, 840.0) == 0


def test_cn_buy_budget_rounds_to_affordable_board_lots():
    engine = _cn_engine()

    assert engine.buy_shares_for_budget(20_000, 37.0) == 500


def test_us_buy_budget_still_allows_single_share_sizing():
    engine = TradingEngine(costs=MoomooAUCosts())

    assert engine.buy_shares_for_budget(20_000, 840.0) == 23


def test_cn_unbuyable_budget_exposes_skip_detail():
    engine = _cn_engine()

    detail = engine.buy_skip_detail_for_budget(20_000, 840.0)

    assert detail is not None
    assert detail["reason"] == "board_lot_unaffordable"
    assert detail["lot_size"] == 100
    assert detail["min_lot_notional"] > 84_000
    assert detail["budget"] == 20_000
    assert detail["price"] == 840.0


def test_cn_affordable_budget_has_no_skip_detail():
    engine = _cn_engine()

    assert engine.buy_skip_detail_for_budget(20_000, 37.0) is None


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
