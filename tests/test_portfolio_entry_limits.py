import pytest

from trading.engine import TradingEngine


def test_buy_respects_gross_exposure_and_cash_reserve():
    engine = TradingEngine(
        max_position_pct=1.0,
        max_gross_exposure_pct=0.80,
        min_cash_reserve_pct=0.20,
    )
    account = engine.create_account("A", initial_cash=10_000)

    trade = engine.execute_signal("A", "AAA", "buy", 100, 100, {"AAA": 100})

    assert trade is not None
    gross = account.get_positions()["AAA"] * 100
    assert gross <= 8_000 + 1e-6
    assert account.cash >= 2_000 - 1e-6


def test_new_buy_is_blocked_when_portfolio_gross_limit_is_full():
    engine = TradingEngine(
        max_position_pct=1.0,
        max_gross_exposure_pct=0.50,
        min_cash_reserve_pct=0.50,
    )
    account = engine.create_account("A", initial_cash=10_000)
    assert engine.execute_signal("A", "AAA", "buy", 100, 100, {"AAA": 100})

    result = engine.execute_signal(
        "A", "BBB", "buy", 10, 100,
        {"AAA": 100, "BBB": 100},
    )
    assert result is None
    assert "BBB" not in account.get_positions()


def test_inconsistent_portfolio_limits_are_rejected():
    with pytest.raises(ValueError, match="inconsistent"):
        TradingEngine(max_gross_exposure_pct=0.90, min_cash_reserve_pct=0.20)
