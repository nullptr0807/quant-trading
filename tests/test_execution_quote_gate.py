from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from data.quotes import RealtimeQuote
from tests.test_persisted_live_cycle import _bare_system


@pytest.mark.parametrize(
    "quote_factory,reason",
    [
        (lambda now: {}, "missing"),
        (lambda now: {"AAA": RealtimeQuote("AAA", 10, "test", now - timedelta(seconds=181), now, True)}, "stale"),
        (lambda now: {"AAA": RealtimeQuote("AAA", 10, "test", now, now, False)}, "untradable"),
        (lambda now: {"AAA": RealtimeQuote("AAA", 0, "test", now, now, True)}, "invalid"),
        (lambda now: {"AAA": RealtimeQuote("AAA", 10, "test", None, now, True)}, "stale"),
    ],
)
def test_legacy_execution_quote_gate_rejects_bad_metadata(
    tmp_path, monkeypatch, quote_factory, reason
):
    system = _bare_system(tmp_path, universe=("AAA",))
    system.benchmarks = []
    now = datetime(2026, 7, 10, 14, 0, tzinfo=timezone.utc)
    system.fetcher = SimpleNamespace(
        get_realtime_quote_metadata=lambda tickers: quote_factory(now)
    )
    monkeypatch.setattr("main._utc_now", lambda: now)

    with pytest.raises(RuntimeError, match=reason):
        system._fetch_realtime_prices(strict_execution=True)


def test_run_once_rejects_scalar_and_cannot_trade_historical_fallback(tmp_path, monkeypatch):
    system = _bare_system(tmp_path, universe=("AAA",))
    system.benchmarks = []
    system.fetcher = SimpleNamespace(get_realtime_quotes=lambda tickers: {"AAA": 10.0})
    monkeypatch.setattr("main.is_market_hours_for", lambda market: True)
    monkeypatch.setattr(system, "fetch_data", lambda: None)
    monkeypatch.setattr(system, "compute_factors", lambda: None)
    monkeypatch.setattr(system, "mine_gp_factors", lambda: None)
    calls = []
    monkeypatch.setattr(system, "run_trading_cycle", lambda: calls.append("trade"))

    with pytest.raises(RuntimeError, match="legacy scalar quotes are display-only"):
        system.run_once(send_report=False)
    assert calls == []


def test_report_only_keeps_legacy_scalar_quotes_read_only(tmp_path, monkeypatch):
    system = _bare_system(tmp_path, universe=("AAA",))
    system.benchmarks = []
    system.fetcher = SimpleNamespace(get_realtime_quotes=lambda tickers: {"AAA": 10.0})
    monkeypatch.setattr("main.is_market_hours_for", lambda market: False)
    monkeypatch.setattr(system, "fetch_data", lambda: None)

    assert system.run_once(send_report=False) is None
    assert system._realtime_prices == {"AAA": 10.0}
