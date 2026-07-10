from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest


def _sina_payload(*, price: str, date: str, time: str) -> str:
    fields = [
        "测试股份", "9.90", "9.80", price, "10.10", "9.70",
        *(["0"] * 24), date, time, "00",
    ]
    return ",".join(fields)


def test_us_quote_metadata_keeps_finnhub_source_timestamp(monkeypatch):
    import data.fetcher as fetcher_module

    class EmptyFastInfoTicker:
        fast_info = {"lastPrice": None}

    fetcher = object.__new__(fetcher_module.DataFetcher)
    fetcher.finnhub_client = SimpleNamespace(
        quote=lambda ticker: {"c": 123.45, "t": 1_786_745_400}
    )
    monkeypatch.setattr(fetcher_module.yf, "Ticker", lambda ticker: EmptyFastInfoTicker())

    quotes = fetcher.get_realtime_quote_metadata(["AAPL"])

    quote = quotes["AAPL"]
    assert quote.price == 123.45
    assert quote.source == "finnhub"
    assert quote.source_timestamp == datetime.fromtimestamp(
        1_786_745_400, tz=timezone.utc
    )
    assert quote.tradable is True
    assert fetcher.get_realtime_quotes(["AAPL"]) == {"AAPL": 123.45}


def test_cn_sina_quote_metadata_keeps_exchange_date_time(monkeypatch):
    import data.cn_fetcher as cn_fetcher

    body = (
        'var hq_str_sz000001="'
        + _sina_payload(price="10.25", date="2026-07-10", time="14:59:30")
        + '";'
    ).encode("gbk")

    class Response:
        def read(self):
            return body

    monkeypatch.setattr(cn_fetcher.urllib.request, "urlopen", lambda *a, **k: Response())
    fetcher = object.__new__(cn_fetcher.CNDataFetcher)

    quotes = fetcher.get_realtime_quote_metadata(["000001.SZ"])

    quote = quotes["000001.SZ"]
    assert quote.price == 10.25
    assert quote.source == "sina"
    assert quote.source_timestamp.isoformat() == "2026-07-10T06:59:30+00:00"
    assert quote.tradable is True


def test_update_prices_rejects_partial_or_stale_held_quotes_and_writes_no_heartbeat(
    tmp_path, monkeypatch
):
    import scripts.update_prices as updater
    from data.quotes import RealtimeQuote
    from data.store import DataStore

    db = tmp_path / "trading.db"
    DataStore(str(db))
    with sqlite3.connect(db) as con:
        con.executemany(
            "INSERT INTO account_meta (account_id,market,status) VALUES (?,?,?)",
            [("A01", "US", "active")],
        )
        con.execute(
            "INSERT INTO account_state (account,cash,initial_cash,updated_at,market) "
            "VALUES ('A01',9000,10000,'2026-07-10T13:00:00+00:00','US')"
        )
        con.executemany(
            "INSERT INTO positions "
            "(account,ticker,shares,avg_cost,total_cost,current_price,updated_at,market) "
            "VALUES ('A01',?,1,100,100,100,'2026-07-10T13:00:00+00:00','US')",
            [("AAPL",), ("MSFT",)],
        )

    now = datetime(2026, 7, 10, 14, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(updater, "_utc_now", lambda: now)
    monkeypatch.setattr(updater, "_is_us_market_hours_now", lambda: True)
    monkeypatch.setattr(updater, "_is_cn_market_hours_now", lambda: False)
    monkeypatch.setattr(updater, "_force_price_update", lambda: False)
    monkeypatch.setattr(
        updater,
        "fetch_quote_metadata",
        lambda tickers: {
            "AAPL": RealtimeQuote(
                ticker="AAPL",
                price=101.0,
                source="finnhub",
                source_timestamp=now - timedelta(minutes=30),
                received_at=now,
                tradable=True,
            )
        },
    )

    with pytest.raises(RuntimeError, match="quote validation failed"):
        updater.update_equity_snapshots(str(db))

    with sqlite3.connect(db) as con:
        assert con.execute("SELECT COUNT(*) FROM operational_health").fetchone()[0] == 0


def test_update_prices_success_writes_durable_market_heartbeat(tmp_path, monkeypatch):
    import scripts.update_prices as updater
    from data.quotes import RealtimeQuote
    from data.store import DataStore

    db = tmp_path / "trading.db"
    DataStore(str(db))
    with sqlite3.connect(db) as con:
        con.execute(
            "INSERT INTO account_meta (account_id,market,status) VALUES ('A01','US','active')"
        )
        con.execute(
            "INSERT INTO account_state (account,cash,initial_cash,updated_at,market) "
            "VALUES ('A01',9000,10000,'2026-07-10T13:00:00+00:00','US')"
        )
        con.execute(
            "INSERT INTO positions "
            "(account,ticker,shares,avg_cost,total_cost,current_price,updated_at,market) "
            "VALUES ('A01','AAPL',1,100,100,100,'2026-07-10T13:00:00+00:00','US')"
        )

    now = datetime(2026, 7, 10, 14, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(updater, "_utc_now", lambda: now)
    monkeypatch.setattr(updater, "_is_us_market_hours_now", lambda: True)
    monkeypatch.setattr(updater, "_is_cn_market_hours_now", lambda: False)
    monkeypatch.setattr(updater, "_force_price_update", lambda: False)
    monkeypatch.setattr(updater, "check_stop_losses", lambda conn, prices, **kwargs: [])
    monkeypatch.setattr(
        updater,
        "fetch_quote_metadata",
        lambda tickers: {
            "AAPL": RealtimeQuote(
                ticker="AAPL", price=101.0, source="finnhub",
                source_timestamp=now - timedelta(seconds=5), received_at=now,
                tradable=True,
            )
        },
    )

    stats = updater.update_equity_snapshots(str(db))

    assert stats["held_coverage"] == 1.0
    with sqlite3.connect(db) as con:
        row = con.execute(
            "SELECT component,market,status,success_at,details "
            "FROM operational_health WHERE component='update_prices' AND market='US'"
        ).fetchone()
    assert row[:4] == ("update_prices", "US", "ok", now.isoformat())
    assert '"coverage": 1.0' in row[4]


def test_fast_cycle_fails_closed_for_missing_or_untradable_held_quote(tmp_path, monkeypatch):
    from data.quotes import RealtimeQuote
    from tests.test_persisted_live_cycle import _bare_system
    from trading.account import _Position

    system = _bare_system(tmp_path, universe=("AAA",))
    account = system.engine.create_account("A01", initial_cash=10_000)
    account._positions["HELD"] = _Position(shares=1, avg_cost=10, total_cost=10)
    now = datetime(2026, 7, 10, 14, 0, tzinfo=timezone.utc)
    system.fetcher = SimpleNamespace(
        get_realtime_quote_metadata=lambda tickers: {
            ticker: RealtimeQuote(
                ticker=ticker, price=10.0, source="test",
                source_timestamp=now, received_at=now,
                tradable=(ticker != "HELD"),
            )
            for ticker in tickers
        }
    )
    monkeypatch.setattr("main.is_market_hours_for", lambda market: True)
    monkeypatch.setattr("main._utc_now", lambda: now)
    monkeypatch.setattr(system, "prepare_fast_live_cycle", lambda: {"HELD", "AAA", "SPY"})

    with pytest.raises(RuntimeError, match="fast-cycle quote validation failed"):
        system.run_fast_live_cycle()


def test_fast_cycle_rejects_stale_prepared_artifact(tmp_path, monkeypatch):
    from tests.test_persisted_live_cycle import _bare_system

    system = _bare_system(tmp_path, universe=("AAA",))
    artifact = tmp_path / "prepared.json"
    artifact.write_text(__import__("json").dumps({
        "market": "US",
        "prepared_at": "2026-07-10T13:00:00+00:00",
        "tickers": ["AAA", "SPY"],
        "prepared_alpha_signals": {},
        "prepared_gp_signals": {},
        "prepared_qlib_scores": {},
    }))
    monkeypatch.setattr("main.is_market_hours_for", lambda market: True)
    monkeypatch.setattr(
        "main._utc_now",
        lambda: datetime(2026, 7, 10, 14, 0, tzinfo=timezone.utc),
    )

    with pytest.raises(RuntimeError, match="artifact stale"):
        system.run_fast_live_cycle(str(artifact))


def test_fast_cycle_rejects_prepared_artifact_with_missing_held_ticker(
    tmp_path, monkeypatch
):
    from tests.test_persisted_live_cycle import _bare_system
    from trading.account import _Position

    system = _bare_system(tmp_path, universe=("AAA",))
    account = system.engine.create_account("A01", initial_cash=10_000)
    account._positions["HELD"] = _Position(shares=1, avg_cost=10, total_cost=10)
    artifact = tmp_path / "prepared.json"
    artifact.write_text(__import__("json").dumps({
        "market": "US",
        "tickers": ["AAA", "SPY"],
        "prepared_alpha_signals": {},
        "prepared_gp_signals": {},
        "prepared_qlib_scores": {},
    }))
    monkeypatch.setattr("main.is_market_hours_for", lambda market: True)

    with pytest.raises(RuntimeError, match="missing active holdings"):
        system.run_fast_live_cycle(str(artifact))


def test_fast_cycle_rejects_untradable_candidate_quote_before_buy(tmp_path, monkeypatch):
    from data.quotes import RealtimeQuote
    from tests.test_persisted_live_cycle import _bare_system

    system = _bare_system(tmp_path, universe=("CANDIDATE",))
    now = datetime(2026, 7, 10, 14, 0, tzinfo=timezone.utc)
    system.fetcher = SimpleNamespace(
        get_realtime_quote_metadata=lambda tickers: {
            ticker: RealtimeQuote(
                ticker=ticker,
                price=10.0,
                source="test",
                source_timestamp=now,
                received_at=now,
                tradable=(ticker != "CANDIDATE"),
            )
            for ticker in tickers
        }
    )
    monkeypatch.setattr("main.is_market_hours_for", lambda market: True)
    monkeypatch.setattr("main._utc_now", lambda: now)
    monkeypatch.setattr(
        system, "prepare_fast_live_cycle", lambda: {"CANDIDATE", "SPY"}
    )

    with pytest.raises(RuntimeError, match="fast-cycle quote validation failed"):
        system.run_fast_live_cycle()


def test_health_check_flags_stale_heartbeat_and_snapshot_only_during_session():
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).parents[1] / "scripts" / "health_check.py"
    spec = importlib.util.spec_from_file_location("hardening_health_check", module_path)
    assert spec and spec.loader
    health_check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(health_check)
    check_live_runtime_health = health_check.check_live_runtime_health

    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.executescript(
        """
        CREATE TABLE operational_health (
          component TEXT, market TEXT, status TEXT, success_at TEXT,
          source_timestamp TEXT, details TEXT,
          PRIMARY KEY(component, market)
        );
        CREATE TABLE account_meta (account_id TEXT, market TEXT, status TEXT);
        CREATE TABLE accounts (name TEXT, market TEXT, timestamp TEXT);
        INSERT INTO account_meta VALUES ('A01','US','active');
        INSERT INTO operational_health VALUES
          ('update_prices','US','ok','2026-07-10T13:00:00+00:00',
           '2026-07-10T12:59:59+00:00','{}');
        INSERT INTO accounts VALUES ('A01','US','2026-07-10T13:00:00+00:00');
        """
    )
    now = datetime(2026, 7, 10, 14, 0, tzinfo=timezone.utc)

    issues = check_live_runtime_health(
        con,
        now=now,
        session_open={"US": True, "CN": False},
        max_heartbeat_age_seconds=180,
        max_snapshot_age_seconds=180,
    )

    assert {issue["check"] for issue in issues} == {
        "update_prices_heartbeat",
        "equity_snapshot_freshness",
    }
    assert all(issue["severity"] == "critical" for issue in issues)

    assert check_live_runtime_health(
        con,
        now=now,
        session_open={"US": False, "CN": False},
        max_heartbeat_age_seconds=180,
        max_snapshot_age_seconds=180,
    ) == []