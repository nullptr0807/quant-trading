import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone

import pandas as pd
import pytest


def test_cn_daily_cache_read_includes_latest_midnight_bar(tmp_path):
    from data.cn_fetcher import CNDataFetcher
    from data.store import DataStore
    from datetime import datetime, timedelta, timezone

    store = DataStore(str(tmp_path / 'cn.db'))
    today = datetime.now(timezone.utc).date()
    rows = pd.DataFrame([
        {
            'ticker': '000001.SZ', 'datetime': (today - timedelta(days=i)).isoformat(),
            'open': 10, 'high': 11, 'low': 9, 'close': 10, 'volume': 100,
        }
        for i in range(3)
    ])
    store.save_prices_bulk(rows, interval='1d')
    fetcher = object.__new__(CNDataFetcher)
    fetcher.store = store

    result = fetcher.get_historical(['000001.SZ'], days=5, interval='1d')

    assert today.isoformat() in set(pd.to_datetime(result['datetime']).dt.date.astype(str))


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        # Before the US close, Friday's incomplete daily bar is not required.
        (datetime(2026, 7, 10, 19, 59, tzinfo=timezone.utc), date(2026, 7, 9)),
        # The same 16:00 local boundary holds in winter (EST, UTC-5).
        (datetime(2026, 1, 9, 20, 59, tzinfo=timezone.utc), date(2026, 1, 8)),
        (datetime(2026, 1, 9, 21, 1, tzinfo=timezone.utc), date(2026, 1, 9)),
        # After 16:00 America/New_York, Friday is the completed target session.
        (datetime(2026, 7, 10, 20, 1, tzinfo=timezone.utc), date(2026, 7, 10)),
        # Weekend targets the most recent completed weekday session.
        (datetime(2026, 7, 11, 15, 0, tzinfo=timezone.utc), date(2026, 7, 10)),
        (datetime(2026, 7, 12, 15, 0, tzinfo=timezone.utc), date(2026, 7, 10)),
        # Monday before close still targets Friday.
        (datetime(2026, 7, 13, 15, 0, tzinfo=timezone.utc), date(2026, 7, 10)),
    ],
)
def test_latest_completed_us_session_uses_new_york_close_and_weekdays(now, expected):
    from data.fetcher import latest_completed_session_date

    assert latest_completed_session_date("US", now) == expected


def test_daily_cache_is_fresh_at_preclose_when_latest_completed_session_exists(monkeypatch):
    import data.fetcher as fetcher_module

    cached = pd.DataFrame([
        {
            "ticker": "AAPL",
            "datetime": "2026-07-09",
            "open": 1,
            "high": 2,
            "low": 1,
            "close": 2,
            "volume": 10,
        },
    ])

    class Store:
        def load_prices(self, *args, **kwargs):
            return cached.copy()

        def get_price_coverage(self, *args, **kwargs):
            return {"AAPL": ("2026-06-01", "2026-07-09", 30)}

    fetcher = object.__new__(fetcher_module.DataFetcher)
    fetcher.store = Store()
    monkeypatch.setattr(
        fetcher_module,
        "_utc_now",
        lambda: datetime(2026, 7, 10, 19, 0, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(
        fetcher,
        "_fetch_yf_batch",
        lambda *args, **kwargs: pytest.fail("completed-session cache should not download"),
    )

    result = fetcher.get_historical(["AAPL"], days=30, interval="1d")

    assert result.equals(cached)


def test_daily_yahoo_request_includes_completed_target_with_exclusive_end(monkeypatch):
    import data.fetcher as fetcher_module

    calls = []

    class Store:
        def load_prices(self, *args, **kwargs):
            return pd.DataFrame()

        def get_price_coverage(self, *args, **kwargs):
            return {}

        def save_prices_bulk(self, *args, **kwargs):
            pass

    fetcher = object.__new__(fetcher_module.DataFetcher)
    fetcher.store = Store()
    monkeypatch.setattr(
        fetcher_module,
        "_utc_now",
        lambda: datetime(2026, 7, 10, 21, 0, tzinfo=timezone.utc),
    )

    def fake_fetch(tickers, start, end, interval, price_mode="adjusted"):
        calls.append((tickers, start, end, interval, price_mode))
        return pd.DataFrame()

    monkeypatch.setattr(fetcher, "_fetch_yf_batch", fake_fetch)

    fetcher.get_historical(["AAPL"], days=30, interval="1d")

    # Yahoo's end is exclusive, so Friday's target requires Saturday as end.
    assert calls[0][2] == "2026-07-11"


class _RecordingHistoricalExecutor:
    instances = []

    def __init__(self, *args, **kwargs):
        self._inner = ThreadPoolExecutor(*args, **kwargs)
        self.shutdown_calls = []
        type(self).instances.append(self)

    def submit(self, *args, **kwargs):
        return self._inner.submit(*args, **kwargs)

    def shutdown(self, *args, **kwargs):
        self.shutdown_calls.append((args, kwargs))
        return self._inner.shutdown(*args, **kwargs)


def _price_frame(ticker):
    return pd.DataFrame([
        {
            "ticker": ticker,
            "datetime": "2026-07-09",
            "open": 1,
            "high": 2,
            "low": 1,
            "close": 2,
            "volume": 10,
        },
    ])


def test_cn_historical_force_refresh_returns_partial_on_total_timeout(monkeypatch):
    import data.cn_fetcher as cn_fetcher

    _RecordingHistoricalExecutor.instances.clear()
    release = threading.Event()

    class Store:
        def save_prices_bulk(self, *args, **kwargs):
            pass

    def fake_hist(ticker, *args, **kwargs):
        if ticker == "HANG.SH":
            release.wait(2)
        return _price_frame(ticker)

    fetcher = object.__new__(cn_fetcher.CNDataFetcher)
    fetcher.store = Store()
    monkeypatch.setattr(cn_fetcher, "ThreadPoolExecutor", _RecordingHistoricalExecutor)
    monkeypatch.setattr(cn_fetcher, "_hist_one", fake_hist)

    started = time.monotonic()
    try:
        result = fetcher.get_historical(
            ["FAST.SH", "HANG.SH"],
            days=5,
            use_cache=False,
            batch_timeout_s=0.05,
        )
    finally:
        release.set()

    assert time.monotonic() - started < 0.5
    assert set(result["ticker"]) == {"FAST.SH"}
    assert _RecordingHistoricalExecutor.instances[0].shutdown_calls == [
        ((), {"wait": False, "cancel_futures": True})
    ]


def test_cn_historical_cache_miss_returns_cached_plus_partial_on_total_timeout(monkeypatch):
    import data.cn_fetcher as cn_fetcher

    _RecordingHistoricalExecutor.instances.clear()
    release = threading.Event()
    cached = _price_frame("CACHED.SH")
    persisted = []

    class Store:
        def load_prices(self, *args, **kwargs):
            if persisted:
                return pd.concat([cached, *persisted], ignore_index=True)
            return cached.copy()

        def get_price_coverage(self, *args, **kwargs):
            return {"CACHED.SH": ("2026-06-01", "2099-01-01", 30)}

        def save_prices_bulk(self, frame, *args, **kwargs):
            persisted.append(frame.copy())

    def fake_hist(ticker, *args, **kwargs):
        if ticker == "HANG.SH":
            release.wait(2)
        return _price_frame(ticker)

    fetcher = object.__new__(cn_fetcher.CNDataFetcher)
    fetcher.store = Store()
    monkeypatch.setattr(cn_fetcher, "ThreadPoolExecutor", _RecordingHistoricalExecutor)
    monkeypatch.setattr(cn_fetcher, "_hist_one", fake_hist)

    started = time.monotonic()
    try:
        result = fetcher.get_historical(
            ["CACHED.SH", "FAST.SH", "HANG.SH"],
            days=5,
            batch_timeout_s=0.05,
        )
    finally:
        release.set()

    assert time.monotonic() - started < 0.5
    assert set(result["ticker"]) == {"CACHED.SH", "FAST.SH"}
    assert _RecordingHistoricalExecutor.instances[0].shutdown_calls == [
        ((), {"wait": False, "cancel_futures": True})
    ]


def test_fetch_data_does_not_duplicate_fetcher_price_persistence(monkeypatch):
    from main import QuantSystem

    raw = _price_frame("AAPL")

    class Fetcher:
        def get_historical(self, tickers, days):
            return raw.copy()

    class Store:
        def save_prices(self, *args, **kwargs):
            pytest.fail("get_historical already persists downloaded prices")

    system = object.__new__(QuantSystem)
    system.universe = ["AAPL"]
    system.market = "US"
    system.fetcher = Fetcher()
    system.store = Store()
    system._adaptive_enabled = False
    monkeypatch.setattr("main.emit_event", lambda *args, **kwargs: None)

    system.fetch_data()

    assert list(system._historical_data) == ["AAPL"]
