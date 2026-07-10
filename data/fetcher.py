"""Data fetching via yfinance (historical) and finnhub (real-time).

Cache-aware: get_historical() first checks the SQLite cache, then fetches only
the missing date range from yfinance, and writes new bars back to the cache.
"""

import os
import logging
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import yfinance as yf
import finnhub
import pandas as pd

from data.store import DataStore

log = logging.getLogger("quant")

# yfinance interval limits (max days of history available)
INTERVAL_MAX_DAYS = {
    "1m": 7,
    "2m": 60,
    "5m": 60,
    "15m": 60,
    "30m": 60,
    "60m": 730,
    "1h": 730,
    "1d": 36500,
}

_MARKET_TIMEZONES = {
    "US": ZoneInfo("America/New_York"),
    "CN": ZoneInfo("Asia/Shanghai"),
}
_MARKET_CLOSES = {
    "US": time(16, 0),
    "CN": time(15, 0),
}


def _utc_now() -> datetime:
    """Return an aware UTC clock value; kept injectable for boundary tests."""
    return datetime.now(timezone.utc)


def latest_completed_session_date(market: str, now: datetime | None = None) -> date:
    """Return the latest completed weekday trading session for ``market``.

    This deliberately has no online exchange-calendar dependency. It is exact
    for close-time and weekend boundaries. On an exchange holiday it is
    conservative: the weekday may be requested, while an empty provider result
    simply leaves the previous cached session intact.
    """
    market = market.upper()
    try:
        market_tz = _MARKET_TIMEZONES[market]
        close_time = _MARKET_CLOSES[market]
    except KeyError as exc:
        raise ValueError(f"unsupported market: {market}") from exc

    now = now or _utc_now()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    local_now = now.astimezone(market_tz)
    target = local_now.date()
    if local_now.weekday() >= 5 or local_now.time() < close_time:
        target -= timedelta(days=1)
    while target.weekday() >= 5:
        target -= timedelta(days=1)
    return target


def _normalize_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize yfinance output to lowercase ohlcv + ticker + datetime columns."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["datetime", "ticker", "open", "high", "low", "close", "volume"])
    # yfinance may return MultiIndex columns when multiple tickers requested
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance MultiIndex layout depends on group_by:
        #   group_by='ticker' → level 0 = ticker, level 1 = field (Open/High/...)
        #   group_by='column' → level 0 = field, level 1 = ticker
        # Try both so single-ticker calls don't silently drop everything.
        if ticker in df.columns.get_level_values(0):
            df = df.xs(ticker, axis=1, level=0)
        elif ticker in df.columns.get_level_values(1):
            df = df.xs(ticker, axis=1, level=1)
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    if len(cols) < 5:
        return pd.DataFrame(columns=["datetime", "ticker", "open", "high", "low", "close", "volume"])
    out = df[cols].copy()
    out.columns = ["open", "high", "low", "close", "volume"]
    out["ticker"] = ticker
    out.index.name = "datetime"
    out = out.reset_index()
    # Drop rows with NaN OHLC (e.g., yfinance returns empty bars sometimes)
    out = out.dropna(subset=["open", "high", "low", "close"])
    return out


def _restore_split_unadjusted_ohlc(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Convert Yahoo's split-adjusted OHLC back to raw market-price scale.

    yfinance daily history is split-adjusted even when auto_adjust=False:
    pre-split CRWD 2026-05 closes near 154 instead of the raw traded ~594.
    For account replay/execution audits we need the raw coordinate, so multiply
    every row before each future split ex-date by that split ratio.
    """
    if df is None or df.empty:
        return df
    try:
        splits = yf.Ticker(ticker).splits
    except Exception:
        return df
    if splits is None or len(splits) == 0:
        return df
    if isinstance(splits, pd.DataFrame):
        if "Stock Splits" not in splits.columns:
            return df
        splits = splits["Stock Splits"]
    out = df.copy()
    dates = pd.to_datetime(out["datetime"], utc=True, errors="coerce").dt.date
    for idx, ratio in splits.items():
        try:
            split_date = pd.to_datetime(idx, utc=True).date()
            ratio_f = float(ratio)
        except Exception:
            continue
        if not ratio_f or abs(ratio_f - 1.0) < 1e-12:
            continue
        mask = dates < split_date
        if mask.any():
            out.loc[mask, ["open", "high", "low", "close"]] = (
                out.loc[mask, ["open", "high", "low", "close"]].astype(float) * ratio_f
            )
    return out


# Note: zero-volume / outlier filtering happens at READ time (DataStore.load_prices
# with skip_zero_volume=True by default), not here. Rationale: DB preserves the
# raw yfinance output verbatim for audit/debug; business reads transparently
# skip the dirty rows. To bypass the filter, pass skip_zero_volume=False.


class DataFetcher:
    def __init__(self, finnhub_api_key: str | None = None):
        key = finnhub_api_key or os.environ.get("FINNHUB_API_KEY", "")
        self.finnhub_client = finnhub.Client(api_key=key)
        self.store = DataStore()

    # ── Historical (cache-aware) ────────────────────────────────────────────

    def get_historical(self, tickers: list[str], days: int = 30,
                       interval: str = "1d", use_cache: bool = True,
                       price_mode: str = "adjusted") -> pd.DataFrame:
        """Download OHLCV history for tickers over the last `days` days.

        Cache-aware: reads from DB first, only fetches missing ticker/date ranges
        from yfinance, then writes new bars back. Pass use_cache=False to force refresh.
        """
        now_utc = _utc_now()
        end = now_utc.replace(tzinfo=None)
        start = end - timedelta(days=days)
        start_str = start.strftime("%Y-%m-%d")
        if interval == "1d":
            target_session = latest_completed_session_date("US", now_utc)
            # yfinance's ``end`` boundary is exclusive. Request through the day
            # after the latest completed session so its bar is included.
            end_str = (target_session + timedelta(days=1)).isoformat()
        else:
            target_session = None
            end_str = end.strftime("%Y-%m-%d")

        # Cap `days` to yfinance limit for the given interval
        max_days = INTERVAL_MAX_DAYS.get(interval, 36500)
        if days > max_days:
            log.warning("days=%d exceeds yfinance limit for interval=%s (max %d), capping.",
                        days, interval, max_days)

        if use_cache:
            # 1) Load whatever the cache has
            cached = self.store.load_prices(tickers, start_str, end_str, interval=interval, price_mode=price_mode)
            coverage = self.store.get_price_coverage(tickers, interval=interval, price_mode=price_mode)

            # 2) Identify tickers with no data or insufficient coverage
            # Expected bars: rough floor based on interval
            if interval == "1d":
                expected = max(1, int(days * 0.6))  # ~60% trading days
            elif interval in ("1h", "60m"):
                expected = max(1, int(days * 0.6 * 6.5))  # ~6.5 bars/day
            elif interval == "15m":
                expected = max(1, int(days * 0.6 * 26))
            elif interval == "5m":
                expected = max(1, int(days * 0.6 * 78))
            else:
                expected = 1

            missing = []
            # Daily freshness is measured against the latest completed US
            # session, not today's still-forming candle. Otherwise every ticker
            # is classified stale throughout regular trading hours.
            for t in tickers:
                cov = coverage.get(t)
                if not cov or cov[2] < expected:
                    missing.append(t)
                    continue
                max_dt = cov[1]
                try:
                    max_d = datetime.fromisoformat(max_dt.replace("Z", "").split("+")[0])
                except Exception:
                    missing.append(t)
                    continue
                if interval == "1d":
                    assert target_session is not None
                    if max_d.date() < target_session:
                        missing.append(t)
                else:
                    # Intraday: stale if older than 3 days.
                    if (end - max_d).days > 3:
                        missing.append(t)

            hit_tickers = len(tickers) - len(missing)
            if not missing:
                log.info(
                    "📦 [%s] %d days | CACHE HIT: %d tickers (%d rows) | DOWNLOAD: 0 | TOTAL: %d rows",
                    interval, days, hit_tickers, len(cached), len(cached),
                )
                return cached

            log.info(
                "📥 [%s] %d days | CACHE HIT: %d/%d tickers (%d rows) | DOWNLOADING %d tickers from yfinance...",
                interval, days, hit_tickers, len(tickers), len(cached), len(missing),
            )
            fetched = self._fetch_yf_batch(missing, start_str, end_str, interval, price_mode=price_mode)
            dl_rows = len(fetched)
            dl_tickers = fetched["ticker"].nunique() if not fetched.empty else 0
            if not fetched.empty:
                self.store.save_prices_bulk(fetched, interval=interval, price_mode=price_mode)
            # Re-load full set from cache now that missing is filled
            final = self.store.load_prices(tickers, start_str, end_str, interval=interval, price_mode=price_mode)
            log.info(
                "📦 [%s] %d days | CACHE HIT: %d tickers (%d rows) | DOWNLOAD: %d tickers (%d rows) | TOTAL: %d rows",
                interval, days, hit_tickers, len(cached), dl_tickers, dl_rows, len(final),
            )
            return final

        # use_cache=False: force full fetch
        fetched = self._fetch_yf_batch(tickers, start_str, end_str, interval, price_mode=price_mode)
        if not fetched.empty:
            self.store.save_prices_bulk(fetched, interval=interval, price_mode=price_mode)
        return fetched

    def _fetch_yf_batch(self, tickers: list[str], start: str, end: str,
                        interval: str, price_mode: str = "adjusted") -> pd.DataFrame:
        """Batch download from yfinance with threading. Returns normalized DataFrame."""
        if not tickers:
            return pd.DataFrame(columns=["datetime", "ticker", "open", "high", "low", "close", "volume"])
        # Intraday intervals: include pre/post-market bars so dashboard +
        # benchmark curves cover 04:00-20:00 ET, not just RTH.
        prepost = interval in ("1m", "2m", "5m", "15m", "30m", "60m", "1h", "90m")
        # yfinance logs per-symbol "possibly delisted" messages at ERROR level
        # through its own logger even when the batch succeeds for >99% of names.
        # Silence that third-party noise here; we log structured coverage below.
        yf_logger = logging.getLogger("yfinance")
        old_yf_level = yf_logger.level
        yf_logger.setLevel(logging.CRITICAL)
        try:
            raw = yf.download(
                tickers, start=start, end=end, interval=interval,
                progress=False, auto_adjust=(price_mode != "raw"), threads=True,
                group_by="ticker", prepost=prepost,
            )
        except Exception as e:
            log.error("yf.download batch failed: %s", e)
            return pd.DataFrame(columns=["datetime", "ticker", "open", "high", "low", "close", "volume"])
        finally:
            yf_logger.setLevel(old_yf_level)

        frames = []
        if len(tickers) == 1:
            df = _normalize_df(raw, tickers[0])
            if not df.empty and price_mode == "raw":
                df = _restore_split_unadjusted_ohlc(df, tickers[0])
            if not df.empty:
                frames.append(df)
        else:
            # MultiIndex: top level = ticker
            if isinstance(raw.columns, pd.MultiIndex):
                top = raw.columns.get_level_values(0).unique()
                for t in top:
                    sub = raw[t] if t in top else None
                    if sub is None or sub.empty:
                        continue
                    df = _normalize_df(sub, t)
                    if not df.empty and price_mode == "raw":
                        df = _restore_split_unadjusted_ohlc(df, t)
                    if not df.empty:
                        frames.append(df)
            else:
                df = _normalize_df(raw, tickers[0])
                if not df.empty and price_mode == "raw":
                    df = _restore_split_unadjusted_ohlc(df, tickers[0])
                if not df.empty:
                    frames.append(df)

        if not frames:
            return pd.DataFrame(columns=["datetime", "ticker", "open", "high", "low", "close", "volume"])
        # NB: zero-volume filtering happens at read time in DataStore.load_prices.
        return pd.concat(frames, ignore_index=True)

    # ── Intraday (direct, no cache) ─────────────────────────────────────────

    def get_intraday(self, tickers: list[str], period: str = "1d",
                     interval: str = "5m") -> pd.DataFrame:
        """Download intraday OHLCV data for tickers (no cache, current-day use)."""
        try:
            raw = yf.download(tickers, period=period, interval=interval,
                              progress=False, auto_adjust=True, threads=True,
                              group_by="ticker")
        except Exception as e:
            log.warning("Intraday batch fetch failed: %s", e)
            return pd.DataFrame(columns=["datetime", "ticker", "open", "high", "low", "close", "volume"])
        frames = []
        if len(tickers) == 1:
            df = _normalize_df(raw, tickers[0])
            if not df.empty:
                frames.append(df)
        elif isinstance(raw.columns, pd.MultiIndex):
            for t in raw.columns.get_level_values(0).unique():
                df = _normalize_df(raw[t], t)
                if not df.empty:
                    frames.append(df)
        if not frames:
            return pd.DataFrame(columns=["datetime", "ticker", "open", "high", "low", "close", "volume"])
        return pd.concat(frames, ignore_index=True)

    # ── Realtime quotes (yfinance fast_info → finnhub fallback) ─────────────

    def get_realtime_quotes(self, tickers: list[str]) -> dict[str, float]:
        """Fetch latest realtime price per ticker (US market).

        Strategy: yfinance Ticker.fast_info.lastPrice (works pre/post market)
        with parallel fan-out, falls back to Finnhub quote for any miss.
        Returns {ticker: price}; tickers we could not quote are absent.

        This is the single source of truth for US realtime quotes — both
        main.py's hourly cycle and scripts/update_prices.py's per-minute
        watchdog go through here so equity values stay consistent.
        """
        if not tickers:
            return {}
        from concurrent.futures import ThreadPoolExecutor

        def _one(ticker: str) -> tuple[str, float | None]:
            # 1) yfinance fast_info (covers pre/post market)
            try:
                px = yf.Ticker(ticker).fast_info.get("lastPrice")
                if px and px > 0:
                    return ticker, float(px)
            except Exception:
                pass
            # 2) Finnhub fallback
            try:
                q = self.finnhub_client.quote(ticker)
                c = q.get("c")
                if c and c > 0:
                    return ticker, float(c)
            except Exception:
                pass
            return ticker, None

        out: dict[str, float] = {}
        with ThreadPoolExecutor(max_workers=16) as ex:
            for tk, px in ex.map(_one, tickers):
                if px is not None:
                    out[tk] = px
        if out:
            log.info("Fetched realtime quotes for %d/%d US tickers", len(out), len(tickers))
        return out

    def get_extended_hours_quote(self, ticker: str) -> dict:
        """Get pre/after market quote from finnhub."""
        try:
            q = self.finnhub_client.quote(ticker)
            return {
                "ticker": ticker,
                "current": q.get("c"),
                "high": q.get("h"),
                "low": q.get("l"),
                "open": q.get("o"),
                "prev_close": q.get("pc"),
                "timestamp": q.get("t"),
                "pre_market": q.get("dp"),
                "change_percent": q.get("dp"),
            }
        except Exception as e:
            return {"ticker": ticker, "error": str(e)}
