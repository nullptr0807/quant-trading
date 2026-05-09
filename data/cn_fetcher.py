"""akshare-backed CN price fetcher.

Same public interface as data.fetcher.DataFetcher.get_historical so the
Orchestrator can swap in/out by market without touching trading logic.

Ticker convention (DB-side): 6-digit code + '.SH' or '.SZ'.
  - Shanghai main board / STAR: 60xxxx, 68xxxx → .SH
  - Shenzhen main / ChiNext:    00xxxx, 30xxxx → .SZ
  - Index 沪深300:               000300.SH (akshare wants 'sh000300')

akshare quirk: column headers are Chinese; we rename to lowercase OHLCV.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pandas as pd

from data.store import DataStore

log = logging.getLogger("quant.cn")

# A-share index codes we treat specially (akshare uses stock_zh_index_daily for these)
INDEX_CODES = {"000300", "000001", "399001", "399006", "000016", "000905"}


def _split_code(ticker: str) -> tuple[str, str]:
    """'600519.SH' → ('600519', 'sh'). Lower-case suffix for akshare API."""
    if "." not in ticker:
        raise ValueError(f"CN ticker must include .SH/.SZ suffix: {ticker}")
    code, suffix = ticker.split(".", 1)
    return code, suffix.lower()


def _hist_one(ticker: str, start: str, end: str, interval: str,
              max_retries: int = 3) -> pd.DataFrame:
    """Fetch one ticker's OHLCV from akshare, return normalized DataFrame.

    Retries transient connection errors with exponential backoff (akshare's
    upstream rate-limits aggressively).
    """
    import time
    import akshare as ak  # local import: optional dependency

    try:
        code, ex = _split_code(ticker)
    except ValueError as e:
        log.warning("skip bad CN ticker: %s", e)
        return pd.DataFrame()

    is_index = code in INDEX_CODES
    last_err = None
    for attempt in range(max_retries):
        try:
            if is_index:
                df = ak.stock_zh_index_daily(symbol=f"{ex}{code}")
                if df is None or df.empty:
                    return pd.DataFrame()
                df = df.rename(columns={"date": "datetime"})
                df["datetime"] = df["datetime"].astype(str)
                df = df[(df["datetime"] >= start) & (df["datetime"] < end)]
            elif interval == "1d":
                # Use sina backend (stock_zh_a_daily) — eastmoney
                # (stock_zh_a_hist) was rate-limiting our SH requests.
                df = ak.stock_zh_a_daily(
                    symbol=f"{ex}{code}",
                    start_date=start.replace("-", ""),
                    end_date=end.replace("-", ""),
                    adjust="qfq",
                )
                if df is None or df.empty:
                    return pd.DataFrame()
                df = df.rename(columns={"date": "datetime"})
                # sina returns: date, open, high, low, close, volume, amount,
                # outstanding_share, turnover — we keep the OHLCV ones.
            elif interval in ("1h", "60m"):
                df = ak.stock_zh_a_hist_min_em(
                    symbol=code,
                    period="60",
                    start_date=f"{start} 09:00:00",
                    end_date=f"{end} 16:00:00",
                    adjust="qfq",
                )
                if df is None or df.empty:
                    return pd.DataFrame()
                df = df.rename(columns={
                    "时间": "datetime",
                    "开盘": "open",
                    "收盘": "close",
                    "最高": "high",
                    "最低": "low",
                    "成交量": "volume",
                })
            else:
                log.warning("CN interval %s not supported yet", interval)
                return pd.DataFrame()

            if df.empty:
                return df
            df["datetime"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            df["ticker"] = ticker
            keep = ["datetime", "ticker", "open", "high", "low", "close", "volume"]
            if "volume" not in df.columns:
                df["volume"] = 0.0
            return df[keep].dropna(subset=["open", "high", "low", "close"])
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                wait = 1.5 * (2 ** attempt)
                log.debug("akshare %s attempt %d/%d failed (%s), retry in %.1fs",
                          ticker, attempt + 1, max_retries, e, wait)
                time.sleep(wait)
                continue
    log.warning("akshare fetch %s failed after %d attempts: %s",
                ticker, max_retries, last_err)
    return pd.DataFrame()


class CNDataFetcher:
    """Cache-aware CN price fetcher mirroring DataFetcher.get_historical."""

    def __init__(self):
        self.store = DataStore()

    def get_historical(self, tickers: list[str], days: int = 30,
                       interval: str = "1d", use_cache: bool = True) -> pd.DataFrame:
        end = datetime.now(timezone.utc).replace(tzinfo=None)
        start = end - timedelta(days=days)
        s, e = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

        if not tickers:
            return pd.DataFrame(columns=["datetime", "ticker", "open", "high", "low", "close", "volume"])

        if use_cache:
            cached = self.store.load_prices(tickers, s, e, interval=interval)
            cov = self.store.get_price_coverage(tickers, interval=interval)
            if interval == "1d":
                expected = max(1, int(days * 0.5))   # ~50% trading days (CN: ~250/yr)
            elif interval in ("1h", "60m"):
                expected = max(1, int(days * 0.5 * 4))   # 4 hourly bars/day in CN
            else:
                expected = 1

            missing: list[str] = []
            for t in tickers:
                c = cov.get(t)
                if not c or c[2] < expected:
                    missing.append(t)
                    continue
                # Freshness: max datetime within 3 days of end
                try:
                    max_d = datetime.fromisoformat(c[1].replace("Z", "").split("+")[0])
                    if (end - max_d).days > 3:
                        missing.append(t)
                except Exception:
                    missing.append(t)

            hit = len(tickers) - len(missing)
            if not missing:
                log.info("📦 [CN %s] %dd | CACHE HIT %d tickers (%d rows)",
                         interval, days, hit, len(cached))
                return cached

            log.info("📥 [CN %s] %dd | CACHE %d/%d (%d rows) | DOWNLOADING %d via akshare...",
                     interval, days, hit, len(tickers), len(cached), len(missing))
            frames: list[pd.DataFrame] = []
            with ThreadPoolExecutor(max_workers=3) as ex_pool:
                for df in ex_pool.map(lambda t: _hist_one(t, s, e, interval), missing):
                    if not df.empty:
                        frames.append(df)
            dl_rows = sum(len(f) for f in frames)
            dl_tickers = len(frames)
            if frames:
                merged = pd.concat(frames, ignore_index=True)
                self.store.save_prices_bulk(merged, interval=interval)
            final = self.store.load_prices(tickers, s, e, interval=interval)
            log.info("📦 [CN %s] %dd | CACHE %d (%d rows) | DOWNLOADED %d (%d rows) | TOTAL %d rows",
                     interval, days, hit, len(cached), dl_tickers, dl_rows, len(final))
            return final

        # use_cache=False — force refresh
        frames: list[pd.DataFrame] = []
        with ThreadPoolExecutor(max_workers=3) as ex_pool:
            for df in ex_pool.map(lambda t: _hist_one(t, s, e, interval), tickers):
                if not df.empty:
                    frames.append(df)
        if not frames:
            return pd.DataFrame(columns=["datetime", "ticker", "open", "high", "low", "close", "volume"])
        merged = pd.concat(frames, ignore_index=True)
        self.store.save_prices_bulk(merged, interval=interval)
        return merged

    # ── Realtime quotes (akshare 1-minute bar) ──────────────────────────────

    def get_realtime_quotes(self, tickers: list[str]) -> dict[str, float]:
        """Fetch latest realtime price per CN ticker via akshare's 1-minute bar.

        yfinance can't quote .SH/.SZ symbols (its CSI300 lives at 000300.SS),
        so this is the canonical CN realtime path. Both main.py's hourly
        cycle and scripts/update_prices.py's per-minute watchdog call this
        so equity values agree at every tick.

        Returns {ticker: price}; tickers we could not quote are absent.
        """
        if not tickers:
            return {}
        try:
            import akshare as ak  # optional dep
        except Exception as e:
            log.warning("akshare unavailable, can't realtime-quote CN: %s", e)
            return {}

        def _one(tk: str) -> tuple[str, float | None]:
            try:
                code, suf = tk.split(".")
                sina = suf.lower() + code  # '000300.SH' -> 'sh000300'
                df = ak.stock_zh_a_minute(symbol=sina, period="1", adjust="")
                if df is None or df.empty:
                    return tk, None
                px = float(df["close"].iloc[-1])
                if px > 0:
                    return tk, px
            except Exception as e:
                log.debug("CN realtime fetch %s failed: %s", tk, e)
            return tk, None

        out: dict[str, float] = {}
        # SINA quote endpoint tolerates parallelism: 16 workers brings 65
        # tickers from ~29s to ~7s, comfortably under a 1-minute cron tick.
        with ThreadPoolExecutor(max_workers=16) as ex_pool:
            for tk, px in ex_pool.map(_one, tickers):
                if px is not None:
                    out[tk] = px
        if out:
            log.info("Fetched CN realtime via akshare for %d/%d tickers",
                     len(out), len(tickers))
        return out


def get_fetcher_for(market: str):
    """Factory: return a fetcher matching DataFetcher's interface for the given market."""
    if market == "CN":
        return CNDataFetcher()
    from data.fetcher import DataFetcher
    return DataFetcher()
