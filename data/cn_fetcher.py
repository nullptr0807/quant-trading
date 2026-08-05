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
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from data.store import DataStore
from data.quotes import RealtimeQuote, prices_from_quotes

log = logging.getLogger("quant.cn")

CN_HISTORICAL_BATCH_TIMEOUT_S = 120.0

# A-share index codes we treat specially (akshare uses stock_zh_index_daily for these)
INDEX_CODES = {"000300", "000001", "399001", "399006", "000016", "000905"}


def _split_code(ticker: str) -> tuple[str, str]:
    """'600519.SH' → ('600519', 'sh'). Lower-case suffix for akshare API."""
    if "." not in ticker:
        raise ValueError(f"CN ticker must include .SH/.SZ suffix: {ticker}")
    code, suffix = ticker.split(".", 1)
    return code, suffix.lower()


def _hist_one(ticker: str, start: str, end: str, interval: str,
              max_retries: int = 3, price_mode: str = "adjusted") -> pd.DataFrame:
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
                    adjust="" if price_mode == "raw" else "qfq",
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
                    adjust="" if price_mode == "raw" else "qfq",
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


def _fetch_historical_batch(tickers: list[str], start: str, end: str,
                            interval: str, price_mode: str,
                            timeout_s: float,
                            max_workers: int = 1) -> list[pd.DataFrame]:
    """Fetch a CN batch within one wall-clock budget and keep partial results."""
    if not tickers:
        return []

    frames: list[pd.DataFrame] = []
    collected = set()
    ex_pool = ThreadPoolExecutor(max_workers=max(1, int(max_workers)))
    futures = {
        ex_pool.submit(
            _hist_one, ticker, start, end, interval, price_mode=price_mode,
        ): ticker
        for ticker in tickers
    }
    try:
        for future in as_completed(futures, timeout=timeout_s):
            ticker = futures[future]
            try:
                frame = future.result(timeout=0)
            except Exception as exc:
                log.debug("CN historical fetch %s failed: %s", ticker, exc)
                continue
            if frame is not None and not frame.empty:
                frames.append(frame)
                collected.add(future)
    except FuturesTimeoutError:
        # A future may finish exactly as as_completed raises; harvest every
        # completed result once more so timeout races do not discard good data.
        for future, ticker in futures.items():
            if future in collected or not future.done() or future.cancelled():
                continue
            try:
                frame = future.result(timeout=0)
            except Exception as exc:
                log.debug("CN historical fetch %s failed: %s", ticker, exc)
                continue
            if frame is not None and not frame.empty:
                frames.append(frame)
        pending = sum(1 for future in futures if not future.done())
        log.warning(
            "CN historical batch timed out after %.1fs: %d/%d still pending; returning %d partial tickers",
            timeout_s, pending, len(tickers), len(frames),
        )
    finally:
        ex_pool.shutdown(wait=False, cancel_futures=True)
    return frames


class CNDataFetcher:
    """Cache-aware CN price fetcher mirroring DataFetcher.get_historical."""

    def __init__(self):
        self.store = DataStore()

    def get_historical(self, tickers: list[str], days: int = 30,
                       interval: str = "1d", use_cache: bool = True,
                       price_mode: str = "adjusted",
                       batch_timeout_s: float = CN_HISTORICAL_BATCH_TIMEOUT_S,
                       historical_workers: int = 1) -> pd.DataFrame:
        end = datetime.now(timezone.utc).replace(tzinfo=None)
        start = end - timedelta(days=days)
        s, e = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        # DataStore.load_prices uses an exclusive end bound. Daily bars are
        # stamped at midnight, so querying through today's date would otherwise
        # silently drop the latest completed CN session.
        load_e = (
            (end + timedelta(days=1)).strftime("%Y-%m-%d")
            if interval == "1d" else e
        )

        if not tickers:
            return pd.DataFrame(columns=["datetime", "ticker", "open", "high", "low", "close", "volume"])

        if use_cache:
            cached = self.store.load_prices(tickers, s, load_e, interval=interval, price_mode=price_mode)
            cov = self.store.get_price_coverage(tickers, interval=interval, price_mode=price_mode)
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
            frames = _fetch_historical_batch(
                missing, s, e, interval, price_mode, batch_timeout_s,
                max_workers=historical_workers,
            )
            dl_rows = sum(len(f) for f in frames)
            dl_tickers = len(frames)
            if frames:
                merged = pd.concat(frames, ignore_index=True)
                self.store.save_prices_bulk(merged, interval=interval, price_mode=price_mode)
            final = self.store.load_prices(tickers, s, load_e, interval=interval, price_mode=price_mode)
            log.info("📦 [CN %s] %dd | CACHE %d (%d rows) | DOWNLOADED %d (%d rows) | TOTAL %d rows",
                     interval, days, hit, len(cached), dl_tickers, dl_rows, len(final))
            return final

        # use_cache=False — force refresh
        frames = _fetch_historical_batch(
            tickers, s, e, interval, price_mode, batch_timeout_s,
            max_workers=historical_workers,
        )
        if not frames:
            return pd.DataFrame(columns=["datetime", "ticker", "open", "high", "low", "close", "volume"])
        merged = pd.concat(frames, ignore_index=True)
        self.store.save_prices_bulk(merged, interval=interval, price_mode=price_mode)
        return merged

    # ── Realtime quotes (Sina batch → akshare 1-minute fallback) ─────────────

    @staticmethod
    def _sina_symbol(ticker: str) -> str:
        """'600519.SH' → 'sh600519'; '000001.SZ' → 'sz000001'."""
        code, suffix = _split_code(ticker)
        return suffix + code

    @staticmethod
    def _parse_sina_price(payload: str) -> float | None:
        """Parse one hq.sinajs.cn quote payload into a last price.

        Stock rows are:
          name,open,prev_close,current,high,low,...,date,time,...
        Index rows (e.g. sh000300) are:
          name,current,prev_close,current,high,low,...
        Prefer field[3] for normal stocks; fall back to field[1] for indices.
        """
        parsed = CNDataFetcher._parse_sina_quote(payload)
        return parsed["price"] if parsed else None

    @staticmethod
    def _parse_sina_quote(payload: str) -> dict | None:
        if not payload:
            return None
        fields = payload.split(",")
        for idx in (3, 1):
            if len(fields) <= idx:
                continue
            try:
                px = float(fields[idx])
            except Exception:
                continue
            if px > 0:
                date_text = fields[30].strip() if len(fields) > 30 else ""
                time_text = fields[31].strip() if len(fields) > 31 else ""
                return {
                    "price": px,
                    "prev_close": (
                        float(fields[2]) if len(fields) > 2 and fields[2] else None
                    ),
                    "volume": (
                        float(fields[8]) if len(fields) > 8 and fields[8] else None
                    ),
                    "date": date_text,
                    "time": time_text,
                }
        return None

    def _fetch_sina_quote_metadata(self, tickers: list[str],
                                   timeout_s: float = 8.0) -> dict[str, RealtimeQuote]:
        """Fetch CN realtime quotes in one HTTP call via hq.sinajs.cn.

        This replaces the old one-akshare-call-per-ticker realtime path. For the
        current CN live book (~88 held tickers), the batch endpoint returns full
        coverage in <1s instead of timing out after 45s with partial coverage.
        """
        if not tickers:
            return {}
        symbols = [self._sina_symbol(tk) for tk in tickers]
        url = "https://hq.sinajs.cn/list=" + ",".join(symbols)
        req = urllib.request.Request(
            url,
            headers={
                "Referer": "https://finance.sina.com.cn",
                "User-Agent": "Mozilla/5.0",
            },
        )
        try:
            raw = urllib.request.urlopen(req, timeout=timeout_s).read()
            text = raw.decode("gbk", errors="replace")
        except Exception as e:
            log.warning("CN Sina batch quote failed: %s", e)
            return {}

        out: dict[str, RealtimeQuote] = {}
        received_at = datetime.now(timezone.utc)
        for ticker, symbol in zip(tickers, symbols):
            m = re.search(rf'var hq_str_{re.escape(symbol)}="([^"]*)";', text)
            if not m:
                continue
            parsed = self._parse_sina_quote(m.group(1))
            if not parsed:
                continue
            source_timestamp = None
            if parsed.get("date") and parsed.get("time"):
                try:
                    source_timestamp = datetime.strptime(
                        f"{parsed['date']} {parsed['time']}", "%Y-%m-%d %H:%M:%S"
                    ).replace(tzinfo=ZoneInfo("Asia/Shanghai")).astimezone(timezone.utc)
                except ValueError:
                    source_timestamp = None
            out[ticker] = RealtimeQuote(
                ticker=ticker, price=float(parsed["price"]), source="sina",
                source_timestamp=source_timestamp, received_at=received_at,
                tradable=source_timestamp is not None,
                prev_close=parsed.get("prev_close"), volume=parsed.get("volume"),
            )
        if out:
            log.info("Fetched CN realtime via Sina batch for %d/%d tickers", len(out), len(tickers))
        return out

    def get_realtime_quote_metadata(self, tickers: list[str]) -> dict[str, RealtimeQuote]:
        if not tickers:
            return {}
        tickers = list(dict.fromkeys(tickers))
        out = self._fetch_sina_quote_metadata(tickers)
        # Timestamp-less fallback prices are display-only and deliberately
        # omitted from metadata trading paths; fail-closed callers will reject
        # missing coverage rather than trade on unverifiable bars.
        return out

    def get_realtime_quotes(self, tickers: list[str]) -> dict[str, float]:
        """Compatibility API returning prices, with legacy AkShare fallback."""
        metadata = self.get_realtime_quote_metadata(tickers)
        out = prices_from_quotes(metadata)
        tickers = list(dict.fromkeys(tickers))
        missing = [tk for tk in tickers if tk not in out]
        if not missing:
            return out

        try:
            import akshare as ak  # optional dep; fallback only
        except Exception as e:
            log.warning("akshare unavailable for CN quote fallback: %s", e)
            return out

        def _one(tk: str) -> tuple[str, float | None]:
            try:
                sina = self._sina_symbol(tk)
                df = ak.stock_zh_a_minute(symbol=sina, period="1", adjust="")
                if df is None or df.empty:
                    return tk, None
                px = float(df["close"].iloc[-1])
                if px > 0:
                    return tk, px
            except Exception as e:
                log.debug("CN realtime fallback fetch %s failed: %s", tk, e)
            return tk, None

        # Fallback should normally be tiny. Keep a hard budget so a broken
        # provider cannot starve update_prices/run_cycle under the shared flock.
        timeout_s = 20.0
        ex_pool = ThreadPoolExecutor(max_workers=min(8, max(1, len(missing))))
        futures = {ex_pool.submit(_one, tk): tk for tk in missing}
        try:
            for fut in as_completed(futures, timeout=timeout_s):
                tk = futures[fut]
                try:
                    _, px = fut.result(timeout=0)
                except Exception as e:
                    log.debug("CN realtime fallback fetch %s failed: %s", tk, e)
                    continue
                if px is not None:
                    out[tk] = px
        except FuturesTimeoutError:
            pending = sum(1 for fut in futures if not fut.done())
            log.warning(
                "CN realtime fallback timed out after %.0fs: %d/%d still pending",
                timeout_s, pending, len(missing),
            )
        finally:
            ex_pool.shutdown(wait=False, cancel_futures=True)

        if missing:
            log.info(
                "CN realtime fallback filled %d/%d missing; total %d/%d",
                len([tk for tk in missing if tk in out]), len(missing), len(out), len(tickers),
            )
        return out


def get_fetcher_for(market: str):
    """Factory: return a fetcher matching DataFetcher's interface for the given market."""
    if market == "CN":
        return CNDataFetcher()
    from data.fetcher import DataFetcher
    return DataFetcher()
