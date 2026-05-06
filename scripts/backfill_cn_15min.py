#!/usr/bin/env python3
"""Backfill 15-minute bars for CSI300 + 000300.SH index from akshare SINA endpoint.

Endpoint: ak.stock_zh_a_minute(symbol='sh600519', period='15', adjust='qfq')
- SINA returns ~1-2y of history per call (~2000 bars).
- We filter to >= 2026-04-15 before saving (only need 4/17 onwards for replay,
  but retain a couple of warm-up days for any look-back factor that would ask).
- Idempotent: prices PK = (ticker, datetime, interval).
- Polite: ThreadPoolExecutor with max_workers=4.
"""
import os
import sys
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = os.path.expanduser("~/quant-trading")
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import akshare as ak

from config.settings import CN_UNIVERSE, BENCHMARKS_BY_MARKET
from data.store import DataStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("cn15m")

CUTOFF_DATE = "2026-04-15"   # keep a tiny warm-up before replay start (4/17)
INTERVAL = "15m"


def to_sina(ticker: str) -> str:
    """'600519.SH' -> 'sh600519' ; '000001.SZ' -> 'sz000001'."""
    code, suffix = ticker.split(".")
    return suffix.lower() + code


def fetch_one(ticker: str) -> pd.DataFrame | None:
    sina_sym = to_sina(ticker)
    try:
        df = ak.stock_zh_a_minute(symbol=sina_sym, period="15", adjust="qfq")
    except Exception as e:
        log.warning("fetch fail %s (%s): %s", ticker, sina_sym, e)
        return None
    if df is None or df.empty:
        return None
    df = df.rename(columns={"day": "datetime"})
    # Filter to >= cutoff
    df = df[df["datetime"] >= CUTOFF_DATE].copy()
    if df.empty:
        return None
    # Coerce numerics
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df["ticker"] = ticker
    return df[["ticker", "datetime", "open", "high", "low", "close", "volume"]]


def main():
    bench = [bm["ticker"] for bm in BENCHMARKS_BY_MARKET["CN"]]
    tickers = list(dict.fromkeys(list(CN_UNIVERSE) + bench))  # dedupe, keep order
    log.info("Backfilling 15m bars for %d tickers (cutoff>=%s)", len(tickers), CUTOFF_DATE)

    store = DataStore()
    total_rows = 0
    ok = 0
    empty = 0
    fail = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(fetch_one, t): t for t in tickers}
        for i, fut in enumerate(as_completed(futures), 1):
            ticker = futures[fut]
            try:
                df = fut.result()
            except Exception as e:
                log.warning("[%d/%d] %s ERROR: %s", i, len(tickers), ticker, e)
                fail += 1
                continue
            if df is None or df.empty:
                log.info("[%d/%d] %s empty", i, len(tickers), ticker)
                empty += 1
                continue
            store.save_prices_bulk(df, interval=INTERVAL)
            total_rows += len(df)
            ok += 1
            if i % 25 == 0 or i == len(tickers):
                log.info("[%d/%d] %s +%d rows (cum %d)",
                         i, len(tickers), ticker, len(df), total_rows)

    elapsed = time.time() - t0
    log.info("Done in %.1fs. ok=%d empty=%d fail=%d total_rows=%d",
             elapsed, ok, empty, fail, total_rows)


if __name__ == "__main__":
    main()
