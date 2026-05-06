#!/usr/bin/env python3
"""Backfill historical prices (1h and 1d) into the SQLite cache.

Usage:
    python -m scripts.backfill_prices              # defaults: 1h 90d + 1d 400d
    python -m scripts.backfill_prices --interval 1h --days 90
    python -m scripts.backfill_prices --interval 1d --days 400
"""
import sys
import os
import argparse
import logging
import time

PROJECT_ROOT = os.path.expanduser("~/quant-trading")
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("backfill")


def backfill(interval: str, days: int, batch_size: int = 50, market: str = "US"):
    if market == "CN":
        from config.settings import CN_UNIVERSE, BENCHMARKS_BY_MARKET
        from data.cn_fetcher import CNDataFetcher
        bench_tickers = [bm["ticker"] for bm in BENCHMARKS_BY_MARKET["CN"]]
        tickers = list(CN_UNIVERSE) + bench_tickers
        fetcher = CNDataFetcher()
        # akshare/sina rate limits — smaller batches help
        batch_size = min(batch_size, 30)
    else:
        from config.settings import STOCK_UNIVERSE
        from data.fetcher import DataFetcher
        tickers = list(STOCK_UNIVERSE) + ["QQQ", "SPY"]
        fetcher = DataFetcher()

    log.info("Backfilling [%s] %d tickers, interval=%s, days=%d",
             market, len(tickers), interval, days)
    t0 = time.time()

    total_rows = 0
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        log.info("Batch %d-%d (%d tickers)...", i, i+len(batch), len(batch))
        df = fetcher.get_historical(batch, days=days, interval=interval, use_cache=False)
        total_rows += len(df)
        log.info("  -> %d rows fetched (cumulative %d)", len(df), total_rows)

    elapsed = time.time() - t0
    log.info("Done in %.1fs. Total rows stored: %d", elapsed, total_rows)

    cov = fetcher.store.get_price_coverage(tickers, interval=interval)
    counts = sorted([(c[2], t) for t, c in cov.items()])
    log.info("Coverage: %d/%d tickers have data", len(cov), len(tickers))
    if counts:
        log.info("  min bars: %d (%s), max bars: %d (%s)",
                 counts[0][0], counts[0][1], counts[-1][0], counts[-1][1])
    missing = [t for t in tickers if t not in cov]
    if missing:
        log.warning("Missing %d tickers (first 20): %s", len(missing), missing[:20])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--interval", default="1h", choices=["1h", "1d", "15m", "5m"])
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--market", default="US", choices=["US", "CN"],
                   help="Which market's universe to backfill (default US)")
    p.add_argument("--all", action="store_true",
                   help="Backfill 1h (90d) + 1d (400d). For CN: 1d only (no intraday hist).")
    args = p.parse_args()

    if args.all:
        if args.market == "CN":
            backfill("1d", max(400, args.days), args.batch_size, market="CN")
        else:
            backfill("5m", 60, args.batch_size, market="US")
            backfill("1h", 90, args.batch_size, market="US")
            backfill("1d", 400, args.batch_size, market="US")
    else:
        backfill(args.interval, args.days, args.batch_size, market=args.market)


if __name__ == "__main__":
    main()
