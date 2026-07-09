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


def _held_tickers_for_market(market: str) -> list[str]:
    """Return currently held tickers for a market.

    The live portfolio can contain legacy/non-universe symbols (e.g. ARM after
    the Russell-1000 universe rotated it out). Daily backfill must still refresh
    those tickers or the ledger watchdog cannot price historical snapshots.
    """
    import sqlite3
    try:
        from data.store import DB_PATH
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            "SELECT DISTINCT ticker FROM positions WHERE market=? ORDER BY ticker",
            (market,),
        ).fetchall()
        conn.close()
        return [r[0] for r in rows]
    except Exception as e:
        log.warning("Could not load held tickers for %s: %s", market, e)
        return []


def _dedupe(seq):
    return list(dict.fromkeys(seq))


def backfill(interval: str, days: int, batch_size: int = 50, market: str = "US", price_mode: str = "adjusted"):
    if market == "CN":
        from config.settings import CN_UNIVERSE, BENCHMARKS_BY_MARKET
        from data.cn_fetcher import CNDataFetcher
        bench_tickers = [bm["ticker"] for bm in BENCHMARKS_BY_MARKET["CN"]]
        tickers = _dedupe(list(CN_UNIVERSE) + _held_tickers_for_market("CN") + bench_tickers)
        fetcher = CNDataFetcher()
        # akshare/sina rate limits — smaller batches help
        batch_size = min(batch_size, 30)
    else:
        from config.settings import STOCK_UNIVERSE
        from data.fetcher import DataFetcher
        tickers = _dedupe(list(STOCK_UNIVERSE) + _held_tickers_for_market("US") + ["QQQ", "SPY"])
        fetcher = DataFetcher()

    log.info("Backfilling [%s] %d tickers, interval=%s, days=%d, price_mode=%s",
             market, len(tickers), interval, days, price_mode)
    t0 = time.time()

    total_rows = 0
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        log.info("Batch %d-%d (%d tickers)...", i, i+len(batch), len(batch))
        df = fetcher.get_historical(batch, days=days, interval=interval, use_cache=False, price_mode=price_mode)
        total_rows += len(df)
        log.info("  -> %d rows fetched (cumulative %d)", len(df), total_rows)

    elapsed = time.time() - t0
    log.info("Done in %.1fs. Total rows stored: %d", elapsed, total_rows)

    cov = fetcher.store.get_price_coverage(tickers, interval=interval, price_mode=price_mode)
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
    p.add_argument("--price-mode", default="adjusted", choices=["adjusted", "raw", "both"],
                   help="adjusted/qfq research prices, raw execution prices, or both")
    p.add_argument("--all", action="store_true",
                   help="Backfill 1h (90d) + 1d (400d). For CN: 1d only (no intraday hist).")
    args = p.parse_args()

    modes = ["adjusted", "raw"] if args.price_mode == "both" else [args.price_mode]
    for mode in modes:
        if args.all:
            if args.market == "CN":
                backfill("1d", max(400, args.days), args.batch_size, market="CN", price_mode=mode)
            else:
                backfill("5m", 60, args.batch_size, market="US", price_mode=mode)
                backfill("1h", 90, args.batch_size, market="US", price_mode=mode)
                backfill("1d", 400, args.batch_size, market="US", price_mode=mode)
        else:
            backfill(args.interval, args.days, args.batch_size, market=args.market, price_mode=mode)


if __name__ == "__main__":
    main()
