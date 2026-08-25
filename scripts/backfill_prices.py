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
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
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


def _ledger_tickers_for_market(market: str, days: int) -> list[str]:
    """Return the bounded ticker set needed to replay recent live ledgers.

    Current holdings cover positions opened before the window. Recent trades
    add symbols sold during the window, which are still needed to value earlier
    snapshots. Benchmarks are added by ``backfill``.
    """
    import sqlite3
    try:
        from data.store import DB_PATH
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max(1, days) + 1)).isoformat()
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            """
            SELECT DISTINCT ticker FROM (
                SELECT p.ticker AS ticker
                FROM positions p
                LEFT JOIN account_meta m ON m.account_id=p.account
                WHERE p.market=? AND p.shares>0
                  AND COALESCE(m.status,'active')!='retired'
                UNION
                SELECT t.ticker AS ticker
                FROM trades t
                LEFT JOIN account_meta m ON m.account_id=t.account
                WHERE t.market=? AND t.timestamp>=?
                  AND COALESCE(m.status,'active')!='retired'
            ) ORDER BY ticker
            """,
            (market, market, cutoff),
        ).fetchall()
        conn.close()
        from config.security_master import canonical_ticker
        now = datetime.now(timezone.utc)
        return sorted({canonical_ticker(r[0], market, now) for r in rows})
    except Exception as e:
        log.warning("Could not load ledger tickers for %s: %s", market, e)
        return _held_tickers_for_market(market)


def _dedupe(seq):
    return list(dict.fromkeys(seq))


def _stale_for_target(coverage: dict, tickers: list[str], target_date: str) -> list[str]:
    """Return tickers whose newest daily bar is older than target_date."""
    stale = []
    for ticker in tickers:
        row = coverage.get(ticker)
        if not row or not row[1]:
            stale.append(ticker)
            continue
        try:
            latest = datetime.fromisoformat(str(row[1]).replace("Z", "").split("+")[0]).date()
        except (TypeError, ValueError):
            stale.append(ticker)
            continue
        if latest.isoformat() < target_date:
            stale.append(ticker)
    return stale


def backfill(interval: str, days: int, batch_size: int = 50, market: str = "US", price_mode: str = "adjusted", scope: str = "universe"):
    if market == "CN":
        from config.settings import CN_UNIVERSE, BENCHMARKS_BY_MARKET
        from data.cn_fetcher import CNDataFetcher
        bench_tickers = [bm["ticker"] for bm in BENCHMARKS_BY_MARKET["CN"]]
        base = _ledger_tickers_for_market("CN", days) if scope == "ledger" else list(CN_UNIVERSE) + _held_tickers_for_market("CN")
        tickers = _dedupe(base + bench_tickers)
        fetcher = CNDataFetcher()
        # akshare/sina rate limits — smaller batches help
        batch_size = min(batch_size, 30)
    else:
        from config.settings import STOCK_UNIVERSE
        from data.fetcher import DataFetcher
        base = _ledger_tickers_for_market("US", days) if scope == "ledger" else list(STOCK_UNIVERSE) + _held_tickers_for_market("US")
        tickers = _dedupe(base + ["QQQ", "SPY"])
        fetcher = DataFetcher()

    log.info("Backfilling [%s] %d tickers, interval=%s, days=%d, price_mode=%s, scope=%s",
             market, len(tickers), interval, days, price_mode, scope)
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

    if interval == "1d":
        from data.fetcher import latest_completed_session_date
        target_date = latest_completed_session_date(market).isoformat()
        stale = _stale_for_target(cov, tickers, target_date)
        if stale:
            # Batch providers can return a non-empty historical frame while
            # silently omitting a few target-session bars. Retry only the stale
            # set once, then verify the persisted cache instead of declaring a
            # false green based on lifetime row counts.
            log.warning(
                "Target-date retry: %d/%d stale for %s (first 20): %s",
                len(stale), len(tickers), target_date, stale[:20],
            )
            fetcher.get_historical(
                stale, days=days, interval=interval, use_cache=False,
                price_mode=price_mode,
            )
            cov = fetcher.store.get_price_coverage(
                tickers, interval=interval, price_mode=price_mode,
            )
            stale = _stale_for_target(cov, tickers, target_date)

        fresh = len(tickers) - len(stale)
        log.info(
            "Target-date coverage: %d/%d at %s", fresh, len(tickers), target_date,
        )
        if stale:
            raise RuntimeError(
                f"target-date coverage incomplete for {market}/{price_mode}/{scope}: "
                f"{fresh}/{len(tickers)} at {target_date}; stale={stale[:20]}"
            )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--interval", default="1h", choices=["1h", "1d", "15m", "5m"])
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--market", default="US", choices=["US", "CN"],
                   help="Which market's universe to backfill (default US)")
    p.add_argument("--price-mode", default="adjusted", choices=["adjusted", "raw", "both"],
                   help="adjusted/qfq research prices, raw execution prices, or both")
    p.add_argument("--scope", default="universe", choices=["universe", "ledger"],
                   help="full research universe or bounded current/recent ledger tickers")
    p.add_argument("--all", action="store_true",
                   help="Backfill 1h (90d) + 1d (400d). For CN: 1d only (no intraday hist).")
    args = p.parse_args()

    modes = ["adjusted", "raw"] if args.price_mode == "both" else [args.price_mode]
    for mode in modes:
        if args.all:
            if args.market == "CN":
                backfill("1d", max(400, args.days), args.batch_size, market="CN", price_mode=mode, scope=args.scope)
            else:
                backfill("5m", 60, args.batch_size, market="US", price_mode=mode, scope=args.scope)
                backfill("1h", 90, args.batch_size, market="US", price_mode=mode, scope=args.scope)
                backfill("1d", 400, args.batch_size, market="US", price_mode=mode, scope=args.scope)
        else:
            backfill(args.interval, args.days, args.batch_size, market=args.market, price_mode=mode, scope=args.scope)


if __name__ == "__main__":
    main()
