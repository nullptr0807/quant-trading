"""Backfill US daily prices to 2020-01-01 for the Russell 1000 universe.

Cache-aware via DataFetcher: only downloads dates not already in trading.db.
Chunks tickers into batches of 100 to stay friendly to yfinance.
"""
from __future__ import annotations
import logging, os, sys, time
sys.path.insert(0, "/home/gexin/quant-trading")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("backfill")

from config.settings import STOCK_UNIVERSE
from data.fetcher import DataFetcher

# Flatten Russell 1000 (sector dict → list)
if isinstance(STOCK_UNIVERSE, dict):
    tickers = sorted({t for v in STOCK_UNIVERSE.values() for t in v})
else:
    tickers = sorted(set(STOCK_UNIVERSE))
log.info("universe: %d tickers", len(tickers))

# 6 years ≈ 2200 days back from today
DAYS = 2200
BATCH = 100

f = DataFetcher()
t0 = time.time()
total_rows = 0
for i in range(0, len(tickers), BATCH):
    batch = tickers[i:i+BATCH]
    log.info("[%d/%d] batch of %d tickers...", i // BATCH + 1,
             (len(tickers) + BATCH - 1) // BATCH, len(batch))
    try:
        df = f.get_historical(batch, days=DAYS, interval="1d", use_cache=True)
        total_rows += len(df)
    except Exception as e:
        log.error("batch failed: %s", e)
        time.sleep(5)
        continue
    time.sleep(1)  # gentle pacing

log.info("DONE. total_rows_returned=%d elapsed=%.1fmin",
         total_rows, (time.time() - t0) / 60)

# Verify
import sqlite3
c = sqlite3.connect("/home/gexin/quant-trading/data/trading.db")
r = c.execute("""SELECT MIN(datetime), MAX(datetime),
                        COUNT(DISTINCT substr(datetime,1,10)),
                        COUNT(DISTINCT ticker)
                 FROM prices WHERE interval='1d'
                 AND ticker NOT LIKE '%.SH'
                 AND ticker NOT LIKE '%.SZ'
                 AND ticker NOT LIKE '%.BJ'
                 AND open IS NOT NULL""").fetchone()
log.info("DB now: min=%s max=%s n_days=%s n_tickers=%s", *r)
