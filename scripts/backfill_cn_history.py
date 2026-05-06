"""Backfill CN A-share daily prices to 2020-01-01 via akshare.

CN universe: whatever's already in trading.db (so we don't expand scope here,
just deepen history for tickers we already trade/track).
"""
from __future__ import annotations
import logging, sys, time, sqlite3
sys.path.insert(0, "/home/gexin/quant-trading")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("backfill_cn")

from data.cn_fetcher import CNDataFetcher

DB = "/home/gexin/quant-trading/data/trading.db"
c = sqlite3.connect(DB)
tickers = sorted({r[0] for r in c.execute(
    "SELECT DISTINCT ticker FROM prices WHERE interval='1d' "
    "AND (ticker LIKE '%.SH' OR ticker LIKE '%.SZ' OR ticker LIKE '%.BJ')"
).fetchall()})
c.close()
log.info("CN universe in DB: %d tickers", len(tickers))

DAYS = 2200    # ~6 years
BATCH = 50

f = CNDataFetcher()
t0 = time.time()
ok = 0; fail = 0
for i in range(0, len(tickers), BATCH):
    batch = tickers[i:i+BATCH]
    log.info("[%d/%d] batch of %d ...", i // BATCH + 1,
             (len(tickers) + BATCH - 1) // BATCH, len(batch))
    try:
        df = f.get_historical(batch, days=DAYS, interval="1d", use_cache=True)
        ok += len(batch)
    except Exception as e:
        log.error("batch failed: %s", e)
        fail += len(batch)
    time.sleep(2)

log.info("DONE ok=%d fail=%d elapsed=%.1fmin", ok, fail, (time.time() - t0) / 60)

c = sqlite3.connect(DB)
r = c.execute("""SELECT MIN(datetime), MAX(datetime),
                        COUNT(DISTINCT substr(datetime,1,10)),
                        COUNT(DISTINCT ticker)
                 FROM prices WHERE interval='1d'
                 AND (ticker LIKE '%.SH' OR ticker LIKE '%.SZ' OR ticker LIKE '%.BJ')
                 AND open IS NOT NULL""").fetchone()
log.info("CN DB now: min=%s max=%s n_days=%s n_tickers=%s", *r)
