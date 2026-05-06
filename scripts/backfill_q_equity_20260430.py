"""
Backfill correct equity for Q01-Q10 between 2026-04-30 13:30:00Z and 2026-05-01 08:00Z.

Plan:
 1. Fetch 5m bars for each of the 61 unique Q-position tickers from yfinance for 04-30..05-01.
 2. For each affected snapshot timestamp T in `accounts` (Q*, equity==10000 in window),
    compute equity = state.cash + sum(shares * price_at_T) using nearest 5m close ≤ T.
    For T after market close (>20:00Z), use 04-30 daily close (last 5m bar of day).
 3. UPDATE accounts SET cash=state.cash, equity=computed WHERE rowid=...
"""
import sqlite3, sys, time
from datetime import datetime, timezone
import yfinance as yf
import pandas as pd

DB = '/home/gexin/quant-trading/data/trading.db'
WIN_START = '2026-04-30T13:30:01'
WIN_END = '2026-05-01T07:59'

c = sqlite3.connect(DB)
c.row_factory = sqlite3.Row

# 1. tickers + holdings per account
positions = {}  # acct -> {ticker: shares}
for r in c.execute("SELECT account, ticker, shares FROM positions WHERE account LIKE 'Q%'"):
    positions.setdefault(r['account'], {})[r['ticker']] = r['shares']
cash_by_acct = {r['account']: r['cash'] for r in c.execute("SELECT account, cash FROM account_state WHERE account LIKE 'Q%'")}
all_tickers = sorted({t for d in positions.values() for t in d})
print(f"{len(positions)} Q accounts, {len(all_tickers)} unique tickers")

# 2. fetch 5m bars 2026-04-30..2026-05-02
yf_norm = lambda t: t.replace('.','-')  # not needed for these but safe
t0 = time.time()
df = yf.download(' '.join(yf_norm(t) for t in all_tickers),
                 start='2026-04-30', end='2026-05-02',
                 interval='5m', progress=False, auto_adjust=False, group_by='ticker', threads=True)
print(f"yf.download took {time.time()-t0:.1f}s, shape={df.shape}")

# Build {ticker: pd.Series of close indexed by tz-naive UTC datetime}
price_series = {}
if isinstance(df.columns, pd.MultiIndex):
    for t in all_tickers:
        try:
            s = df[yf_norm(t)]['Close'].dropna()
            if not s.empty:
                if s.index.tz is not None:
                    s.index = s.index.tz_convert('UTC').tz_localize(None)
                price_series[t] = s
        except Exception as e:
            print('skip', t, e)
else:  # single ticker
    s = df['Close'].dropna()
    if s.index.tz is not None:
        s.index = s.index.tz_convert('UTC').tz_localize(None)
    price_series[all_tickers[0]] = s

print(f"got price series for {len(price_series)}/{len(all_tickers)} tickers")
missing = set(all_tickers)-set(price_series)
if missing:
    print('MISSING:', missing)

# fallback: yesterday's avg_cost? No - use 04-29 daily close
fallback_close = {}
for t in missing:
    row = c.execute("SELECT close FROM prices WHERE ticker=? AND interval='1d' AND datetime='2026-04-29 00:00:00'", (t,)).fetchone()
    if row:
        fallback_close[t] = row['close']
print(f"fallback (04-29 close) for {len(fallback_close)}/{len(missing)} missing")

def price_at(ticker, ts):
    """ts: naive UTC datetime"""
    s = price_series.get(ticker)
    if s is not None:
        # nearest <= ts
        loc = s.index.searchsorted(ts, side='right') - 1
        if loc >= 0:
            return float(s.iloc[loc])
        # before market open: use first bar
        return float(s.iloc[0])
    return fallback_close.get(ticker)

# 3. find affected rows
affected = c.execute(
    "SELECT id, name, timestamp FROM accounts "
    "WHERE name LIKE 'Q%' AND timestamp > ? AND timestamp < ? AND ABS(equity-10000)<0.01",
    (WIN_START, WIN_END)
).fetchall()
print(f"affected rows: {len(affected)}")

updates = []
unresolved = 0
for r in affected:
    acct = r['name']
    ts = r['timestamp']
    # parse iso
    ts_clean = ts.replace('Z','+00:00') if ts.endswith('Z') else ts
    dt = datetime.fromisoformat(ts_clean)
    if dt.tzinfo:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    holdings = positions.get(acct, {})
    cash = cash_by_acct.get(acct, 10000.0)
    equity = cash
    ok = True
    for t, sh in holdings.items():
        p = price_at(t, dt)
        if p is None:
            ok = False
            break
        equity += sh * p
    if ok:
        updates.append((round(cash,4), round(equity,4), r['id']))
    else:
        unresolved += 1

print(f"prepared {len(updates)} updates, unresolved: {unresolved}")
# sample
for u in updates[:3]:
    print('  sample:', u)
for u in updates[-3:]:
    print('  sample:', u)

# Write
cur = c.cursor()
cur.executemany("UPDATE accounts SET cash=?, equity=? WHERE id=?", updates)
c.commit()
print(f"committed {cur.rowcount} updates")
