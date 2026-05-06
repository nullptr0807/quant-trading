"""Sweep stop-loss variants on real trade history.

Variants tested:
  - fixed     : current production (each account's strat.stop_loss)
  - vol_kX    : vol-adaptive with k = X.X, clipped to [floor, cap]
  - trail_X   : trailing stop — sell if drawdown from running peak >= X.X%
  - hybrid    : fixed stop AND trailing stop, whichever fires first

For each variant, replay every BUY→SELL pair and compare PnL.
"""
from __future__ import annotations
import math, sqlite3, statistics, sys
from collections import defaultdict
from datetime import datetime, timedelta

sys.path.insert(0, "/home/gexin/quant-trading")
from accounts.strategies import STRATEGIES
from accounts.gp_strategies import GP_STRATEGIES
from accounts.qlib_strategies import QLIB_STRATEGIES

DB = "/home/gexin/quant-trading/data/trading.db"
LOOKBACK = 20
MIN_OBS = 10
HORIZON = 60

FIXED_STOP = {}
for s in STRATEGIES: FIXED_STOP[s.id] = s.stop_loss
for g in GP_STRATEGIES: FIXED_STOP[g.id] = g.stop_loss
for q in QLIB_STRATEGIES: FIXED_STOP[q.id] = q.stop_loss


def sigma_at(conn, ticker, asof):
    rows = conn.execute(
        "SELECT close FROM prices WHERE interval='1d' AND ticker=? "
        "AND datetime < ? AND close > 0 ORDER BY datetime DESC LIMIT ?",
        (ticker, asof, LOOKBACK + 1)).fetchall()
    if len(rows) < MIN_OBS + 1: return None
    closes = [r[0] for r in rows][::-1]
    rets = [math.log(closes[i]/closes[i-1]) for i in range(1, len(closes))]
    if len(rets) < MIN_OBS: return None
    m = sum(rets)/len(rets)
    return math.sqrt(sum((r-m)**2 for r in rets)/(len(rets)-1))


def simulate(conn, account, ticker, entry_dt, entry_px,
             real_exit_dt, real_exit_px, variant: dict):
    """Return simulated exit PnL for one variant."""
    horizon_end = (datetime.fromisoformat(entry_dt.split(' ')[0])
                   + timedelta(days=HORIZON)).isoformat()
    end_dt = min(real_exit_dt, horizon_end) if real_exit_dt else horizon_end
    rows = conn.execute(
        "SELECT datetime, low, high, close FROM prices WHERE interval='1d' "
        "AND ticker=? AND datetime > ? AND datetime <= ? ORDER BY datetime ASC",
        (ticker, entry_dt, end_dt)).fetchall()

    # Compute stop_pct based on variant
    kind = variant["kind"]
    if kind == "fixed":
        stop_pct = FIXED_STOP.get(account, 0.03)
        trail_pct = None
    elif kind == "vol":
        sig = sigma_at(conn, ticker, entry_dt)
        if sig is None:
            stop_pct = FIXED_STOP.get(account, 0.03)
        else:
            stop_pct = max(variant["floor"], min(variant["cap"], variant["k"] * sig))
        trail_pct = None
    elif kind == "trail":
        stop_pct = None      # no fixed stop
        trail_pct = variant["trail"]
    elif kind == "hybrid":
        stop_pct = FIXED_STOP.get(account, 0.03)
        trail_pct = variant["trail"]
    else:
        raise ValueError(kind)

    peak = entry_px
    sim_exit_px = None
    for dt, low, high, close in rows:
        if high is not None and high > peak:
            peak = high
        # Check stops in priority: fixed/vol stop first (worst case low), then trail
        if stop_pct is not None and low is not None:
            stop_price = entry_px * (1 - stop_pct)
            if low <= stop_price:
                sim_exit_px = stop_price
                break
        if trail_pct is not None and low is not None:
            trail_price = peak * (1 - trail_pct)
            # Only valid after we've moved up enough that trail is below entry stop
            if low <= trail_price:
                sim_exit_px = trail_price
                break

    if sim_exit_px is None:
        sim_exit_px = real_exit_px if real_exit_px else (rows[-1][3] if rows else entry_px)
    return (sim_exit_px - entry_px) / entry_px


def stats(xs):
    n = len(xs)
    return dict(
        n=n,
        mean=statistics.mean(xs) if xs else 0,
        median=statistics.median(xs) if xs else 0,
        std=statistics.stdev(xs) if n > 1 else 0,
        win=sum(1 for x in xs if x > 0)/n if xs else 0,
        sharpe=(statistics.mean(xs)/statistics.stdev(xs)*math.sqrt(252)
                if n > 1 and statistics.stdev(xs) > 0 else 0),
    )


def main():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT account, ticker, side, price, timestamp FROM trades "
        "WHERE side IN ('buy','sell') ORDER BY account, ticker, timestamp"
    ).fetchall()
    holdings = defaultdict(list)
    pairs = []
    for r in rows:
        key = (r["account"], r["ticker"])
        if r["side"] == "buy":
            holdings[key].append((r["timestamp"], r["price"]))
        else:
            if not holdings[key]: continue
            bd, bp = holdings[key].pop(0)
            pairs.append((r["account"], r["ticker"], bd, bp,
                          r["timestamp"], r["price"]))
    print(f"matched buy/sell pairs: {len(pairs)}")

    variants = [
        ("fixed (current)",      dict(kind="fixed")),
        ("vol k=0.8",            dict(kind="vol", k=0.8, floor=0.010, cap=0.06)),
        ("vol k=1.0",            dict(kind="vol", k=1.0, floor=0.010, cap=0.06)),
        ("vol k=1.5",            dict(kind="vol", k=1.5, floor=0.015, cap=0.06)),
        ("vol k=1.0 cap4%",      dict(kind="vol", k=1.0, floor=0.010, cap=0.04)),
        ("vol k=0.8 cap3%",      dict(kind="vol", k=0.8, floor=0.010, cap=0.03)),
        ("trail 3%",             dict(kind="trail", trail=0.03)),
        ("trail 5%",             dict(kind="trail", trail=0.05)),
        ("trail 8%",             dict(kind="trail", trail=0.08)),
        ("hybrid fixed+trail5%", dict(kind="hybrid", trail=0.05)),
        ("hybrid fixed+trail8%", dict(kind="hybrid", trail=0.08)),
    ]

    rows = []
    for name, v in variants:
        pnls = [simulate(conn, *p[:6], v) for p in pairs]
        rows.append((name, stats(pnls)))

    print(f"\n{'variant':<22} {'mean%':>8} {'med%':>7} {'win%':>6} {'std%':>7} {'sharpe':>7}")
    print("-"*60)
    for name, s in rows:
        print(f"{name:<22} {s['mean']*100:>+8.3f} {s['median']*100:>+7.3f} "
              f"{s['win']*100:>6.1f} {s['std']*100:>7.3f} {s['sharpe']:>7.2f}")


if __name__ == "__main__":
    main()
