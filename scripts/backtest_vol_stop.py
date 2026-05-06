"""Backtest: fixed vs vol-adaptive stop loss on real trade history.

For each historical BUY trade, simulate what would have happened with
a vol-adaptive stop instead of the account's fixed stop:
  1. Take entry price + entry date from real trades
  2. Compute σ_20 as of entry date → vol_stop_pct
  3. Walk daily prices forward until either:
     - Vol stop hit  → exit at stop price (paper sell)
     - Real exit happened (signal or fixed-stop sell)  → exit at real price
     - 60 days elapsed → exit at last close
  4. Compare PnL of vol-adaptive exit vs the actual exit

Outputs distribution of PnL diffs and counts of "saved by tighter stop"
vs "killed by whipsaw".
"""
from __future__ import annotations
import math, sqlite3, statistics
from collections import defaultdict
from datetime import datetime, timedelta

DB = "/home/gexin/quant-trading/data/trading.db"

# Same constants as production
LOOKBACK = 20
K = 1.5
FLOOR = 0.015
CAP = 0.06
MIN_OBS = 10
HORIZON = 60   # cap simulated holding at 60 days

# Sweep multiple k values
import sys
sys.path.insert(0, "/home/gexin/quant-trading")
from accounts.strategies import STRATEGIES
from accounts.gp_strategies import GP_STRATEGIES
from accounts.qlib_strategies import QLIB_STRATEGIES
FIXED_STOP = {}
for s in STRATEGIES: FIXED_STOP[s.id] = s.stop_loss
for g in GP_STRATEGIES: FIXED_STOP[g.id] = g.stop_loss
for q in QLIB_STRATEGIES: FIXED_STOP[q.id] = q.stop_loss


def sigma_at(conn, ticker, asof_date):
    rows = conn.execute(
        "SELECT close FROM prices WHERE interval='1d' AND ticker=? "
        "AND datetime < ? AND close > 0 "
        "ORDER BY datetime DESC LIMIT ?",
        (ticker, asof_date, LOOKBACK + 1),
    ).fetchall()
    if len(rows) < MIN_OBS + 1:
        return None
    closes = [r[0] for r in rows][::-1]
    rets = [math.log(closes[i]/closes[i-1]) for i in range(1, len(closes))]
    if len(rets) < MIN_OBS:
        return None
    m = sum(rets)/len(rets)
    var = sum((r-m)**2 for r in rets)/(len(rets)-1)
    return math.sqrt(var)


def simulate_one(conn, account, ticker, entry_date, entry_price,
                 real_exit_date, real_exit_price, real_pnl_pct):
    sig = sigma_at(conn, ticker, entry_date)
    if sig is None:
        vol_stop = FIXED_STOP.get(account, 0.03)
    else:
        vol_stop = max(FLOOR, min(CAP, K * sig))
    fixed_stop = FIXED_STOP.get(account, 0.03)

    # Walk daily lows from entry_date to real_exit_date (or +60d cap)
    horizon_end = (datetime.fromisoformat(entry_date.split(' ')[0]) +
                   timedelta(days=HORIZON)).isoformat()
    end_date = min(real_exit_date, horizon_end) if real_exit_date else horizon_end
    rows = conn.execute(
        "SELECT datetime, low, close FROM prices WHERE interval='1d' "
        "AND ticker=? AND datetime > ? AND datetime <= ? "
        "ORDER BY datetime ASC",
        (ticker, entry_date, end_date),
    ).fetchall()

    # If vol stop is HIT during holding period, exit there.
    # We approximate: if any day's `low` <= entry*(1-vol_stop), assume filled at stop price.
    vol_exit_price = None
    vol_exit_date = None
    for dt, low, close in rows:
        if low is None:
            continue
        if low <= entry_price * (1 - vol_stop):
            vol_exit_price = entry_price * (1 - vol_stop)   # assume filled at stop
            vol_exit_date = dt
            break
    if vol_exit_price is None:
        # No vol-stop hit → use real exit (or last close in window)
        if real_exit_price:
            vol_exit_price = real_exit_price
            vol_exit_date = real_exit_date
        elif rows:
            vol_exit_price = rows[-1][2]
            vol_exit_date = rows[-1][0]
        else:
            return None

    vol_pnl = (vol_exit_price - entry_price) / entry_price
    return dict(
        account=account, ticker=ticker, entry_date=entry_date,
        sigma=sig, vol_stop=vol_stop, fixed_stop=fixed_stop,
        real_pnl=real_pnl_pct, vol_pnl=vol_pnl,
        diff=vol_pnl - real_pnl_pct,
        vol_stop_triggered=(vol_exit_date != real_exit_date),
    )


def main():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    # Pair each BUY with the next SELL of same (account, ticker)
    rows = conn.execute(
        "SELECT account, ticker, side, shares, price, timestamp "
        "FROM trades WHERE side IN ('buy','sell') "
        "ORDER BY account, ticker, timestamp"
    ).fetchall()

    holdings = defaultdict(list)   # (account,ticker) -> [(buy_dt, buy_px, shares)]
    pairs = []
    for r in rows:
        key = (r["account"], r["ticker"])
        if r["side"] == "buy":
            holdings[key].append((r["timestamp"], r["price"], r["shares"]))
        else:
            # Match against earliest open buy (FIFO)
            if not holdings[key]:
                continue
            buy_dt, buy_px, _ = holdings[key].pop(0)
            pnl = (r["price"] - buy_px) / buy_px
            pairs.append((r["account"], r["ticker"], buy_dt, buy_px,
                          r["timestamp"], r["price"], pnl))

    print(f"matched buy/sell pairs: {len(pairs)}")
    sims = []
    for acc, tk, bd, bp, sd, sp, pnl in pairs:
        s = simulate_one(conn, acc, tk, bd, bp, sd, sp, pnl)
        if s:
            sims.append(s)
    print(f"simulated: {len(sims)}")

    # Aggregate
    real_pnls = [s["real_pnl"] for s in sims]
    vol_pnls = [s["vol_pnl"] for s in sims]
    diffs = [s["diff"] for s in sims]
    triggered = [s for s in sims if s["vol_stop_triggered"]]

    def stats(xs):
        return dict(
            mean=statistics.mean(xs),
            median=statistics.median(xs),
            std=statistics.stdev(xs) if len(xs) > 1 else 0,
            wins=sum(1 for x in xs if x > 0)/len(xs) if xs else 0,
        )

    print("\n=== PnL per trade (FIFO buy→sell pairs) ===")
    print(f"               n={len(sims)}")
    print(f"REAL  fixed:   mean={stats(real_pnls)['mean']*100:+.2f}%  "
          f"median={stats(real_pnls)['median']*100:+.2f}%  "
          f"win={stats(real_pnls)['wins']*100:.1f}%")
    print(f"SIM   vol:     mean={stats(vol_pnls)['mean']*100:+.2f}%  "
          f"median={stats(vol_pnls)['median']*100:+.2f}%  "
          f"win={stats(vol_pnls)['wins']*100:.1f}%")
    print(f"DIFF (vol-real): mean={stats(diffs)['mean']*100:+.3f}%  "
          f"median={stats(diffs)['median']*100:+.3f}%")

    # How often did vol stop fire when fixed didn't (or vice versa)?
    saved = sum(1 for s in sims if s["diff"] > 0.001 and s["vol_stop_triggered"])
    hurt  = sum(1 for s in sims if s["diff"] < -0.001 and s["vol_stop_triggered"])
    print(f"\nvol stop triggered earlier in {len(triggered)} trades:")
    print(f"  → improved outcome (diff > +0.1%): {saved}")
    print(f"  → worse outcome   (diff < -0.1%): {hurt}")

    # Stop pct distribution
    stops = [s["vol_stop"] for s in sims]
    sigmas = [s["sigma"] for s in sims if s["sigma"] is not None]
    print(f"\nvol_stop pct distribution (1.5σ clipped to [{FLOOR*100:.1f}%,{CAP*100:.1f}%]):")
    print(f"  Q1={sorted(stops)[len(stops)//4]*100:.2f}%  "
          f"median={sorted(stops)[len(stops)//2]*100:.2f}%  "
          f"Q3={sorted(stops)[3*len(stops)//4]*100:.2f}%")
    print(f"  hit floor: {sum(1 for s in stops if s<=FLOOR+1e-6)} "
          f"({sum(1 for s in stops if s<=FLOOR+1e-6)/len(stops)*100:.1f}%)")
    print(f"  hit cap:   {sum(1 for s in stops if s>=CAP-1e-6)} "
          f"({sum(1 for s in stops if s>=CAP-1e-6)/len(stops)*100:.1f}%)")

    # Per-account breakdown (just averages)
    by_acc = defaultdict(list)
    for s in sims:
        by_acc[s["account"]].append(s["diff"])
    print("\nPer-account mean PnL diff (vol - fixed):")
    for acc, ds in sorted(by_acc.items()):
        if len(ds) < 5: continue
        print(f"  {acc}: n={len(ds):3d}  mean_diff={sum(ds)/len(ds)*100:+.3f}%")


if __name__ == "__main__":
    main()
