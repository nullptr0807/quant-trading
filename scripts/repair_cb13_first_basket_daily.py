"""Repair CB13 first-basket historical equity rows using daily close prices.

The first CB13 position set (bought 2026-05-01, sold 2026-06-08) suffered many
intraday avg-cost / partial-price snapshot artifacts. For overview/history this
script recomputes every account snapshot between the first buy and first sell
from the actual trade book and 1d close prices at-or-before the snapshot date.

Non-destructive: backs up affected rows before UPDATE.
"""
from __future__ import annotations

import argparse, json, os, shutil, sqlite3
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "trading.db")
ACCOUNT = "CB13"
MARKET = "CN"
INITIAL_CASH = 100000.0


def _date(ts: str) -> str:
    return ts[:10]


def _load_prev_close(conn: sqlite3.Connection, tickers: list[str]) -> dict[str, list[tuple[str, float]]]:
    out = {}
    for tk in tickers:
        rows = conn.execute(
            "SELECT substr(datetime,1,10) d, close FROM prices "
            "WHERE ticker=? AND interval='1d' ORDER BY datetime",
            (tk,),
        ).fetchall()
        out[tk] = [(r["d"], float(r["close"])) for r in rows if r["close"] is not None]
    return out


def _price_at(series: list[tuple[str, float]], d: str) -> float | None:
    # Series is small enough for linear scan; use last close <= d.
    last = None
    for dd, px in series:
        if dd <= d:
            last = px
        else:
            break
    return last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.apply and not args.dry_run:
        args.dry_run = True

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        trades = [dict(r) for r in conn.execute(
            "SELECT * FROM trades WHERE account=? ORDER BY timestamp,id", (ACCOUNT,)
        )]
        if not trades:
            raise SystemExit("no trades")
        first_buy_ts = min(t["timestamp"] for t in trades if t["side"].lower() == "buy")
        first_sell_ts = min(t["timestamp"] for t in trades if t["side"].lower() == "sell")
        first_buys = [t for t in trades if t["timestamp"] >= first_buy_ts and t["timestamp"] < first_sell_ts and t["side"].lower() == "buy"]
        tickers = [t["ticker"] for t in first_buys]
        px_series = _load_prev_close(conn, tickers)

        cash = INITIAL_CASH
        shares: dict[str, float] = {}
        for t in first_buys:
            sh = float(t["shares"]); px = float(t["price"]); cost = float(t.get("cost") or 0) + float(t.get("slippage") or 0)
            cash -= sh * px + cost
            shares[t["ticker"]] = shares.get(t["ticker"], 0.0) + sh

        rows = [dict(r) for r in conn.execute(
            "SELECT id,timestamp,cash,equity FROM accounts "
            "WHERE name=? AND market=? AND timestamp>=? AND timestamp<? ORDER BY timestamp",
            (ACCOUNT, MARKET, first_buy_ts, first_sell_ts),
        )]
        repairs = []
        for r in rows:
            d = _date(r["timestamp"])
            equity = cash
            missing = []
            for tk, sh in shares.items():
                px = _price_at(px_series.get(tk, []), d)
                if px is None:
                    missing.append(tk)
                    continue
                equity += sh * px
            if missing:
                continue
            equity = round(equity, 4)
            if abs(equity - float(r["equity"])) < 100.0:
                continue
            repairs.append({
                "id": int(r["id"]),
                "timestamp": r["timestamp"],
                "old_cash": float(r["cash"]),
                "old_equity": float(r["equity"]),
                "new_cash": round(cash, 4),
                "new_equity": equity,
            })
        print(f"window={first_buy_ts} -> {first_sell_ts} rows={len(rows)} repairs={len(repairs)} tickers={tickers} cash={cash:.4f}")
        for x in repairs[:10]: print(x)
        if len(repairs) > 10:
            print('...')
            for x in repairs[-8:]: print(x)
        if not args.apply:
            print('DRY RUN — no DB changes')
            return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        now_iso = datetime.now(timezone.utc).isoformat()
        db_backup = f"{DB_PATH}.bak_cb13_first_basket_daily_{stamp}"
        shutil.copy2(DB_PATH, db_backup)
        backup_table = f"accounts_backup_cb13_first_basket_daily_{stamp}"
        conn.execute(f"CREATE TABLE {backup_table} AS SELECT *, ? AS backup_ts FROM accounts WHERE 0", (now_iso,))
        ids = [r["id"] for r in repairs]
        if ids:
            placeholders = ','.join(['?'] * len(ids))
            conn.execute(
                f"INSERT INTO {backup_table} SELECT *, ? AS backup_ts FROM accounts WHERE id IN ({placeholders})",
                [now_iso, *ids],
            )
            for r in repairs:
                conn.execute("UPDATE accounts SET cash=?, equity=? WHERE id=?", (r["new_cash"], r["new_equity"], r["id"]))
        conn.execute(
            """
            INSERT INTO events (ts, category, severity, account, ticker, title, detail, market)
            VALUES (?, 'data', 'warn', ?, NULL, ?, ?, ?)
            """,
            (now_iso, ACCOUNT, "🧹 CB13 首篮子权益曲线按日线回填", json.dumps({
                "reason": "recomputed CB13 first-basket account snapshots from trade book + 1d close prices",
                "rows_repaired": len(repairs),
                "backup_table": backup_table,
                "db_backup": db_backup,
                "window": [first_buy_ts, first_sell_ts],
                "tickers": tickers,
            }, ensure_ascii=False), MARKET),
        )
        conn.commit()
        print('APPLIED')
        print('db_backup', db_backup)
        print('backup_table', backup_table)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
