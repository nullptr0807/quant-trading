"""Export trading.db price data to qlib bin format.

qlib expects a directory layout:
    <provider_uri>/
        calendars/day.txt          # one ISO date per line, sorted ascending
        instruments/all.txt        # tab-separated: ticker  start_date  end_date
        features/<ticker_lower>/<field>.day.bin    # float32 LE, [start_idx, *values]

Each `.bin` is a contiguous slice of the calendar starting at `start_idx`,
NaN-filled for any missing trading day inside the slice.

Source of truth: data/trading.db `prices` table (interval='1d', market='US').
Run via:
    python -m factors.qlib_export                   # full rebuild
    python -m factors.qlib_export --incremental     # appends only

Idempotent: full mode wipes the target dir first.
"""
from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.expanduser("~/quant-trading")
sys.path.insert(0, PROJECT_ROOT)

DB_PATH = os.path.join(PROJECT_ROOT, "data", "trading.db")
QLIB_DIR = os.path.expanduser("~/.qlib/qlib_data/us_data")

# Fields qlib's Alpha158 expects (we have all of these in `prices` table).
# qlib also expects `factor` (adjustment factor). Since our prices are already
# adjusted (yfinance auto_adjust=True), we write factor=1.0 throughout.
FIELDS = ["open", "high", "low", "close", "volume", "factor"]

log = logging.getLogger("qlib_export")


def _connect():
    # Export is read-heavy and can overlap dashboard/realtime writers. Use WAL +
    # a generous busy timeout so transient writes don't abort the whole retrain.
    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=60000")
    conn.row_factory = sqlite3.Row
    return conn


def _fetch_universe_df(conn, market: str = "US",
                       universe: list[str] | None = None) -> pd.DataFrame:
    """Pull all 1d bars for `market` into a long DataFrame.

    Market is inferred from ticker suffix (no `market` column on `prices`):
        CN: ticker ends in .SH/.SZ/.BJ
        US: everything else

    If `universe` is provided, restrict to that list (avoids exporting
    historical/delisted tickers that are no longer in STOCK_UNIVERSE).
    """
    if market == "CN":
        suffix_clause = "(ticker LIKE '%.SH' OR ticker LIKE '%.SZ' OR ticker LIKE '%.BJ')"
    else:
        suffix_clause = "(ticker NOT LIKE '%.SH' AND ticker NOT LIKE '%.SZ' AND ticker NOT LIKE '%.BJ')"

    q = f"""
        SELECT ticker, datetime, open, high, low, close, volume
          FROM prices
         WHERE interval='1d' AND {suffix_clause}
           AND open IS NOT NULL AND close IS NOT NULL
    """
    df = pd.read_sql_query(q, conn)
    if df.empty:
        return df
    if universe is not None:
        keep = set(universe)
        df = df[df["ticker"].isin(keep)].copy()
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None).dt.normalize()
    df = df.drop_duplicates(["ticker", "datetime"], keep="last")
    df = df.sort_values(["ticker", "datetime"]).reset_index(drop=True)
    return df


def _write_calendar(out_dir: Path, dates: list[pd.Timestamp]) -> None:
    cal_dir = out_dir / "calendars"
    cal_dir.mkdir(parents=True, exist_ok=True)
    txt = "\n".join(d.strftime("%Y-%m-%d") for d in dates) + "\n"
    (cal_dir / "day.txt").write_text(txt)
    log.info("calendar/day.txt: %d entries", len(dates))


def _write_instruments(out_dir: Path, ticker_ranges: dict[str, tuple[pd.Timestamp, pd.Timestamp]]) -> None:
    inst_dir = out_dir / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for tk in sorted(ticker_ranges):
        start, end = ticker_ranges[tk]
        rows.append(f"{tk}\t{start:%Y-%m-%d}\t{end:%Y-%m-%d}")
    (inst_dir / "all.txt").write_text("\n".join(rows) + "\n")
    log.info("instruments/all.txt: %d tickers", len(ticker_ranges))


def _write_feature_bin(path: Path, start_idx: int, values: np.ndarray) -> None:
    """Write a single ticker/field .bin: float32 LE, [start_idx, *values]."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = np.concatenate([[start_idx], values.astype(np.float32)]).astype("<f4")
    with path.open("wb") as f:
        payload.tofile(f)


def _build_features_for_ticker(
    sub: pd.DataFrame,
    cal_index: dict[pd.Timestamp, int],
) -> tuple[int, dict[str, np.ndarray]]:
    """Reindex one ticker's bars onto the global calendar.

    Returns (start_idx, {field: contiguous_values}).
    Values inside the [start_idx, end_idx] slice that the ticker doesn't
    cover get NaN (e.g. a non-trading day).
    """
    sub = sub.set_index("datetime")
    first_dt = sub.index.min()
    last_dt = sub.index.max()
    start_idx = cal_index[first_dt]
    end_idx = cal_index[last_dt]
    n = end_idx - start_idx + 1

    # Build slice of calendar dates for this ticker
    cal_dates = sorted(cal_index, key=lambda d: cal_index[d])
    slice_dates = cal_dates[start_idx : end_idx + 1]

    sub_reindexed = sub.reindex(slice_dates)
    out: dict[str, np.ndarray] = {}
    for field in ("open", "high", "low", "close", "volume"):
        out[field] = sub_reindexed[field].to_numpy(dtype=np.float64)
    out["factor"] = np.ones(n, dtype=np.float64)
    return start_idx, out


def export(market: str = "US", out_dir: str = QLIB_DIR, wipe: bool = True,
           universe: list[str] | None = None) -> dict:
    """Full export. Returns summary dict.

    If `universe` is None, defaults to current STOCK_UNIVERSE (US) or
    CN_UNIVERSE (CN) plus benchmarks — keeps the qlib bin dir aligned
    with the live trading universe and avoids historical/delisted noise.
    """
    out = Path(out_dir)
    if wipe and out.exists():
        import shutil
        log.warning("wiping %s", out)
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    if universe is None:
        if market == "US":
            from config.settings import STOCK_UNIVERSE
            universe = list(STOCK_UNIVERSE) + ["QQQ", "SPY"]
        else:
            from config.settings import CN_UNIVERSE
            universe = list(CN_UNIVERSE) + ["000300.SH"]

    conn = _connect()
    try:
        df = _fetch_universe_df(conn, market=market, universe=universe)
    finally:
        conn.close()
    if df.empty:
        raise RuntimeError(f"no 1d data found for market={market}")

    # Global calendar = union of all trading days seen across the universe
    all_dates = sorted(df["datetime"].unique())
    cal_index = {d: i for i, d in enumerate(all_dates)}
    _write_calendar(out, all_dates)

    # Per-ticker write
    ranges: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    n_tickers = df["ticker"].nunique()
    for i, (tk, sub) in enumerate(df.groupby("ticker", sort=True), 1):
        start_idx, fields = _build_features_for_ticker(sub, cal_index)
        feat_dir = out / "features" / tk.lower()
        for field, values in fields.items():
            _write_feature_bin(feat_dir / f"{field}.day.bin", start_idx, values)
        ranges[tk] = (sub["datetime"].min(), sub["datetime"].max())
        if i % 200 == 0 or i == n_tickers:
            log.info("  features written: %d/%d", i, n_tickers)

    _write_instruments(out, ranges)
    summary = {
        "market": market,
        "out_dir": str(out),
        "tickers": n_tickers,
        "calendar_days": len(all_dates),
        "first_date": str(all_dates[0].date()),
        "last_date": str(all_dates[-1].date()),
    }
    log.info("export done: %s", summary)
    return summary


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--market", default="US", choices=["US", "CN"])
    p.add_argument("--out", default=None, help="override provider_uri (defaults: ~/.qlib/qlib_data/us_data | cn_data)")
    p.add_argument("--no-wipe", action="store_true", help="incremental-style: don't wipe target dir first")
    args = p.parse_args()

    out_dir = args.out or os.path.expanduser(f"~/.qlib/qlib_data/{args.market.lower()}_data")
    export(market=args.market, out_dir=out_dir, wipe=not args.no_wipe)


if __name__ == "__main__":
    main()
