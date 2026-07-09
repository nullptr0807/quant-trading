"""SQLite storage for prices, trades, accounts, positions, factors, and adaptive state."""

import sqlite3
import json
import os
import time
import logging
from datetime import datetime, timezone

import pandas as pd

log = logging.getLogger("quant.store")

DB_PATH = os.path.expanduser("~/quant-trading/data/trading.db")
_CN_SUFFIXES = (".SH", ".SZ", ".BJ")


def _is_cn_ticker(ticker: str) -> bool:
    return str(ticker or "").upper().endswith(_CN_SUFFIXES)


def _market_for_trade(account: str, ticker: str, market: str) -> str:
    """Normalize market for a trade row.

    Historical bug class: CN tickers like ``300782.SZ`` were occasionally
    persisted with ``market='US'`` by non-main execution paths, which then made
    account-level replay/audits miss the corresponding sell trades.  Ticker
    suffix is authoritative for CN A-shares; never allow a .SH/.SZ/.BJ trade to
    be stored as US.
    """
    if _is_cn_ticker(ticker):
        if market != "CN":
            log.warning(
                "correcting trade market for CN ticker: account=%s ticker=%s market=%s -> CN",
                account, ticker, market,
            )
        return "CN"
    return market or "US"


def _connect(db_path: str | None = None) -> sqlite3.Connection:
    # Dashboard backtests and live price/update jobs can legitimately touch the
    # shared cache at the same time.  SQLite's default 5s timeout (and some
    # callers' 1s PRAGMA busy_timeout) is too brittle for bulk price writes.
    conn = sqlite3.connect(db_path or DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db(db_path: str | None = None):
    """Create all tables if they don't exist."""
    conn = _connect(db_path)
    c = conn.cursor()

    # 价格数据 (支持多 interval: '1d', '1h', '5m' 等)
    c.execute("""CREATE TABLE IF NOT EXISTS prices (
        ticker TEXT NOT NULL,
        datetime TEXT NOT NULL,
        interval TEXT NOT NULL DEFAULT '1d',
        open REAL, high REAL, low REAL, close REAL, volume REAL,
        PRIMARY KEY (ticker, datetime, interval)
    )""")
    c.execute("CREATE INDEX IF NOT EXISTS idx_prices_ticker_interval ON prices(ticker, interval, datetime)")

    # 交易记录（每笔交易）
    c.execute("""CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        account TEXT NOT NULL,
        ticker TEXT NOT NULL,
        side TEXT NOT NULL,
        shares REAL NOT NULL,
        price REAL NOT NULL,
        cost REAL NOT NULL,
        slippage REAL DEFAULT 0,
        timestamp TEXT NOT NULL
    )""")
    c.execute("CREATE INDEX IF NOT EXISTS idx_trades_account ON trades(account)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(timestamp)")

    # 账户快照（每小时权益曲线）
    c.execute("""CREATE TABLE IF NOT EXISTS accounts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        cash REAL NOT NULL,
        equity REAL NOT NULL,
        timestamp TEXT NOT NULL
    )""")
    c.execute("CREATE INDEX IF NOT EXISTS idx_accounts_name ON accounts(name)")

    # 当前持仓（最新状态，用于重启恢复）
    c.execute("""CREATE TABLE IF NOT EXISTS positions (
        account TEXT NOT NULL,
        ticker TEXT NOT NULL,
        shares REAL NOT NULL,
        avg_cost REAL NOT NULL,
        total_cost REAL NOT NULL DEFAULT 0,
        current_price REAL,
        updated_at TEXT,
        PRIMARY KEY (account, ticker)
    )""")

    # 账户现金余额（用于重启恢复）
    c.execute("""CREATE TABLE IF NOT EXISTS account_state (
        account TEXT PRIMARY KEY,
        cash REAL NOT NULL,
        initial_cash REAL NOT NULL DEFAULT 10000,
        updated_at TEXT NOT NULL
    )""")

    # 持仓历史快照（时间序列）
    c.execute("""CREATE TABLE IF NOT EXISTS positions_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        account TEXT NOT NULL,
        ticker TEXT NOT NULL,
        shares REAL NOT NULL,
        avg_cost REAL NOT NULL,
        market_price REAL,
        market_value REAL,
        unrealized_pnl REAL,
        timestamp TEXT NOT NULL
    )""")
    c.execute("CREATE INDEX IF NOT EXISTS idx_poshist_acct ON positions_history(account, timestamp)")

    # 因子值（按ticker/日期/group/factor存储）
    c.execute("""CREATE TABLE IF NOT EXISTS factor_values (
        ticker TEXT NOT NULL,
        date TEXT NOT NULL,
        factor_name TEXT NOT NULL,
        value REAL,
        factor_group TEXT NOT NULL DEFAULT 'alpha158',
        PRIMARY KEY (ticker, date, factor_name, factor_group)
    )""")
    c.execute("CREATE INDEX IF NOT EXISTS idx_fv_date ON factor_values(date)")

    # 自适应换仓状态
    c.execute("""CREATE TABLE IF NOT EXISTS adaptive_state (
        account TEXT PRIMARY KEY,
        last_rebalance TEXT,
        rebalance_hours REAL,
        sigma_ref REAL,
        updated_at TEXT
    )""")

    # 市场收益率序列（用于自适应换仓）
    c.execute("""CREATE TABLE IF NOT EXISTS market_returns (
        date TEXT PRIMARY KEY,
        avg_return REAL NOT NULL
    )""")

    conn.commit()
    conn.close()


class DataStore:
    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or DB_PATH
        init_db(self.db_path)

    def _conn(self) -> sqlite3.Connection:
        return _connect(self.db_path)

    # ── Prices ──────────────────────────────────────────────────────────────

    def save_prices(self, ticker_or_df, df: pd.DataFrame | None = None,
                    interval: str = "1d"):
        """Save price data. Two signatures:
            save_prices(df, interval=...)           — df must have 'ticker' column
            save_prices(ticker, df, interval=...)   — df indexed by datetime with ohlcv cols
        """
        conn = self._conn()
        if df is None:
            frame = ticker_or_df
            rows = [
                (row["ticker"], str(row.get("datetime", row.name)), interval,
                 row["open"], row["high"], row["low"], row["close"], row["volume"])
                for _, row in frame.iterrows()
            ]
        else:
            ticker = ticker_or_df
            rows = [
                (ticker, str(idx), interval,
                 row["open"], row["high"], row["low"], row["close"], row["volume"])
                for idx, row in df.iterrows()
            ]
        try:
            for attempt in range(3):
                try:
                    conn.executemany(
                        "INSERT OR REPLACE INTO prices (ticker,datetime,interval,open,high,low,close,volume) "
                        "VALUES (?,?,?,?,?,?,?,?)", rows,
                    )
                    conn.commit()
                    break
                except sqlite3.OperationalError as e:
                    if "locked" not in str(e).lower() or attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
        finally:
            conn.close()

    def save_prices_bulk(self, df: pd.DataFrame, interval: str = "1d"):
        """Bulk save. df must have columns: ticker, datetime, open, high, low, close, volume."""
        if df is None or df.empty:
            return
        conn = self._conn()
        rows = [
            (r["ticker"], str(r["datetime"]), interval,
             r["open"], r["high"], r["low"], r["close"], r["volume"])
            for _, r in df.iterrows()
        ]
        try:
            for attempt in range(3):
                try:
                    conn.executemany(
                        "INSERT OR REPLACE INTO prices (ticker,datetime,interval,open,high,low,close,volume) "
                        "VALUES (?,?,?,?,?,?,?,?)", rows,
                    )
                    conn.commit()
                    break
                except sqlite3.OperationalError as e:
                    if "locked" not in str(e).lower() or attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
        finally:
            conn.close()

    def load_prices(self, tickers: list[str], start: str, end: str,
                    interval: str = "1d",
                    skip_zero_volume: bool = True) -> pd.DataFrame:
        """Load cached prices from DB. Returns DataFrame with ticker/datetime/ohlcv columns.

        skip_zero_volume (default True): for intraday intervals (1m/2m/5m/15m/30m),
        filter out rows where volume==0. These are Yahoo's pre/post-market
        placeholder bars synthesized from single odd-lot ECN prints — they
        cause spike-and-revert artifacts (e.g. SPY 657.39 on 2026-04-21 21:15
        sandwiched by 706.x bars) in equity/benchmark curves.

        Empirical evidence (2026-04 audit): 100% of 1371 detected spike outliers
        had volume==0 and occurred outside RTH. Real RTH bars always carry
        volume. Daily/hourly bars (1d/1h) are aggregated by Yahoo and clean,
        so filter does not apply.

        DB rows are NOT deleted — pass skip_zero_volume=False to see raw data
        for audit/debug.
        """
        if not tickers:
            return pd.DataFrame(columns=["datetime", "ticker", "open", "high", "low", "close", "volume"])
        conn = self._conn()
        placeholders = ",".join("?" * len(tickers))
        # Apply zero-volume filter only on guarded intraday intervals
        intraday_guarded = interval in ("1m", "2m", "5m", "15m", "30m")
        vol_clause = " AND COALESCE(volume,0) > 0" if (skip_zero_volume and intraday_guarded) else ""
        query = (
            f"SELECT ticker, datetime, open, high, low, close, volume FROM prices "
            f"WHERE interval=? AND datetime>=? AND datetime<? AND ticker IN ({placeholders})"
            f"{vol_clause} "
            f"ORDER BY ticker, datetime"
        )
        df = pd.read_sql_query(query, conn, params=[interval, start, end, *tickers])
        conn.close()
        return df

    def get_price_coverage(self, tickers: list[str], interval: str = "1d") -> dict:
        """Return {ticker: (min_datetime, max_datetime, count)} for given interval."""
        if not tickers:
            return {}
        conn = self._conn()
        placeholders = ",".join("?" * len(tickers))
        rows = conn.execute(
            f"SELECT ticker, MIN(datetime), MAX(datetime), COUNT(*) FROM prices "
            f"WHERE interval=? AND ticker IN ({placeholders}) GROUP BY ticker",
            [interval, *tickers],
        ).fetchall()
        conn.close()
        return {r[0]: (r[1], r[2], r[3]) for r in rows}

    # ── Trades ──────────────────────────────────────────────────────────────

    def save_trade(self, account: str, ticker: str, side: str, shares: float,
                   price: float, cost: float, slippage: float = 0.0,
                   timestamp: str | None = None, market: str = "US"):
        """Insert a trade record."""
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        market = _market_for_trade(account, ticker, market)
        conn = self._conn()
        conn.execute(
            "INSERT INTO trades (account,ticker,side,shares,price,cost,slippage,timestamp,market) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (account, ticker, side, shares, price, cost, slippage, ts, market),
        )
        conn.commit()
        conn.close()

    def get_trades(self, account: str | None = None, limit: int = 1000,
                   market: str = "US") -> pd.DataFrame:
        """Return trades, optionally filtered by account, scoped to one market."""
        conn = self._conn()
        if account:
            df = pd.read_sql_query(
                "SELECT * FROM trades WHERE account=? AND market=? ORDER BY timestamp DESC LIMIT ?",
                conn, params=(account, market, limit))
        else:
            df = pd.read_sql_query(
                "SELECT * FROM trades WHERE market=? ORDER BY timestamp DESC LIMIT ?",
                conn, params=(market, limit))
        conn.close()
        return df

    # ── Account Snapshots (equity curve) ────────────────────────────────────

    def save_snapshot(self, name: str, cash: float, equity: float,
                      positions: list[dict] | None = None,
                      timestamp: str | None = None, market: str = "US"):
        """Save an account equity snapshot and optionally update positions."""
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        conn = self._conn()
        conn.execute(
            "INSERT INTO accounts (name,cash,equity,timestamp,market) VALUES (?,?,?,?,?)",
            (name, cash, equity, ts, market),
        )
        if positions:
            for p in positions:
                conn.execute(
                    "INSERT OR REPLACE INTO positions (account,ticker,shares,avg_cost,total_cost,current_price,updated_at,market) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (name, p["ticker"], p["shares"], p["avg_cost"],
                     p.get("total_cost", 0), p.get("current_price"), ts, market),
                )
        conn.commit()
        conn.close()

    def get_account_history(self, name: str, market: str = "US") -> pd.DataFrame:
        """Return account snapshots as a DataFrame."""
        conn = self._conn()
        df = pd.read_sql_query(
            "SELECT * FROM accounts WHERE name=? AND market=? ORDER BY timestamp",
            conn, params=(name, market),
        )
        conn.close()
        return df

    # ── Account State (cash + positions for restart recovery) ───────────────

    def save_account_state(self, account: str, cash: float,
                           initial_cash: float = 10000.0, market: str = "US"):
        """Save current cash balance for restart recovery."""
        ts = datetime.now(timezone.utc).isoformat()
        conn = self._conn()
        conn.execute(
            "INSERT OR REPLACE INTO account_state (account, cash, initial_cash, updated_at, market) "
            "VALUES (?,?,?,?,?)",
            (account, cash, initial_cash, ts, market),
        )
        conn.commit()
        conn.close()

    def save_positions(self, account: str, positions: list[dict], market: str = "US"):
        """Save current positions (overwrite). Also append to history."""
        ts = datetime.now(timezone.utc).isoformat()
        conn = self._conn()
        # Clear old positions for this (account, market)
        conn.execute("DELETE FROM positions WHERE account=? AND market=?", (account, market))
        for p in positions:
            conn.execute(
                "INSERT INTO positions (account,ticker,shares,avg_cost,total_cost,current_price,updated_at,market) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (account, p["ticker"], p["shares"], p["avg_cost"],
                 p.get("total_cost", 0), p.get("current_price"), ts, market),
            )
            # History
            conn.execute(
                "INSERT INTO positions_history (account,ticker,shares,avg_cost,market_price,market_value,unrealized_pnl,timestamp,market) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (account, p["ticker"], p["shares"], p["avg_cost"],
                 p.get("current_price"), p.get("market_value"), p.get("unrealized_pnl"), ts, market),
            )
        conn.commit()
        conn.close()

    def save_account_full_state(
        self,
        account: str,
        cash: float,
        initial_cash: float,
        positions: list[dict],
        *,
        equity: float | None = None,
        timestamp: str | None = None,
        market: str = "US",
    ):
        """Atomically persist account cash + positions, optionally equity snapshot.

        This is the safe post-trade state writer. Cash and positions must not be
        saved by separate code paths: if a held ticker is missing a quote, we may
        skip the equity snapshot, but cash and holdings still have to advance
        together so the next realtime updater does not combine new cash with old
        positions.
        """
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        conn = self._conn()
        try:
            conn.execute("BEGIN")
            conn.execute(
                "INSERT OR REPLACE INTO account_state (account, cash, initial_cash, updated_at, market) "
                "VALUES (?,?,?,?,?)",
                (account, cash, initial_cash, ts, market),
            )
            conn.execute("DELETE FROM positions WHERE account=? AND market=?", (account, market))
            for p in positions:
                conn.execute(
                    "INSERT INTO positions (account,ticker,shares,avg_cost,total_cost,current_price,updated_at,market) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (account, p["ticker"], p["shares"], p["avg_cost"],
                     p.get("total_cost", 0), p.get("current_price"), ts, market),
                )
                conn.execute(
                    "INSERT INTO positions_history (account,ticker,shares,avg_cost,market_price,market_value,unrealized_pnl,timestamp,market) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (account, p["ticker"], p["shares"], p["avg_cost"],
                     p.get("current_price"), p.get("market_value"), p.get("unrealized_pnl"), ts, market),
                )
            if equity is not None:
                conn.execute(
                    "INSERT INTO accounts (name,cash,equity,timestamp,market) VALUES (?,?,?,?,?)",
                    (account, cash, equity, ts, market),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def load_account_state(self, account: str, market: str = "US") -> dict | None:
        """Load saved account state. Returns {cash, initial_cash} or None."""
        conn = self._conn()
        row = conn.execute(
            "SELECT cash, initial_cash FROM account_state WHERE account=? AND market=?",
            (account, market),
        ).fetchone()
        conn.close()
        if row:
            return {"cash": row[0], "initial_cash": row[1]}
        return None

    def load_positions(self, account: str, market: str = "US") -> list[dict]:
        """Load saved positions for an account."""
        conn = self._conn()
        rows = conn.execute(
            "SELECT ticker, shares, avg_cost, total_cost FROM positions WHERE account=? AND market=?",
            (account, market),
        ).fetchall()
        conn.close()
        return [{"ticker": r[0], "shares": r[1], "avg_cost": r[2], "total_cost": r[3]} for r in rows]

    # ── Factor Values ──────────────────────────────────────────────────────

    def save_factor_values(self, ticker: str, date: str, factors: dict[str, float],
                           group: str = "alpha158"):
        """Save computed factor values for a ticker on a date."""
        conn = self._conn()
        for name, value in factors.items():
            if value is not None and pd.notna(value):
                conn.execute(
                    "INSERT OR REPLACE INTO factor_values (ticker,date,factor_name,value,factor_group) "
                    "VALUES (?,?,?,?,?)",
                    (ticker, date, name, float(value), group),
                )
        conn.commit()
        conn.close()

    def save_factor_df(self, factor_dict: dict, group: str = "alpha158"):
        """Save factors from {ticker: DataFrame} dict. DataFrame columns = factor names.

        Bulk-written with observability. The old row-by-row nested execute loop
        made large Alpha158 writes slow and opaque; if a cron wrapper killed the
        process later, logs only said "Computed factors" without proving the DB
        write landed. This method now logs the attempted row count, coverage, max
        factor date, and elapsed write time.
        """
        t0 = time.time()
        rows: list[tuple[str, str, str, float, str]] = []
        tickers_seen = 0
        min_date = None
        max_date = None
        for ticker, df in factor_dict.items():
            if df is None or df.empty:
                continue
            tickers_seen += 1
            for idx, row in df.iterrows():
                date_str = str(idx)[:10]  # YYYY-MM-DD
                min_date = date_str if min_date is None else min(min_date, date_str)
                max_date = date_str if max_date is None else max(max_date, date_str)
                for col in df.columns:
                    val = row[col]
                    if val is not None and pd.notna(val):
                        rows.append((ticker, date_str, col, float(val), group))

        if not rows:
            log.warning(
                "save_factor_df group=%s: no non-NaN factor rows (tickers=%d)",
                group, tickers_seen,
            )
            return {"rows": 0, "tickers": tickers_seen, "min_date": min_date, "max_date": max_date}

        conn = self._conn()
        try:
            conn.executemany(
                "INSERT OR REPLACE INTO factor_values (ticker,date,factor_name,value,factor_group) "
                "VALUES (?,?,?,?,?)",
                rows,
            )
            conn.commit()
        finally:
            conn.close()
        elapsed = time.time() - t0
        log.info(
            "save_factor_df group=%s wrote rows=%d tickers=%d dates=%s..%s elapsed=%.1fs",
            group, len(rows), tickers_seen, min_date, max_date, elapsed,
        )
        return {"rows": len(rows), "tickers": tickers_seen, "min_date": min_date, "max_date": max_date, "elapsed_s": elapsed}

    def get_factor_values(self, ticker: str, date: str | None = None,
                          group: str | None = None) -> pd.DataFrame:
        """Get factor values for a ticker."""
        conn = self._conn()
        query = "SELECT date, factor_name, value, factor_group FROM factor_values WHERE ticker=?"
        params = [ticker]
        if date:
            query += " AND date=?"
            params.append(date)
        if group:
            query += " AND factor_group=?"
            params.append(group)
        df = pd.read_sql_query(query + " ORDER BY date", conn, params=params)
        conn.close()
        return df

    # ── Adaptive Rebalance State ───────────────────────────────────────────

    def save_adaptive_state(self, account: str, last_rebalance: str | None,
                            rebalance_hours: float | None = None,
                            sigma_ref: float | None = None, market: str = "US"):
        """Save adaptive rebalance state for an account."""
        ts = datetime.now(timezone.utc).isoformat()
        conn = self._conn()
        conn.execute(
            "INSERT OR REPLACE INTO adaptive_state (account, last_rebalance, rebalance_hours, sigma_ref, updated_at, market) "
            "VALUES (?,?,?,?,?,?)",
            (account, last_rebalance, rebalance_hours, sigma_ref, ts, market),
        )
        conn.commit()
        conn.close()

    def load_adaptive_state(self, account: str, market: str = "US") -> dict | None:
        """Load adaptive state. Returns {last_rebalance, rebalance_hours, sigma_ref} or None."""
        conn = self._conn()
        row = conn.execute(
            "SELECT last_rebalance, rebalance_hours, sigma_ref FROM adaptive_state WHERE account=? AND market=?",
            (account, market),
        ).fetchone()
        conn.close()
        if row:
            return {"last_rebalance": row[0], "rebalance_hours": row[1], "sigma_ref": row[2]}
        return None

    def load_all_adaptive_state(self, market: str = "US") -> dict:
        """Load all adaptive states for a market. Returns {account_id: {last_rebalance, ...}}."""
        conn = self._conn()
        rows = conn.execute(
            "SELECT account, last_rebalance, rebalance_hours, sigma_ref FROM adaptive_state WHERE market=?",
            (market,),
        ).fetchall()
        conn.close()
        return {
            r[0]: {"last_rebalance": r[1], "rebalance_hours": r[2], "sigma_ref": r[3]}
            for r in rows
        }

    def save_market_returns(self, returns: list[tuple[str, float]]):
        """Save daily market returns. [(date_str, avg_return), ...]"""
        conn = self._conn()
        for date_str, ret in returns:
            conn.execute(
                "INSERT OR REPLACE INTO market_returns (date, avg_return) VALUES (?,?)",
                (date_str, ret),
            )
        conn.commit()
        conn.close()

    def load_market_returns(self, limit: int = 120) -> list[float]:
        """Load recent market returns as a list of floats."""
        conn = self._conn()
        rows = conn.execute(
            "SELECT avg_return FROM market_returns ORDER BY date DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        # Reverse to chronological order
        return [r[0] for r in reversed(rows)]
