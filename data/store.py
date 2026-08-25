"""SQLite storage for prices, trades, accounts, positions, factors, and adaptive state."""

import sqlite3
import json
import math
import os
import time
import logging
from datetime import datetime, timezone

import pandas as pd

log = logging.getLogger("quant.store")

DB_PATH = os.path.expanduser("~/quant-trading/data/trading.db")
_CN_SUFFIXES = (".SH", ".SZ", ".BJ")
PRICE_MODE_TABLE = {
    "adjusted": "prices",
    "qfq": "prices",
    "research": "prices",
    "raw": "prices_raw",
    "execution": "prices_raw",
}


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


def _price_table(price_mode: str = "adjusted") -> str:
    try:
        return PRICE_MODE_TABLE[str(price_mode or "adjusted").lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported price_mode={price_mode!r}; expected adjusted/qfq/raw") from exc


def _price_quality_issue(row: tuple) -> tuple[str, str] | None:
    """Return issue type/detail for impossible OHLC; never mutate raw input."""
    try:
        ticker, dt, _interval, open_, high, low, close, _volume = row
        values = [float(open_), float(high), float(low), float(close)]
        if not all(pd.notna(value) and math.isfinite(value) for value in values):
            return "invalid_ohlc", "null/non-finite OHLC"
        open_, high, low, close = values
        if min(values) <= 0:
            return "invalid_ohlc", "non-positive OHLC"
        if high < max(open_, low, close) or low > min(open_, high, close):
            return "invalid_ohlc", "high/low envelope invariant violated"
    except (TypeError, ValueError):
        return "invalid_ohlc", "non-numeric OHLC"
    return None


def _record_price_quality_issues(conn: sqlite3.Connection, rows: list[tuple], price_mode: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    issues = []
    for row in rows:
        issue = _price_quality_issue(row)
        if issue:
            issues.append((price_mode, row[0], row[1], row[2], issue[0], now, issue[1]))
        else:
            # A later provider refresh may repair a previously invalid bar.
            # Clear only the matching marker; the original audit row has just
            # been replaced by the valid provider payload in the same txn.
            conn.execute(
                "DELETE FROM price_quality_issues "
                "WHERE price_mode=? AND ticker=? AND datetime=? AND interval=? "
                "AND issue_type='invalid_ohlc'",
                (price_mode, row[0], row[1], row[2]),
            )
    if issues:
        conn.executemany(
            "INSERT OR REPLACE INTO price_quality_issues "
            "(price_mode,ticker,datetime,interval,issue_type,detected_at,detail) VALUES (?,?,?,?,?,?,?)",
            issues,
        )


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

    # Raw/unadjusted OHLCV for execution/account replay.  The legacy `prices`
    # table intentionally remains the adjusted/qfq research series so existing
    # factor and qlib code keeps its historical semantics.
    c.execute("""CREATE TABLE IF NOT EXISTS prices_raw (
        ticker TEXT NOT NULL,
        datetime TEXT NOT NULL,
        interval TEXT NOT NULL DEFAULT '1d',
        open REAL, high REAL, low REAL, close REAL, volume REAL,
        PRIMARY KEY (ticker, datetime, interval)
    )""")
    c.execute("CREATE INDEX IF NOT EXISTS idx_prices_raw_ticker_interval ON prices_raw(ticker, interval, datetime)")

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

    c.execute("""CREATE TABLE IF NOT EXISTS universe_membership (
        market TEXT NOT NULL,
        date TEXT NOT NULL,
        ticker TEXT NOT NULL,
        source TEXT NOT NULL,
        universe_hash TEXT NOT NULL,
        recorded_at TEXT NOT NULL,
        PRIMARY KEY (market,date,ticker)
    )""")
    c.execute(
        "CREATE INDEX IF NOT EXISTS idx_universe_membership_market_date "
        "ON universe_membership(market,date,ticker)"
    )

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

    # Trades and account state are market-scoped. Keep fresh databases aligned
    # with the deployed schema; existing databases already receive these columns
    # from scripts/migrate_add_market.py.
    market_scoped = (
        "trades", "accounts", "positions", "account_state",
        "positions_history", "adaptive_state",
    )
    for table in market_scoped:
        columns = {row[1] for row in c.execute(f"PRAGMA table_info({table})")}
        if "market" not in columns:
            c.execute(
                f"ALTER TABLE {table} ADD COLUMN market TEXT NOT NULL DEFAULT 'US'"
            )

    # Trade events share the same connection/transaction as the ledger write.
    c.execute("""CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        category TEXT NOT NULL,
        severity TEXT NOT NULL DEFAULT 'info',
        account TEXT,
        ticker TEXT,
        title TEXT NOT NULL,
        detail TEXT,
        market TEXT NOT NULL DEFAULT 'US'
    )""")
    event_columns = {row[1] for row in c.execute("PRAGMA table_info(events)")}
    if "market" not in event_columns:
        c.execute("ALTER TABLE events ADD COLUMN market TEXT NOT NULL DEFAULT 'US'")
    c.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts DESC)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_events_cat ON events(category)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_trades_market_ts ON trades(market, timestamp)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_events_market_ts ON events(market, ts)")

    # Immutable audit trail for explicit open-position corporate-action repairs.
    # The repair CLI also creates this table transactionally for older databases.
    c.execute("""CREATE TABLE IF NOT EXISTS corporate_action_repairs (
        action_key TEXT PRIMARY KEY,
        account TEXT NOT NULL, market TEXT NOT NULL, ticker TEXT NOT NULL,
        ex_date TEXT NOT NULL, action_type TEXT NOT NULL, ratio REAL NOT NULL,
        source TEXT NOT NULL, raw TEXT, before_state TEXT NOT NULL,
        after_state TEXT NOT NULL, raw_price_timestamp TEXT NOT NULL,
        backup_handle TEXT NOT NULL, backup_sha256 TEXT NOT NULL,
        applied_at TEXT NOT NULL
    )""")
    c.execute(
        "CREATE INDEX IF NOT EXISTS idx_corporate_action_repairs_scope "
        "ON corporate_action_repairs(market,account,ticker,ex_date)"
    )
    c.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_corporate_action_repair_event "
        "ON corporate_action_repairs(account,market,ticker,ex_date,action_type)"
    )

    c.execute("""CREATE TABLE IF NOT EXISTS operational_health (
        component TEXT NOT NULL,
        market TEXT NOT NULL,
        status TEXT NOT NULL,
        success_at TEXT NOT NULL,
        source_timestamp TEXT,
        details TEXT,
        PRIMARY KEY(component, market)
    )""")
    c.execute("""CREATE TABLE IF NOT EXISTS account_meta (
        account_id TEXT PRIMARY KEY,
        strategy_name TEXT DEFAULT '', description TEXT DEFAULT '',
        "group" TEXT DEFAULT '', factors TEXT DEFAULT '',
        initial_cash REAL DEFAULT 10000, created_at TEXT,
        status TEXT DEFAULT 'active', market TEXT NOT NULL DEFAULT 'US',
        retired_at TEXT, retire_reason TEXT,
        runtime_status TEXT NOT NULL DEFAULT 'ready',
        runtime_reason TEXT, runtime_detail TEXT, runtime_updated_at TEXT
    )""")
    meta_columns = {row[1] for row in c.execute("PRAGMA table_info(account_meta)")}
    for name, ddl in (
        ("runtime_status", "TEXT NOT NULL DEFAULT 'ready'"),
        ("runtime_reason", "TEXT"), ("runtime_detail", "TEXT"),
        ("runtime_updated_at", "TEXT"),
    ):
        if name not in meta_columns:
            c.execute(f"ALTER TABLE account_meta ADD COLUMN {name} {ddl}")

    c.execute("""CREATE TABLE IF NOT EXISTS price_quality_issues (
        price_mode TEXT NOT NULL, ticker TEXT NOT NULL, datetime TEXT NOT NULL,
        interval TEXT NOT NULL, issue_type TEXT NOT NULL, detected_at TEXT NOT NULL,
        detail TEXT,
        PRIMARY KEY(price_mode,ticker,datetime,interval,issue_type)
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
                    interval: str = "1d", price_mode: str = "adjusted"):
        """Save price data.

        price_mode='adjusted' writes the legacy research/qfq table (`prices`);
        price_mode='raw' writes the execution/account-replay table (`prices_raw`).
        """
        table = _price_table(price_mode)
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
                        f"INSERT OR REPLACE INTO {table} (ticker,datetime,interval,open,high,low,close,volume) "
                        "VALUES (?,?,?,?,?,?,?,?)", rows,
                    )
                    _record_price_quality_issues(conn, rows, str(price_mode).lower())
                    conn.commit()
                    break
                except sqlite3.OperationalError as e:
                    if "locked" not in str(e).lower() or attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
        finally:
            conn.close()

    def save_prices_bulk(self, df: pd.DataFrame, interval: str = "1d", price_mode: str = "adjusted"):
        """Bulk save. df must have columns: ticker, datetime, open, high, low, close, volume."""
        if df is None or df.empty:
            return
        table = _price_table(price_mode)
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
                        f"INSERT OR REPLACE INTO {table} (ticker,datetime,interval,open,high,low,close,volume) "
                        "VALUES (?,?,?,?,?,?,?,?)", rows,
                    )
                    _record_price_quality_issues(conn, rows, str(price_mode).lower())
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
                    skip_zero_volume: bool = True,
                    price_mode: str = "adjusted",
                    quality_mode: str = "research") -> pd.DataFrame:
        """Load cached prices.

        ``quality_mode='research'`` (default) quarantines impossible OHLC rows;
        ``'audit'`` returns the original persisted rows unchanged.  Quarantine is
        read-time only so provider evidence remains available for replay/audit.
        """
        if not tickers:
            return pd.DataFrame(columns=["datetime", "ticker", "open", "high", "low", "close", "volume"])
        table = _price_table(price_mode)
        conn = self._conn()
        placeholders = ",".join("?" * len(tickers))
        # Apply zero-volume filter only on guarded intraday intervals
        intraday_guarded = interval in ("1m", "2m", "5m", "15m", "30m")
        vol_clause = " AND COALESCE(volume,0) > 0" if (skip_zero_volume and intraday_guarded) else ""
        if quality_mode not in {"research", "audit"}:
            conn.close()
            raise ValueError("quality_mode must be 'research' or 'audit'")
        quality_clause = ""
        if quality_mode == "research":
            quality_clause = (
                " AND NOT EXISTS (SELECT 1 FROM price_quality_issues q "
                f"WHERE q.price_mode=? AND q.ticker={table}.ticker "
                f"AND q.datetime={table}.datetime AND q.interval={table}.interval)"
            )
        query = (
            f"SELECT ticker, datetime, open, high, low, close, volume FROM {table} "
            f"WHERE interval=? AND datetime>=? AND datetime<? AND ticker IN ({placeholders})"
            f"{vol_clause}{quality_clause} "
            f"ORDER BY ticker, datetime"
        )
        params = [interval, start, end, *tickers]
        if quality_mode == "research":
            params.append(str(price_mode).lower())
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df

    def get_price_coverage(self, tickers: list[str], interval: str = "1d", price_mode: str = "adjusted") -> dict:
        """Return {ticker: (min_datetime, max_datetime, count)} for given interval/price mode."""
        if not tickers:
            return {}
        table = _price_table(price_mode)
        conn = self._conn()
        placeholders = ",".join("?" * len(tickers))
        rows = conn.execute(
            f"SELECT ticker, MIN(datetime), MAX(datetime), COUNT(*) FROM {table} "
            f"WHERE interval=? AND ticker IN ({placeholders}) GROUP BY ticker",
            [interval, *tickers],
        ).fetchall()
        conn.close()
        return {r[0]: (r[1], r[2], r[3]) for r in rows}

    def count_price_quality_issues(self, *, interval: str | None = None,
                                   price_mode: str | None = None) -> dict[str, int]:
        conn = self._conn()
        clauses, params = [], []
        if interval:
            clauses.append("interval=?"); params.append(interval)
        if price_mode:
            clauses.append("price_mode=?"); params.append(str(price_mode).lower())
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = conn.execute(
            "SELECT issue_type,COUNT(*) FROM price_quality_issues" + where + " GROUP BY issue_type",
            params,
        ).fetchall()
        conn.close()
        return {str(row[0]): int(row[1]) for row in rows}

    def set_account_runtime_status(self, account: str, market: str, status: str,
                                   reason: str | None = None,
                                   detail: dict | None = None) -> None:
        """Publish execution readiness separately from lifecycle ``status``."""
        conn = self._conn()
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT OR IGNORE INTO account_meta (account_id,market,created_at) VALUES (?,?,?)",
            (account, market, now),
        )
        conn.execute(
            "UPDATE account_meta SET runtime_status=?,runtime_reason=?,runtime_detail=?,runtime_updated_at=? "
            "WHERE account_id=? AND market=?",
            (status, reason, json.dumps(detail, ensure_ascii=False) if detail is not None else None,
             now, account, market),
        )
        conn.commit()
        conn.close()

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

    def save_trade_with_state(
        self,
        *,
        account: str,
        trade: dict,
        cash: float,
        initial_cash: float,
        positions: list[dict],
        event: dict,
        market: str = "US",
    ) -> None:
        """Atomically persist one trade, its event, and resulting account state."""
        ts = datetime.now(timezone.utc).isoformat()
        ticker = trade["ticker"]
        market = _market_for_trade(account, ticker, market)
        expected_tickers = {
            str(position["ticker"])
            for position in positions
            if float(position.get("shares", 0) or 0) > 0
        }
        expected_position_count = len(expected_tickers)
        conn = self._conn()
        try:
            conn.execute("BEGIN")
            db_tickers = {
                str(row[0]) for row in conn.execute(
                    "SELECT ticker FROM positions WHERE account=? AND market=? AND shares>0",
                    (account, market),
                ).fetchall()
            }
            current_position_count = len(db_tickers)
            # A trade may add or fully remove only its own ticker. Every other
            # holding must already match between restored memory and durable DB.
            if (db_tickers - {ticker}) != (expected_tickers - {ticker}):
                raise RuntimeError(
                    f"atomic trade precondition failed for {account}/{market}: "
                    f"memory_tickers={sorted(expected_tickers)} db_tickers={sorted(db_tickers)}"
                )
            # A first trade can legitimately create the first position. Otherwise
            # the DB and restored in-memory book must agree before we overwrite
            # the full position set in this transaction.
            allowed_counts = {expected_position_count}
            if trade["side"] == "buy" and expected_position_count > 0:
                allowed_counts.add(expected_position_count - 1)
            if trade["side"] == "sell":
                allowed_counts.add(expected_position_count + 1)
            if current_position_count not in allowed_counts:
                raise RuntimeError(
                    f"atomic trade precondition failed for {account}/{market}: "
                    f"post_trade_memory_positions={expected_position_count} "
                    f"db_positions={current_position_count}"
                )
            conn.execute(
                "INSERT INTO trades (account,ticker,side,shares,price,cost,slippage,timestamp,market) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    account, ticker, trade["side"], trade["shares"], trade["price"],
                    trade.get("total_fees", 0),
                    trade.get(
                        "slippage_cost",
                        abs(float(trade.get("exec_price", trade["price"])) - float(trade["price"]))
                        * float(trade["shares"]),
                    ),
                    ts, market,
                ),
            )
            conn.execute(
                "INSERT INTO events (ts,category,severity,account,ticker,title,detail,market) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    ts, "trade", "info", account, ticker, event["title"],
                    json.dumps(event.get("detail"), ensure_ascii=False)
                    if event.get("detail") is not None else None,
                    market,
                ),
            )
            conn.execute(
                "INSERT OR REPLACE INTO account_state "
                "(account,cash,initial_cash,updated_at,market) VALUES (?,?,?,?,?)",
                (account, cash, initial_cash, ts, market),
            )
            conn.execute(
                "DELETE FROM positions WHERE account=? AND market=?",
                (account, market),
            )
            for position in positions:
                conn.execute(
                    "INSERT INTO positions "
                    "(account,ticker,shares,avg_cost,total_cost,current_price,updated_at,market) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (
                        account, position["ticker"], position["shares"],
                        position["avg_cost"], position.get("total_cost", 0),
                        position.get("current_price"), ts, market,
                    ),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def load_position_prices(self, account: str, market: str = "US") -> dict[str, float]:
        """Return last durable positive marks for one account's positions."""
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT ticker,current_price FROM positions "
                "WHERE account=? AND market=? AND current_price>0",
                (account, market),
            ).fetchall()
            return {str(ticker): float(price) for ticker, price in rows}
        finally:
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

    def load_trade_lots(self, account: str, market: str = "US") -> dict[str, list[dict]]:
        """Reconstruct currently-held acquisition lots from the durable trade log.

        FIFO replay is conservative for CN T+1.  If dirty history cannot explain
        the current position, callers must fail closed rather than invent a buy
        date that would make shares sellable.
        """
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT ticker,side,shares,timestamp FROM trades "
                "WHERE account=? AND market=? ORDER BY timestamp,id",
                (account, market),
            ).fetchall()
        finally:
            conn.close()
        lots: dict[str, list[dict]] = {}
        for ticker, side, raw_shares, timestamp in rows:
            shares = float(raw_shares or 0)
            queue = lots.setdefault(str(ticker), [])
            if str(side).lower() == "buy":
                queue.append({"shares": shares, "bought_at": str(timestamp)})
                continue
            if str(side).lower() != "sell":
                continue
            remaining = shares
            kept = []
            for lot in queue:
                qty = float(lot["shares"])
                used = min(qty, remaining)
                qty -= used
                remaining -= used
                if qty > 1e-9:
                    kept.append({**lot, "shares": qty})
            lots[str(ticker)] = kept
        return lots

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
