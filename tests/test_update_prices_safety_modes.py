import importlib.util
import sqlite3
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "update_prices.py"
SPEC = importlib.util.spec_from_file_location("hardening_update_prices", MODULE_PATH)
assert SPEC and SPEC.loader
update_prices = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(update_prices)


def _seed_account(db_path):
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE account_meta (
                account_id TEXT PRIMARY KEY, market TEXT, status TEXT
            );
            CREATE TABLE account_state (
                account TEXT, cash REAL, initial_cash REAL, updated_at TEXT, market TEXT,
                PRIMARY KEY (account, market)
            );
            CREATE TABLE positions (
                account TEXT, ticker TEXT, shares REAL, avg_cost REAL, total_cost REAL,
                current_price REAL, updated_at TEXT, market TEXT,
                PRIMARY KEY (account, ticker, market)
            );
            CREATE TABLE accounts (
                id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, cash REAL, equity REAL,
                timestamp TEXT, market TEXT
            );
            CREATE TABLE trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT, account TEXT, ticker TEXT, side TEXT,
                shares REAL, price REAL, cost REAL, slippage REAL, timestamp TEXT, market TEXT
            );
            CREATE TABLE events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, category TEXT, severity TEXT,
                account TEXT, ticker TEXT, title TEXT, detail TEXT, market TEXT
            );
            CREATE TABLE operational_health (
                component TEXT, market TEXT, status TEXT, success_at TEXT,
                source_timestamp TEXT, details TEXT,
                PRIMARY KEY (component, market)
            );
            CREATE TABLE positions_history (
                account TEXT, ticker TEXT, market_price REAL, timestamp TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO account_meta (account_id,market,status) VALUES ('A01','US','active')"
        )
        conn.execute(
            "INSERT INTO account_state (account,cash,initial_cash,updated_at,market) "
            "VALUES ('A01',9000,10000,'2026-07-10T00:00:00+00:00','US')"
        )
        conn.execute(
            "INSERT INTO positions "
            "(account,ticker,shares,avg_cost,total_cost,current_price,updated_at,market) "
            "VALUES ('A01','AAPL',10,100,1000,100,'2026-07-10T00:00:00+00:00','US')"
        )
        conn.execute(
            "INSERT INTO accounts (name,cash,equity,timestamp,market) "
            "VALUES ('A01',9000,10000,'2026-07-10T00:00:00+00:00','US')"
        )


def _table_counts(db_path):
    with sqlite3.connect(db_path) as conn:
        return {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("trades", "events", "accounts", "positions", "account_state")
        }


def test_dry_run_fetches_and_reports_but_leaves_database_byte_identical(tmp_path, monkeypatch):
    db = tmp_path / "trading.db"
    _seed_account(db)
    before_bytes = db.read_bytes()
    before_counts = _table_counts(db)

    monkeypatch.setattr(update_prices, "_is_us_market_hours_now", lambda: True)
    monkeypatch.setattr(update_prices, "_is_us_regular_session_now", lambda: True)
    monkeypatch.setattr(update_prices, "_is_cn_market_hours_now", lambda: False)
    now = update_prices._utc_now()
    monkeypatch.setattr(
        update_prices,
        "fetch_quote_metadata",
        lambda tickers: {
            "AAPL": __import__("data.quotes", fromlist=["RealtimeQuote"]).RealtimeQuote(
                ticker="AAPL", price=90.0, source="test",
                source_timestamp=now, received_at=now, tradable=True,
            )
        },
    )
    monkeypatch.setitem(update_prices.STOP_LOSS_BY_ACCT, "A01", 0.05)

    stats = update_prices.update_equity_snapshots(str(db), dry_run=True)

    assert stats["dry_run"] is True
    assert stats["would_stop_losses"] == 1
    assert stats["stop_losses"] == 0
    assert stats["accounts_updated"] == 0
    assert _table_counts(db) == before_counts
    assert db.read_bytes() == before_bytes


def test_no_trades_updates_snapshots_without_selling(tmp_path, monkeypatch):
    db = tmp_path / "trading.db"
    _seed_account(db)
    before_counts = _table_counts(db)

    monkeypatch.setattr(update_prices, "_is_us_market_hours_now", lambda: True)
    monkeypatch.setattr(update_prices, "_is_us_regular_session_now", lambda: True)
    monkeypatch.setattr(update_prices, "_is_cn_market_hours_now", lambda: False)
    now = update_prices._utc_now()
    monkeypatch.setattr(
        update_prices,
        "fetch_quote_metadata",
        lambda tickers: {
            "AAPL": __import__("data.quotes", fromlist=["RealtimeQuote"]).RealtimeQuote(
                ticker="AAPL", price=90.0, source="test",
                source_timestamp=now, received_at=now, tradable=True,
            )
        },
    )
    monkeypatch.setitem(update_prices.STOP_LOSS_BY_ACCT, "A01", 0.05)

    stats = update_prices.update_equity_snapshots(str(db), no_trades=True)

    assert stats["no_trades"] is True
    assert stats["would_stop_losses"] == 1
    assert stats["stop_losses"] == 0
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0] == 1
        assert conn.execute("SELECT cash FROM account_state WHERE account='A01'").fetchone()[0] == 9000
        assert conn.execute("SELECT current_price FROM positions WHERE account='A01'").fetchone()[0] == 90
        assert conn.execute("SELECT COUNT(*) FROM accounts").fetchone()[0] == before_counts["accounts"] + 1


def test_no_trades_routes_risk_regime_to_selected_database(tmp_path, monkeypatch):
    db = tmp_path / "trading.db"
    _seed_account(db)
    now = update_prices._utc_now()
    monkeypatch.setattr(update_prices, "_is_us_market_hours_now", lambda: True)
    monkeypatch.setattr(update_prices, "_is_us_regular_session_now", lambda: True)
    monkeypatch.setattr(update_prices, "_is_cn_market_hours_now", lambda: False)
    monkeypatch.setattr(
        update_prices,
        "fetch_quote_metadata",
        lambda tickers: {
            "AAPL": __import__("data.quotes", fromlist=["RealtimeQuote"]).RealtimeQuote(
                ticker="AAPL", price=101.0, source="test",
                source_timestamp=now, received_at=now, tradable=True,
            )
        },
    )
    from trading import risk_regime

    original = risk_regime.DB_PATH
    seen = []
    monkeypatch.setattr(
        risk_regime,
        "evaluate_and_update",
        lambda *, db_path: seen.append(db_path) or {"transitioned": False},
    )

    update_prices.update_equity_snapshots(str(db), no_trades=True)

    assert seen == [str(db)]
    assert risk_regime.DB_PATH == original


def test_cli_requires_explicit_live_flag_and_forwards_safety_modes(monkeypatch):
    calls = []
    monkeypatch.setattr(
        update_prices,
        "update_equity_snapshots",
        lambda db_path, *, dry_run, no_trades: calls.append((db_path, dry_run, no_trades)) or {},
    )

    assert update_prices.main(["--db", "/tmp/test.db", "--dry-run"]) == 0
    assert calls.pop() == ("/tmp/test.db", True, False)

    assert update_prices.main(["--db", "/tmp/test.db", "--no-trades"]) == 0
    assert calls.pop() == ("/tmp/test.db", False, True)

    assert update_prices.main(["--db", "/tmp/test.db", "--live"]) == 0
    assert calls.pop() == ("/tmp/test.db", False, False)
