from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone


def test_ledger_scope_includes_active_holdings_recent_sales_and_excludes_retired(tmp_path, monkeypatch):
    import data.store as store
    from scripts.backfill_prices import _ledger_tickers_for_market

    db = tmp_path / "trading.db"
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE account_meta (account_id TEXT, market TEXT, status TEXT);
        CREATE TABLE positions (account TEXT, market TEXT, ticker TEXT, shares REAL);
        CREATE TABLE trades (account TEXT, market TEXT, ticker TEXT, timestamp TEXT);
        INSERT INTO account_meta VALUES
          ('A01','US','active'), ('A99','US','retired'), ('CA01','CN','active');
        INSERT INTO positions VALUES
          ('A01','US','HELD',2), ('A99','US','RETIRED_HELD',3),
          ('CA01','CN','000001.SZ',100);
        """
    )
    recent = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    con.executemany(
        "INSERT INTO trades VALUES (?,?,?,?)",
        [
            ('A01','US','RECENTLY_SOLD',recent),
            ('A01','US','OLD_SOLD',old),
            ('A99','US','RETIRED_SOLD',recent),
        ],
    )
    con.commit()
    con.close()
    monkeypatch.setattr(store, "DB_PATH", str(db))

    assert _ledger_tickers_for_market("US", 5) == ["HELD", "RECENTLY_SOLD"]


def test_us_universe_uses_current_people_inc_symbol():
    from config.settings import STOCK_UNIVERSE

    assert "PPLI" in STOCK_UNIVERSE
    assert "IAC" not in STOCK_UNIVERSE


def test_iac_symbol_change_is_effective_only_from_sec_change_date():
    from config.security_master import canonical_ticker, ticker_lifecycle_block_reason

    assert canonical_ticker("IAC", "US", "2026-06-02T23:59:59+00:00") == "IAC"
    assert canonical_ticker("IAC", "US", "2026-06-03T00:00:00+00:00") == "PPLI"
    assert ticker_lifecycle_block_reason("IAC", "US", "2026-07-10T14:00:00+00:00") == "symbol_changed_to_PPLI"
    assert ticker_lifecycle_block_reason("PPLI", "US", "2026-07-10T14:00:00+00:00") is None


def test_avb_symbol_change_starts_when_vmrk_history_takes_over():
    from config.security_master import canonical_ticker, ticker_lifecycle_block_reason
    from config.settings import STOCK_UNIVERSE

    assert canonical_ticker("AVB", "US", "2026-08-24T23:59:59+00:00") == "AVB"
    assert canonical_ticker("AVB", "US", "2026-08-25T00:00:00+00:00") == "VMRK"
    assert ticker_lifecycle_block_reason("AVB", "US", "2026-08-25T00:00:00+00:00") == "symbol_changed_to_VMRK"
    assert "VMRK" in STOCK_UNIVERSE and "AVB" not in STOCK_UNIVERSE


def test_replay_inventory_canonicalizes_avb_to_vmrk_at_cutoff():
    from scripts.ledger_watchdog import _canonicalize_position_symbols

    before = {"AVB": {"shares": 27.93, "total_cost": 1808.673375}}
    assert set(_canonicalize_position_symbols(before, "US", "2026-08-24T23:59:59+00:00")) == {"AVB"}
    after = _canonicalize_position_symbols(before, "US", "2026-08-25T00:00:00+00:00")
    assert after == {"VMRK": {"shares": 27.93, "total_cost": 1808.673375}}


def test_main_execution_gate_rejects_old_symbol_even_with_a_price(monkeypatch):
    import main

    events = []
    monkeypatch.setattr(main, "emit_event", lambda *a, **k: events.append((a, k)))

    class Engine:
        def execute_signal(self, *args, **kwargs):
            raise AssertionError("inactive symbol reached execution engine")

    system = object.__new__(main.QuantSystem)
    system.market = "US"
    system.engine = Engine()  # type: ignore[assignment]

    result = main.QuantSystem._execute_live_signal(
        system, "F15", "IAC", "sell", 54, 42.24, {"IAC": 42.24}
    )
    assert result is None
    assert events and events[0][1]["detail"]["reason"] == "symbol_changed_to_PPLI"


def test_stop_loss_gate_rejects_old_symbol_even_with_fresh_quote(tmp_path, monkeypatch):
    import scripts.update_prices as updater
    from data.store import DataStore

    db = tmp_path / "trading.db"
    DataStore(str(db))
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO account_meta(account_id,market,status) VALUES ('F15','US','active')")
        con.execute(
            """INSERT INTO account_state(account,cash,initial_cash,updated_at,market)
               VALUES ('F15',5000,10000,'2026-07-10T14:00:00+00:00','US')"""
        )
        con.execute(
            """INSERT INTO positions(account,ticker,shares,avg_cost,total_cost,current_price,market)
               VALUES ('F15','IAC',54,42.28,2283.12,42.24,'US')"""
        )
    monkeypatch.setattr(updater, "_is_market_hours_now", lambda: True)
    monkeypatch.setattr(updater, "_is_market_open_for", lambda market: True)
    monkeypatch.setattr(updater, "STOP_LOSS_BY_ACCT", {"F15": 0.01})
    monkeypatch.setattr(
        updater, "_now_utc",
        lambda: datetime(2026, 7, 10, 14, 0, tzinfo=timezone.utc),
    )
    with sqlite3.connect(db) as con:
        con.row_factory = sqlite3.Row
        assert updater.check_stop_losses(con, {"IAC": 1.0}, db_path=str(db)) == []
    with sqlite3.connect(db) as con:
        assert con.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 0
        detail = con.execute("SELECT detail FROM events WHERE ticker='IAC' AND category='guard'").fetchone()[0]
        assert "symbol_changed_to_PPLI" in detail


def test_history_price_lookup_rejects_stale_intraday_bar():
    from scripts.ledger_watchdog import _price_at_meta

    bar_ts = datetime.fromisoformat("2026-07-10T14:00:00+00:00").timestamp()
    history = {"AAPL": [(bar_ts, 100.0, "5m")]}
    assert _price_at_meta(
        history, "AAPL", "2026-07-10T14:10:00+00:00",
        max_age_by_interval={"5m": 600.0},
    ) == (100.0, "5m")
    assert _price_at_meta(
        history, "AAPL", "2026-07-10T14:10:01+00:00",
        max_age_by_interval={"5m": 600.0},
    ) is None


def test_symbol_migration_is_dry_run_by_default_and_audited_on_apply(tmp_path):
    from scripts.migrate_security_symbol import migrate

    db = tmp_path / "trading.db"
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE positions (
          account TEXT, ticker TEXT, shares REAL, avg_cost REAL, total_cost REAL,
          current_price REAL, updated_at TEXT, market TEXT,
          PRIMARY KEY(account,ticker,market)
        );
        CREATE TABLE events (
          ts TEXT, category TEXT, severity TEXT, account TEXT, ticker TEXT,
          title TEXT, detail TEXT, market TEXT
        );
        INSERT INTO positions VALUES
          ('F15','IAC',54,42.28,2283.25,42.24,'2026-07-09T23:55:00+00:00','US');
        """
    )
    con.commit()
    con.close()

    result = migrate(str(db), "US", "IAC")
    assert result["dry_run"] is True
    with sqlite3.connect(db) as check:
        assert check.execute("SELECT ticker FROM positions").fetchone()[0] == "IAC"

    result = migrate(str(db), "US", "IAC", apply=True)
    assert result["migrated"] == 1
    with sqlite3.connect(db) as check:
        assert check.execute("SELECT ticker FROM positions").fetchone()[0] == "PPLI"
        assert check.execute("SELECT old_ticker,new_ticker FROM security_symbol_migrations").fetchone() == ("IAC", "PPLI")
        assert check.execute("SELECT category,ticker FROM events").fetchone() == ("corporate_action", "PPLI")


def test_backfill_module_uses_its_checkout_as_project_root():
    from pathlib import Path
    import scripts.backfill_prices as backfill

    assert Path(backfill.PROJECT_ROOT) == Path(backfill.__file__).resolve().parents[1]
