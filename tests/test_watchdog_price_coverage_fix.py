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
    from config.security_master import canonical_ticker

    assert canonical_ticker("IAC", "US", "2026-06-02T23:59:59+00:00") == "IAC"
    assert canonical_ticker("IAC", "US", "2026-06-03T00:00:00+00:00") == "PPLI"


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
