from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone


def _seed_us_stop_account(db):
    from data.store import DataStore

    DataStore(str(db))
    with sqlite3.connect(db) as con:
        con.execute(
            "INSERT INTO account_meta (account_id,market,status) VALUES ('A01','US','active')"
        )
        con.execute(
            "INSERT INTO account_state (account,cash,initial_cash,updated_at,market) "
            "VALUES ('A01',9000,10000,'2026-07-10T10:00:00+00:00','US')"
        )
        con.execute(
            "INSERT INTO positions "
            "(account,ticker,shares,avg_cost,total_cost,current_price,updated_at,market) "
            "VALUES ('A01','AAPL',10,100,1000,100,'2026-07-10T10:00:00+00:00','US')"
        )


def test_us_premarket_live_mode_collects_but_never_trades(tmp_path, monkeypatch):
    import scripts.update_prices as updater
    from data.quotes import RealtimeQuote

    db = tmp_path / "trading.db"
    _seed_us_stop_account(db)
    now = datetime(2026, 7, 10, 10, 0, tzinfo=timezone.utc)  # 06:00 ET
    monkeypatch.setattr(updater, "_utc_now", lambda: now)
    monkeypatch.setattr(updater, "_now_utc", lambda: now)
    monkeypatch.setattr(updater, "_is_us_market_hours_now", lambda: True)
    monkeypatch.setattr(updater, "_is_us_regular_session_now", lambda: False)
    monkeypatch.setattr(updater, "_is_cn_market_hours_now", lambda: False)
    monkeypatch.setattr(updater, "_force_price_update", lambda: False)
    monkeypatch.setattr(updater, "STOP_LOSS_BY_ACCT", {"A01": 0.05})
    monkeypatch.setattr(
        updater,
        "fetch_quote_metadata",
        lambda tickers: {
            "AAPL": RealtimeQuote(
                ticker="AAPL", price=80, source="test",
                source_timestamp=now - timedelta(seconds=5), received_at=now,
                tradable=True,
            )
        },
    )

    stats = updater.update_equity_snapshots(str(db))

    assert stats["stop_losses"] == 0
    with sqlite3.connect(db) as con:
        assert con.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 0
        assert con.execute(
            "SELECT current_price FROM positions WHERE account='A01' AND ticker='AAPL'"
        ).fetchone()[0] == 80


def test_us_rth_live_mode_can_execute_fresh_stop(tmp_path, monkeypatch):
    import scripts.update_prices as updater
    from data.quotes import RealtimeQuote

    db = tmp_path / "trading.db"
    _seed_us_stop_account(db)
    now = datetime(2026, 7, 10, 14, 0, tzinfo=timezone.utc)  # 10:00 ET
    monkeypatch.setattr(updater, "_utc_now", lambda: now)
    monkeypatch.setattr(updater, "_now_utc", lambda: now)
    monkeypatch.setattr(updater, "_is_us_market_hours_now", lambda: True)
    monkeypatch.setattr(updater, "_is_us_regular_session_now", lambda: True)
    monkeypatch.setattr(updater, "_is_cn_market_hours_now", lambda: False)
    monkeypatch.setattr(updater, "_force_price_update", lambda: False)
    monkeypatch.setattr(updater, "STOP_LOSS_BY_ACCT", {"A01": 0.05})
    monkeypatch.setattr(
        updater,
        "fetch_quote_metadata",
        lambda tickers: {
            "AAPL": RealtimeQuote(
                ticker="AAPL", price=80, source="test",
                source_timestamp=now - timedelta(seconds=5), received_at=now,
                tradable=True,
            )
        },
    )

    stats = updater.update_equity_snapshots(str(db))

    assert stats["stop_losses"] == 1
    with sqlite3.connect(db) as con:
        assert con.execute(
            "SELECT COUNT(*) FROM trades WHERE account='A01' AND ticker='AAPL' AND side='sell'"
        ).fetchone()[0] == 1
