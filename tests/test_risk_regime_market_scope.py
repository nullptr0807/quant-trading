from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from data.store import DataStore
from trading import risk_regime


def _seed(path):
    DataStore(str(path))
    now = datetime.now(timezone.utc)
    with sqlite3.connect(path) as con:
        con.executemany(
            "INSERT INTO account_meta(account_id,market,status) VALUES(?,?,?)",
            [("U", "US", "active"), ("C", "CN", "active"),
             ("R", "US", "research"), ("OLD", "US", "retired")],
        )
        rows = []
        for days, values in [(2, {"U": 100, "C": 100, "R": 1000, "OLD": 1000}),
                             (1, {"U": 96, "C": 80, "R": 100, "OLD": 100})]:
            ts = (now - timedelta(days=days)).isoformat()
            for name, equity in values.items():
                market = "CN" if name == "C" else "US"
                rows.append((name, 0, equity, ts, market))
        con.executemany(
            "INSERT INTO accounts(name,cash,equity,timestamp,market) VALUES(?,?,?,?,?)", rows
        )


def test_cn_drawdown_cannot_arm_us_and_events_are_market_scoped(tmp_path, monkeypatch):
    db = tmp_path / "db.sqlite"
    _seed(db)
    monkeypatch.setattr(risk_regime, "_send_telegram", lambda message: None)

    us = risk_regime.evaluate_and_update(market="US", db_path=str(db))
    cn = risk_regime.evaluate_and_update(market="CN", db_path=str(db))

    assert us["state"] == "DISARMED"
    assert us["drawdown"] == pytest.approx(0.04)
    assert cn["state"] == "ARMED"
    with sqlite3.connect(db) as con:
        assert con.execute(
            "SELECT market FROM events WHERE category='risk'"
        ).fetchall() == [("CN",)]


def test_same_account_name_cross_market_does_not_mix(tmp_path):
    db = tmp_path / "db.sqlite"
    DataStore(str(db))
    now = datetime.now(timezone.utc)
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO account_meta(account_id,market,status) VALUES('SAME','US','active')")
        con.executemany(
            "INSERT INTO accounts(name,cash,equity,timestamp,market) VALUES('SAME',0,?,?,?)",
            [
                (100, (now - timedelta(days=2)).isoformat(), "US"),
                (96, (now - timedelta(days=1)).isoformat(), "US"),
                (1000, (now - timedelta(days=2)).isoformat(), "CN"),
                (1, (now - timedelta(days=1)).isoformat(), "CN"),
            ],
        )
        assert risk_regime._portfolio_drawdown(con, market="US") == pytest.approx(0.04)


def test_legacy_singleton_migration_recomputes_instead_of_copying_armed(tmp_path):
    db = tmp_path / "db.sqlite"
    _seed(db)
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE risk_regime(id INTEGER PRIMARY KEY, state TEXT, armed_at TEXT, recovery_streak INTEGER, last_drawdown REAL, last_check_at TEXT)")
        con.execute("INSERT INTO risk_regime VALUES(1,'ARMED',NULL,0,.0789,NULL)")

    state = risk_regime.get_state(market="US", db_path=str(db))

    assert state["state"] == "DISARMED"
    with sqlite3.connect(db) as con:
        assert con.execute("SELECT state FROM risk_regime WHERE market='CN'").fetchone()[0] == "ARMED"
        detail, market = con.execute(
            "SELECT detail,market FROM events WHERE title LIKE 'Risk regime schema migration%' AND market='ALL'"
        ).fetchone()
        assert 'legacy state was not copied' in detail
        assert market == "ALL"


def test_all_public_apis_require_market():
    with pytest.raises(TypeError):
        risk_regime.get_state()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        risk_regime.evaluate_and_update()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        risk_regime.get_effective_trailing_stop()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        risk_regime.status_line()  # type: ignore[call-arg]
