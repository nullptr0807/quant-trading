import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from data.store import DataStore


def _audit_module():
    import scripts.audit_trailing_stop_contamination as audit
    return audit


def _seed(path):
    DataStore(str(path))
    now = datetime.now(timezone.utc)
    trade_ts = (now - timedelta(hours=1)).isoformat()
    with sqlite3.connect(path) as con:
        con.execute("INSERT INTO account_meta(account_id,market,status) VALUES('A01','US','active')")
        con.executemany(
            "INSERT INTO accounts(name,cash,equity,timestamp,market) VALUES('A01',0,?,?, 'US')",
            [(100, (now - timedelta(days=2)).isoformat()),
             (96, (now - timedelta(days=1)).isoformat())],
        )
        con.execute(
            "INSERT INTO trades(account,ticker,side,shares,price,cost,slippage,timestamp,market) "
            "VALUES('A01','AAA','sell',1,96,0,0,?,'US')", (trade_ts,),
        )
        con.execute(
            "INSERT INTO events(ts,category,severity,account,ticker,title,detail,market) "
            "VALUES(?,'trade','info','A01','AAA','sell',?,'US')",
            (trade_ts, json.dumps({"reason": "trailing_stop"})),
        )
    return trade_ts


def test_default_classification_is_read_only_and_marks_us_disarmed_fill(tmp_path):
    db = tmp_path / "db.sqlite"
    _seed(db)
    before = db.read_bytes()

    rows = _audit_module().classify(str(db))

    assert len(rows) == 1
    assert rows[0]["counterfactual_state"] == "DISARMED"
    assert rows[0]["polluted"] is True
    assert rows[0]["us_only_drawdown"] == pytest.approx(0.04)
    assert db.read_bytes() == before


def test_apply_requires_confirmation_backup_is_non_destructive_and_idempotent(tmp_path):
    db = tmp_path / "db.sqlite"
    _seed(db)
    rows = _audit_module().classify(str(db))
    backup = tmp_path / "backup.sqlite"

    audit = _audit_module()
    with pytest.raises(ValueError, match="confirmation"):
        audit.apply_quarantine(str(db), rows, backup_path=str(backup), confirmation="no")
    assert not backup.exists()

    assert audit.apply_quarantine(
        str(db), rows, backup_path=str(backup), confirmation=audit.CONFIRMATION
    ) == 1
    assert backup.exists()
    with sqlite3.connect(db) as con:
        trade_before = con.execute("SELECT * FROM trades").fetchall()
        assert con.execute("SELECT trade_id FROM trailing_stop_quarantine").fetchall() == [(1,)]
        assert con.execute(
            "SELECT market FROM events WHERE category='audit'"
        ).fetchall() == [("US",)]

    second_backup = tmp_path / "backup2.sqlite"
    assert audit.apply_quarantine(
        str(db), rows, backup_path=str(second_backup), confirmation=audit.CONFIRMATION
    ) == 0
    with sqlite3.connect(db) as con:
        assert con.execute("SELECT * FROM trades").fetchall() == trade_before
        assert con.execute("SELECT COUNT(*) FROM events WHERE category='audit'").fetchone()[0] == 1
