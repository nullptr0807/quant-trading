import sqlite3

import pytest


def test_universe_snapshot_is_idempotent_and_refuses_history_rewrite(tmp_path, monkeypatch):
    import scripts.snapshot_universe as snap
    db=tmp_path/'u.db'
    from data.store import DataStore
    DataStore(str(db))
    with sqlite3.connect(db) as con:
        con.executemany(
            "INSERT INTO prices(ticker,datetime,interval,open,high,low,close,volume) "
            "VALUES (?,?,'1d',1,1,1,1,1)",
            [('A','2026-08-04'),('B','2026-08-04')],
        )
    monkeypatch.setitem(snap.UNIVERSES,'US',['A','B'])
    first=snap.snapshot(str(db),'US')
    second=snap.snapshot(str(db),'US')
    assert first['count']==second['count']==2
    with sqlite3.connect(db) as con:
        assert con.execute("SELECT COUNT(*) FROM universe_membership").fetchone()[0]==2

    monkeypatch.setitem(snap.UNIVERSES,'US',['A'])
    with pytest.raises(RuntimeError,match='refusing to rewrite'):
        snap.snapshot(str(db),'US')
