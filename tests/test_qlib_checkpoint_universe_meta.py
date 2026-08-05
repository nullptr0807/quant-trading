import sqlite3


def test_checkpoint_universe_meta_requires_exact_pit_membership(tmp_path, monkeypatch):
    import factors.qlib_signal as signal
    from config import settings
    from data.store import DataStore
    db=tmp_path/'q.db';DataStore(str(db))
    monkeypatch.setattr(signal,'DB_PATH',str(db))
    monkeypatch.setitem(settings.UNIVERSES,'US',['A','B'])
    with sqlite3.connect(db) as con:
        con.executemany(
            "INSERT INTO universe_membership VALUES ('US','2026-08-04',?,'test','hash','2026-08-05')",
            [('A',),('B',)],
        )
    meta=signal._checkpoint_universe_meta('US','2026-08-04')
    assert meta['point_in_time_complete'] is True
    with sqlite3.connect(db) as con:
        con.execute("DELETE FROM universe_membership WHERE ticker='B'")
    assert signal._checkpoint_universe_meta('US','2026-08-04')['point_in_time_complete'] is False
