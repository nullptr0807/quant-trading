import sqlite3
import pytest


def test_price_quality_backfill_is_dry_by_default_and_non_destructive(tmp_path):
    from data.store import DataStore
    from scripts.backfill_price_quality import scan,CONFIRM
    db=tmp_path/'p.db';DataStore(str(db))
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO prices VALUES ('BAD','2026-01-01','1d',10,9,8,11,1)")
    assert scan(str(db))=={'adjusted':1,'raw':0}
    with sqlite3.connect(db) as con:
        assert con.execute('SELECT COUNT(*) FROM price_quality_issues').fetchone()[0]==0
    with pytest.raises(ValueError,match='confirmation'):
        scan(str(db),True,'wrong')
    assert scan(str(db),True,CONFIRM)=={'adjusted':1,'raw':0}
    with sqlite3.connect(db) as con:
        assert con.execute('SELECT COUNT(*) FROM price_quality_issues').fetchone()[0]==1
        assert con.execute("SELECT COUNT(*) FROM prices WHERE ticker='BAD'").fetchone()[0]==1

    with sqlite3.connect(db) as con:
        con.execute("UPDATE prices SET high=12 WHERE ticker='BAD'")
    assert scan(str(db),True,CONFIRM)=={'adjusted':0,'raw':0}
    with sqlite3.connect(db) as con:
        assert con.execute('SELECT COUNT(*) FROM price_quality_issues').fetchone()[0]==0
