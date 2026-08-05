import sqlite3

import pytest


def _db(path):
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE prices(ticker TEXT,datetime TEXT,interval TEXT);
        CREATE TABLE factor_values(ticker TEXT,date TEXT,factor_name TEXT,value REAL,factor_group TEXT);
        """
    )
    con.executemany("INSERT INTO prices VALUES (?,?,'1d')", [('A','2026-08-04'),('B','2026-08-04')])
    return con


def test_verify_qlib_rejects_sparse_latest_publication(tmp_path, monkeypatch):
    import scripts.verify_qlib_scores as verify
    monkeypatch.setitem(verify.UNIVERSES, 'US', ['A','B'])
    db=tmp_path/'q.db'; con=_db(db)
    con.execute("INSERT INTO factor_values VALUES ('A','2026-08-04','qlib_Q01_score',1,'qlib')")
    con.commit();con.close()
    with pytest.raises(RuntimeError, match='publication incomplete'):
        verify.verify(str(db),'US',min_coverage=1.0,model_ids=['Q01'])


def test_verify_qlib_accepts_complete_target_date(tmp_path, monkeypatch):
    import scripts.verify_qlib_scores as verify
    monkeypatch.setitem(verify.UNIVERSES, 'US', ['A','B'])
    db=tmp_path/'q.db'; con=_db(db)
    con.executemany(
        "INSERT INTO factor_values VALUES (?,'2026-08-04','qlib_Q01_score',1,'qlib')",
        [('A',),('B',)],
    )
    con.commit();con.close()
    result=verify.verify(str(db),'US',min_coverage=1.0,model_ids=['Q01'])
    assert result['date']=='2026-08-04'
    assert result['coverage']=={'Q01':2}
