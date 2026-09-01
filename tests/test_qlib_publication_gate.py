import os
import sqlite3
from types import SimpleNamespace

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
        verify.verify(str(db),'US',min_coverage=1.0,model_ids=['Q01'],verify_checkpoints=False)


def test_verify_qlib_accepts_complete_target_date(tmp_path, monkeypatch):
    import scripts.verify_qlib_scores as verify
    monkeypatch.setitem(verify.UNIVERSES, 'US', ['A','B'])
    db=tmp_path/'q.db'; con=_db(db)
    con.executemany(
        "INSERT INTO factor_values VALUES (?,'2026-08-04','qlib_Q01_score',1,'qlib')",
        [('A',),('B',)],
    )
    con.commit();con.close()
    result=verify.verify(str(db),'US',min_coverage=1.0,model_ids=['Q01'],verify_checkpoints=False)
    assert result['date']=='2026-08-04'
    assert result['coverage']=={'Q01':2}


def test_verify_empty_frozen_manifest_is_successful_noop_without_prices(tmp_path):
    import scripts.verify_qlib_scores as verify
    db = tmp_path / 'empty.db'
    assert verify.verify(str(db), 'US', model_ids=[], verify_checkpoints=False) == {
        'market': 'US', 'models': [], 'skipped': 'no_active_qlib_accounts'
    }


def test_verify_qlib_rejects_checkpoint_without_pit_proof(tmp_path, monkeypatch):
    import scripts.verify_qlib_scores as verify
    import factors.qlib_checkpoint as checkpoint
    monkeypatch.setitem(verify.UNIVERSES, 'US', ['A','B'])
    db=tmp_path/'q.db'; con=_db(db)
    con.executemany(
        "INSERT INTO factor_values VALUES (?,'2026-08-04','qlib_Q01_score',1,'qlib')",
        [('A',),('B',)],
    )
    con.commit();con.close()
    root=tmp_path/'checkpoints'; monkeypatch.setattr(checkpoint,'CHECKPOINT_ROOT',root)
    folder=root/'US'/'Q01';folder.mkdir(parents=True)
    (folder/'2026-08-04.pkl').write_bytes(b'payload')
    (folder/'2026-08-04.json').write_text('{"extra":{"point_in_time_complete":false}}')
    with pytest.raises(RuntimeError,match='checkpoint/PIT verification failed'):
        verify.verify(str(db),'US',min_coverage=1.0,model_ids=['Q01'])


def test_publication_marker_binds_live_gate_to_verified_file_state(tmp_path, monkeypatch):
    import factors.qlib_checkpoint as checkpoint
    import scripts.verify_qlib_scores as verify
    root=tmp_path/'checkpoints'
    monkeypatch.setattr(checkpoint,'CHECKPOINT_ROOT',root)
    folder=root/'US'/'Q01';folder.mkdir(parents=True)
    pkl=folder/'2026-08-04.pkl';pkl.write_bytes(b'payload')
    sidecar=folder/'2026-08-04.json';sidecar.write_text(
        '{"self_test":{"expected_score":1},"extra":{"point_in_time_complete":true,"universe_count":2}}'
    )
    ready,reason=checkpoint.checkpoint_ready_for_publication('Q01','2026-08-04','US',2)
    assert ready is False and reason=='checkpoint_publication_marker_missing'
    monkeypatch.setattr(
        verify,'_checkpoint_probe',
        lambda *args:SimpleNamespace(returncode=0,stdout='',stderr=''),
    )
    verify.write_publication_marker({
        'market':'US','date':'2026-08-04','checkpoints':{'Q01':'verified'}
    })
    assert checkpoint.checkpoint_ready_for_publication('Q01','2026-08-04','US',2)==(True,'ok')
    before=pkl.stat();pkl.write_bytes(b'PAYLOAD')
    os.utime(pkl,ns=(before.st_atime_ns,before.st_mtime_ns))
    ready,reason=checkpoint.checkpoint_ready_for_publication('Q01','2026-08-04','US',2)
    assert ready is False and reason=='checkpoint_hash_mismatch'


def test_publication_marker_refuses_unloadable_payload(tmp_path, monkeypatch):
    import factors.qlib_checkpoint as checkpoint
    import scripts.verify_qlib_scores as verify
    root=tmp_path/'checkpoints';monkeypatch.setattr(checkpoint,'CHECKPOINT_ROOT',root)
    folder=root/'US'/'Q01';folder.mkdir(parents=True)
    (folder/'2026-08-04.pkl').write_bytes(b'not-a-joblib-checkpoint')
    (folder/'2026-08-04.json').write_text(
        '{"self_test":{"expected_score":1},"extra":{"point_in_time_complete":true,"universe_count":2}}'
    )
    with pytest.raises(RuntimeError,match='refused unverified'):
        verify.write_publication_marker({
            'market':'US','date':'2026-08-04','checkpoints':{'Q01':'verified'}
        })
