import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace


def test_purge_retired_assets_preserves_active_and_cross_market_q(tmp_path, monkeypatch):
    import scripts.purge_retired_model_assets as purge

    root = tmp_path
    (root / 'factors/factor_miner_gp/mining_logs').mkdir(parents=True)
    (root / 'data/qlib_checkpoints/US/Q02').mkdir(parents=True)
    (root / 'data/qlib_checkpoints/US/Q02/x.pkl').write_bytes(b'model')
    (root / 'data/qlib_checkpoints/US/publication-2026-01-01.json').write_text(
        json.dumps({'market': 'US', 'models': {'Q01': {'x': 1}, 'Q02': {'x': 2}}})
    )
    gp = root / 'factors/mined_alphas_per_account.json'
    gp.write_text(json.dumps({'B01': [{'name': 'keep'}], 'B02': [{'name': 'drop'}], 'CB04': [{'name': 'drop'}]}))
    fm = root / 'factors/factor_miner_gp/mined_alphas_f.json'
    fm.write_text(json.dumps({'F11': [{'name': 'drop'}], 'F12': [{'name': 'keep'}]}))
    memory = root / 'factors/factor_miner_gp/mining_memory.json'
    memory.write_text(json.dumps({'recommended': {'x': 1}, 'recent_reports': [{'account_id': 'F11'}, {'account_id': 'F12'}]}))
    (root / 'factors/factor_miner_gp/mining_logs/20260101_F11.json').write_text('{}')
    dedicated = root / 'data/model-backup.db'
    dedicated.write_bytes(b'old model backup')

    db = root / 'data/trading.db'
    con = sqlite3.connect(db)
    con.executescript('''
      CREATE TABLE account_meta(account_id TEXT,market TEXT,"group" TEXT,status TEXT,factors TEXT);
      CREATE TABLE factor_values(ticker TEXT,date TEXT,factor_name TEXT,value REAL,factor_group TEXT);
      CREATE TABLE factor_values_backup(ticker TEXT,date TEXT,factor_name TEXT,value REAL,factor_group TEXT);
      CREATE TABLE events(ts TEXT,category TEXT,severity TEXT,account TEXT,ticker TEXT,title TEXT,detail TEXT,market TEXT);
    ''')
    con.executemany('INSERT INTO account_meta VALUES (?,?,?,?,?)', [
        ('B01','US','B','active','keep'), ('B02','US','B','retired','drop'),
        ('CB04','CN','B','retired','drop'), ('F11','US','F','retired','drop'),
        ('F12','US','F','active','keep'), ('Q02','US','Q','retired','drop'),
        ('A09','US','A','retired','drop'),
        ('CQ02','CN','Q','active','keep'),
    ])
    rows = [
        ('AAPL','2026-01-01','x',1,'gp_B01'),
        ('AAPL','2026-01-01','x',1,'gp_B02'),
        ('000001.SZ','2026-01-01','x',1,'gp_CB04'),
        ('AAPL','2026-01-01','x',1,'fmgp_F11'),
        ('AAPL','2026-01-01','qlib_Q02_score',1,'qlib'),
        ('000001.SZ','2026-01-01','qlib_Q02_score',1,'qlib'),
    ]
    con.executemany('INSERT INTO factor_values VALUES (?,?,?,?,?)', rows)
    con.executemany('INSERT INTO factor_values_backup VALUES (?,?,?,?,?)', rows)
    con.commit(); con.close()

    monkeypatch.setattr(purge, 'ROOT', root)
    monkeypatch.setattr(purge, 'GP_FILES', [gp])
    monkeypatch.setattr(purge, 'FM_FILES', [fm])
    monkeypatch.setattr(purge, 'DEDICATED_DB_BACKUPS', [dedicated])
    result = purge.purge(db, True)
    assert result['apply'] is True

    assert set(json.loads(gp.read_text())) == {'B01'}
    assert set(json.loads(fm.read_text())) == {'F12'}
    assert json.loads(memory.read_text())['recent_reports'] == [{'account_id': 'F12'}]
    assert not (root / 'factors/factor_miner_gp/mining_logs/20260101_F11.json').exists()
    assert not (root / 'data/qlib_checkpoints/US/Q02').exists()
    assert set(json.loads((root / 'data/qlib_checkpoints/US/publication-2026-01-01.json').read_text())['models']) == {'Q01'}
    assert not dedicated.exists()

    con = sqlite3.connect(db)
    for table in ('factor_values', 'factor_values_backup'):
        remaining = set(con.execute(f'SELECT ticker,factor_group FROM {table}'))
        assert remaining == {('AAPL', 'gp_B01'), ('000001.SZ', 'qlib')}
    assert con.execute("SELECT factors FROM account_meta WHERE account_id='B02'").fetchone()[0] == ''
    assert con.execute("SELECT factors FROM account_meta WHERE account_id='A09'").fetchone()[0] == ''
    assert con.execute("SELECT factors FROM account_meta WHERE account_id='CQ02'").fetchone()[0] == 'keep'
    con.close()


def test_missing_gp_strategies_skips_retired_and_fails_closed(tmp_path):
    from main import QuantSystem

    db = tmp_path / 'trading.db'
    con = sqlite3.connect(db)
    con.execute('CREATE TABLE account_meta(account_id TEXT,market TEXT,status TEXT)')
    con.executemany('INSERT INTO account_meta VALUES (?,?,?)', [('B01','US','active'), ('B02','US','retired')])
    con.commit(); con.close()

    class Store:
        def _conn(self):
            return sqlite3.connect(db)

    system = object.__new__(QuantSystem)
    system.store = Store()
    system.market = 'US'
    system.gp_strategies = [SimpleNamespace(id='B01', mining_backend='gplearn'), SimpleNamespace(id='B02', mining_backend='gplearn')]
    system._per_account_mined = {}
    system._gp_signal_source_id = lambda g: g.id
    system._active_mined_count = lambda factors: len(factors or [])
    system._factor_miner_retry_due = lambda factors: False
    assert [g.id for g in system._missing_gp_strategies()] == ['B01']

    con = sqlite3.connect(db)
    con.execute("UPDATE account_meta SET status=NULL WHERE account_id='B01'")
    con.commit(); con.close()
    try:
        system._missing_gp_strategies()
        assert False, 'invalid status must fail closed'
    except RuntimeError as exc:
        assert 'status invalid' in str(exc)


def test_account_meta_bootstrap_does_not_restore_retired_factor_metadata(tmp_path):
    from data.store import DataStore
    from main import QuantSystem

    db = tmp_path / 'trading.db'
    DataStore(str(db))
    con = sqlite3.connect(db)
    con.execute(
        "INSERT INTO account_meta(account_id,market,status,factors) VALUES ('A09','US','retired','')"
    )
    con.commit(); con.close()

    system = object.__new__(QuantSystem)
    system.db_path = str(db)
    system.market = 'US'
    system.initial_cash = 10_000
    system.strategies = [SimpleNamespace(
        id='A09', name='retired', strategy_type='x', factor_names=['RSI_14'],
    )]
    system.gp_strategies = []
    system.qlib_strategies = []
    system.benchmarks = []
    system._ensure_account_meta_rows()
    con = sqlite3.connect(db)
    assert con.execute("SELECT factors FROM account_meta WHERE account_id='A09'").fetchone()[0] == ''
    con.close()
