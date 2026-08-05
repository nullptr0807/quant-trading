from types import SimpleNamespace
import sqlite3

from main import QuantSystem


class Store:
    def __init__(self):
        self.rows = []

    def set_account_runtime_status(self, *args):
        self.rows.append(args)


def test_gp_runtime_status_distinguishes_ready_and_non_tradeable():
    system = object.__new__(QuantSystem)
    system.market = "US"
    system.store = Store()
    system.gp_strategies = [SimpleNamespace(id="F14"), SimpleNamespace(id="F15"), SimpleNamespace(id="F16")]
    system._per_account_mined = {
        "F14": [{"status": "mining_failed", "active": False, "reason": "NO_ADMISSIBLE_FACTOR"}],
        "F15": [{"expression": "add(X0,X1)", "active": True}],
        "F16": [],
    }

    system._publish_gp_runtime_statuses()

    rows = {row[0]: row for row in system.store.rows}
    assert rows["F14"][2:4] == ("non_tradeable", "NO_ADMISSIBLE_FACTOR")
    assert rows["F15"][2] == "ready"
    assert rows["F15"][4]["active_factors"] == 1
    assert rows["F16"][2:4] == ("non_tradeable", "empty_config")


def test_qlib_runtime_status_tracks_checkpoint_publication(tmp_path, monkeypatch):
    db=tmp_path/'q.db';con=sqlite3.connect(db)
    con.executescript(
        "CREATE TABLE prices(ticker TEXT,datetime TEXT,interval TEXT);"
        "CREATE TABLE factor_values(ticker TEXT,date TEXT,factor_name TEXT,value REAL,factor_group TEXT);"
    )
    con.execute("INSERT INTO prices VALUES('A','2026-08-04','1d')")
    con.execute("INSERT INTO factor_values VALUES('A','2026-08-04','qlib_Q01_score',1,'qlib')")
    con.commit();con.close()
    system=object.__new__(QuantSystem);system.market='US';system.db_path=str(db);system.universe=['A']
    system.store=Store();system.qlib_strategies=[SimpleNamespace(id='Q01')]
    monkeypatch.setattr(
        'factors.qlib_checkpoint.checkpoint_ready_for_publication',
        lambda *a,**k:(False,'checkpoint_pit_incomplete'),
    )
    system._publish_qlib_runtime_statuses()
    assert system.store.rows[0][2]=='non_tradeable'
    assert system.store.rows[0][3]=='checkpoint_pit_incomplete'
