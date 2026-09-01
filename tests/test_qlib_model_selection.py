import sqlite3

import pytest

from scripts.qlib_model_selection import select_qlib_model_ids


def _db(path):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE account_meta(account_id TEXT,market TEXT,status TEXT)")
    con.executemany(
        "INSERT INTO account_meta VALUES (?,?,?)",
        [
            ("Q01", "US", "active"),
            ("Q02", "US", "retired"),
            ("CQ01", "CN", "retired"),
            ("CQ02", "CN", "active"),
        ],
    )
    con.commit()
    con.close()


def test_default_selection_is_market_scoped_and_excludes_retired(tmp_path):
    db = tmp_path / "trading.db"
    _db(db)
    assert select_qlib_model_ids(db, "US", ["Q01", "Q02"]) == ["Q01"]
    assert select_qlib_model_ids(db, "CN", ["Q01", "Q02"]) == ["Q02"]


def test_explicit_models_override_retirement_for_manual_recovery(tmp_path):
    db = tmp_path / "trading.db"
    _db(db)
    assert select_qlib_model_ids(db, "US", ["Q01", "Q02"], ["Q02"]) == ["Q02"]


def test_default_selection_fails_closed_on_missing_lifecycle_metadata(tmp_path):
    db = tmp_path / "trading.db"
    _db(db)
    with pytest.raises(RuntimeError, match="missing accounts"):
        select_qlib_model_ids(db, "US", ["Q01", "Q02", "Q03"])


def test_explicit_selection_rejects_unknown_model(tmp_path):
    db = tmp_path / "trading.db"
    _db(db)
    with pytest.raises(ValueError, match="unknown Qlib model"):
        select_qlib_model_ids(db, "US", ["Q01", "Q02"], ["Q99"])


@pytest.mark.parametrize("bad_status", [None, "", "paused", "ACTIVE"])
def test_default_selection_fails_closed_on_invalid_status(tmp_path, bad_status):
    db = tmp_path / "trading.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE account_meta(account_id TEXT,market TEXT,status TEXT)")
    con.execute("INSERT INTO account_meta VALUES ('Q01','US',?)", (bad_status,))
    con.commit(); con.close()
    with pytest.raises(RuntimeError, match="status invalid"):
        select_qlib_model_ids(db, "US", ["Q01"])


def test_live_shape_full_market_selection(tmp_path):
    db = tmp_path / "trading.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE account_meta(account_id TEXT,market TEXT,status TEXT)")
    ids = [f"Q{i:02d}" for i in range(1, 11)]
    retired = {"Q02", "Q05", "Q06", "Q07"}
    con.executemany("INSERT INTO account_meta VALUES (?, 'US', ?)",
                    [(x, "retired" if x in retired else "active") for x in ids])
    con.executemany("INSERT INTO account_meta VALUES (?, 'CN', 'active')", [("C" + x,) for x in ids])
    con.commit(); con.close()
    assert select_qlib_model_ids(db, "US", ids) == ["Q01", "Q03", "Q04", "Q08", "Q09", "Q10"]
    assert select_qlib_model_ids(db, "CN", ids) == ids