from __future__ import annotations

import sqlite3
from pathlib import Path


def _schema(con):
    con.executescript(
        """
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY, account TEXT, ticker TEXT, side TEXT,
            shares REAL, price REAL, cost REAL, slippage REAL,
            timestamp TEXT, market TEXT
        );
        CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, category TEXT,
            severity TEXT, account TEXT, ticker TEXT, title TEXT,
            detail TEXT, market TEXT
        );
        """
    )


def test_quarantine_apply_archives_only_and_is_idempotent(tmp_path, monkeypatch):
    from scripts import quarantine_cn_t1_history as q

    db = tmp_path / "trading.db"
    backup = tmp_path / "full-backup.zst"
    backup.write_bytes(b"verified elsewhere")
    with sqlite3.connect(db) as con:
        _schema(con)
        con.executemany(
            "INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?)",
            [
                (1, "CA01", "000001.SZ", "buy", 100, 10, 0, 0, "2026-07-01T02:00:00+00:00", "CN"),
                (2, "CA01", "000001.SZ", "sell", 100, 9, 0, 0, "2026-07-01T05:00:00+00:00", "CN"),
            ],
        )
    monkeypatch.setattr(q, "DB", db)

    first = q.run(apply=True, backup=str(backup))
    second = q.run(apply=True, backup=str(backup))

    assert first["violations"] == 1
    assert second["already_applied"] is True
    with sqlite3.connect(db) as con:
        assert con.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 2
        assert con.execute("SELECT COUNT(*) FROM data_quality_quarantine").fetchone()[0] == 1
        table = con.execute("SELECT archive_table FROM data_quality_quarantine").fetchone()[0]
        assert con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0] == 1
        assert con.execute(f'SELECT id FROM "{table}"').fetchone()[0] == 2
