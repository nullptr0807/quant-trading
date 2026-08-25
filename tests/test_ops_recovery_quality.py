import json
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def test_scope_aware_backfill_lifecycle_and_legacy_compatibility():
    from scripts.health_check import parse_backfill_log_events

    events = parse_backfill_log_events(
        "===== Backfill [US/1d/raw/ledger] start 2026-07-10T01:00:00Z days=5 =====\n"
        "===== Backfill [US/1d/raw/ledger] OK 2026-07-10T01:02:00Z =====\n"
        "2026-07-10 02:00:00 [INFO] Backfilling [CN] 300 tickers, interval=1d, days=5, price_mode=adjusted\n"
        "2026-07-10 02:03:00 [INFO] Done in 180.0s\n"
    )
    assert [(e["market"], e["scope"], e["event"]) for e in events] == [
        ("US", "ledger", "start"), ("US", "ledger", "ok"),
        ("CN", "universe", "start"), ("CN", "universe", "ok"),
    ]


def test_backfill_health_keys_include_scope():
    from scripts.health_check import evaluate_backfill_log_events, parse_backfill_log_events

    events = parse_backfill_log_events(
        "===== Backfill [US/1d/raw/ledger] FAIL 2026-07-10T01:00:00Z exit=1 =====\n"
        "===== Backfill [US/1d/raw/universe] OK 2026-07-10T01:02:00Z =====\n"
    )
    issues = evaluate_backfill_log_events(
        events, now=datetime(2026, 7, 10, 3, tzinfo=timezone.utc),
        max_success_age_seconds=86400, max_incomplete_seconds=60,
    )
    assert issues == [dict(issues[0], scope="ledger")]
    assert issues[0]["status"] == "failed"


def test_issue_transitions_distinguish_new_continuing_and_recovered():
    from scripts.health_check import classify_issue_transitions

    previous = {"disk:root": "critical", "old:thing": "warning"}
    current = [
        {"check": "disk", "scope": "root", "severity": "critical"},
        {"check": "new", "market": "US", "severity": "warning"},
    ]
    transitions, state = classify_issue_transitions(previous, current)
    assert {x["transition"] for x in transitions} == {"continuing", "new", "recovered"}
    assert state == {"disk:root": "critical", "new:US": "warning"}


def test_inactive_symbols_are_explicitly_blocked_without_guessing_mapping():
    from config.security_master import (
        SECURITY_LIFECYCLE, active_universe_tickers, ticker_lifecycle_block_reason,
    )

    expected = {"APLS", "BK", "BLD", "CTRA", "CWEN-A", "JHG", "MASI", "NSA"}
    assert expected <= {ticker for market, ticker in SECURITY_LIFECYCLE if market == "US"}
    for ticker in expected:
        assert ticker_lifecycle_block_reason(ticker, "US", "2026-08-01T00:00:00Z") == "temporarily_unavailable"
        assert SECURITY_LIFECYCLE[("US", ticker)]["replacement_ticker"] is None

    all_unavailable = {ticker for market, ticker in SECURITY_LIFECYCLE if market == "US"}
    assert active_universe_tickers(
        ["AAPL", *sorted(all_unavailable)], "US", "2026-08-25T00:00:00Z"
    ) == ["AAPL"]


def test_f_account_runtime_state_is_dashboard_consumable(tmp_path):
    from data.store import DataStore

    db = tmp_path / "runtime.db"
    store = DataStore(str(db))
    store.set_account_runtime_status("F14", "US", "blocked", "NO_ADMISSIBLE_FACTOR", {"retry_after": "later"})
    with sqlite3.connect(db) as con:
        row = con.execute(
            "SELECT runtime_status,runtime_reason,runtime_detail FROM account_meta WHERE account_id='F14'"
        ).fetchone()
    assert row[0:2] == ("blocked", "NO_ADMISSIBLE_FACTOR")
    assert "retry_after" in row[2]


def test_ohlc_quarantine_filters_research_but_preserves_audit_rows(tmp_path):
    from data.store import DataStore

    store = DataStore(str(tmp_path / "quality.db"))
    rows = pd.DataFrame([
        {"ticker":"OK","datetime":"2026-07-10","open":10,"high":12,"low":9,"close":11,"volume":100},
        {"ticker":"BAD","datetime":"2026-07-10","open":10,"high":9,"low":8,"close":11,"volume":100},
    ])
    store.save_prices_bulk(rows, interval="1d")
    research = store.load_prices(["OK", "BAD"], "2026-07-01", "2026-07-11", interval="1d")
    audit = store.load_prices(["OK", "BAD"], "2026-07-01", "2026-07-11", interval="1d", quality_mode="audit")
    assert set(research.ticker) == {"OK"}
    assert set(audit.ticker) == {"OK", "BAD"}
    assert store.count_price_quality_issues(interval="1d")["invalid_ohlc"] == 1

    repaired = rows.loc[rows.ticker == "BAD"].copy()
    repaired.loc[:, ["open", "high", "low", "close"]] = [10, 12, 8, 11]
    store.save_prices_bulk(repaired, interval="1d")
    research = store.load_prices(["BAD"], "2026-07-01", "2026-07-11", interval="1d")
    assert set(research.ticker) == {"BAD"}
    assert store.count_price_quality_issues(interval="1d").get("invalid_ohlc", 0) == 0


def test_online_backup_compresses_hashes_and_restore_checks(tmp_path):
    from scripts.backup_trading_db import create_backup

    db = tmp_path / "source.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE x(id INTEGER PRIMARY KEY, value TEXT)")
    con.execute("INSERT INTO x(value) VALUES ('durable')")
    con.commit(); con.close()

    result = create_backup(db, tmp_path / "backups", keep=2)
    artifact = Path(result["path"])
    assert artifact.exists()
    assert artifact.stat().st_mode & 0o777 == 0o600
    assert Path(str(artifact) + ".sha256").exists()
    manifest_path = Path(str(artifact) + ".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    assert manifest_path.stat().st_mode & 0o777 == 0o600
    assert manifest["source_path"] == str(db.resolve())
    assert manifest["artifact_sha256"] == result["sha256"]
    assert manifest["pre_state_fingerprint"]["sha256"] == result["pre_state_fingerprint"]
    assert result["quick_check"] == result["restore_drill"] == "ok"
    assert sorted(p.suffix for p in (tmp_path / "backups").iterdir()) == ['.json', '.sha256', '.zst']


def test_online_backup_removes_snapshot_wal_sidecars(tmp_path, monkeypatch):
    from scripts import backup_trading_db as backup

    db = tmp_path / "source.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE t(x INTEGER)")
    con.execute("INSERT INTO t VALUES (1)")
    con.commit(); con.close()

    original = backup._quick_check
    def checking_with_sidecars(path):
        original(path)
        Path(str(path) + "-wal").touch()
        Path(str(path) + "-shm").touch()

    monkeypatch.setattr(backup, "_quick_check", checking_with_sidecars)
    result = backup.create_backup(db, tmp_path / "backups", keep=2)

    artifact = Path(result["path"])
    raw = Path(str(artifact)[:-4])
    assert not Path(str(raw) + "-wal").exists()
    assert not Path(str(raw) + "-shm").exists()


def test_permissions_are_dry_run_by_default_and_apply_is_explicit(tmp_path):
    root = tmp_path / "quant"
    (root / "logs").mkdir(parents=True)
    secret = root / ".env"; secret.write_text("TOKEN=x\n"); secret.chmod(0o664)
    log = root / "logs/a.log"; log.write_text("x"); log.chmod(0o644)
    script = Path(__file__).parents[1] / "scripts/harden_permissions.py"

    dry = subprocess.run([str(Path(__import__('sys').executable)), str(script), "--root", str(root)], capture_output=True, text=True, check=True)
    assert "DRY_RUN changes=2" in dry.stdout
    assert secret.stat().st_mode & 0o777 == 0o664
    subprocess.run([str(Path(__import__('sys').executable)), str(script), "--root", str(root), "--apply"], check=True)
    assert secret.stat().st_mode & 0o777 == 0o600
    assert log.stat().st_mode & 0o777 == 0o600


def test_operational_wrappers_have_private_umask_and_backup_restore_gate():
    root = Path(__file__).parents[1]
    for name in ("run_scheduled_job.sh", "refresh_factors_daily.sh", "backup_trading_db_daily.sh", "rotate_logs_daily.sh"):
        text = (root / "scripts" / name).read_text()
        assert "umask 077" in text
    assert "run_module_force_exit.py" in (root / "scripts/refresh_factors_daily.sh").read_text()
    assert "run_module_force_exit.py" in (root / "scripts/backfill_prices_daily.sh").read_text()
    backfill_wrapper = (root / "scripts/backfill_prices_daily.sh").read_text()
    assert "HEALTH_WRITE_FAILED" in backfill_wrapper
    assert "exit 70" in backfill_wrapper
    assert "record_operational_health.py" in backfill_wrapper
    assert "|| true" not in backfill_wrapper
    qlib_wrapper = (root / "scripts/qlib_retrain_daily.sh").read_text()
    assert "LOCK_TIMEOUT" in qlib_wrapper
    assert "QLIB_TOTAL_TIMEOUT_SECONDS" in qlib_wrapper
    assert "verify_qlib_scores" in qlib_wrapper
    backup = (root / "scripts/backup_trading_db.py").read_text()
    assert "source.backup" not in backup  # connection object is named src
    assert "src.backup(dst" in backup
    assert "zstd\", \"-t" in backup
    assert "_quick_check(restored)" in backup


def test_generic_scheduler_fails_closed_when_health_write_fails(tmp_path):
    root = tmp_path / "quant"
    (root / "scripts").mkdir(parents=True)
    (root / "data").mkdir()
    fake_python = root / "fake-python"
    fake_python.write_text("#!/bin/sh\nexit 1\n")
    fake_python.chmod(0o755)
    runner = Path(__file__).parents[1] / "scripts" / "run_scheduled_job.sh"
    log = root / "scheduler.log"

    proc = subprocess.run(
        ["/bin/bash", str(runner), "corporate_action_gate", "US", str(root / "gate.lock"),
         "1", "10", str(log), "--", "/bin/true"],
        env={**__import__('os').environ, "QUANT_PROJECT_ROOT": str(root),
             "QUANT_PYTHON": str(fake_python)},
        text=True, capture_output=True,
    )

    assert proc.returncode == 70
    assert "scheduler health write failed component=corporate_action_gate" in log.read_text()


def test_force_exit_runner_does_not_wait_for_orphan_provider_thread(tmp_path):
    module = tmp_path / "hanging_provider.py"
    module.write_text(
        "import threading,time\n"
        "threading.Thread(target=lambda: time.sleep(30), daemon=False).start()\n"
        "print('module_done', flush=True)\n"
    )
    root = Path(__file__).parents[1]
    env = dict(__import__('os').environ, PYTHONPATH=str(tmp_path))
    proc = subprocess.run(
        [__import__('sys').executable, str(root / 'scripts/run_module_force_exit.py'), 'hanging_provider'],
        capture_output=True, text=True, timeout=3, env=env,
    )
    assert proc.returncode == 0
    assert 'module_done' in proc.stdout
