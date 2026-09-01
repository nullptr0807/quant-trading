#!/usr/bin/env python3
"""Purge account-specific model assets for retired accounts.

General disaster-recovery database backups are intentionally out of scope.
Dry-run is the default; --apply performs secure SQLite deletes and file removal.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GP_FILES = [
    ROOT / "factors/mined_alphas_per_account.json",
    ROOT / "factors/mined_alphas_per_account.json.bak.20260429_063042",
]
FM_FILES = [
    ROOT / "factors/factor_miner_gp/mined_alphas_f.json",
    ROOT / "factors/factor_miner_gp/mined_alphas_f.json.bak_pre_failure_markers",
]
DEDICATED_DB_BACKUPS = [
    ROOT / "data/trading.db.bak_factor_values_pk_20260707T055112Z",
    ROOT / "data/trading.db.bak_refresh_factors_20260707T050116Z",
    ROOT / "data/trading.db.bak_refresh_other_factors_20260707T051752Z",
    ROOT / "data/trading.db.bak_refresh_other_factors_20260707T052358Z",
]
CN_SUFFIXES = (".SH", ".SZ", ".BJ")


def _write_json(path: Path, value) -> None:
    tmp = path.with_name(path.name + ".purge-tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def _market_clause(market: str) -> str:
    cn = "(ticker LIKE '%.SH' OR ticker LIKE '%.SZ' OR ticker LIKE '%.BJ')"
    return cn if market == "CN" else f"NOT {cn}"


def inventory(db_path: Path) -> dict:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT account_id,market,\"group\" FROM account_meta "
        "WHERE status='retired' ORDER BY market,account_id"
    ).fetchall()
    con.close()
    retired = [dict(r) for r in rows]
    gp = [r["account_id"] for r in retired if r["group"] == "B"]
    fm = [r["account_id"] for r in retired if r["group"] == "F"]
    qlib = {}
    for r in retired:
        if r["group"] == "Q":
            model = r["account_id"][1:] if r["market"] == "CN" and r["account_id"].startswith("C") else r["account_id"]
            qlib.setdefault(r["market"], []).append(model)
    return {"retired": retired, "gp": gp, "fm": fm, "qlib": qlib}


def purge(db_path: Path, apply: bool) -> dict:
    inv = inventory(db_path)
    gp, fm, qlib = set(inv["gp"]), set(inv["fm"]), inv["qlib"]
    groups = {f"gp_{x}" for x in gp} | {f"fmgp_{x}" for x in fm}
    result = {"apply": apply, "inventory": inv, "json_keys": {}, "files": [], "db_rows": {}}

    for path, keys in [(p, gp) for p in GP_FILES] + [(p, fm) for p in FM_FILES]:
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        removed = sorted(set(data) & keys)
        result["json_keys"][str(path.relative_to(ROOT))] = removed
        if apply and removed:
            for key in removed:
                del data[key]
            _write_json(path, data)

    memory_path = ROOT / "factors/factor_miner_gp/mining_memory.json"
    if memory_path.exists():
        memory = json.loads(memory_path.read_text())
        reports = memory.get("recent_reports") or []
        removed_reports = [r for r in reports if r.get("account_id") in fm]
        result["json_keys"][str(memory_path.relative_to(ROOT))] = [r.get("account_id") for r in removed_reports]
        if apply and removed_reports:
            memory["recent_reports"] = [r for r in reports if r.get("account_id") not in fm]
            _write_json(memory_path, memory)

    log_dir = ROOT / "factors/factor_miner_gp/mining_logs"
    retired_log_ids = gp | fm
    if log_dir.exists():
        for path in log_dir.glob("*.json"):
            if any(re.search(rf"_{re.escape(account)}\.json$", path.name) for account in retired_log_ids):
                result["files"].append(str(path.relative_to(ROOT)))
                if apply:
                    path.unlink()

    checkpoint_root = ROOT / "data/qlib_checkpoints"
    for market, models in qlib.items():
        for model in models:
            path = checkpoint_root / market / model
            if path.exists():
                file_count = sum(1 for p in path.rglob("*") if p.is_file())
                byte_count = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
                result["files"].append({"path": str(path.relative_to(ROOT)), "files": file_count, "bytes": byte_count})
                if apply:
                    shutil.rmtree(path)
        for marker in (checkpoint_root / market).glob("publication-*.json"):
            data = json.loads(marker.read_text())
            model_map = data.get("models") or {}
            removed = sorted(set(model_map) & set(models))
            if removed:
                result["json_keys"][str(marker.relative_to(ROOT))] = removed
                if apply:
                    for model in removed:
                        del model_map[model]
                    _write_json(marker, data)

    for path in DEDICATED_DB_BACKUPS:
        if path.exists():
            result["files"].append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size})
            if apply:
                path.unlink()

    con = sqlite3.connect(db_path, timeout=120)
    con.execute("PRAGMA busy_timeout=120000")
    if apply:
        con.execute("PRAGMA secure_delete=ON")
    factor_tables = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'factor_values%'"
    )]
    for table in factor_tables:
        count = 0
        if groups:
            marks = ",".join("?" for _ in groups)
            count += con.execute(
                f'SELECT COUNT(*) FROM "{table}" WHERE factor_group IN ({marks})', sorted(groups)
            ).fetchone()[0]
            if apply:
                con.execute(f'DELETE FROM "{table}" WHERE factor_group IN ({marks})', sorted(groups))
        for market, models in qlib.items():
            clause = _market_clause(market)
            names = [f"qlib_{model}_score" for model in models]
            if not names:
                continue
            marks = ",".join("?" for _ in names)
            count += con.execute(
                f'SELECT COUNT(*) FROM "{table}" WHERE factor_group=\'qlib\' '
                f'AND factor_name IN ({marks}) AND {clause}', names
            ).fetchone()[0]
            if apply:
                con.execute(
                    f'DELETE FROM "{table}" WHERE factor_group=\'qlib\' '
                    f'AND factor_name IN ({marks}) AND {clause}', names
                )
        result["db_rows"][table] = count

    retired_ids = [r["account_id"] for r in inv["retired"]]
    if retired_ids:
        marks = ",".join("?" for _ in retired_ids)
        params = [*retired_ids]
        result["db_rows"]["account_meta_factor_fields"] = con.execute(
            f"SELECT COUNT(*) FROM account_meta WHERE status='retired' AND account_id IN ({marks}) "
            "AND COALESCE(factors,'')<>''", params
        ).fetchone()[0]
        if apply:
            con.execute(
                f"UPDATE account_meta SET factors='' WHERE status='retired' AND account_id IN ({marks})", params
            )
    if apply:
        now = datetime.now(timezone.utc).isoformat()
        con.execute(
            "INSERT INTO events(ts,category,severity,account,ticker,title,detail,market) "
            "VALUES (?, 'lifecycle', 'info', NULL, NULL, ?, ?, 'ALL')",
            (now, "Retired model assets purged", json.dumps({
                "retired_accounts": retired_ids,
                "general_disaster_recovery_backups_preserved": True,
            }, ensure_ascii=False)),
        )
        con.commit()
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    con.close()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(ROOT / "data/trading.db"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(purge(Path(args.db), args.apply), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
