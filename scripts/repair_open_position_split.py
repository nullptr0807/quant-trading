#!/usr/bin/env python3
"""Preview or atomically repair one open-position stock split.

Trades remain immutable.  ``--apply`` requires a verified compressed SQLite
backup produced by ``backup_trading_db.py``; preview is the default and performs
no writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backup_trading_db import MANIFEST_FORMAT, database_fingerprint


def _action_key(account: str, market: str, ticker: str, ex_date: str, ratio: float) -> str:
    value = json.dumps(
        [account, market, ticker, ex_date, format(ratio, ".12g")],
        separators=(",", ":"),
    )
    return hashlib.sha256(value.encode()).hexdigest()


def _position(con: sqlite3.Connection, account: str, market: str, ticker: str) -> dict[str, Any]:
    row = con.execute(
        "SELECT shares,avg_cost,total_cost,current_price,updated_at FROM positions "
        "WHERE account=? AND market=? AND ticker=?",
        (account, market, ticker),
    ).fetchone()
    if row is None or float(row[0] or 0) <= 0:
        raise RuntimeError(f"no open position for {account}/{market}/{ticker}")
    return {
        "shares": float(row[0]), "avg_cost": float(row[1]),
        "total_cost": float(row[2]),
        "current_price": float(row[3]) if row[3] is not None else None,
        "updated_at": row[4],
    }


def _unchanged_ex_date_entitlement(
    con: sqlite3.Connection, account: str, market: str, ticker: str,
    ex_date: str, current_shares: float,
) -> float:
    rows = con.execute(
        "SELECT side,shares,timestamp FROM trades WHERE account=? AND market=? AND ticker=? "
        "ORDER BY timestamp,id",
        (account, market, ticker),
    ).fetchall()
    cutoff = datetime.strptime(ex_date, "%Y-%m-%d").date()
    shares = 0.0
    for side, qty, timestamp in rows:
        try:
            trade_date = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00")).date()
        except ValueError as exc:
            raise RuntimeError(f"invalid trade timestamp in entitlement ledger: {timestamp}") from exc
        if trade_date >= cutoff:
            raise RuntimeError("post-ex-date trade prevents safe split repair")
        side = str(side).lower()
        if side not in {"buy", "sell"}:
            raise RuntimeError(f"unsupported trade side in entitlement ledger: {side}")
        shares += float(qty) if side == "buy" else -float(qty)
    if shares <= 1e-9:
        raise RuntimeError("trade ledger does not prove shares held before split")
    if not math.isclose(current_shares, shares, rel_tol=1e-10, abs_tol=1e-9):
        raise RuntimeError(
            "current position does not equal ex-date entitlement; safe split repair is blocked"
        )
    return shares


def _post_split_raw_price(con: sqlite3.Connection, ticker: str, ex_date: str) -> tuple[float, str]:
    row = con.execute(
        "SELECT close,datetime FROM prices_raw WHERE ticker=? AND interval='1d' "
        "AND datetime>=? AND close>0 ORDER BY datetime DESC LIMIT 1",
        (ticker, ex_date),
    ).fetchone()
    if row is None or not math.isfinite(float(row[0])):
        raise RuntimeError(f"missing post-split raw price for {ticker} on/after {ex_date}")
    return float(row[0]), str(row[1])


def _quick_check(path: Path) -> None:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        result = con.execute("PRAGMA quick_check").fetchone()[0]
    finally:
        con.close()
    if result != "ok":
        raise RuntimeError(f"backup quick_check failed: {result}")


def _verify_backup(
    handle: Path, expected: dict[str, Any], *, db_path: Path,
    live_con: sqlite3.Connection, account: str, market: str, ticker: str,
) -> tuple[str, str]:
    handle = handle.resolve()
    sidecar = Path(str(handle) + ".sha256")
    manifest_path = Path(str(handle) + ".manifest.json")
    if (
        not handle.is_file() or not sidecar.is_file() or not manifest_path.is_file()
        or handle.suffix != ".zst"
    ):
        raise RuntimeError(
            "--apply requires a verified backup handle (.zst plus .sha256 and manifest)"
        )
    sidecar_parts = sidecar.read_text().strip().split()
    if len(sidecar_parts) != 2 or sidecar_parts[1] != handle.name:
        raise RuntimeError("verified backup checksum sidecar is not bound to artifact name")
    expected_digest = sidecar_parts[0]
    digest = hashlib.sha256(handle.read_bytes()).hexdigest()
    if expected_digest != digest:
        raise RuntimeError("verified backup checksum mismatch")
    try:
        manifest = json.loads(manifest_path.read_text())
        fingerprint = manifest["pre_state_fingerprint"]["sha256"]
        fingerprint_algorithm = manifest["pre_state_fingerprint"]["algorithm"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("verified backup manifest is invalid") from exc
    if manifest.get("format") != MANIFEST_FORMAT:
        raise RuntimeError("verified backup manifest format is unsupported")
    if Path(str(manifest.get("source_path", ""))).resolve() != db_path.resolve():
        raise RuntimeError("verified backup manifest canonical source does not match target database")
    if manifest.get("artifact_name") != handle.name or manifest.get("artifact_sha256") != digest:
        raise RuntimeError("verified backup manifest does not match artifact")
    if fingerprint_algorithm != "sha256-sqlite-iterdump-v1":
        raise RuntimeError("verified backup manifest fingerprint algorithm is unsupported")
    subprocess.run(["zstd", "-t", str(handle)], check=True, capture_output=True)
    with tempfile.TemporaryDirectory() as td:
        restored = Path(td) / "restored.db"
        with restored.open("wb") as output:
            subprocess.run(["zstd", "-dc", str(handle)], check=True, stdout=output)
        _quick_check(restored)
        con = sqlite3.connect(f"file:{restored}?mode=ro", uri=True)
        try:
            backup_state = _position(con, account, market, ticker)
            restored_fingerprint = database_fingerprint(con)
        finally:
            con.close()
    if restored_fingerprint != fingerprint:
        raise RuntimeError("verified backup manifest full pre-state fingerprint mismatch")
    if database_fingerprint(live_con) != fingerprint:
        raise RuntimeError("verified backup does not match current full pre-state fingerprint")
    for field in ("shares", "avg_cost", "total_cost", "current_price", "updated_at"):
        if backup_state[field] != expected[field]:
            raise RuntimeError(
                f"verified backup does not match current pre-repair position: {field}"
            )
    return digest, fingerprint


def _table_exists(con: sqlite3.Connection, name: str) -> bool:
    return con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _existing_repair(con: sqlite3.Connection, action_key: str) -> dict[str, Any] | None:
    if not _table_exists(con, "corporate_action_repairs"):
        return None
    row = con.execute(
        "SELECT applied_at,after_state,backup_handle FROM corporate_action_repairs WHERE action_key=?",
        (action_key,),
    ).fetchone()
    if row is None:
        return None
    return {
        "mode": "apply", "changed": False, "already_applied": True,
        "action_key": action_key, "applied_at": row[0],
        "after": json.loads(row[1]), "backup_handle": row[2],
    }


def _create_audit_table(con: sqlite3.Connection) -> None:
    con.execute(
        """CREATE TABLE IF NOT EXISTS corporate_action_repairs (
        action_key TEXT PRIMARY KEY,
        account TEXT NOT NULL, market TEXT NOT NULL, ticker TEXT NOT NULL,
        ex_date TEXT NOT NULL, action_type TEXT NOT NULL, ratio REAL NOT NULL,
        source TEXT NOT NULL, raw TEXT, before_state TEXT NOT NULL,
        after_state TEXT NOT NULL, raw_price_timestamp TEXT NOT NULL,
        backup_handle TEXT NOT NULL, backup_sha256 TEXT NOT NULL,
        applied_at TEXT NOT NULL
        )"""
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_corporate_action_repairs_scope "
        "ON corporate_action_repairs(market,account,ticker,ex_date)"
    )
    con.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_corporate_action_repair_event "
        "ON corporate_action_repairs(account,market,ticker,ex_date,action_type)"
    )


def repair_open_split(
    *, db_path: Path, account: str, market: str, ticker: str, ex_date: str,
    ratio: float, source: str, raw: Any = None, apply: bool = False,
    backup_handle: Path | None = None, now: datetime | None = None,
) -> dict[str, Any]:
    db_path = Path(db_path).resolve()
    market = market.upper().strip()
    ticker = ticker.upper().strip()
    if not math.isfinite(ratio) or ratio <= 0 or abs(ratio - 1.0) < 1e-12:
        raise ValueError("ratio must be finite, positive, and not 1")
    try:
        datetime.strptime(ex_date, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("ex_date must be YYYY-MM-DD") from exc
    key = _action_key(account, market, ticker, ex_date, ratio)

    con = sqlite3.connect(db_path, timeout=30)
    try:
        existing = _existing_repair(con, key)
        if existing is not None:
            return existing
        before = _position(con, account, market, ticker)
        held_before = _unchanged_ex_date_entitlement(
            con, account, market, ticker, ex_date, before["shares"]
        )
        raw_price, raw_price_ts = _post_split_raw_price(con, ticker, ex_date)
        new_shares = before["shares"] * ratio
        if new_shares <= 0 or not math.isfinite(new_shares):
            raise RuntimeError("split calculation produced invalid shares")
        after = {
            **before,
            "shares": new_shares,
            "avg_cost": before["avg_cost"] / ratio,
            "current_price": raw_price,
        }
        result = {
            "mode": "apply" if apply else "preview", "changed": False,
            "already_applied": False, "action_key": key, "account": account,
            "market": market, "ticker": ticker, "ex_date": ex_date,
            "ratio": ratio, "held_shares_before": held_before,
            "before": before, "after": after,
            "raw_price_timestamp": raw_price_ts,
            "trades_immutable": True,
        }
        if not apply:
            return result
        if backup_handle is None:
            raise RuntimeError("--apply requires a verified backup handle")
        backup_digest, backup_fingerprint = _verify_backup(
            Path(backup_handle), before, db_path=db_path, live_con=con,
            account=account, market=market, ticker=ticker,
        )
        applied_at = (now or datetime.now(timezone.utc)).isoformat()
        after["updated_at"] = applied_at

        con.execute("BEGIN IMMEDIATE")
        if database_fingerprint(con) != backup_fingerprint:
            raise RuntimeError("database changed after full pre-state backup verification")
        # Fail closed if another writer changed the position after preview/backup verification.
        if _position(con, account, market, ticker) != before:
            raise RuntimeError("position changed during repair preflight")
        _create_audit_table(con)
        con.execute(
            "UPDATE positions SET shares=?,avg_cost=?,total_cost=?,current_price=?,updated_at=? "
            "WHERE account=? AND market=? AND ticker=?",
            (new_shares, after["avg_cost"], before["total_cost"], raw_price,
             applied_at, account, market, ticker),
        )
        state = con.execute(
            "SELECT cash FROM account_state WHERE account=? AND market=?",
            (account, market),
        ).fetchone()
        if state is None:
            raise RuntimeError("missing market-scoped account_state")
        cash = float(state[0])
        con.execute(
            "UPDATE account_state SET updated_at=? WHERE account=? AND market=?",
            (applied_at, account, market),
        )
        con.execute(
            "INSERT INTO positions_history "
            "(account,ticker,shares,avg_cost,market_price,market_value,unrealized_pnl,timestamp,market) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (account, ticker, new_shares, after["avg_cost"], raw_price,
             new_shares * raw_price, new_shares * raw_price - before["total_cost"],
             applied_at, market),
        )
        values = con.execute(
            "SELECT shares,current_price FROM positions WHERE account=? AND market=?",
            (account, market),
        ).fetchall()
        if any(row[1] is None or float(row[1]) <= 0 for row in values):
            raise RuntimeError("cannot publish repaired snapshot with missing position marks")
        equity = cash + sum(float(row[0]) * float(row[1]) for row in values)
        con.execute(
            "INSERT INTO accounts(name,cash,equity,timestamp,market) VALUES (?,?,?,?,?)",
            (account, cash, equity, applied_at, market),
        )
        raw_json = json.dumps(raw, ensure_ascii=False, sort_keys=True) if raw is not None else None
        before_json = json.dumps(before, ensure_ascii=False, sort_keys=True)
        after_json = json.dumps(after, ensure_ascii=False, sort_keys=True)
        con.execute(
            "INSERT INTO corporate_action_repairs "
            "(action_key,account,market,ticker,ex_date,action_type,ratio,source,raw,"
            "before_state,after_state,raw_price_timestamp,backup_handle,backup_sha256,applied_at) "
            "VALUES (?,?,?,?,?,'split',?,?,?,?,?,?,?,?,?)",
            (key, account, market, ticker, ex_date, ratio, source, raw_json,
             before_json, after_json, raw_price_ts, str(Path(backup_handle).resolve()),
             backup_digest, applied_at),
        )
        detail = {
            "action_key": key, "ex_date": ex_date, "ratio": ratio,
            "source": source, "before": before, "after": after,
            "raw_price_timestamp": raw_price_ts,
            "backup_handle": str(Path(backup_handle).resolve()),
            "backup_sha256": backup_digest, "trades_immutable": True,
        }
        con.execute(
            "INSERT INTO events(ts,category,severity,account,ticker,title,detail,market) "
            "VALUES (?,'corporate_action_repair','warning',?,?,?, ?,?)",
            (applied_at, account, ticker,
             f"Applied audited split repair {ticker} {ratio:g}:1",
             json.dumps(detail, ensure_ascii=False, sort_keys=True), market),
        )
        con.commit()
        result.update({
            "changed": True, "after": after, "applied_at": applied_at,
            "backup_handle": str(Path(backup_handle).resolve()),
            "backup_sha256": backup_digest, "equity": equity,
        })
        return result
    except Exception:
        if con.in_transaction:
            con.rollback()
        raise
    finally:
        con.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview/apply an audited open-position split repair")
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "trading.db")
    parser.add_argument("--account", required=True)
    parser.add_argument("--market", required=True)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--ex-date", required=True)
    parser.add_argument("--ratio", required=True, type=float)
    parser.add_argument("--source", required=True)
    parser.add_argument("--raw-json")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup-handle", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.apply and args.backup_handle is None:
        raise SystemExit("--apply requires --backup-handle from backup_trading_db.py")
    raw = json.loads(args.raw_json) if args.raw_json else None
    result = repair_open_split(
        db_path=args.db, account=args.account, market=args.market, ticker=args.ticker,
        ex_date=args.ex_date, ratio=args.ratio, source=args.source, raw=raw,
        apply=args.apply, backup_handle=args.backup_handle,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
