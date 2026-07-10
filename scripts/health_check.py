#!/usr/bin/env python3
"""Quant system health checks for prices, factors, quotes, snapshots, and ledger.

Read-only by default. Exits non-zero on critical issues so cron/CI can alert.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
DB_PATH = PROJECT_ROOT / "data" / "trading.db"
CN_SUFFIXES = (".SH", ".SZ", ".BJ")
ROLL20_FACTORS = [
    "ROC_20", "MA_RATIO_20", "VMOM_20", "VSTD_20", "STD_20", "BBPOS_20", "BETA_20",
]
BACKFILL_LOG = PROJECT_ROOT / "logs" / "backfill.log"
_BACKFILL_WRAPPER_RE = re.compile(
    r"^===== Backfill \[(?P<market>US|CN)/(?P<interval>[^/\]]+)/(?P<price_mode>[^\]]+)\] "
    r"(?P<event>start|OK|FAIL) (?P<timestamp>\S+)(?: .*?exit=(?P<exit_code>\d+))? =====$"
)
_LEGACY_BACKFILL_START_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:,\d+)? \[INFO\] "
    r"Backfilling \[(?P<market>US|CN)\].*?interval=(?P<interval>[^, ]+)"
    r"(?:.*?price_mode=(?P<price_mode>\w+))?\s*$"
)
_LEGACY_BACKFILL_OK_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:,\d+)? \[INFO\] Done in "
)


def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=30000")
    return con


def _is_cn(ticker: str) -> bool:
    return ticker.upper().endswith(CN_SUFFIXES)


def _table_exists(con: sqlite3.Connection, table: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def _market_clause(market: str) -> str:
    if market == "CN":
        return "ticker LIKE '%.SH' OR ticker LIKE '%.SZ' OR ticker LIKE '%.BJ'"
    return "ticker NOT LIKE '%.SH' AND ticker NOT LIKE '%.SZ' AND ticker NOT LIKE '%.BJ'"


def _normalize_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        return _normalize_utc(datetime.fromisoformat(text))
    except ValueError:
        try:
            return datetime.strptime(text[:19], "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            return None


def _latest_by_ticker(
    con: sqlite3.Connection,
    table: str,
    market: str,
    tickers: Iterable[str],
) -> dict[str, str]:
    wanted = sorted(set(tickers))
    if not wanted or not _table_exists(con, table):
        return {}
    placeholders = ",".join("?" for _ in wanted)
    rows = con.execute(
        f"""
        SELECT ticker, MAX(substr(datetime,1,10)) AS max_date
        FROM {table}
        WHERE interval='1d' AND ({_market_clause(market)})
          AND ticker IN ({placeholders})
        GROUP BY ticker
        """,
        wanted,
    ).fetchall()
    return {str(r["ticker"]): str(r["max_date"]) for r in rows if r["max_date"]}


def _active_held_tickers(con: sqlite3.Connection, market: str) -> list[str]:
    if not _table_exists(con, "positions"):
        return []
    has_meta = _table_exists(con, "account_meta")
    if has_meta:
        rows = con.execute(
            """
            SELECT DISTINCT p.ticker
            FROM positions p
            LEFT JOIN account_meta m
              ON m.account_id=p.account AND COALESCE(m.market,p.market,'US')=COALESCE(p.market,'US')
            WHERE COALESCE(p.market,'US')=? AND COALESCE(p.shares,0)>0
              AND COALESCE(m.status,'active')='active'
            ORDER BY p.ticker
            """,
            (market,),
        ).fetchall()
    else:
        rows = con.execute(
            """
            SELECT DISTINCT ticker FROM positions
            WHERE COALESCE(market,'US')=? AND COALESCE(shares,0)>0
            ORDER BY ticker
            """,
            (market,),
        ).fetchall()
    return [str(r[0]) for r in rows]


def _price_set_issues(
    *,
    con: sqlite3.Connection,
    market: str,
    price_mode: str,
    tickers: Iterable[str],
    target_date: str,
) -> list[dict]:
    expected = sorted(set(tickers))
    if not expected:
        return []
    table = "prices_raw" if price_mode == "raw" else "prices"
    latest = _latest_by_ticker(con, table, market, expected)
    covered = sorted(latest)
    missing = sorted(set(expected) - set(covered))
    stale = sorted(t for t, d in latest.items() if d < target_date)
    issues: list[dict] = []
    if missing:
        coverage = len(covered) / len(expected)
        issues.append({
            "severity": "critical" if coverage < 0.8 else "warning",
            "check": "price_1d_coverage",
            "market": market,
            "price_mode": price_mode,
            "table": table,
            "scope": "active_holdings" if price_mode == "raw" else "universe",
            "expected": len(expected),
            "covered": len(covered),
            "coverage": round(coverage, 4),
            "missing_sample": missing[:20],
        })
    if stale:
        freshest = max((latest[t] for t in stale), default=None)
        issues.append({
            "severity": "critical" if len(stale) == len(expected) else "warning",
            "check": "price_1d_freshness",
            "market": market,
            "price_mode": price_mode,
            "table": table,
            "scope": "active_holdings" if price_mode == "raw" else "universe",
            "target_date": target_date,
            "freshest_stale_date": freshest,
            "stale": len(stale),
            "expected": len(expected),
            "stale_sample": stale[:20],
        })
    return issues


def check_price_1d_health(
    con: sqlite3.Connection,
    *,
    universe_by_market: dict[str, Iterable[str]],
    target_dates: dict[str, str],
) -> list[dict]:
    """Check adjusted universe and raw active-holding 1d coverage/freshness."""
    issues: list[dict] = []
    for market in ("US", "CN"):
        target = target_dates[market]
        issues.extend(_price_set_issues(
            con=con,
            market=market,
            price_mode="adjusted",
            tickers=universe_by_market.get(market, ()),
            target_date=target,
        ))
        issues.extend(_price_set_issues(
            con=con,
            market=market,
            price_mode="raw",
            tickers=_active_held_tickers(con, market),
            target_date=target,
        ))
    return issues


def check_schema(con: sqlite3.Connection) -> list[dict]:
    """Validate safety-critical DB schema invariants."""
    issues: list[dict] = []
    try:
        rows = con.execute("PRAGMA table_info(factor_values)").fetchall()
        pk_cols = [r[1] for r in sorted((r for r in rows if r[5]), key=lambda r: r[5])]
    except Exception as e:
        return [{"severity": "critical", "check": "schema", "table": "factor_values", "detail": repr(e)}]

    expected = ["ticker", "date", "factor_name", "factor_group"]
    if pk_cols != expected:
        issues.append({
            "severity": "critical",
            "check": "schema",
            "table": "factor_values",
            "detail": "factor_values primary key must include factor_group to prevent GP/F/Q cross-account overwrites",
            "expected_pk": expected,
            "actual_pk": pk_cols,
        })
    return issues


def latest_trading_date(con: sqlite3.Connection, market: str) -> str | None:
    clause = _market_clause(market)
    row = con.execute(
        f"SELECT MAX(substr(datetime,1,10)) FROM prices WHERE interval='1d' AND ({clause})"
    ).fetchone()
    return row[0] if row else None


def check_alpha20(con: sqlite3.Connection, warn_lag_days: int = 3) -> list[dict]:
    issues: list[dict] = []
    # Compare 20d factors to each market's latest price date. factor_values has no market column.
    for market in ("US", "CN"):
        latest_px = latest_trading_date(con, market)
        if not latest_px:
            issues.append({"severity": "critical", "check": "price_latest", "market": market, "detail": "no 1d prices"})
            continue
        if market == "CN":
            t_clause = "ticker LIKE '%.SH' OR ticker LIKE '%.SZ' OR ticker LIKE '%.BJ'"
        else:
            t_clause = "ticker NOT LIKE '%.SH' AND ticker NOT LIKE '%.SZ' AND ticker NOT LIKE '%.BJ'"
        rows = con.execute(
            f"""
            SELECT factor_name, MAX(date) max_date, COUNT(DISTINCT ticker) tickers
            FROM factor_values
            WHERE factor_group='alpha158'
              AND factor_name IN ({','.join('?' for _ in ROLL20_FACTORS)})
              AND ({t_clause})
            GROUP BY factor_name
            """,
            ROLL20_FACTORS,
        ).fetchall()
        got = {r["factor_name"]: r for r in rows}
        for fn in ROLL20_FACTORS:
            r = got.get(fn)
            if not r or not r["max_date"]:
                issues.append({"severity": "critical", "check": "alpha20", "market": market, "factor": fn, "detail": "missing"})
                continue
            lag = (datetime.fromisoformat(latest_px) - datetime.fromisoformat(r["max_date"])).days
            if lag > warn_lag_days:
                issues.append({
                    "severity": "critical" if lag > 7 else "warning",
                    "check": "alpha20",
                    "market": market,
                    "factor": fn,
                    "latest_price_date": latest_px,
                    "factor_date": r["max_date"],
                    "lag_days": lag,
                    "tickers": r["tickers"],
                })
    return issues


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _active_factor_count(factors: list[dict] | None) -> int:
    return sum(
        1 for f in (factors or [])
        if f.get("expression") and f.get("active", True) is not False
    )


def _failure_marker_count(factors: list[dict] | None) -> int:
    return sum(1 for f in (factors or []) if f.get("status") == "mining_failed")


def _count_where(con: sqlite3.Connection, sql: str, params: tuple) -> int:
    try:
        row = con.execute(sql, params).fetchone()
        return int(row[0] or 0) if row else 0
    except sqlite3.OperationalError:
        return 0


def _has_account_activity(con: sqlite3.Connection, account: str, market: str) -> bool:
    checks = [
        ("SELECT COUNT(*) FROM account_state WHERE account=? AND market=?", (account, market)),
        ("SELECT COUNT(*) FROM positions WHERE account=? AND market=?", (account, market)),
        ("SELECT COUNT(*) FROM trades WHERE account=? AND market=?", (account, market)),
        ("SELECT COUNT(*) FROM accounts WHERE name=? AND market=?", (account, market)),
    ]
    return any(_count_where(con, sql, params) > 0 for sql, params in checks)


def _lag_days(latest: str, observed: str | None) -> int | None:
    if not latest or not observed:
        return None
    try:
        return (datetime.fromisoformat(latest) - datetime.fromisoformat(observed)).days
    except Exception:
        return None


def check_model_factor_freshness(con: sqlite3.Connection) -> list[dict]:
    """Check Qlib / GP / FactorMiner persisted signal freshness."""
    issues: list[dict] = []
    legacy = _load_json(PROJECT_ROOT / "factors" / "mined_alphas_per_account.json")
    fmgp = _load_json(PROJECT_ROOT / "factors" / "factor_miner_gp" / "mined_alphas_f.json")

    meta_rows = con.execute(
        "SELECT account_id, market, \"group\" AS grp, status FROM account_meta "
        "WHERE COALESCE(status,'active')='active' AND \"group\" IN ('Q','B','F') "
        "ORDER BY market, account_id"
    ).fetchall()

    for r in meta_rows:
        acct = r["account_id"]
        market = r["market"] or "US"
        grp = r["grp"]
        # Ignore inert metadata placeholders (e.g. C01) with no operational rows.
        if not _has_account_activity(con, acct, market):
            continue
        latest_px = latest_trading_date(con, market)
        if not latest_px:
            continue

        if grp == "Q":
            base_id = acct[1:] if market == "CN" and acct.startswith("C") else acct
            factor_name = f"qlib_{base_id}_score"
            if market == "CN":
                t_clause = "ticker LIKE '%.SH' OR ticker LIKE '%.SZ' OR ticker LIKE '%.BJ'"
            else:
                t_clause = "ticker NOT LIKE '%.SH' AND ticker NOT LIKE '%.SZ' AND ticker NOT LIKE '%.BJ'"
            row = con.execute(
                f"SELECT MAX(date) max_date, COUNT(DISTINCT ticker) tickers "
                f"FROM factor_values WHERE factor_group='qlib' AND factor_name=? AND ({t_clause})",
                (factor_name,),
            ).fetchone()
            max_date = row["max_date"] if row else None
            lag = _lag_days(latest_px, max_date)
            if not max_date:
                issues.append({"severity": "critical", "check": "qlib_factor", "account": acct, "detail": "missing persisted score"})
            else:
                # Mirror main.py::_load_qlib_scores live-trading stale gate:
                # US Q scores may lag latest 1d prices by at most 2 calendar
                # days; CN allows 3 for exchange-calendar/provider lag.  Health
                # must warn before or exactly when live Q accounts would no-op.
                max_lag = 3 if market == "CN" else 2
                if lag is not None and lag > max_lag:
                    issues.append({
                        "severity": "critical" if lag > max_lag + 4 else "warning",
                        "check": "qlib_factor",
                        "account": acct,
                        "market": market,
                        "factor_date": max_date,
                        "latest_price_date": latest_px,
                        "lag_days": lag,
                        "max_lag_days": max_lag,
                        "tickers": row["tickers"],
                    })
            continue

        mined = fmgp.get(acct) if grp == "F" else legacy.get(acct)
        active_count = _active_factor_count(mined)
        failure_count = _failure_marker_count(mined)
        if active_count == 0:
            # FactorMiner can legitimately have no admitted factor. Treat explicit
            # failure/inactive markers as known experimental states, not system
            # freshness warnings. Unknown empty state remains a warning.
            if grp == "F" and failure_count:
                continue
            inactive_count = sum(
                1 for f in (mined or [])
                if f.get("active") is False and f.get("expression")
            )
            if grp == "F" and inactive_count:
                continue
            issues.append({
                "severity": "warning",
                "check": "model_factor",
                "account": acct,
                "group": grp,
                "status": "mining_failed" if failure_count else "no_active_factor",
                "failure_markers": failure_count,
            })
            continue

        factor_group = f"fmgp_{acct}" if grp == "F" else f"gp_{acct}"
        row = con.execute(
            "SELECT MAX(date) max_date, COUNT(DISTINCT ticker) tickers, COUNT(*) rows "
            "FROM factor_values WHERE factor_group=?",
            (factor_group,),
        ).fetchone()
        max_date = row["max_date"] if row else None
        lag = _lag_days(latest_px, max_date)
        if not max_date:
            issues.append({
                "severity": "warning",
                "check": "model_factor",
                "account": acct,
                "group": grp,
                "factor_group": factor_group,
                "detail": "active mined factors exist but no persisted runtime values",
                "active_factors": active_count,
            })
        elif lag is not None and lag > 3:
            issues.append({
                # Persisted GP/F factor_values are diagnostics, not the live trading
                # source of truth (main.py recomputes runtime matrices in memory each
                # cycle). Treat stale persisted rows as warning; Alpha/Q remain stricter.
                "severity": "warning",
                "check": "model_factor",
                "account": acct,
                "group": grp,
                "factor_group": factor_group,
                "factor_date": max_date,
                "latest_price_date": latest_px,
                "lag_days": lag,
                "active_factors": active_count,
                "tickers": row["tickers"],
            })
    return issues


def check_snapshot_diff(con: sqlite3.Connection, tolerance: float = 1.0) -> list[dict]:
    issues: list[dict] = []
    rows = con.execute(
        """
        WITH latest AS (
          SELECT a.* FROM accounts a
          JOIN (
            SELECT name, COALESCE(market,'US') market, MAX(timestamp) ts
            FROM accounts GROUP BY name, COALESCE(market,'US')
          ) x ON a.name=x.name AND COALESCE(a.market,'US')=x.market AND a.timestamp=x.ts
        ), posmv AS (
          SELECT account, COALESCE(market,'US') market,
                 SUM(shares*COALESCE(current_price,avg_cost)) mv
          FROM positions GROUP BY account, COALESCE(market,'US')
        )
        SELECT m.market, m.account_id, latest.timestamp,
               latest.equity latest_equity,
               account_state.cash + COALESCE(posmv.mv,0) computed,
               latest.equity - (account_state.cash + COALESCE(posmv.mv,0)) diff
        FROM account_meta m
        JOIN latest ON latest.name=m.account_id AND COALESCE(latest.market,m.market)=m.market
        JOIN account_state ON account_state.account=m.account_id AND COALESCE(account_state.market,m.market)=m.market
        LEFT JOIN posmv ON posmv.account=m.account_id AND posmv.market=m.market
        WHERE COALESCE(m.status,'active')='active'
        ORDER BY ABS(diff) DESC
        """
    ).fetchall()
    for r in rows:
        if abs(float(r["diff"])) > tolerance:
            issues.append({
                "severity": "critical",
                "check": "equity_snapshot",
                "market": r["market"],
                "account": r["account_id"],
                "diff": r["diff"],
                "latest_timestamp": r["timestamp"],
            })
    return issues


def cn_active_held_tickers(con: sqlite3.Connection) -> list[str]:
    return [
        r[0]
        for r in con.execute(
            """
            SELECT DISTINCT p.ticker
            FROM positions p LEFT JOIN account_meta m ON m.account_id=p.account
            WHERE COALESCE(m.status,'active')!='retired'
              AND COALESCE(m.market,p.market,'US')='CN'
            ORDER BY p.ticker
            """
        ).fetchall()
    ]


def check_cn_quotes(min_coverage: float = 0.95) -> list[dict]:
    issues: list[dict] = []
    try:
        from data.cn_fetcher import CNDataFetcher
        with _conn() as con:
            tickers = cn_active_held_tickers(con)
        if not tickers:
            return issues
        out = CNDataFetcher().get_realtime_quotes(tickers)
        coverage = len(out) / len(tickers)
        if coverage < min_coverage:
            issues.append({
                "severity": "critical" if coverage < 0.5 else "warning",
                "check": "cn_quote_coverage",
                "quoted": len(out),
                "requested": len(tickers),
                "coverage": round(coverage, 4),
                "missing_sample": [t for t in tickers if t not in out][:10],
            })
    except Exception as e:
        issues.append({"severity": "critical", "check": "cn_quote_coverage", "detail": repr(e)})
    return issues


def parse_backfill_log_events(text: str) -> list[dict]:
    """Parse wrapper and legacy backfill lifecycle lines without filesystem I/O."""
    events: list[dict] = []
    pending_legacy: dict | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = _BACKFILL_WRAPPER_RE.match(line)
        if match:
            item = match.groupdict()
            item["event"] = item["event"].lower()
            item["price_mode"] = item["price_mode"].lower()
            item["exit_code"] = int(item["exit_code"]) if item.get("exit_code") else None
            item["timestamp"] = _parse_datetime(item["timestamp"])
            events.append(item)
            continue
        match = _LEGACY_BACKFILL_START_RE.match(line)
        if match:
            item = match.groupdict()
            item.update({
                "event": "start",
                "price_mode": (item.get("price_mode") or "adjusted").lower(),
                "exit_code": None,
                "timestamp": _parse_datetime(item["timestamp"]),
            })
            events.append(item)
            pending_legacy = item
            continue
        match = _LEGACY_BACKFILL_OK_RE.match(line)
        if match and pending_legacy:
            events.append({
                **{k: pending_legacy[k] for k in ("market", "interval", "price_mode")},
                "event": "ok",
                "timestamp": _parse_datetime(match.group("timestamp")),
                "exit_code": 0,
            })
            pending_legacy = None
    return events


def evaluate_backfill_log_events(
    events: list[dict],
    *,
    now: datetime,
    max_success_age_seconds: float,
    max_incomplete_seconds: float,
) -> list[dict]:
    """Classify stale/failed/incomplete backfill lifecycle events."""
    now = _normalize_utc(now)
    latest: dict[tuple[str, str, str], dict[str, datetime | None]] = {}
    for event in events:
        ts = event.get("timestamp")
        if not isinstance(ts, datetime):
            continue
        key = (str(event["market"]), str(event["interval"]), str(event["price_mode"]))
        state = latest.setdefault(key, {"start": None, "success": None, "failure": None})
        kind = str(event.get("event", "")).lower()
        if kind == "start":
            state["start"] = ts
        elif kind == "ok":
            state["success"] = max(ts, state["success"]) if state["success"] else ts
        elif kind == "fail":
            state["failure"] = max(ts, state["failure"]) if state["failure"] else ts

    issues: list[dict] = []
    for (market, interval, price_mode), state in sorted(latest.items()):
        start = state["start"]
        success = state["success"]
        failure = state["failure"]
        if failure and (not success or failure > success):
            issues.append({
                "severity": "critical",
                "check": "backfill_log_stale",
                "status": "failed",
                "market": market,
                "interval": interval,
                "price_mode": price_mode,
                "failure_at": failure.isoformat(),
            })
            continue
        if start and (not success or start > success):
            age = (now - start).total_seconds()
            if age > max_incomplete_seconds:
                issues.append({
                    "severity": "critical",
                    "check": "backfill_log_stale",
                    "status": "incomplete",
                    "market": market,
                    "interval": interval,
                    "price_mode": price_mode,
                    "started_at": start.isoformat(),
                    "age_seconds": round(age, 1),
                })
            continue
        if success:
            age = (now - success).total_seconds()
            if age > max_success_age_seconds:
                issues.append({
                    "severity": "warning",
                    "check": "backfill_log_stale",
                    "status": "stale_success",
                    "market": market,
                    "interval": interval,
                    "price_mode": price_mode,
                    "last_success_at": success.isoformat(),
                    "age_seconds": round(age, 1),
                })
    return issues


def parse_process_rows(text: str) -> list[dict]:
    """Parse ``ps -eo pid=,etimes=,args=`` output without shell coupling."""
    rows: list[dict] = []
    for line in text.splitlines():
        parts = line.strip().split(maxsplit=2)
        if len(parts) != 3:
            continue
        try:
            rows.append({"pid": int(parts[0]), "elapsed_seconds": int(parts[1]), "command": parts[2]})
        except ValueError:
            continue
    return rows


def evaluate_backfill_processes(
    rows: list[dict], *, max_runtime_seconds: float
) -> list[dict]:
    issues: list[dict] = []
    for row in rows:
        command = str(row.get("command", ""))
        elapsed = int(row.get("elapsed_seconds", 0) or 0)
        if "scripts.backfill_prices" not in command or elapsed <= max_runtime_seconds:
            continue
        issues.append({
            "severity": "critical",
            "check": "backfill_process_stale",
            "pid": int(row["pid"]),
            "elapsed_seconds": elapsed,
            "command": command[:500],
        })
    return issues


def check_backfill_runtime(
    *,
    log_path: Path = BACKFILL_LOG,
    now: datetime | None = None,
    max_success_age_seconds: float = 36 * 3600,
    max_incomplete_seconds: float = 30 * 60,
    max_process_runtime_seconds: float = 30 * 60,
) -> list[dict]:
    """Read current process/log state and delegate classifications to pure functions."""
    now = now or datetime.now(timezone.utc)
    issues: list[dict] = []
    try:
        text = log_path.read_text(errors="replace")
    except FileNotFoundError:
        issues.append({
            "severity": "warning",
            "check": "backfill_log_stale",
            "status": "missing_log",
            "path": str(log_path),
        })
    else:
        issues.extend(evaluate_backfill_log_events(
            parse_backfill_log_events(text),
            now=now,
            max_success_age_seconds=max_success_age_seconds,
            max_incomplete_seconds=max_incomplete_seconds,
        ))
    try:
        proc = subprocess.run(
            ["ps", "-eo", "pid=,etimes=,args="],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
        if proc.returncode == 0:
            issues.extend(evaluate_backfill_processes(
                parse_process_rows(proc.stdout),
                max_runtime_seconds=max_process_runtime_seconds,
            ))
        else:
            issues.append({
                "severity": "warning",
                "check": "backfill_process_probe",
                "exit_code": proc.returncode,
                "detail": proc.stderr[-500:],
            })
    except Exception as exc:
        issues.append({
            "severity": "warning",
            "check": "backfill_process_probe",
            "detail": repr(exc),
        })
    return issues


def run_ledger(market: str) -> list[dict]:
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "ledger_watchdog.py"), "--market", market, "--history-days", "0", "--quiet-ok", "--fail-on-critical"]
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True, timeout=180)
    if proc.returncode == 0:
        return []
    return [{
        "severity": "critical",
        "check": "ledger_watchdog",
        "market": market,
        "exit_code": proc.returncode,
        "output_tail": (proc.stdout + proc.stderr)[-2000:],
    }]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    ap.add_argument("--skip-quotes", action="store_true", help="Skip live CN quote probe")
    ap.add_argument("--skip-ledger", action="store_true", help="Skip ledger watchdog subprocesses")
    ap.add_argument("--skip-price-health", action="store_true", help="Skip adjusted/raw 1d freshness and coverage checks")
    ap.add_argument("--skip-backfill-runtime", action="store_true", help="Skip backfill log/process stale checks")
    args = ap.parse_args()

    issues: list[dict] = []
    with _conn() as con:
        issues.extend(check_schema(con))
        issues.extend(check_alpha20(con))
        issues.extend(check_model_factor_freshness(con))
        issues.extend(check_snapshot_diff(con))
        if not args.skip_price_health:
            from config.settings import CN_UNIVERSE, STOCK_UNIVERSE
            from data.fetcher import latest_completed_session_date

            now = datetime.now(timezone.utc)
            targets = {
                market: latest_completed_session_date(market, now).isoformat()
                for market in ("US", "CN")
            }
            issues.extend(check_price_1d_health(
                con,
                universe_by_market={"US": STOCK_UNIVERSE, "CN": CN_UNIVERSE},
                target_dates=targets,
            ))
    if not args.skip_backfill_runtime:
        issues.extend(check_backfill_runtime())
    if not args.skip_quotes:
        issues.extend(check_cn_quotes())
    if not args.skip_ledger:
        issues.extend(run_ledger("US"))
        issues.extend(run_ledger("CN"))

    result = {
        "ok": not any(i["severity"] == "critical" for i in issues),
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "issues": issues,
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        if result["ok"] and not issues:
            print("OK: quant health checks passed")
        elif result["ok"]:
            print(f"WARN: {len(issues)} non-critical issue(s)")
        else:
            print(f"FAIL: {sum(i['severity']=='critical' for i in issues)} critical, {sum(i['severity']=='warning' for i in issues)} warning")
        for i in issues:
            print(json.dumps(i, ensure_ascii=False))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
