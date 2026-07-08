#!/usr/bin/env python3
"""Quant system health checks for prices, factors, quotes, snapshots, and ledger.

Read-only by default. Exits non-zero on critical issues so cron/CI can alert.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
DB_PATH = PROJECT_ROOT / "data" / "trading.db"
CN_SUFFIXES = (".SH", ".SZ", ".BJ")
ROLL20_FACTORS = [
    "ROC_20", "MA_RATIO_20", "VMOM_20", "VSTD_20", "STD_20", "BBPOS_20", "BETA_20",
]


def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=30000")
    return con


def _is_cn(ticker: str) -> bool:
    return ticker.upper().endswith(CN_SUFFIXES)


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
    if market == "CN":
        clause = "ticker LIKE '%.SH' OR ticker LIKE '%.SZ' OR ticker LIKE '%.BJ'"
    else:
        clause = "ticker NOT LIKE '%.SH' AND ticker NOT LIKE '%.SZ' AND ticker NOT LIKE '%.BJ'"
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
            elif lag is not None and lag > 3:
                issues.append({
                    "severity": "critical" if lag > 7 else "warning",
                    "check": "qlib_factor",
                    "account": acct,
                    "factor_date": max_date,
                    "latest_price_date": latest_px,
                    "lag_days": lag,
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


def run_ledger(market: str) -> list[dict]:
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "ledger_watchdog.py"), "--market", market, "--history-days", "0", "--quiet-ok"]
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
    args = ap.parse_args()

    issues: list[dict] = []
    with _conn() as con:
        issues.extend(check_schema(con))
        issues.extend(check_alpha20(con))
        issues.extend(check_model_factor_freshness(con))
        issues.extend(check_snapshot_diff(con))
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
