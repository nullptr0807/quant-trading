#!/usr/bin/env python3
"""Fail-closed summary gate for the read-only corporate-action audit.

This module deliberately has no position-adjustment path. It invokes the existing
read-only audit, evaluates its structured summary, and returns non-zero when an
open position crossed a split/bonus event or provider fetch coverage is below the
configured threshold.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def evaluate_audit_result(
    result: dict[str, Any], *, min_fetch_coverage: float = 1.0
) -> dict[str, Any]:
    """Return a deterministic alert/exit decision from an audit result."""
    summary = result.get("summary") or {}
    scanned = max(int(summary.get("tickers_scanned", 0) or 0), 0)
    fetch_errors = max(int(summary.get("fetch_errors", 0) or 0), 0)
    critical_actions = max(
        int(summary.get("critical_open_share_actions", 0) or 0), 0
    )
    # tickers_scanned is the requested ticker count. Fetch errors are a subset of
    # it, not extra attempts.
    attempted = scanned
    successful = max(scanned - fetch_errors, 0)
    fetch_coverage = successful / attempted if attempted else 0.0

    issues: list[dict[str, Any]] = []
    if critical_actions:
        issues.append({
            "severity": "critical",
            "check": "open_position_share_action",
            "count": critical_actions,
            "detail": "open position crossed split/bonus action; manual ledger review required",
        })
    if fetch_errors or fetch_coverage < min_fetch_coverage:
        issues.append({
            "severity": "critical",
            "check": "fetch_coverage",
            "tickers_scanned": scanned,
            "fetch_errors": fetch_errors,
            "coverage": round(fetch_coverage, 6),
            "minimum": min_fetch_coverage,
        })
    if attempted == 0:
        issues.append({
            "severity": "critical",
            "check": "fetch_coverage",
            "tickers_scanned": scanned,
            "fetch_errors": fetch_errors,
            "coverage": 0.0,
            "minimum": min_fetch_coverage,
            "detail": "audit scanned no tickers",
        })

    ok = not issues
    return {
        "ok": ok,
        "exit_code": 0 if ok else 2,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "summary": summary,
        "paths": result.get("paths") or {},
        "issues": issues,
    }


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    """Run the existing network-backed audit; the shell wrapper owns OS bounds."""
    from scripts.audit_corporate_actions import run_audit

    markets = {x.strip().upper() for x in args.markets.split(",") if x.strip()}
    focus = {x.strip() for x in args.accounts.split(",") if x.strip()} or None
    result = run_audit(markets, focus, args.start, args.end, args.max_tickers)
    return evaluate_audit_result(
        result, min_fetch_coverage=args.min_fetch_coverage
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only, fail-closed corporate-action audit summary gate"
    )
    parser.add_argument("--markets", default="US,CN")
    parser.add_argument("--accounts", default="")
    parser.add_argument("--start", required=True)
    parser.add_argument(
        "--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d")
    )
    parser.add_argument("--max-tickers", type=int, default=None)
    parser.add_argument("--min-fetch-coverage", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    result = run_check(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
