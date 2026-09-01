"""Daily Qlib retrain orchestrator.

Trains 10 different models sequentially in isolated subprocesses (so each
release its memory before the next starts — VM only has 4GB RAM).

Schedule: cron @ 23:00 UTC = 09:00 Sydney (AEST +10) = post-US-close,
pre-CN-open. ~5h window before next US session.

Usage:
    python -m scripts.qlib_retrain                      # all 10 models, US
    python -m scripts.qlib_retrain --models Q01,Q02     # subset
    python -m scripts.qlib_retrain --market CN          # CN mirror
    python -m scripts.qlib_retrain --skip-export        # reuse existing qlib bin

Pipeline:
    1. (optional) qlib_export.py — refresh ~/.qlib/qlib_data/<market>_data
    2. For each model: subprocess `python -m factors.qlib_signal --model Qxx`
    3. Per-model log + summary; failures logged but don't abort the batch.
    4. Final summary printed; written to /tmp/qlib_retrain_<date>.log.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = os.path.expanduser("~/quant-trading")
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("qlib_retrain")


def run_export(market: str, db_path: str) -> int:
    log.info("[export] refreshing qlib bin for %s", market)
    t0 = time.time()
    env = os.environ.copy()
    env["QUANT_DB_PATH"] = str(Path(db_path).expanduser().resolve())
    r = subprocess.run(
        [sys.executable, "-m", "factors.qlib_export", "--market", market],
        cwd=PROJECT_ROOT, capture_output=True, text=True, env=env,
    )
    elapsed = time.time() - t0
    if r.returncode != 0:
        log.error("[export] FAILED in %.1fs: %s", elapsed, r.stderr[-1000:])
        return r.returncode
    # Echo the last few lines of export log (summary)
    for line in r.stdout.strip().splitlines()[-5:]:
        log.info("  %s", line)
    log.info("[export] OK in %.1fs", elapsed)
    return 0


def run_one_model(model_id: str, market: str,
                  train_days: int, valid_days: int, predict_days: int,
                  log_dir: Path, db_path: str,
                  model_timeout_seconds: int | None = None) -> dict:
    """Run one model in a subprocess. Stream stdout to log file."""
    log.info("[%s] training... (subprocess, timeout=%s)",
             model_id, model_timeout_seconds or "none")
    t0 = time.time()
    log_file = log_dir / f"{model_id}.log"

    cmd = [
        sys.executable, "-m", "factors.qlib_signal",
        "--model", model_id,
        "--market", market,
        "--train-days", str(train_days),
        "--valid-days", str(valid_days),
        "--predict-days", str(predict_days),
    ]
    returncode = 0
    timed_out = False
    with log_file.open("w") as lf:
        lf.write(f"# {datetime.now(timezone.utc).isoformat()}\n")
        lf.write(f"# cmd: {' '.join(cmd)}\n")
        lf.write(f"# timeout_seconds: {model_timeout_seconds or 'none'}\n")
        lf.flush()
        try:
            env = os.environ.copy()
            env["QUANT_DB_PATH"] = str(Path(db_path).expanduser().resolve())
            r = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=lf,
                stderr=subprocess.STDOUT,
                timeout=model_timeout_seconds,
                env=env,
            )
            returncode = r.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = 124
            lf.write(
                f"\n# TIMEOUT after {model_timeout_seconds}s at "
                f"{datetime.now(timezone.utc).isoformat()}\n"
            )
            lf.flush()
    elapsed = time.time() - t0

    summary = {
        "model_id": model_id,
        "elapsed_s": round(elapsed, 1),
        "exit_code": returncode,
        "timed_out": timed_out,
        "log_file": str(log_file),
    }
    if returncode != 0:
        # Tail log for inline error visibility
        tail = log_file.read_text().splitlines()[-15:]
        log.error("[%s] FAILED in %.1fs (exit=%d%s). Tail:\n  %s",
                  model_id, elapsed, returncode,
                  ", timeout" if timed_out else "", "\n  ".join(tail))
    else:
        log.info("[%s] OK in %.1fs", model_id, elapsed)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default=None,
                   help="comma-separated model ids (default: all Q01..Q10)")
    p.add_argument("--market", default="US", choices=["US", "CN"])
    p.add_argument("--db", default=os.environ.get("QUANT_DB_PATH", "data/trading.db"),
                   help="trading DB used for market-scoped active-account selection")
    p.add_argument("--model-manifest", default="")
    p.add_argument("--train-days", type=int, default=360)
    p.add_argument("--valid-days", type=int, default=60)
    p.add_argument("--predict-days", type=int, default=5)
    p.add_argument("--model-timeout-seconds", type=int,
                   default=int(os.environ.get("QLIB_MODEL_TIMEOUT_SECONDS", "10800")),
                   help="per-model subprocess timeout; default 10800s (3h)")
    p.add_argument("--skip-export", action="store_true",
                   help="reuse existing ~/.qlib/qlib_data/<market>_data dir")
    args = p.parse_args()

    from factors.qlib_signal import MODEL_SPECS
    from scripts.qlib_model_selection import select_qlib_model_ids
    available = [s.id for s in MODEL_SPECS]
    requested = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None
    ids = select_qlib_model_ids(args.db, args.market, available, requested)
    if args.model_manifest:
        Path(args.model_manifest).write_text(json.dumps({
            "market": args.market,
            "db": str(Path(args.db).expanduser().resolve()),
            "models": ids,
            "selected_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2))
    log.info("retrain plan: market=%s models=%s", args.market, ids)

    # Per-day log dir
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    log_dir = Path(f"/tmp/qlib_retrain_{args.market}_{today}")
    log_dir.mkdir(parents=True, exist_ok=True)
    log.info("per-model logs: %s", log_dir)

    if not ids:
        summary_path = log_dir / "summary.json"
        summary_path.write_text(json.dumps({
            "market": args.market,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "n_ok": 0, "n_total": 0, "batch_elapsed_s": 0.0,
            "models": [], "skipped": "no_active_qlib_accounts",
        }, indent=2))
        log.info("no active Qlib accounts for %s; export/training skipped", args.market)
        log.info("summary: %s", summary_path)
        raise SystemExit(3)

    # 1. Export
    if not args.skip_export:
        rc = run_export(args.market, args.db)
        if rc != 0:
            log.error("export failed, aborting batch")
            sys.exit(1)
    else:
        log.info("[export] SKIPPED (reuse existing bin)")

    # 2. Per-model subprocesses (sequential — memory isolation)
    t_batch = time.time()
    summaries = []
    for mid in ids:
        s = run_one_model(
            mid, args.market,
            args.train_days, args.valid_days, args.predict_days,
            log_dir, args.db,
            args.model_timeout_seconds,
        )
        summaries.append(s)

    # 3. Print summary table
    batch_elapsed = time.time() - t_batch
    n_ok = sum(1 for s in summaries if s["exit_code"] == 0)
    log.info("=" * 60)
    log.info("BATCH SUMMARY: %d/%d ok, total %.1fmin",
             n_ok, len(summaries), batch_elapsed / 60)
    for s in summaries:
        flag = "✓" if s["exit_code"] == 0 else "✗"
        log.info("  %s %s  %5.1fs  %s", flag, s["model_id"], s["elapsed_s"], s["log_file"])

    # 4. Save JSON summary for downstream consumers
    summary_path = log_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "market": args.market,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "n_ok": n_ok, "n_total": len(summaries),
        "batch_elapsed_s": round(batch_elapsed, 1),
        "models": summaries,
    }, indent=2))
    log.info("summary: %s", summary_path)

    sys.exit(0 if n_ok == len(summaries) else 2)


if __name__ == "__main__":
    main()
