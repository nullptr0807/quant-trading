#!/usr/bin/env bash
# Market-isolated, bounded factor refresh; no trading/backfill side effects.
set -euo pipefail
umask 077
MARKET=${1:-}
case "$MARKET" in US|CN) ;; *) echo "usage: $0 US|CN [--alpha-only]" >&2; exit 2;; esac
shift
ROOT=${QUANT_PROJECT_ROOT:-/home/gexin/quant-trading}
PYTHON=${QUANT_PYTHON:-$ROOT/venv/bin/python}
LOG=${QUANT_FACTOR_LOG:-$ROOT/logs/factor_refresh.log}
TIMEOUT=${QUANT_FACTOR_TIMEOUT_SECONDS:-1800}
mkdir -p "$(dirname "$LOG")"
# A heartbeat marks scheduler invocation even when the market lock cannot be acquired.
echo "FACTOR_HEARTBEAT market=$MARKET scheduled=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$LOG"
exec bash "$ROOT/scripts/run_scheduled_job.sh" factor_refresh "$MARKET" "/tmp/quant_factor_refresh_${MARKET}.lock" 30 "$TIMEOUT" "$LOG" -- \
  "$PYTHON" "$ROOT/scripts/run_module_force_exit.py" scripts.refresh_factors --market "$MARKET" "$@"
