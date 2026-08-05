#!/bin/bash
set -euo pipefail
cd /home/gexin/quant-trading

# Check if it's a weekday (Mon=1..Fri=5)
DOW=$(date -u +%u)
if [ "$DOW" -gt 5 ]; then
    echo "Weekend, skipping"
    exit 0
fi

MARKET="US"
DRY_RUN=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --market) MARKET="${2:-}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
case "$MARKET" in US|CN) ;; *) echo "Invalid market: $MARKET" >&2; exit 2 ;; esac

PYTHON=/home/gexin/quant-trading/venv/bin/python
PREPARED="/tmp/quant_fast_cycle_${MARKET}_$$.json"
trap 'rm -f "$PREPARED" "$PREPARED.tmp"' EXIT
PREPARE_TIMEOUT_SECONDS=${QUANT_FAST_PREPARE_TIMEOUT_SECONDS:-90}
TRADE_TIMEOUT_SECONDS=${QUANT_RUN_CYCLE_TIMEOUT_SECONDS:-120}

# Expensive read-only DB ranking happens OUTSIDE the shared writer lock, so the
# per-minute updater can continue quote/stop-loss work. Only bounded quote fetch,
# trades and state writes are inside /tmp/quant_run_cycle.lock.
/usr/bin/timeout --kill-after=15s "${PREPARE_TIMEOUT_SECONDS}s" \
    "$PYTHON" -m scripts.prepare_fast_cycle --market "$MARKET" --output "$PREPARED"

export QUANT_FAST_PREPARED_PATH="$PREPARED"
MAIN_ARGS=(main.py --fast-cycle --market "$MARKET")
if [ "$DRY_RUN" -eq 1 ]; then MAIN_ARGS+=(--dry-run); fi
ROOT=${QUANT_PROJECT_ROOT:-/home/gexin/quant-trading}
exec bash "$ROOT/scripts/run_scheduled_job.sh" run_cycle "$MARKET" /tmp/quant_run_cycle.lock 30 "$TRADE_TIMEOUT_SECONDS" \
    "$ROOT/logs/cron.log" -- "$PYTHON" "${MAIN_ARGS[@]}"
