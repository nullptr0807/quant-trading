#!/usr/bin/env bash
# Intra-hour rebalance trigger. Read-only persisted-signal preparation happens
# outside the shared writer lock; the bounded quote/trade phase is locked.
set -euo pipefail

cd /home/gexin/quant-trading
DOW=$(date -u +%u)
if [ "$DOW" -gt 5 ]; then exit 0; fi

MARKET="US"
if [ "${1:-}" = "--market" ]; then
    MARKET="${2:-}"
fi
case "$MARKET" in US|CN) ;; *) echo "Invalid market: $MARKET" >&2; exit 2 ;; esac

PYTHON=/home/gexin/quant-trading/venv/bin/python
PREPARED="/tmp/quant_fast_cycle_${MARKET}_$$.json"
trap 'rm -f "$PREPARED" "$PREPARED.tmp"' EXIT
PREPARE_TIMEOUT_SECONDS=${QUANT_FAST_PREPARE_TIMEOUT_SECONDS:-90}
TRADE_TIMEOUT_SECONDS=${QUANT_RUN_CYCLE_TIMEOUT_SECONDS:-120}

/usr/bin/timeout --kill-after=15s "${PREPARE_TIMEOUT_SECONDS}s" \
    "$PYTHON" -m scripts.prepare_fast_cycle --market "$MARKET" --output "$PREPARED" \
    >> /home/gexin/quant-trading/logs/cron.log 2>&1

export QUANT_FAST_PREPARED_PATH="$PREPARED"
ROOT=${QUANT_PROJECT_ROOT:-/home/gexin/quant-trading}
exec bash "$ROOT/scripts/run_scheduled_job.sh" run_cycle_quiet "$MARKET" /tmp/quant_run_cycle.lock 0 "$TRADE_TIMEOUT_SECONDS" \
    "$ROOT/logs/cron.log" -- "$PYTHON" main.py --fast-cycle --market "$MARKET"
