#!/usr/bin/env bash
# Bounded market-wide writer lock: adjusted/raw and intervals never write concurrently.
set -euo pipefail
umask 077
ROOT=${QUANT_PROJECT_ROOT:-/home/gexin/quant-trading}
cd "$ROOT"
MARKET=${1:-}; INTERVAL=${2:-}; DAYS=${3:-}; PRICE_MODE=${4:-adjusted}; SCOPE=${5:-universe}
case "$MARKET" in US|CN) ;; *) echo "usage: $0 US|CN 1d|1h|15m|5m DAYS adjusted|raw|both universe|ledger" >&2; exit 2;; esac
case "$INTERVAL" in 1d|1h|15m|5m) ;; *) echo "invalid interval: $INTERVAL" >&2; exit 2;; esac
case "$PRICE_MODE" in adjusted|raw|both) ;; *) echo "invalid price mode: $PRICE_MODE" >&2; exit 2;; esac
case "$SCOPE" in universe|ledger) ;; *) echo "invalid scope: $SCOPE" >&2; exit 2;; esac
[[ "$DAYS" =~ ^[0-9]+$ ]] || { echo "invalid days: $DAYS" >&2; exit 2; }
PYTHON=${QUANT_PYTHON:-$ROOT/venv/bin/python}; LOG=${QUANT_BACKFILL_LOG:-$ROOT/logs/backfill.log}
LOCK=${QUANT_BACKFILL_LOCK:-/tmp/quant_backfill_${MARKET}.lock}
WAIT=${QUANT_BACKFILL_LOCK_WAIT_SECONDS:-30}; TIMEOUT=${QUANT_BACKFILL_TIMEOUT_SECONDS:-900}
mkdir -p "$(dirname "$LOG")"; scheduled=$(date -u +%Y-%m-%dT%H:%M:%SZ)
exec 9>"$LOCK"
if ! /usr/bin/flock -w "$WAIT" 9; then
  end=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "===== Backfill [$MARKET/$INTERVAL/$PRICE_MODE/$SCOPE] FAIL $end exit=75 reason=lock_timeout scheduled=$scheduled =====" >>"$LOG"
  "$PYTHON" scripts/record_operational_health.py --component "backfill_${INTERVAL}_${PRICE_MODE}_${SCOPE}" --market "$MARKET" --status lock_timeout --scheduled-at "$scheduled" --stopped-at "$end" --exit-code 75 --detail '{"reason":"market_writer_lock_timeout"}' >>"$LOG" 2>&1 || true
  exit 75
fi
start=$(date -u +%Y-%m-%dT%H:%M:%SZ); t0=$(date +%s)
echo "===== Backfill [$MARKET/$INTERVAL/$PRICE_MODE/$SCOPE] start $start days=$DAYS =====" >>"$LOG"
set +e
/usr/bin/timeout --kill-after=30s "${TIMEOUT}s" "$PYTHON" "$ROOT/scripts/run_module_force_exit.py" scripts.backfill_prices --market "$MARKET" --interval "$INTERVAL" --days "$DAYS" --price-mode "$PRICE_MODE" --scope "$SCOPE" >>"$LOG" 2>&1
rc=$?
set -e
end=$(date -u +%Y-%m-%dT%H:%M:%SZ); duration=$(($(date +%s)-t0))
if [[ $rc -eq 0 ]]; then event=OK; status=ok; else event=FAIL; status=failed; fi
echo "===== Backfill [$MARKET/$INTERVAL/$PRICE_MODE/$SCOPE] $event $end exit=$rc duration=${duration}s =====" >>"$LOG"
"$PYTHON" scripts/record_operational_health.py --component "backfill_${INTERVAL}_${PRICE_MODE}_${SCOPE}" --market "$MARKET" --status "$status" --scheduled-at "$scheduled" --started-at "$start" --stopped-at "$end" --exit-code "$rc" --duration "$duration" >>"$LOG" 2>&1 || true
exit "$rc"
