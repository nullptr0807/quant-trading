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
WAIT=${QUANT_BACKFILL_LOCK_WAIT_SECONDS:-30}
# CN's per-ticker Sina/akshare daily endpoint is materially slower and can
# consume several bounded 120s partial-batch windows. Keep the outer timeout
# long enough for all 301 names while retaining a hard wall-clock limit.
if [[ "$MARKET" == "CN" ]]; then DEFAULT_TIMEOUT=1800; else DEFAULT_TIMEOUT=900; fi
TIMEOUT=${QUANT_BACKFILL_TIMEOUT_SECONDS:-$DEFAULT_TIMEOUT}
mkdir -p "$(dirname "$LOG")"; scheduled=$(date -u +%Y-%m-%dT%H:%M:%SZ)
exec 9>"$LOCK"
if ! /usr/bin/flock -w "$WAIT" 9; then
  end=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "===== Backfill [$MARKET/$INTERVAL/$PRICE_MODE/$SCOPE] FAIL $end exit=75 reason=lock_timeout scheduled=$scheduled =====" >>"$LOG"
  if ! "$PYTHON" scripts/record_operational_health.py --component "backfill_${INTERVAL}_${PRICE_MODE}_${SCOPE}" --market "$MARKET" --status lock_timeout --scheduled-at "$scheduled" --stopped-at "$end" --exit-code 75 --detail '{"reason":"market_writer_lock_timeout"}' >>"$LOG" 2>&1; then
    echo "HEALTH_WRITE_FAILED component=backfill_${INTERVAL}_${PRICE_MODE}_${SCOPE} market=$MARKET" >>"$LOG"
    exit 70
  fi
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
source_ts=""
detail='{}'
if [[ "$INTERVAL" == "1d" ]]; then
  set +e
  source_ts=$("$PYTHON" -c "from data.fetcher import latest_completed_session_date; print(latest_completed_session_date('$MARKET').isoformat())" 2>>"$LOG")
  source_rc=$?
  set -e
  if [[ $source_rc -ne 0 || -z "$source_ts" ]]; then
    status=failed
    [[ $rc -ne 0 ]] || rc=69
    source_ts=""
    detail='{"reason":"target_session_resolution_failed"}'
  else
    detail=$(printf '{"target_date":"%s"}' "$source_ts")
  fi
fi
health_args=(scripts/record_operational_health.py --component "backfill_${INTERVAL}_${PRICE_MODE}_${SCOPE}" --market "$MARKET" --status "$status" --scheduled-at "$scheduled" --started-at "$start" --stopped-at "$end" --exit-code "$rc" --duration "$duration" --detail "$detail")
if [[ -n "$source_ts" ]]; then health_args+=(--source-timestamp "$source_ts"); fi
if ! "$PYTHON" "${health_args[@]}" >>"$LOG" 2>&1; then
  echo "HEALTH_WRITE_FAILED component=backfill_${INTERVAL}_${PRICE_MODE}_${SCOPE} market=$MARKET child_exit=$rc" >>"$LOG"
  exit 70
fi
exit "$rc"
