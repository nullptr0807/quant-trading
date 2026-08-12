#!/usr/bin/env bash
# Usage: run_scheduled_job.sh COMPONENT MARKET LOCK WAIT_SECONDS TIMEOUT_SECONDS LOG -- command...
set -uo pipefail
umask 077
[[ $# -ge 8 && "$7" == "--" ]] || { echo "usage: $0 component market lock wait timeout log -- command..." >&2; exit 2; }
COMPONENT=$1; MARKET=$2; LOCK=$3; WAIT=$4; TIMEOUT=$5; LOG=$6; shift 7
ROOT=${QUANT_PROJECT_ROOT:-/home/gexin/quant-trading}
PYTHON=${QUANT_PYTHON:-$ROOT/venv/bin/python}
mkdir -p "$(dirname "$LOG")"
SCHEDULED=$(date -u +%Y-%m-%dT%H:%M:%SZ)
record() { "$PYTHON" "$ROOT/scripts/record_operational_health.py" --db "$ROOT/data/trading.db" --component "$COMPONENT" --market "$MARKET" --scheduled-at "$SCHEDULED" "$@" >>"$LOG" 2>&1 || echo "scheduler health write failed component=$COMPONENT" >>"$LOG"; }
alert() {
  local text=$1 result="disabled"
  if [[ -n "${TELEGRAM_BOT_TOKEN:-}" && -n "${TELEGRAM_CHAT_ID:-}" ]]; then
    if curl -fsS --max-time 10 -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" -d "chat_id=${TELEGRAM_CHAT_ID}" --data-urlencode "text=$text" >/dev/null; then result=ok; else result=failed; fi
  fi
  echo "ALERT_DELIVERY component=$COMPONENT result=$result at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$LOG"
}
exec 9>"$LOCK"
if ! /usr/bin/flock -w "$WAIT" 9; then
  NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  echo "SCHEDULER_LOCK_TIMEOUT component=$COMPONENT market=$MARKET scheduled=$SCHEDULED at=$NOW wait=${WAIT}s" >>"$LOG"
  record --status lock_timeout --stopped-at "$NOW" --exit-code 75 --detail '{"reason":"flock_timeout"}'
  alert "⚠️ Quant $COMPONENT/$MARKET lock timeout on $(hostname)"
  exit 75
fi
START=$(date -u +%Y-%m-%dT%H:%M:%SZ); T0=$(date +%s)
echo "SCHEDULER_START component=$COMPONENT market=$MARKET scheduled=$SCHEDULED actual=$START" >>"$LOG"
/usr/bin/timeout --kill-after=30s "${TIMEOUT}s" "$@" >>"$LOG" 2>&1
RC=$?
STOP=$(date -u +%Y-%m-%dT%H:%M:%SZ); T1=$(date +%s); DURATION=$((T1-T0))
if [[ $RC -eq 0 ]]; then STATUS=ok; else STATUS=failed; fi
echo "SCHEDULER_STOP component=$COMPONENT market=$MARKET exit=$RC duration=${DURATION}s at=$STOP status=$STATUS" >>"$LOG"
record --status "$STATUS" --started-at "$START" --stopped-at "$STOP" --exit-code "$RC" --duration "$DURATION"
if [[ $RC -ne 0 ]]; then alert "⚠️ Quant $COMPONENT/$MARKET failed on $(hostname), exit=$RC"; fi
exit "$RC"
