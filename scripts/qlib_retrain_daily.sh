#!/bin/bash
# Daily Qlib model retrain wrapper with one durable lifecycle record per attempt.
set -euo pipefail

ROOT=${QUANT_PROJECT_ROOT:-/home/gexin/quant-trading}
PYTHON=${QUANT_PYTHON:-$ROOT/venv/bin/python}
DB=${QUANT_DB_PATH:-$ROOT/data/trading.db}
cd "$ROOT"
source "$ROOT/venv/bin/activate"
# Best-effort alert credentials for failure notifications. Missing env is OK;
# the wrapper still logs and exits non-zero for cron/systemd visibility.
if [ -f "$HOME/.hermes/.env" ]; then
    set -a
    # shellcheck source=/dev/null
    source "$HOME/.hermes/.env"
    set +a
fi

LOG_DIR=${QUANT_LOG_DIR:-$ROOT/logs}
LOG="$LOG_DIR/qlib_retrain.log"
mkdir -p "$LOG_DIR"

MARKET="US"
MODELS=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --market) MARKET="${2:-}"; shift 2 ;;
        --models) MODELS="${2:-}"; shift 2 ;;
        --locked) shift ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done
case "$MARKET" in US|CN) ;; *) echo "Invalid market: $MARKET" >&2; exit 2 ;; esac

scheduled="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
record_health() {
    local status=$1 rc=$2 detail=$3 started=$4 stopped=$5 duration=$6
    local health_rc
    "$PYTHON" scripts/record_operational_health.py \
        --db "$DB" --component qlib_retrain --market "$MARKET" \
        --status "$status" --scheduled-at "$scheduled" \
        --started-at "$started" --stopped-at "$stopped" \
        --exit-code "$rc" --duration "$duration" --detail "$detail" \
        >> "$LOG" 2>&1 && return 0
    health_rc=$?
    echo "qlib scheduler health write failed market=$MARKET exit=$health_rc" >> "$LOG"
    return "$health_rc"
}

# Separate market locks prevent same-market overlap but allow CN/US maintenance
# to be scheduled independently if needed.
LOCK=${QLIB_LOCK_PATH:-/tmp/quant_qlib_retrain_${MARKET}.lock}
LOCK_CONFLICT_RC=200
if [ "${QUANT_QLIB_ALREADY_LOCKED:-0}" != "1" ]; then
    lock_args=("$0" --market "$MARKET")
    if [ -n "$MODELS" ]; then
        lock_args+=(--models "$MODELS")
    fi
    exec 9>"$LOCK"
    set +e
    /usr/bin/flock -E "$LOCK_CONFLICT_RC" -w "${QLIB_LOCK_WAIT_SECONDS:-5}" \
        9
    rc=$?
    set -e
    if [ "$rc" -eq "$LOCK_CONFLICT_RC" ]; then
        now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "===== Qlib retrain [$MARKET] LOCK_TIMEOUT $now =====" >> "$LOG"
        record_health lock_timeout 75 '{"reason":"lock_timeout"}' "$now" "$now" 0 || true
        exit 75
    fi
    if [ "$rc" -ne 0 ]; then
        exit "$rc"
    fi
    export QUANT_QLIB_ALREADY_LOCKED=1
    exec "${lock_args[@]}"
fi

run_args=(--market "$MARKET" --model-timeout-seconds "${QLIB_MODEL_TIMEOUT_SECONDS:-10800}")
if [ -n "$MODELS" ]; then
    run_args+=(--models "$MODELS")
fi
verify_args=(--market "$MARKET")
if [ -n "$MODELS" ]; then
    verify_args+=(--models "$MODELS")
fi

start="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
t0=$(date +%s)
total_timeout=${QLIB_TOTAL_TIMEOUT_SECONDS:-14400}
verify_timeout=${QLIB_VERIFY_TIMEOUT_SECONDS:-600}
echo "===== Qlib retrain [$MARKET] start $start args=${run_args[*]} =====" >> "$LOG"
set +e
/usr/bin/timeout --kill-after=60s "${total_timeout}s" \
    "$PYTHON" -m scripts.qlib_retrain "${run_args[@]}" >> "$LOG" 2>&1
rc=$?
phase=retrain
if [ "$rc" -eq 0 ]; then
    phase=verify
    elapsed=$(($(date +%s)-t0))
    remaining=$((total_timeout-elapsed))
    if [ "$remaining" -le 0 ]; then
        rc=124
    else
        verify_budget=$verify_timeout
        if [ "$remaining" -lt "$verify_budget" ]; then
            verify_budget=$remaining
        fi
        /usr/bin/timeout --kill-after=15s "${verify_budget}s" \
            "$PYTHON" -m scripts.verify_qlib_scores "${verify_args[@]}" >> "$LOG" 2>&1
        rc=$?
    fi
fi
set -e
end="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
duration=$(($(date +%s)-t0))
if [ "$rc" -eq 0 ]; then
    if ! record_health ok 0 '{"phase":"verified"}' "$start" "$end" "$duration"; then
        echo "===== Qlib retrain [$MARKET] FAIL  $end exit=70 phase=health_write =====" >> "$LOG"
        exit 70
    fi
    echo "===== Qlib retrain [$MARKET] OK    $end =====" >> "$LOG"
    exit 0
fi

echo "===== Qlib retrain [$MARKET] FAIL  $end exit=$rc phase=$phase =====" >> "$LOG"
reason=retrain_or_verify_failed
if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    reason=timeout
fi
record_health failed "$rc" "{\"reason\":\"$reason\",\"phase\":\"$phase\"}" "$start" "$end" "$duration" || true
if [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID:-}" ]; then
    curl -fsS -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d "chat_id=${TELEGRAM_CHAT_ID}" \
        --data-urlencode "text=⚠️ Qlib retrain [$MARKET] failed on $(hostname). Check $LOG" \
        >/dev/null 2>&1 || true
fi
exit "$rc"
