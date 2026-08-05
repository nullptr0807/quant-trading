#!/bin/bash
# Daily Qlib model retrain wrapper.
# Runs one market per invocation; cron schedules US/CN separately so a slow US
# Transformer cannot make CN scores stale before the next CN session.
set -euo pipefail

cd /home/gexin/quant-trading
source venv/bin/activate
# Best-effort alert credentials for failure notifications. Missing env is OK;
# the wrapper still logs and exits non-zero for cron/systemd visibility.
if [ -f "$HOME/.hermes/.env" ]; then
    set -a
    # shellcheck source=/dev/null
    source "$HOME/.hermes/.env"
    set +a
fi

LOG_DIR=/home/gexin/quant-trading/logs
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

# Separate market locks prevent same-market overlap but allow CN/US maintenance
# to be scheduled independently if needed.
LOCK="/tmp/quant_qlib_retrain_${MARKET}.lock"
if [ "${QUANT_QLIB_ALREADY_LOCKED:-0}" != "1" ]; then
    export QUANT_QLIB_ALREADY_LOCKED=1
    lock_args=("$0" --market "$MARKET")
    if [ -n "$MODELS" ]; then
        lock_args+=(--models "$MODELS")
    fi
    exec /usr/bin/flock -n "$LOCK" "${lock_args[@]}"
fi

run_args=(--market "$MARKET" --model-timeout-seconds "${QLIB_MODEL_TIMEOUT_SECONDS:-10800}")
if [ -n "$MODELS" ]; then
    run_args+=(--models "$MODELS")
fi

verify_args=(--market "$MARKET")
if [ -n "$MODELS" ]; then
    verify_args+=(--models "$MODELS")
fi

echo "===== Qlib retrain [$MARKET] start $(date -u +%Y-%m-%dT%H:%M:%SZ) args=${run_args[*]} =====" >> "$LOG_DIR/qlib_retrain.log"
if python -m scripts.qlib_retrain "${run_args[@]}" >> "$LOG_DIR/qlib_retrain.log" 2>&1 \
   && python -m scripts.verify_qlib_scores "${verify_args[@]}" >> "$LOG_DIR/qlib_retrain.log" 2>&1; then
    echo "===== Qlib retrain [$MARKET] OK    $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" >> "$LOG_DIR/qlib_retrain.log"
else
    echo "===== Qlib retrain [$MARKET] FAIL  $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" >> "$LOG_DIR/qlib_retrain.log"
    if [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID:-}" ]; then
        curl -fsS -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
            -d "chat_id=${TELEGRAM_CHAT_ID}" \
            --data-urlencode "text=⚠️ Qlib retrain [$MARKET] failed on $(hostname). Check $LOG_DIR/qlib_retrain.log" \
            >/dev/null 2>&1 || true
    fi
    exit 1
fi
