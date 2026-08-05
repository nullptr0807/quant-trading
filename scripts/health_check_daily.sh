#!/usr/bin/env bash
# Daily quant-system health check wrapper.
# Runs read-only checks and alerts on critical failures.
set -euo pipefail

cd /home/gexin/quant-trading
source venv/bin/activate

# Best-effort alert credentials. Missing env is OK; the wrapper still exits
# non-zero so cron/systemd can detect failure.
if [ -f "$HOME/.hermes/.env" ]; then
    set -a
    # shellcheck source=/dev/null
    source "$HOME/.hermes/.env"
    set +a
fi

LOG_DIR=/home/gexin/quant-trading/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/health_check.log"
LOCK=/tmp/quant_health_check.lock

if [ "${QUANT_HEALTH_ALREADY_LOCKED:-0}" != "1" ]; then
    export QUANT_HEALTH_ALREADY_LOCKED=1
    exec /usr/bin/flock -n "$LOCK" "$0" "$@"
fi

stamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "===== Quant health start $stamp =====" >> "$LOG"

tmp_json="$(mktemp /tmp/quant-health.XXXXXX.json)"
trap 'rm -f "$tmp_json"' EXIT
set +e
/usr/bin/timeout --kill-after=15s "${QUANT_HEALTH_TIMEOUT_SECONDS:-600}s" \
    python scripts/health_check.py --json >"$tmp_json" 2>>"$LOG"
rc=$?
set -e
cat "$tmp_json" >>"$LOG"
transition_msg="$(python scripts/classify_health_alerts.py \
    --input "$tmp_json" --state data/health_alert_state.json 2>>"$LOG" || true)"
if [ "$rc" -ne 0 ] && [ -z "$transition_msg" ]; then
    transition_msg="health_check_process_failed exit=$rc"
fi

end="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ "$rc" -eq 0 ]; then
    echo "===== Quant health OK    $end =====" >> "$LOG"
else
    echo "===== Quant health FAIL  $end exit=$rc =====" >> "$LOG"
fi

if [ -n "$transition_msg" ] && [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID:-}" ]; then
    if curl -fsS --max-time 15 -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d "chat_id=${TELEGRAM_CHAT_ID}" \
        --data-urlencode "text=Quant health transitions on $(hostname): ${transition_msg}" \
        >/dev/null 2>&1; then
        echo "ALERT_DELIVERY health result=ok at=$end" >>"$LOG"
    else
        echo "ALERT_DELIVERY health result=failed at=$end" >>"$LOG"
    fi
elif [ -n "$transition_msg" ]; then
    echo "ALERT_DELIVERY health result=disabled at=$end" >>"$LOG"
fi
exit "$rc"
