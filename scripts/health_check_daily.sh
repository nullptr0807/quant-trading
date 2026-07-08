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

set +e
/usr/bin/timeout --kill-after=15s "${QUANT_HEALTH_TIMEOUT_SECONDS:-600}s" \
    python scripts/health_check.py --json >> "$LOG" 2>&1
rc=$?
set -e

end="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ "$rc" -eq 0 ]; then
    echo "===== Quant health OK    $end =====" >> "$LOG"
    exit 0
fi

echo "===== Quant health FAIL  $end exit=$rc =====" >> "$LOG"

if [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID:-}" ]; then
    tail_msg="$(tail -c 2500 "$LOG" 2>/dev/null || true)"
    curl -fsS -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d "chat_id=${TELEGRAM_CHAT_ID}" \
        --data-urlencode "text=⚠️ Quant health check failed on $(hostname) (exit=$rc). Tail:
${tail_msg}" \
        >/dev/null 2>&1 || true
fi

exit "$rc"
