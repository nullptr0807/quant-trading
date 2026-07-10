#!/usr/bin/env bash
# Daily read-only corporate-action audit. Never adjusts holdings automatically.
set -euo pipefail

cd /home/gexin/quant-trading
PYTHON=/home/gexin/quant-trading/venv/bin/python
LOG=/home/gexin/quant-trading/logs/corporate_actions.log
LOCK=/tmp/quant_corporate_action_audit.lock
TIMEOUT_SECONDS=${QUANT_CORPORATE_ACTION_TIMEOUT_SECONDS:-1200}

if [ "${QUANT_CORP_ACTION_ALREADY_LOCKED:-0}" != "1" ]; then
    export QUANT_CORP_ACTION_ALREADY_LOCKED=1
    exec /usr/bin/flock -n "$LOCK" "$0" "$@"
fi

start="${1:-$(date -u -d '14 days ago' +%F)}"
end="${2:-$(date -u +%F)}"
echo "===== Corporate-action audit start $(date -u +%Y-%m-%dT%H:%M:%SZ) range=$start..$end =====" >> "$LOG"
set +e
/usr/bin/timeout --kill-after=30s "${TIMEOUT_SECONDS}s" \
    "$PYTHON" scripts/corporate_action_check.py \
    --markets US,CN --start "$start" --end "$end" --min-fetch-coverage 1.0 \
    >> "$LOG" 2>&1
rc=$?
set -e

end_ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ "$rc" -eq 0 ]; then
    echo "===== Corporate-action audit OK $end_ts =====" >> "$LOG"
    exit 0
fi

echo "===== Corporate-action audit FAIL $end_ts exit=$rc =====" >> "$LOG"
if [ -f "$HOME/.hermes/.env" ]; then
    set -a
    # shellcheck source=/dev/null
    source "$HOME/.hermes/.env"
    set +a
fi
if [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID:-}" ]; then
    curl -fsS -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d "chat_id=${TELEGRAM_CHAT_ID}" \
        --data-urlencode "text=⚠️ Corporate-action audit requires review on $(hostname), exit=$rc. No holdings were changed." \
        >/dev/null 2>&1 || true
fi
exit "$rc"
