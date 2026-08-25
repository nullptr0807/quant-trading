#!/usr/bin/env bash
# Daily corporate-action gate. The runner owns /usr/bin/flock and /usr/bin/timeout.
# Alert delivery is centralized there via TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID.
set -euo pipefail
umask 077

ROOT=${QUANT_PROJECT_ROOT:-/home/gexin/quant-trading}
PYTHON=${QUANT_PYTHON:-$ROOT/venv/bin/python}
RUNNER="$ROOT/scripts/run_scheduled_job.sh"
LOG="$ROOT/logs/corporate_actions.log"
LOCK=/tmp/quant_corporate_action_audit.lock
FAST_TIMEOUT=${QUANT_CORPORATE_ACTION_FAST_TIMEOUT_SECONDS:-300}
FULL_TIMEOUT=${QUANT_CORPORATE_ACTION_TIMEOUT_SECONDS:-1200}
start="${1:-$(date -u -d '14 days ago' +%F)}"
end="${2:-$(date -u +%F)}"

if [[ -f "$HOME/.hermes/.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "$HOME/.hermes/.env"
    set +a
fi

# The safety gate is deliberately bounded to held and recently traded symbols.
"$RUNNER" corporate_action_gate ALL "$LOCK" 5 "$FAST_TIMEOUT" "$LOG" -- \
    "$PYTHON" "$ROOT/scripts/corporate_action_check.py" \
    --scope fast --markets US,CN --start "$start" --end "$end" \
    --min-fetch-coverage 1.0

# Full historical-ledger coverage is separately observable and optional. It can
# remain enabled as a slower secondary scan without delaying the primary gate.
if [[ "${QUANT_CORPORATE_ACTION_FULL_SCAN:-0}" == "1" ]]; then
    "$RUNNER" corporate_action_full_scan ALL "$LOCK" 5 "$FULL_TIMEOUT" "$LOG" -- \
        "$PYTHON" "$ROOT/scripts/corporate_action_check.py" \
        --scope full --markets US,CN --start "$start" --end "$end" \
        --min-fetch-coverage 1.0
fi
