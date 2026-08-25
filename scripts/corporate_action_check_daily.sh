#!/usr/bin/env bash
# Daily corporate-action gate. The runner owns /usr/bin/flock and /usr/bin/timeout.
# Alert delivery is centralized there via TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID.
set -euo pipefail
umask 077

ROOT=${QUANT_PROJECT_ROOT:-/home/gexin/quant-trading}
PYTHON=${QUANT_PYTHON:-$ROOT/venv/bin/python}
RUNNER="$ROOT/scripts/run_scheduled_job.sh"
LOG="$ROOT/logs/corporate_actions.log"
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

# Run both markets fail-late with independent locks/health rows: one provider
# failure must not suppress the other market's corporate-action gate.
overall=0
for market in CN US; do
    set +e
    "$RUNNER" corporate_action_gate "$market" "/tmp/quant_corporate_action_audit_${market}.lock" \
        5 "$FAST_TIMEOUT" "$LOG" -- \
        "$PYTHON" "$ROOT/scripts/corporate_action_check.py" \
        --scope fast --markets "$market" --start "$start" --end "$end" \
        --min-fetch-coverage 1.0
    rc=$?
    set -e
    if [[ $rc -ne 0 && $overall -eq 0 ]]; then overall=$rc; fi
done

# Full historical-ledger coverage is separately observable and optional. It can
# remain enabled as a slower secondary scan without delaying the primary gate.
if [[ "${QUANT_CORPORATE_ACTION_FULL_SCAN:-0}" == "1" ]]; then
    for market in CN US; do
        set +e
        "$RUNNER" corporate_action_full_scan "$market" "/tmp/quant_corporate_action_audit_${market}.lock" \
            5 "$FULL_TIMEOUT" "$LOG" -- \
            "$PYTHON" "$ROOT/scripts/corporate_action_check.py" \
            --scope full --markets "$market" --start "$start" --end "$end" \
            --min-fetch-coverage 1.0
        rc=$?
        set -e
        if [[ $rc -ne 0 && $overall -eq 0 ]]; then overall=$rc; fi
    done
fi

exit "$overall"
