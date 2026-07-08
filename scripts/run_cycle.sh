#!/bin/bash
set -euo pipefail
cd ~/quant-trading
source venv/bin/activate
export $(grep TELEGRAM_BOT_TOKEN ~/.hermes/.env)

# Check if it's a weekday (Mon=1..Fri=5)
DOW=$(date -u +%u)
if [ "$DOW" -gt 5 ]; then
    echo "Weekend, skipping"
    exit 0
fi

# Share a lock with run_cycle_quiet.sh and update_prices.sh so trading cycles
# cannot overlap price snapshots or one another. The OS-level timeout prevents a
# stuck data provider / model path from holding /tmp/quant_run_cycle.lock forever.
TIMEOUT_SECONDS=${QUANT_RUN_CYCLE_TIMEOUT_SECONDS:-1200}
exec /usr/bin/flock -w 30 /tmp/quant_run_cycle.lock \
    /usr/bin/timeout --kill-after=30s "${TIMEOUT_SECONDS}s" \
    python main.py --cycle-no-report 2>&1
