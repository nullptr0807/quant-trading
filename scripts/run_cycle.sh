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

# Share a lock with run_cycle_quiet.sh so intra-hour :15/:30/:45 runs can't
# overlap the hourly report run. -w 30 waits up to 30s for any in-flight run.
exec /usr/bin/flock -w 30 /tmp/quant_run_cycle.lock -c '
    python main.py --cycle-no-report 2>&1
'
