#!/usr/bin/env bash
# Intra-hour rebalance trigger. Runs at :15, :30, :45 — skips :00 which is
# handled by run_cycle.sh (full report). Uses flock to prevent overlap with
# the top-of-hour run and update_prices.sh.
set -euo pipefail

cd /home/gexin/quant-trading
source venv/bin/activate
export $(grep TELEGRAM_BOT_TOKEN ~/.hermes/.env 2>/dev/null || true)

DOW=$(date -u +%u)
if [ "$DOW" -gt 5 ]; then exit 0; fi

TIMEOUT_SECONDS=${QUANT_RUN_CYCLE_TIMEOUT_SECONDS:-1200}
exec /usr/bin/flock -n /tmp/quant_run_cycle.lock \
    /usr/bin/timeout --kill-after=30s "${TIMEOUT_SECONDS}s" \
    python main.py --cycle-no-report >> /home/gexin/quant-trading/logs/cron.log 2>&1
