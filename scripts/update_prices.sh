#!/usr/bin/env bash
# Per-minute realtime price & equity updater. Called by cron every minute
# during US market hours. Non-blocking: if yfinance is slow this run may
# overlap the next — we use flock to prevent pile-up.
set -euo pipefail

cd /home/gexin/quant-trading

# Load env (bot token etc) — ignore failure
set -a
[ -f ~/.hermes/.env ] && . ~/.hermes/.env
set +a

# Disable all Telegram notifications from quant system (Jay Chou monitor handles its own).
export QUANT_DISABLE_TELEGRAM=1

# Use venv python directly (no `source activate` — flock spawns /bin/sh which
# breaks bash-only activate semantics, leading to "python: not found").
PYTHON=/home/gexin/quant-trading/venv/bin/python

# Use the SAME lock as run_cycle.sh / run_cycle_quiet.sh so we never
# overlap a main.py cycle (which races on positions/cash and can roll back
# our stop-loss writes). Wait up to 8s; if a long cycle is still running,
# skip this tick rather than block.
exec /usr/bin/flock -w 8 /tmp/quant_run_cycle.lock \
    "$PYTHON" -m scripts.update_prices \
    >> /home/gexin/quant-trading/logs/update_prices.log 2>&1
