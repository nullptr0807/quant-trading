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
#
# Hard wall-clock guard: realtime quote providers (especially akshare/Sina for
# CN) can occasionally hang inside a worker thread. Without an OS-level timeout
# the Python process can hold /tmp/quant_run_cycle.lock indefinitely and freeze
# both price updates and trading cycles. `timeout` kills the child and releases
# the flock; the next cron tick can retry.
TIMEOUT_SECONDS=${QUANT_UPDATE_TIMEOUT_SECONDS:-110}
ROOT=${QUANT_PROJECT_ROOT:-/home/gexin/quant-trading}
# During a controlled execution freeze, keep quotes/equity/risk observability
# alive without allowing protective sells.  Production normally defaults to
# live; crontab must opt into no-trades explicitly while the freeze is active.
UPDATE_MODE=${QUANT_UPDATE_MODE:-live}
case "$UPDATE_MODE" in
    live|no-trades) ;;
    *) echo "invalid QUANT_UPDATE_MODE: $UPDATE_MODE (expected live|no-trades)" >&2; exit 2 ;;
esac
exec bash "$ROOT/scripts/run_scheduled_job.sh" update_prices ALL /tmp/quant_run_cycle.lock 8 "$TIMEOUT_SECONDS" \
    "$ROOT/logs/update_prices.log" -- "$PYTHON" -m scripts.update_prices "--$UPDATE_MODE"
