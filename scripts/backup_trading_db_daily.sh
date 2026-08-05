#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT=${QUANT_PROJECT_ROOT:-/home/gexin/quant-trading}
PYTHON=${QUANT_PYTHON:-$ROOT/venv/bin/python}
LOG=${QUANT_BACKUP_LOG:-$ROOT/logs/backup.log}
TIMEOUT=${QUANT_BACKUP_TIMEOUT_SECONDS:-1800}
exec bash "$ROOT/scripts/run_scheduled_job.sh" database_backup ALL /tmp/quant_database_backup.lock 5 "$TIMEOUT" "$LOG" -- \
  "$PYTHON" "$ROOT/scripts/backup_trading_db.py" --source "$ROOT/data/trading.db" --out-dir "$ROOT/data/backups/daily" --keep 7
