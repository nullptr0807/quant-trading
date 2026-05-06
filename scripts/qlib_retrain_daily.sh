#!/bin/bash
# Daily Qlib model retrain — runs at 23:00 UTC (Sydney 09:00 AEST)
# Outside US (closes 20:00 UTC) and CN (closes 07:00 UTC) trading hours.
# Trains 10 Q-models for US + 10 for CN in subprocesses (memory isolation).
set -euo pipefail

cd /home/gexin/quant-trading
source venv/bin/activate

LOG_DIR=/home/gexin/quant-trading/logs
mkdir -p "$LOG_DIR"

run_market() {
    local mkt="$1"
    echo "===== Qlib retrain [$mkt] start $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" >> "$LOG_DIR/qlib_retrain.log"
    if python -m scripts.qlib_retrain --market "$mkt" >> "$LOG_DIR/qlib_retrain.log" 2>&1; then
        echo "===== Qlib retrain [$mkt] OK    $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" >> "$LOG_DIR/qlib_retrain.log"
    else
        echo "===== Qlib retrain [$mkt] FAIL  $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" >> "$LOG_DIR/qlib_retrain.log"
    fi
}

# US runs first (richer data, validates pipeline). CN second.
run_market US
run_market CN || true   # don't let CN failure mask US success
