#!/usr/bin/env bash
# Bounded, non-overlapping adjusted/raw price backfill with explicit success/failure logging.
set -euo pipefail

cd /home/gexin/quant-trading

MARKET="${1:-}"
INTERVAL="${2:-}"
DAYS="${3:-}"
PRICE_MODE="${4:-adjusted}"
SCOPE="${5:-universe}"
case "$MARKET" in US|CN) ;; *) echo "usage: $0 US|CN 1d|1h|5m DAYS adjusted|raw|both universe|ledger" >&2; exit 2 ;; esac
case "$INTERVAL" in 1d|1h|15m|5m) ;; *) echo "invalid interval: $INTERVAL" >&2; exit 2 ;; esac
case "$PRICE_MODE" in adjusted|raw|both) ;; *) echo "invalid price mode: $PRICE_MODE" >&2; exit 2 ;; esac
case "$SCOPE" in universe|ledger) ;; *) echo "invalid scope: $SCOPE" >&2; exit 2 ;; esac
[[ "$DAYS" =~ ^[0-9]+$ ]] || { echo "invalid days: $DAYS" >&2; exit 2; }

PYTHON=/home/gexin/quant-trading/venv/bin/python
LOG=/home/gexin/quant-trading/logs/backfill.log
LOCK="/tmp/quant_backfill_${MARKET}_${INTERVAL}_${PRICE_MODE}.lock"
TIMEOUT_SECONDS=${QUANT_BACKFILL_TIMEOUT_SECONDS:-900}

if [ "${QUANT_BACKFILL_ALREADY_LOCKED:-0}" != "1" ]; then
    export QUANT_BACKFILL_ALREADY_LOCKED=1
    exec /usr/bin/flock -n "$LOCK" "$0" "$MARKET" "$INTERVAL" "$DAYS" "$PRICE_MODE" "$SCOPE"
fi

stamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "===== Backfill [$MARKET/$INTERVAL/$PRICE_MODE/$SCOPE] start $stamp days=$DAYS =====" >> "$LOG"
set +e
/usr/bin/timeout --kill-after=30s "${TIMEOUT_SECONDS}s" \
    "$PYTHON" -m scripts.backfill_prices \
    --market "$MARKET" --interval "$INTERVAL" --days "$DAYS" --price-mode "$PRICE_MODE" --scope "$SCOPE" \
    >> "$LOG" 2>&1
rc=$?
set -e
end="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ "$rc" -eq 0 ]; then
    echo "===== Backfill [$MARKET/$INTERVAL/$PRICE_MODE/$SCOPE] OK $end =====" >> "$LOG"
    exit 0
fi

echo "===== Backfill [$MARKET/$INTERVAL/$PRICE_MODE/$SCOPE] FAIL $end exit=$rc =====" >> "$LOG"
if [ -f "$HOME/.hermes/.env" ]; then
    set -a
    # shellcheck source=/dev/null
    source "$HOME/.hermes/.env"
    set +a
fi
if [ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID:-}" ]; then
    curl -fsS -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d "chat_id=${TELEGRAM_CHAT_ID}" \
        --data-urlencode "text=⚠️ Quant backfill [$MARKET/$INTERVAL/$PRICE_MODE] failed on $(hostname), exit=$rc" \
        >/dev/null 2>&1 || true
fi
exit "$rc"
