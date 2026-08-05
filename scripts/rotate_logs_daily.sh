#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT=${QUANT_PROJECT_ROOT:-/home/gexin/quant-trading}
exec /usr/sbin/logrotate -s "$HOME/.quant-logrotate.state" "$ROOT/config/quant-logrotate.conf"
