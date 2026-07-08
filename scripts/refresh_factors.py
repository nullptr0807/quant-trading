#!/usr/bin/env python3
"""Refresh persisted factor_values without trading.

This is the safe factor-production job: price cache -> Alpha158 -> GP/F runtime
factors -> factor_values.  It does NOT trade, rebalance, write account snapshots,
or send Telegram reports.  Use it when factor freshness is stale but we do not
want to invoke the full trading engine.

Examples:
    python -m scripts.refresh_factors --market US --alpha-only
    python -m scripts.refresh_factors --market CN
"""
from __future__ import annotations

import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.expanduser("~/quant-trading")
sys.path.insert(0, PROJECT_ROOT)

from main import QuantSystem  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--market", choices=["US", "CN"], default="US")
    p.add_argument("--alpha-only", action="store_true", help="Skip GP/F runtime factor persistence")
    args = p.parse_args()

    t0 = time.time()
    system = QuantSystem(market=args.market)
    system.fetch_data()
    system.compute_factors()
    if not args.alpha_only:
        system.mine_gp_factors()
    elapsed = time.time() - t0
    print(
        f"refresh_factors market={args.market} alpha_only={args.alpha_only} "
        f"done in {elapsed:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
