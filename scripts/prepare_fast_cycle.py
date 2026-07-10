#!/usr/bin/env python3
"""Prepare a bounded persisted-signal quote universe for the fast live cycle."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from main import QuantSystem  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", required=True, choices=["US", "CN"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    system = QuantSystem(args.market)
    tickers = sorted(system.prepare_fast_live_cycle())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps({
        "market": args.market,
        "tickers": tickers,
        "prepared_alpha_signals": system._prepared_alpha_signals,
        "prepared_gp_signals": system._prepared_gp_signals,
        "prepared_qlib_scores": system._prepared_qlib_scores,
    }, ensure_ascii=False))
    tmp.replace(output)
    print(f"prepared market={args.market} tickers={len(tickers)} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
