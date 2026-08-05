#!/usr/bin/env python3
"""Fail closed unless Qlib scores were published for the latest market date."""
from __future__ import annotations
import argparse
import sqlite3
from pathlib import Path

from config.settings import UNIVERSES


def verify(db_path: str, market: str, min_coverage: float = 0.80, model_ids=None) -> dict:
    market = market.upper()
    universe = list(UNIVERSES[market])
    models = list(model_ids or [f"Q{i:02d}" for i in range(1, 11)])
    if not universe:
        raise RuntimeError(f"empty {market} universe")
    con = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True)
    try:
        marks = ','.join('?' for _ in universe)
        target = con.execute(
            f"SELECT MAX(substr(datetime,1,10)) FROM prices "
            f"WHERE interval='1d' AND ticker IN ({marks})", universe,
        ).fetchone()[0]
        if not target:
            raise RuntimeError(f"no adjusted daily target date for {market}")
        required = max(1, int(len(universe) * min_coverage + 0.999999))
        coverage = {}
        for model in models:
            name = f"qlib_{model}_score"
            count = con.execute(
                f"SELECT COUNT(DISTINCT ticker) FROM factor_values "
                f"WHERE factor_group='qlib' AND factor_name=? AND date=? "
                f"AND value IS NOT NULL AND ticker IN ({marks})",
                (name, target, *universe),
            ).fetchone()[0]
            coverage[model] = int(count)
        failed = {model: count for model, count in coverage.items() if count < required}
        if failed:
            raise RuntimeError(
                f"{market} Qlib publication incomplete for {target}: "
                f"required={required}/{len(universe)} failed={failed}"
            )
        return {"market": market, "date": target, "required": required, "coverage": coverage}
    finally:
        con.close()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--db', default='data/trading.db')
    p.add_argument('--market', choices=['US','CN'], required=True)
    p.add_argument('--models', default='')
    p.add_argument('--min-coverage', type=float, default=.80)
    a = p.parse_args()
    models = [x.strip() for x in a.models.split(',') if x.strip()] or None
    result = verify(a.db, a.market, a.min_coverage, models)
    print(f"QLIB_PUBLISH_OK market={result['market']} date={result['date']} coverage={result['coverage']}")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
