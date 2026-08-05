#!/usr/bin/env python3
"""Persist an immutable point-in-time universe snapshot for the latest price date."""
from __future__ import annotations
import argparse
import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from config.settings import UNIVERSES
from data.store import DataStore


def snapshot(db_path: str, market: str, source: str = 'configured_universe') -> dict:
    market = market.upper()
    tickers = sorted(set(UNIVERSES[market]))
    if not tickers:
        raise RuntimeError(f'empty {market} universe')
    DataStore(db_path)  # schema only
    con = sqlite3.connect(db_path, timeout=30)
    try:
        marks=','.join('?' for _ in tickers)
        effective = con.execute(
            f"SELECT MAX(substr(datetime,1,10)) FROM prices WHERE interval='1d' "
            f"AND ticker IN ({marks})", tickers,
        ).fetchone()[0]
        if not effective:
            raise RuntimeError(f'no daily price date for {market}')
        digest=hashlib.sha256('\n'.join(tickers).encode()).hexdigest()
        now=datetime.now(timezone.utc).isoformat()
        existing=con.execute(
            'SELECT DISTINCT universe_hash FROM universe_membership WHERE market=? AND date=?',
            (market,effective),
        ).fetchall()
        if existing and {r[0] for r in existing}!={digest}:
            raise RuntimeError(
                f'refusing to rewrite PIT universe {market}/{effective} with a different hash'
            )
        con.executemany(
            'INSERT OR IGNORE INTO universe_membership '
            '(market,date,ticker,source,universe_hash,recorded_at) VALUES (?,?,?,?,?,?)',
            [(market,effective,t,source,digest,now) for t in tickers],
        )
        con.commit()
        count=con.execute(
            'SELECT COUNT(*) FROM universe_membership WHERE market=? AND date=?',
            (market,effective),
        ).fetchone()[0]
        if count!=len(tickers):
            raise RuntimeError(f'incomplete snapshot {count}/{len(tickers)}')
        return {'market':market,'date':effective,'count':count,'hash':digest}
    finally:
        con.close()


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument('--db',default='data/trading.db')
    p.add_argument('--market',choices=['US','CN'],required=True)
    p.add_argument('--source',default='configured_universe')
    a=p.parse_args(); result=snapshot(a.db,a.market,a.source)
    print('UNIVERSE_SNAPSHOT_OK '+' '.join(f'{k}={v}' for k,v in result.items()))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
