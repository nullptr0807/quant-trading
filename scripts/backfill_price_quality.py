#!/usr/bin/env python3
"""Backfill OHLC quarantine markers without deleting provider rows."""
from __future__ import annotations
import argparse
import sqlite3
from datetime import datetime, timezone

CONFIRM='BACKFILL OHLC QUARANTINE'


def scan(db: str, apply: bool=False, confirm: str='') -> dict:
    if apply and confirm!=CONFIRM:
        raise ValueError(f'confirmation must equal {CONFIRM}')
    con=sqlite3.connect(db,timeout=60)
    try:
        totals={}
        for mode,table in [('adjusted','prices'),('raw','prices_raw')]:
            if not con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",(table,)).fetchone():
                continue
            predicate=(
                "open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL OR "
                "open<=0 OR high<=0 OR low<=0 OR close<=0 OR "
                "high<MAX(open,close,low) OR low>MIN(open,close,high)"
            )
            count=con.execute(f'SELECT COUNT(*) FROM {table} WHERE {predicate}').fetchone()[0]
            totals[mode]=count
            if apply:
                now=datetime.now(timezone.utc).isoformat()
                # Synchronize markers to current persisted provider rows. This
                # removes stale tombstones left behind after a valid refetch,
                # while preserving every raw OHLC row for audit.
                con.execute(
                    "DELETE FROM price_quality_issues "
                    "WHERE price_mode=? AND issue_type='invalid_ohlc'",
                    (mode,),
                )
                con.execute(f"""
                    INSERT OR REPLACE INTO price_quality_issues
                    (price_mode,ticker,datetime,interval,issue_type,detected_at,detail)
                    SELECT ?,ticker,datetime,interval,'invalid_ohlc',?,
                           'historical OHLC invariant violation'
                    FROM {table} WHERE {predicate}
                """,(mode,now))
        if apply: con.commit()
        else: con.rollback()
        return totals
    finally:
        con.close()


def main() -> int:
    p=argparse.ArgumentParser();p.add_argument('--db',default='data/trading.db')
    p.add_argument('--apply',action='store_true');p.add_argument('--confirm',default='')
    a=p.parse_args();print(('APPLIED' if a.apply else 'DRY_RUN'),scan(a.db,a.apply,a.confirm));return 0
if __name__=='__main__': raise SystemExit(main())
