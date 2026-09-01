#!/usr/bin/env python3
"""Fail closed unless Qlib scores were published for the latest market date."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from config.settings import UNIVERSES


def _sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_probe(
    model: str, target: str, market: str, checkpoint_root: Path | None = None,
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    if checkpoint_root is not None:
        env['QLIB_CHECKPOINT_ROOT'] = str(Path(checkpoint_root).resolve())
    return subprocess.run(
        [
            sys.executable, '-c',
            'import sys; from factors.qlib_checkpoint import load_checkpoint; '
            'load_checkpoint(sys.argv[1],sys.argv[2],market=sys.argv[3],verify=True)',
            model, target, market,
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env, capture_output=True, text=True, timeout=600,
    )


def verify(db_path: str, market: str, min_coverage: float = 0.80, model_ids=None,
           verify_checkpoints: bool = True) -> dict:
    market = market.upper()
    models = list([f"Q{i:02d}" for i in range(1, 11)] if model_ids is None else model_ids)
    if not models:
        return {"market": market, "models": [], "skipped": "no_active_qlib_accounts"}
    universe = list(UNIVERSES[market])
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
        checkpoint_status = {}
        if verify_checkpoints:
            from factors.qlib_checkpoint import CHECKPOINT_ROOT
            checkpoint_failures = {}
            for model in models:
                sidecar = CHECKPOINT_ROOT / market / model / f"{target}.json"
                pkl = CHECKPOINT_ROOT / market / model / f"{target}.pkl"
                try:
                    meta = json.loads(sidecar.read_text())
                    extra = meta.get('extra') or {}
                    if extra.get('point_in_time_complete') is not True:
                        raise RuntimeError('point_in_time_complete is not true')
                    if int(extra.get('universe_count') or 0) != len(universe):
                        raise RuntimeError(
                            f"universe_count={extra.get('universe_count')} expected={len(universe)}"
                        )
                    if not pkl.exists() or pkl.stat().st_size <= 0:
                        raise RuntimeError('checkpoint payload missing or empty')
                    probe = _checkpoint_probe(model, target, market, CHECKPOINT_ROOT)
                    if probe.returncode != 0:
                        raise RuntimeError((probe.stdout + probe.stderr)[-1500:])
                    checkpoint_status[model] = 'verified'
                except Exception as exc:
                    checkpoint_failures[model] = str(exc)
            if checkpoint_failures:
                raise RuntimeError(
                    f"{market} Qlib checkpoint/PIT verification failed for {target}: "
                    f"{checkpoint_failures}"
                )
        return {
            "market": market, "date": target, "required": required,
            "coverage": coverage, "checkpoints": checkpoint_status,
        }
    finally:
        con.close()


def write_publication_marker(result: dict) -> Path:
    from factors.qlib_checkpoint import CHECKPOINT_ROOT
    market=result['market'];date=result['date']
    marker_path=CHECKPOINT_ROOT/market/f'publication-{date}.json'
    marker_path.parent.mkdir(parents=True,exist_ok=True)
    existing={}
    if marker_path.exists():
        try: existing=json.loads(marker_path.read_text())
        except Exception: existing={}
    records=dict(existing.get('models') or {})
    for model,status in (result.get('checkpoints') or {}).items():
        if status!='verified': continue
        pkl=CHECKPOINT_ROOT/market/model/f'{date}.pkl'
        sidecar=CHECKPOINT_ROOT/market/model/f'{date}.json'
        pre={'pkl':_sha256(pkl),'json':_sha256(sidecar)}
        probe=_checkpoint_probe(model,date,market,CHECKPOINT_ROOT)
        if probe.returncode!=0:
            raise RuntimeError(
                f'publication marker refused unverified {market}/{model}/{date}: '
                +(probe.stdout+probe.stderr)[-1500:]
            )
        post={'pkl':_sha256(pkl),'json':_sha256(sidecar)}
        if pre!=post:
            raise RuntimeError(f'checkpoint changed during verification: {market}/{model}/{date}')
        ps,js=pkl.stat(),sidecar.stat()
        records[model]={
            'pkl_size':ps.st_size,'pkl_mtime_ns':ps.st_mtime_ns,'pkl_sha256':post['pkl'],
            'json_size':js.st_size,'json_mtime_ns':js.st_mtime_ns,'json_sha256':post['json'],
        }
    doc={'market':market,'date':date,'verified_at':datetime.now(timezone.utc).isoformat(),'models':records}
    tmp=marker_path.with_suffix('.tmp');tmp.write_text(json.dumps(doc,indent=2));os.chmod(tmp,0o600);tmp.replace(marker_path);os.chmod(marker_path,0o600)
    return marker_path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--db', default='data/trading.db')
    p.add_argument('--market', choices=['US','CN'], required=True)
    p.add_argument('--models', default='')
    p.add_argument('--model-manifest', default='')
    p.add_argument('--min-coverage', type=float, default=.80)
    a = p.parse_args()
    from factors.qlib_signal import MODEL_SPECS
    from scripts.qlib_model_selection import select_qlib_model_ids
    if a.model_manifest:
        manifest = json.loads(Path(a.model_manifest).read_text())
        if manifest.get('market') != a.market or not isinstance(manifest.get('models'), list):
            raise RuntimeError('invalid or wrong-market Qlib model manifest')
        if Path(manifest.get('db', '')).resolve() != Path(a.db).resolve():
            raise RuntimeError('Qlib model manifest DB mismatch')
        models = manifest['models']
    else:
        requested = [x.strip() for x in a.models.split(',') if x.strip()] if a.models else None
        models = select_qlib_model_ids(
            a.db, a.market, [spec.id for spec in MODEL_SPECS], requested,
        )
    result = verify(a.db, a.market, a.min_coverage, models)
    if result.get('skipped'):
        print(f"QLIB_PUBLISH_SKIPPED market={a.market} reason={result['skipped']}")
        return 0
    marker = write_publication_marker(result)
    print(f"QLIB_PUBLISH_OK market={result['market']} date={result['date']} coverage={result['coverage']} marker={marker}")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
