#!/usr/bin/env python3
"""Audit or apply private permissions for quant secrets and operational data."""
from __future__ import annotations
import argparse
import os
from pathlib import Path

ROOT = Path('/home/gexin/quant-trading')


def candidates(root: Path):
    files = [root / '.env', root / 'data/trading.db', root / 'data/trading.db-wal', root / 'data/trading.db-shm']
    for folder in (root / 'logs', root / 'data/backups'):
        if folder.exists():
            for base, _dirs, names in os.walk(folder):
                for name in names:
                    files.append(Path(base) / name)
    return [p for p in files if p.exists() and p.is_file()]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--apply', action='store_true')
    args = p.parse_args()
    changed = []
    for path in candidates(args.root):
        mode = path.stat().st_mode & 0o777
        if mode != 0o600:
            changed.append({'path': str(path), 'old': oct(mode), 'new': '0o600'})
            if args.apply:
                os.chmod(path, 0o600)
    for folder in (args.root / 'logs', args.root / 'data/backups'):
        if folder.exists() and args.apply:
            os.chmod(folder, 0o700)
    print(('APPLIED' if args.apply else 'DRY_RUN'), f'changes={len(changed)}')
    for row in changed:
        print(row)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
