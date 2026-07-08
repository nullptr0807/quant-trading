#!/usr/bin/env python3
"""Compress old trading.db backup files into data/backups/archive.

Non-destructive: only moves a backup after zstd succeeds and the archive exists.
Keeps the newest N raw backups in data/ for immediate rollback convenience.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ARCHIVE = DATA / "backups" / "archive"
PATTERNS = ["trading.db.bak*", "*.db.bak*"]


def backup_files() -> list[Path]:
    files: dict[Path, None] = {}
    for pat in PATTERNS:
        for p in DATA.glob(pat):
            if p.is_file() and not p.name.endswith(".zst"):
                files[p] = None
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, default=3, help="keep newest N raw backups in data/")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    zstd = shutil.which("zstd")
    if not zstd:
        raise SystemExit("zstd not found")
    ARCHIVE.mkdir(parents=True, exist_ok=True)

    files = backup_files()
    to_archive = files[args.keep:]
    print(f"found={len(files)} keep_raw={min(args.keep, len(files))} archive={len(to_archive)}")
    for p in to_archive:
        dest = ARCHIVE / (p.name + ".zst")
        print(f"{p} -> {dest}")
        if args.dry_run:
            continue
        if not dest.exists():
            subprocess.run([zstd, "-T0", "-19", "--rm", "-o", str(dest), str(p)], check=True)
        else:
            # Archive already exists: remove raw duplicate only after confirming archive is non-empty.
            if dest.stat().st_size <= 0:
                raise RuntimeError(f"existing archive is empty: {dest}")
            p.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
