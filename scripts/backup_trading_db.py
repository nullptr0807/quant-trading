#!/usr/bin/env python3
"""Create and verify a compressed SQLite online backup.

The source remains online in WAL mode. Success requires: SQLite online backup,
quick_check on the snapshot, zstd integrity, SHA256, and a restore-drill
quick_check after decompression. Plain snapshots are always removed.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sqlite3
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def _quick_check(path: Path) -> None:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        result = con.execute("PRAGMA quick_check").fetchone()[0]
    finally:
        con.close()
    if result != "ok":
        raise RuntimeError(f"quick_check failed for {path}: {result}")


def _remove_sqlite_sidecars(path: Path) -> None:
    """Remove transient WAL/SHM files created while verifying a snapshot."""
    for suffix in ("-wal", "-shm"):
        Path(str(path) + suffix).unlink(missing_ok=True)


def create_backup(source: Path, out_dir: Path, *, keep: int = 7) -> dict:
    source = source.resolve(); out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(out_dir, 0o700)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw = out_dir / f"trading.db.{stamp}"
    compressed = Path(str(raw) + ".zst")
    digest_file = Path(str(compressed) + ".sha256")
    if raw.exists() or compressed.exists():
        raise FileExistsError(compressed)

    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True, timeout=60)
    dst = sqlite3.connect(raw)
    try:
        src.backup(dst, pages=8192, sleep=0.05)
        dst.commit()
    finally:
        dst.close(); src.close()
    try:
        _quick_check(raw)
        subprocess.run(["zstd", "-T0", "-6", "-f", str(raw), "-o", str(compressed)], check=True)
        subprocess.run(["zstd", "-t", str(compressed)], check=True)
        h = hashlib.sha256()
        with compressed.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        digest_file.write_text(f"{h.hexdigest()}  {compressed.name}\n")
        os.chmod(compressed, 0o600); os.chmod(digest_file, 0o600)

        # Restore drill proves the compressed artifact, not merely the raw temp.
        with tempfile.TemporaryDirectory(dir=out_dir) as td:
            restored = Path(td) / "restored.db"
            with restored.open("wb") as output:
                subprocess.run(["zstd", "-dc", str(compressed)], stdout=output, check=True)
            _quick_check(restored)

        backups = sorted(out_dir.glob("trading.db.*.zst"), key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in backups[max(1, keep):]:
            stale.unlink(missing_ok=True)
            Path(str(stale) + ".sha256").unlink(missing_ok=True)
            _remove_sqlite_sidecars(Path(str(stale)[:-4]))
        return {"path": str(compressed), "sha256": h.hexdigest(), "quick_check": "ok", "restore_drill": "ok"}
    except Exception:
        compressed.unlink(missing_ok=True); digest_file.unlink(missing_ok=True)
        raise
    finally:
        raw.unlink(missing_ok=True)
        _remove_sqlite_sidecars(raw)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/trading.db")
    parser.add_argument("--out-dir", default="data/backups/daily")
    parser.add_argument("--keep", type=int, default=7)
    args = parser.parse_args()
    result = create_backup(Path(args.source), Path(args.out_dir), keep=args.keep)
    print("BACKUP_OK " + " ".join(f"{k}={v}" for k, v in result.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
