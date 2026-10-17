"""A verified, compressed snapshot of the SQLite database into a backup folder (typically a
synced drive).

The database holds every crawled posting and every evaluation — the LLM spend behind them is the
expensive part to rebuild — and it's far too large for a git host. ``VACUUM INTO`` writes a
consistent, compacted copy even while the database is open, into a scratch folder beside the
database. The copy must pass SQLite's integrity check; only then is it gzip-compressed into the
backup folder, replacing the previous snapshot in one step — so a failed or interrupted backup
never overwrites the last good one, and the synced folder never sees an unverified file.
Compression matters there: a synced drive keeps every version for a while (Google Drive: 30 days),
and the database shrinks about eightfold (109 MB to 14 MB, measured).
"""
from __future__ import annotations

import gzip
import os
import shutil
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path


class BackupError(RuntimeError):
    """The snapshot couldn't be written or failed verification; the previous one was kept."""


def backup_database(db_path: Path, target_dir: Path) -> Path:
    """Snapshot ``db_path``, verify it, and write it gzip-compressed to ``target_dir/<name>.gz``;
    returns that path. Restore with ``gzip -d`` or Python's ``gzip`` module."""
    if not db_path.exists():
        raise BackupError(f"no database at {db_path}")
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{db_path.name}.gz"
    partial = target.with_name(target.name + ".partial")
    with tempfile.TemporaryDirectory(dir=db_path.parent) as scratch:
        snapshot = Path(scratch) / db_path.name
        try:
            with closing(sqlite3.connect(db_path)) as connection:
                connection.execute("VACUUM INTO ?", (str(snapshot),))
            problem = _integrity_problem(snapshot)
        except sqlite3.Error as exc:
            raise BackupError(f"snapshot failed ({exc}); the previous backup was kept") from exc
        if problem is not None:
            raise BackupError(f"snapshot failed its integrity check ({problem}); the previous backup was kept")
        try:
            with open(snapshot, "rb") as source, gzip.open(partial, "wb", compresslevel=6) as compressed:
                shutil.copyfileobj(source, compressed, 1 << 20)
        except OSError as exc:
            partial.unlink(missing_ok=True)
            raise BackupError(f"writing the snapshot failed ({exc}); the previous backup was kept") from exc
    os.replace(partial, target)
    return target


def _integrity_problem(path: Path) -> str | None:
    """None when SQLite's full integrity check passes, else its first finding."""
    with closing(sqlite3.connect(path)) as connection:
        result = connection.execute("PRAGMA integrity_check").fetchone()[0]
    return None if result == "ok" else str(result)
