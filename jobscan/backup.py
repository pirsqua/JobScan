"""A consistent snapshot of the SQLite database into a backup folder (typically a synced drive).

The database holds every crawled posting and every evaluation — the LLM spend behind them is the
expensive part to rebuild — and it's far too large for a git host. ``VACUUM INTO`` writes a
consistent, compacted copy even while the database is open; it goes to a temporary file first and
replaces the previous snapshot only once complete, so an interrupted backup never leaves a
half-written one behind.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path


def backup_database(db_path: Path, target_dir: Path) -> Path:
    """Snapshot ``db_path`` to ``target_dir/<name>``; returns the snapshot's path."""
    if not db_path.exists():
        raise FileNotFoundError(f"no database at {db_path}")
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / db_path.name
    partial = target.with_name(target.name + ".partial")
    partial.unlink(missing_ok=True)
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("VACUUM INTO ?", (str(partial),))
    finally:
        connection.close()
    os.replace(partial, target)
    return target
