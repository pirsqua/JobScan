from __future__ import annotations

import gzip
import shutil
import sqlite3

import pytest

from jobscan import backup
from jobscan.backup import BackupError, backup_database
from jobscan.db import Database
from jobscan.models import AtsType, Company


def seed(settings) -> None:
    with Database(settings.db_path) as db:
        db.upsert_company(Company(name="Acme", domain=None, careers_url=None, ats_type=AtsType.GREENHOUSE, board_id="acme"))


def restore(snapshot, tmp_path):
    restored = tmp_path / "restored.db"
    with gzip.open(snapshot, "rb") as source, open(restored, "wb") as out:
        shutil.copyfileobj(source, out)
    return restored


def test_snapshot_is_a_complete_compressed_copy_and_replaces_the_previous_one(settings, tmp_path):
    seed(settings)
    target_dir = tmp_path / "drive"
    target_dir.mkdir()
    (target_dir / f"{settings.db_path.name}.gz").write_bytes(gzip.compress(b"an older snapshot"))

    snapshot = backup_database(settings.db_path, target_dir)

    assert snapshot.name == "jobscan.db.gz"
    with sqlite3.connect(restore(snapshot, tmp_path)) as copy:
        assert copy.execute("select name from companies").fetchall() == [("Acme",)]
    assert sorted(p.name for p in target_dir.iterdir()) == ["jobscan.db.gz"]  # no partial, no plain copy
    assert not [p for p in settings.db_path.parent.iterdir() if p.is_dir() and p.name.startswith("tmp")]  # scratch removed


def test_a_snapshot_failing_its_integrity_check_never_replaces_the_previous_one(settings, tmp_path, monkeypatch):
    seed(settings)
    target_dir = tmp_path / "drive"
    target_dir.mkdir()
    previous = target_dir / f"{settings.db_path.name}.gz"
    previous.write_bytes(b"the last good snapshot")
    monkeypatch.setattr(backup, "_integrity_problem", lambda path: "*** in database main *** Page 7 is never used")

    with pytest.raises(BackupError, match="previous backup was kept"):
        backup_database(settings.db_path, target_dir)

    assert previous.read_bytes() == b"the last good snapshot"
    assert sorted(p.name for p in target_dir.iterdir()) == ["jobscan.db.gz"]


def test_a_missing_database_is_reported_not_created(settings, tmp_path):
    with pytest.raises(BackupError, match="no database"):
        backup_database(tmp_path / "nope.db", tmp_path / "drive")
    assert not (tmp_path / "nope.db").exists()
