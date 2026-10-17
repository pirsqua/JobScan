from __future__ import annotations

import sqlite3

import pytest

from jobscan.backup import backup_database
from jobscan.db import Database
from jobscan.models import AtsType, Company


def test_snapshot_is_a_complete_copy_and_replaces_the_previous_one(settings, tmp_path):
    with Database(settings.db_path) as db:
        db.upsert_company(Company(name="Acme", domain=None, careers_url=None, ats_type=AtsType.GREENHOUSE, board_id="acme"))
    target_dir = tmp_path / "drive"
    target_dir.mkdir()
    (target_dir / settings.db_path.name).write_text("an older snapshot", encoding="utf-8")

    snapshot = backup_database(settings.db_path, target_dir)

    with sqlite3.connect(snapshot) as copy:
        assert copy.execute("select name from companies").fetchall() == [("Acme",)]
    assert not list(target_dir.glob("*.partial"))


def test_a_missing_database_is_reported_not_created(settings, tmp_path):
    with pytest.raises(FileNotFoundError):
        backup_database(tmp_path / "nope.db", tmp_path / "drive")
