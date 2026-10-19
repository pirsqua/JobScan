from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from jobscan.config import Settings, ModelPricing
from jobscan.db import Database
from jobscan.models import AtsType, Company


EXAMPLE_PROFILE = Path(__file__).resolve().parent.parent / "config" / "candidate_profile.example.yaml"


@pytest.fixture(autouse=True)
def _isolate_personal_paths(monkeypatch, tmp_path: Path) -> None:
    """No test reads the owner's personal files or writes a backup: .env never overrides a
    variable that's already set, so pinning these keeps every test — including ones that call
    jobscan.cli.main, which loads real settings — on the committed example and temp paths, and
    makes a fresh clone with no .env behave the same."""
    monkeypatch.setenv("JOBSCAN_PROFILE_PATH", str(EXAMPLE_PROFILE))
    monkeypatch.setenv("JOBSCAN_APPLICATIONS_PATH", str(tmp_path / "applications.yaml"))
    monkeypatch.setenv("JOBSCAN_REGISTRY_PATH", "")
    monkeypatch.setenv("JOBSCAN_BACKUP_DIR", "")


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        db_path=tmp_path / "jobscan.db",
        output_dir=tmp_path / "out",
        anthropic_model="claude-sonnet-5",
        anthropic_max_retries=1,
        anthropic_timeout_seconds=5,
        anthropic_pricing={
            "claude-sonnet-5": ModelPricing(input_per_million=3.0, output_per_million=15.0),
            "claude-haiku-test": ModelPricing(input_per_million=0.8, output_per_million=4.0),
        },
        # None by default so existing tests exercise the full-evaluation path unchanged; tests
        # covering triage explicitly override this with dataclasses.replace.
        triage_model=None,
        min_base_salary=170000,
        candidate_state="WA",
        candidate_state_name="Washington",
        log_level="WARNING",
        http_timeout_seconds=5,
        http_user_agent="JobScan-Test/0.1",
        # One at a time, so stub clients hand out their queued responses in a fixed order; the
        # parallel path has its own tests.
        evaluate_workers=1,
        anthropic_api_key=None,
        # The committed example, never the candidate's real (private) profile.
        profile_path=EXAMPLE_PROFILE,
        # Never the real applications file — tests that need applications write their own.
        applications_path=tmp_path / "applications.yaml",
    )


@pytest.fixture()
def db(settings: Settings):
    database = Database(settings.db_path)
    yield database
    database.close()


@pytest.fixture()
def sample_company() -> Company:
    return Company(
        name="Acme Corp",
        domain="acme.example.com",
        careers_url="https://acme.example.com/careers",
        ats_type=AtsType.GREENHOUSE,
        board_id="acme",
        discovery_source="test fixture",
    )


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
