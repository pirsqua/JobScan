from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from jobscan.config import Settings, ModelPricing
from jobscan.db import Database
from jobscan.models import AtsType, Company


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
        anthropic_api_key=None,
        profile_path=Path(__file__).resolve().parent.parent / "config" / "candidate_profile.yaml",
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
