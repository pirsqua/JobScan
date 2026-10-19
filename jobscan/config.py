"""Application configuration.

Secrets (API keys) come from environment variables / a local .env file. Facts about the candidate
— salary floor, employment-eligible state, time zone — come from the candidate profile, which is
personal and lives outside the repository (``JOBSCAN_PROFILE_PATH``), as do the applications and
registry files. Everything else (model name, pricing table, ...) lives in config/settings.yaml so
it can be tuned without touching code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SETTINGS_PATH = REPO_ROOT / "config" / "settings.yaml"
DEFAULT_PROFILE_PATH = REPO_ROOT / "config" / "candidate_profile.yaml"
DEFAULT_APPLICATIONS_PATH = REPO_ROOT / "data" / "applications.yaml"


@dataclass
class ModelPricing:
    input_per_million: float
    output_per_million: float


@dataclass
class Settings:
    db_path: Path
    output_dir: Path
    anthropic_model: str
    anthropic_max_retries: int
    anthropic_timeout_seconds: int
    anthropic_pricing: dict[str, ModelPricing]
    triage_model: str | None
    min_base_salary: int
    candidate_state: str
    candidate_state_name: str
    log_level: str
    http_timeout_seconds: int
    http_user_agent: str
    anthropic_api_key: str | None = None
    profile_path: Path = DEFAULT_PROFILE_PATH
    applications_path: Path = DEFAULT_APPLICATIONS_PATH
    candidate_time_zone: str = "Pacific"
    # Time zones whose working hours, required or preferred, rule a posting out — the triage guard's
    # copy of the profile's working-hours filter. Empty: none.
    candidate_excluded_time_zones: tuple[str, ...] = ()
    crawl_workers: int = 4
    registry_path: Path | None = None
    backup_dir: Path | None = None
    # The thinking.type sent with each model's calls (None leaves the model's default); which value
    # turns thinking off depends on the model, so it's configured beside the model name.
    anthropic_thinking: str | None = None
    triage_thinking: str | None = None

    def pricing_for(self, model_name: str) -> ModelPricing | None:
        return self.anthropic_pricing.get(model_name)


def load_settings(
    settings_path: Path | str | None = None,
    env: dict | None = None,
    load_dotenv_file: bool = True,
) -> Settings:
    """Load settings.yaml merged with environment variable overrides.

    ``env`` defaults to ``os.environ``; a dict can be passed in tests for isolation.
    """
    if load_dotenv_file:
        load_dotenv(REPO_ROOT / ".env", override=False)

    env_map = env if env is not None else os.environ
    path = Path(settings_path) if settings_path else DEFAULT_SETTINGS_PATH
    raw: dict = {}
    if path.exists():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    def env_or(key: str, default):
        return env_map.get(key, default)

    db_path = Path(env_or("JOBSCAN_DB_PATH", raw.get("db_path", "data/jobscan.db")))
    if not db_path.is_absolute():
        db_path = REPO_ROOT / db_path

    output_dir = Path(env_or("JOBSCAN_OUTPUT_DIR", raw.get("output_dir", "out")))
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir

    profile_path = Path(env_or("JOBSCAN_PROFILE_PATH", DEFAULT_PROFILE_PATH))
    candidate = _candidate_facts(profile_path, raw)
    registry_raw = env_or("JOBSCAN_REGISTRY_PATH", raw.get("registry_path"))
    backup_raw = env_or("JOBSCAN_BACKUP_DIR", raw.get("backup_dir"))

    pricing_raw = raw.get("anthropic_pricing", {}) or {}
    pricing = {
        name: ModelPricing(
            input_per_million=float(vals["input_per_million"]),
            output_per_million=float(vals["output_per_million"]),
        )
        for name, vals in pricing_raw.items()
    }

    return Settings(
        db_path=db_path,
        output_dir=output_dir,
        anthropic_model=env_or("JOBSCAN_ANTHROPIC_MODEL", raw.get("anthropic_model", "claude-sonnet-5-5")),
        anthropic_max_retries=int(raw.get("anthropic_max_retries", 3)),
        anthropic_timeout_seconds=int(raw.get("anthropic_timeout_seconds", 60)),
        anthropic_pricing=pricing,
        triage_model=env_or("JOBSCAN_TRIAGE_MODEL", raw.get("triage_model")) or None,
        anthropic_thinking=env_or("JOBSCAN_ANTHROPIC_THINKING", raw.get("anthropic_thinking")) or None,
        triage_thinking=env_or("JOBSCAN_TRIAGE_THINKING", raw.get("triage_thinking")) or None,
        min_base_salary=int(candidate["min_base_salary"]),
        candidate_state=str(candidate["candidate_state"]),
        candidate_state_name=str(candidate["candidate_state_name"]),
        candidate_time_zone=str(candidate["candidate_time_zone"]),
        candidate_excluded_time_zones=candidate["excluded_time_zones"],
        log_level=env_or("JOBSCAN_LOG_LEVEL", raw.get("log_level", "INFO")),
        http_timeout_seconds=int(raw.get("http_timeout_seconds", 30)),
        http_user_agent=raw.get("http_user_agent", "JobScan/0.1"),
        crawl_workers=int(raw.get("crawl_workers", 4)),
        anthropic_api_key=env_or("ANTHROPIC_API_KEY", None),
        profile_path=profile_path,
        applications_path=Path(env_or("JOBSCAN_APPLICATIONS_PATH", DEFAULT_APPLICATIONS_PATH)),
        registry_path=Path(registry_raw) if registry_raw else None,
        backup_dir=Path(backup_raw) if backup_raw else None,
    )


_CANDIDATE_KEYS = ("min_base_salary", "candidate_state", "candidate_state_name", "candidate_time_zone")


class SettingsError(RuntimeError):
    """Raised when a required candidate fact is missing from the profile."""


def _candidate_facts(profile_path: Path, raw_settings: dict) -> dict:
    """The candidate's salary floor, state and time zone, read from the profile; an older
    settings.yaml that still carries them is honoured as a fallback. ``excluded_time_zones`` is
    optional and read from the profile only."""
    profile: dict = {}
    if profile_path.exists():
        profile = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    facts = {key: profile.get(key, raw_settings.get(key)) for key in _CANDIDATE_KEYS}
    missing = [key for key, value in facts.items() if value is None]
    if missing:
        raise SettingsError(
            f"{', '.join(missing)} missing — set them in the candidate profile ({profile_path}); "
            "see config/candidate_profile.example.yaml"
        )
    facts["excluded_time_zones"] = tuple(str(zone) for zone in profile.get("excluded_time_zones") or ())
    return facts
