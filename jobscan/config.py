"""Application configuration.

Secrets (API keys) come from environment variables / a local .env file. Everything else
(model name, pricing table, salary threshold, ...) lives in config/settings.yaml so it can be
tuned without touching code, per the project's own rule against hard-coding the model name.
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
        anthropic_model=env_or("JOBSCAN_ANTHROPIC_MODEL", raw.get("anthropic_model", "claude-sonnet-5")),
        anthropic_max_retries=int(raw.get("anthropic_max_retries", 3)),
        anthropic_timeout_seconds=int(raw.get("anthropic_timeout_seconds", 60)),
        anthropic_pricing=pricing,
        triage_model=env_or("JOBSCAN_TRIAGE_MODEL", raw.get("triage_model")) or None,
        min_base_salary=int(raw.get("min_base_salary", 170000)),
        candidate_state=raw.get("candidate_state", "WA"),
        candidate_state_name=raw.get("candidate_state_name", "Washington"),
        log_level=env_or("JOBSCAN_LOG_LEVEL", raw.get("log_level", "INFO")),
        http_timeout_seconds=int(raw.get("http_timeout_seconds", 30)),
        http_user_agent=raw.get("http_user_agent", "JobScan/0.1"),
        anthropic_api_key=env_or("ANTHROPIC_API_KEY", None),
        profile_path=Path(env_or("JOBSCAN_PROFILE_PATH", DEFAULT_PROFILE_PATH)),
    )
