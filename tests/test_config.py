from __future__ import annotations

from pathlib import Path

import pytest

from jobscan.config import SettingsError, load_settings

SAMPLE_YAML = """
db_path: custom/jobscan.db
output_dir: custom_out
anthropic_model: claude-test-model
anthropic_max_retries: 5
anthropic_timeout_seconds: 90
anthropic_pricing:
  claude-test-model:
    input_per_million: 1.5
    output_per_million: 7.5
log_level: DEBUG
http_timeout_seconds: 15
http_user_agent: "Test-Agent/1.0"
"""

PROFILE_YAML = """
headline: Test candidate
min_base_salary: 150000
candidate_state: OR
candidate_state_name: Oregon
candidate_time_zone: Pacific
"""


@pytest.fixture()
def env(tmp_path: Path) -> dict:
    """An isolated environment whose profile is a temporary file, never the real one."""
    profile = tmp_path / "profile.yaml"
    profile.write_text(PROFILE_YAML, encoding="utf-8")
    return {"JOBSCAN_PROFILE_PATH": str(profile)}


@pytest.fixture()
def settings_path(tmp_path: Path) -> Path:
    path = tmp_path / "settings.yaml"
    path.write_text(SAMPLE_YAML, encoding="utf-8")
    return path


class TestLoadSettings:
    def test_parses_yaml_fields(self, settings_path: Path, env: dict):
        settings = load_settings(settings_path=settings_path, env=env, load_dotenv_file=False)

        assert settings.anthropic_model == "claude-test-model"
        assert settings.anthropic_max_retries == 5
        assert settings.anthropic_timeout_seconds == 90
        assert settings.log_level == "DEBUG"
        assert settings.http_timeout_seconds == 15
        assert settings.http_user_agent == "Test-Agent/1.0"

    def test_candidate_facts_come_from_the_profile(self, settings_path: Path, env: dict):
        settings = load_settings(settings_path=settings_path, env=env, load_dotenv_file=False)

        assert settings.min_base_salary == 150000
        assert (settings.candidate_state, settings.candidate_state_name, settings.candidate_time_zone) == \
            ("OR", "Oregon", "Pacific")

    def test_an_older_settings_file_still_supplies_facts_the_profile_lacks(self, tmp_path: Path):
        settings_path = tmp_path / "settings.yaml"
        settings_path.write_text(SAMPLE_YAML + "min_base_salary: 140000\n", encoding="utf-8")
        profile = tmp_path / "profile.yaml"
        profile.write_text("candidate_state: OR\ncandidate_state_name: Oregon\ncandidate_time_zone: Pacific\n",
                           encoding="utf-8")

        settings = load_settings(settings_path=settings_path, env={"JOBSCAN_PROFILE_PATH": str(profile)},
                                 load_dotenv_file=False)

        assert settings.min_base_salary == 140000

    def test_missing_candidate_facts_are_an_error_naming_them(self, settings_path: Path, tmp_path: Path):
        profile = tmp_path / "profile.yaml"
        profile.write_text("min_base_salary: 150000\n", encoding="utf-8")

        with pytest.raises(SettingsError, match="candidate_state, candidate_state_name, candidate_time_zone"):
            load_settings(settings_path=settings_path, env={"JOBSCAN_PROFILE_PATH": str(profile)},
                          load_dotenv_file=False)

    def test_builds_pricing_table(self, settings_path: Path, env: dict):
        settings = load_settings(settings_path=settings_path, env=env, load_dotenv_file=False)

        pricing = settings.pricing_for("claude-test-model")
        assert pricing is not None
        assert pricing.input_per_million == 1.5
        assert pricing.output_per_million == 7.5

    def test_unknown_model_has_no_pricing(self, settings_path: Path, env: dict):
        settings = load_settings(settings_path=settings_path, env=env, load_dotenv_file=False)

        assert settings.pricing_for("some-other-model") is None

    def test_relative_paths_resolved_to_absolute(self, settings_path: Path, env: dict):
        settings = load_settings(settings_path=settings_path, env=env, load_dotenv_file=False)

        assert settings.db_path.is_absolute()
        assert settings.db_path.as_posix().endswith("custom/jobscan.db")
        assert settings.output_dir.is_absolute()
        assert settings.output_dir.as_posix().endswith("custom_out")

    def test_missing_settings_file_falls_back_to_defaults(self, tmp_path: Path, env: dict):
        settings = load_settings(settings_path=tmp_path / "does_not_exist.yaml", env=env, load_dotenv_file=False)

        assert settings.anthropic_model == "claude-sonnet-5"
        assert settings.registry_path is None

    def test_env_vars_override_yaml_values(self, settings_path: Path, env: dict):
        settings = load_settings(
            settings_path=settings_path,
            env={
                **env,
                "JOBSCAN_ANTHROPIC_MODEL": "claude-env-override",
                "JOBSCAN_DB_PATH": "env/jobscan.db",
                "JOBSCAN_LOG_LEVEL": "WARNING",
                "ANTHROPIC_API_KEY": "sk-test-key",
                "JOBSCAN_REGISTRY_PATH": "elsewhere/companies.yaml",
            },
            load_dotenv_file=False,
        )

        assert settings.anthropic_model == "claude-env-override"
        assert settings.db_path.as_posix().endswith("env/jobscan.db")
        assert settings.log_level == "WARNING"
        assert settings.anthropic_api_key == "sk-test-key"
        assert settings.registry_path.as_posix() == "elsewhere/companies.yaml"
        # YAML values not targeted by an env var stay as configured.
        assert settings.http_timeout_seconds == 15

    def test_no_api_key_defaults_to_none(self, settings_path: Path, env: dict):
        settings = load_settings(settings_path=settings_path, env=env, load_dotenv_file=False)

        assert settings.anthropic_api_key is None
