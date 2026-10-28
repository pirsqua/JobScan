from __future__ import annotations

from pathlib import Path

import pytest

from jobscan import config
from jobscan.config import SettingsError, load_settings

SAMPLE_YAML = """
db_path: custom/jobscan.db
output_dir: custom_out
anthropic_model: claude-test-model
anthropic_thinking: between_tools
triage_model: claude-test-small
triage_thinking: disabled
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
excluded_time_zones: [Eastern, Central]
"""


def data_dir_with_profile(tmp_path: Path, profile_yaml: str, name: str = "candidate_profile.yaml") -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / name).write_text(profile_yaml, encoding="utf-8")
    return data_dir


@pytest.fixture()
def env(tmp_path: Path) -> dict:
    """An isolated environment whose data folder is a temporary one, never the real one."""
    return {"JOBSCAN_DATA_DIR": str(data_dir_with_profile(tmp_path, PROFILE_YAML))}


@pytest.fixture()
def settings_path(tmp_path: Path) -> Path:
    path = tmp_path / "settings.yaml"
    path.write_text(SAMPLE_YAML, encoding="utf-8")
    return path


class TestLoadSettings:
    def test_parses_yaml_fields(self, settings_path: Path, env: dict):
        settings = load_settings(settings_path=settings_path, env=env, load_dotenv_file=False)

        assert settings.anthropic_model == "claude-test-model"
        assert (settings.anthropic_thinking, settings.triage_thinking) == ("between_tools", "disabled")
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
        assert settings.candidate_excluded_time_zones == ("Eastern", "Central")

    def test_excluded_time_zones_are_optional(self, settings_path: Path, tmp_path: Path):
        data_dir = data_dir_with_profile(tmp_path, PROFILE_YAML.replace("excluded_time_zones: [Eastern, Central]\n", ""))
        settings = load_settings(settings_path=settings_path, env={"JOBSCAN_DATA_DIR": str(data_dir)},
                                 load_dotenv_file=False)
        assert settings.candidate_excluded_time_zones == ()

    def test_missing_candidate_facts_are_an_error_naming_them(self, settings_path: Path, tmp_path: Path):
        data_dir = data_dir_with_profile(tmp_path, "min_base_salary: 150000\n")

        with pytest.raises(SettingsError, match="candidate_state, candidate_state_name, candidate_time_zone"):
            load_settings(settings_path=settings_path, env={"JOBSCAN_DATA_DIR": str(data_dir)},
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

        assert settings.anthropic_model == "claude-sonnet-5-5"
        assert settings.anthropic_thinking is None  # the model's own default
        assert settings.registry_path == Path(env["JOBSCAN_DATA_DIR"]) / "jobscan_companies.yaml"

    def test_env_vars_override_yaml_values(self, settings_path: Path, env: dict):
        settings = load_settings(
            settings_path=settings_path,
            env={
                **env,
                "JOBSCAN_ANTHROPIC_MODEL": "claude-env-override",
                "JOBSCAN_DB_PATH": "env/jobscan.db",
                "JOBSCAN_LOG_LEVEL": "WARNING",
                "ANTHROPIC_API_KEY": "sk-test-key",
            },
            load_dotenv_file=False,
        )

        assert settings.anthropic_model == "claude-env-override"
        assert settings.db_path.as_posix().endswith("env/jobscan.db")
        assert settings.log_level == "WARNING"
        assert settings.anthropic_api_key == "sk-test-key"
        # YAML values not targeted by an env var stay as configured.
        assert settings.http_timeout_seconds == 15

    def test_no_api_key_defaults_to_none(self, settings_path: Path, env: dict):
        settings = load_settings(settings_path=settings_path, env=env, load_dotenv_file=False)

        assert settings.anthropic_api_key is None


class TestPersonalFiles:
    def test_personal_files_live_in_the_data_folder_which_is_also_the_backup_folder(
            self, settings_path: Path, env: dict):
        data_dir = Path(env["JOBSCAN_DATA_DIR"])
        settings = load_settings(settings_path=settings_path, env=env, load_dotenv_file=False)

        assert settings.data_dir == data_dir
        assert settings.profile_path == data_dir / "candidate_profile.yaml"
        assert settings.applications_path == data_dir / "applications.yaml"
        assert settings.registry_path == data_dir / "jobscan_companies.yaml"
        assert settings.backup_dir == data_dir

    def test_file_names_come_from_settings(self, tmp_path: Path):
        settings_path = tmp_path / "settings.yaml"
        settings_path.write_text(
            SAMPLE_YAML + "personal_files:\n  profile: me.yaml\n  registry: employers.yaml\n", encoding="utf-8")
        data_dir = data_dir_with_profile(tmp_path, PROFILE_YAML, name="me.yaml")

        settings = load_settings(settings_path=settings_path, env={"JOBSCAN_DATA_DIR": str(data_dir)},
                                 load_dotenv_file=False)

        assert settings.profile_path == data_dir / "me.yaml"
        assert settings.registry_path == data_dir / "employers.yaml"
        assert settings.applications_path == data_dir / "applications.yaml"  # not renamed: the default

    def test_without_a_data_folder_the_in_repo_default_is_used_and_nothing_is_backed_up(
            self, settings_path: Path, tmp_path: Path, monkeypatch):
        default = data_dir_with_profile(tmp_path, PROFILE_YAML)
        monkeypatch.setattr(config, "DEFAULT_DATA_DIR", default)

        settings = load_settings(settings_path=settings_path, env={}, load_dotenv_file=False)

        assert settings.profile_path == default / "candidate_profile.yaml"
        assert settings.backup_dir is None

    def test_the_old_per_file_variables_are_an_error_pointing_at_the_data_folder(
            self, settings_path: Path, env: dict):
        with pytest.raises(SettingsError, match="JOBSCAN_PROFILE_PATH is no longer read.*JOBSCAN_DATA_DIR"):
            load_settings(settings_path=settings_path, env={**env, "JOBSCAN_PROFILE_PATH": "elsewhere/profile.yaml"},
                          load_dotenv_file=False)
