from __future__ import annotations

from pathlib import Path

from jobscan.config import load_settings

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
min_base_salary: 150000
candidate_state: OR
candidate_state_name: Oregon
log_level: DEBUG
http_timeout_seconds: 15
http_user_agent: "Test-Agent/1.0"
"""


class TestLoadSettings:
    def test_parses_yaml_fields(self, tmp_path: Path):
        settings_path = tmp_path / "settings.yaml"
        settings_path.write_text(SAMPLE_YAML, encoding="utf-8")

        settings = load_settings(settings_path=settings_path, env={}, load_dotenv_file=False)

        assert settings.anthropic_model == "claude-test-model"
        assert settings.anthropic_max_retries == 5
        assert settings.anthropic_timeout_seconds == 90
        assert settings.min_base_salary == 150000
        assert settings.candidate_state == "OR"
        assert settings.candidate_state_name == "Oregon"
        assert settings.log_level == "DEBUG"
        assert settings.http_timeout_seconds == 15
        assert settings.http_user_agent == "Test-Agent/1.0"

    def test_builds_pricing_table(self, tmp_path: Path):
        settings_path = tmp_path / "settings.yaml"
        settings_path.write_text(SAMPLE_YAML, encoding="utf-8")

        settings = load_settings(settings_path=settings_path, env={}, load_dotenv_file=False)

        pricing = settings.pricing_for("claude-test-model")
        assert pricing is not None
        assert pricing.input_per_million == 1.5
        assert pricing.output_per_million == 7.5

    def test_unknown_model_has_no_pricing(self, tmp_path: Path):
        settings_path = tmp_path / "settings.yaml"
        settings_path.write_text(SAMPLE_YAML, encoding="utf-8")

        settings = load_settings(settings_path=settings_path, env={}, load_dotenv_file=False)

        assert settings.pricing_for("some-other-model") is None

    def test_relative_paths_resolved_to_absolute(self, tmp_path: Path):
        settings_path = tmp_path / "settings.yaml"
        settings_path.write_text(SAMPLE_YAML, encoding="utf-8")

        settings = load_settings(settings_path=settings_path, env={}, load_dotenv_file=False)

        assert settings.db_path.is_absolute()
        assert settings.db_path.as_posix().endswith("custom/jobscan.db")
        assert settings.output_dir.is_absolute()
        assert settings.output_dir.as_posix().endswith("custom_out")

    def test_missing_settings_file_falls_back_to_defaults(self, tmp_path: Path):
        missing_path = tmp_path / "does_not_exist.yaml"

        settings = load_settings(settings_path=missing_path, env={}, load_dotenv_file=False)

        assert settings.anthropic_model == "claude-sonnet-5"
        assert settings.min_base_salary == 170000
        assert settings.candidate_state == "WA"
        assert settings.candidate_state_name == "Washington"

    def test_env_vars_override_yaml_values(self, tmp_path: Path):
        settings_path = tmp_path / "settings.yaml"
        settings_path.write_text(SAMPLE_YAML, encoding="utf-8")

        settings = load_settings(
            settings_path=settings_path,
            env={
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
        assert settings.min_base_salary == 150000

    def test_no_api_key_defaults_to_none(self, tmp_path: Path):
        settings_path = tmp_path / "settings.yaml"
        settings_path.write_text(SAMPLE_YAML, encoding="utf-8")

        settings = load_settings(settings_path=settings_path, env={}, load_dotenv_file=False)

        assert settings.anthropic_api_key is None
