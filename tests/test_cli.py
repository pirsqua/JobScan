"""CLI-level tests: argument parsing, command dispatch, and printed output.

`main()` builds a real AnthropicClient from settings/env with no injection seam, so LLM-touching
behavior is covered at the `evaluate_all()` level in test_evaluate.py instead of faking a client
through the CLI layer. The `run` tests below set an empty API key, so no client is built at all.
"""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx
import yaml

from jobscan.cli import main
from jobscan.config import load_settings
from jobscan.db import Database
from jobscan.models import AtsType, ClassificationSource, Company, CompanyClassification, RawPosting
from jobscan.normalize import normalize_posting

COMPANIES_CSV = """name,domain,ats_type,board_id,classification
Acme Corp,acme.example.com,greenhouse,acme,product
"""


@pytest.fixture(autouse=True)
def _clean_jobscan_env(monkeypatch):
    # main() loads settings from the real environment/.env — strip anything that could make
    # these tests depend on the host machine's ambient configuration. (The personal-file paths
    # are pinned to the example and temp paths by conftest's _isolate_personal_paths.)
    for key in ("JOBSCAN_DB_PATH", "JOBSCAN_OUTPUT_DIR", "JOBSCAN_ANTHROPIC_MODEL", "JOBSCAN_LOG_LEVEL",
                "JOBSCAN_ANTHROPIC_THINKING", "JOBSCAN_TRIAGE_MODEL", "JOBSCAN_TRIAGE_THINKING"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture()
def settings_path(tmp_path: Path) -> Path:
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text(
        yaml.safe_dump(
            {
                "db_path": str(tmp_path / "jobscan.db"),
                "output_dir": str(tmp_path / "out"),
                "log_level": "WARNING",
            }
        ),
        encoding="utf-8",
    )
    return settings_file


class TestCompaniesCommands:
    def test_import_then_list(self, settings_path: Path, tmp_path: Path, capsys):
        csv_path = tmp_path / "companies.csv"
        csv_path.write_text(COMPANIES_CSV, encoding="utf-8")

        exit_code = main(["--settings", str(settings_path), "companies", "import", str(csv_path)])
        assert exit_code == 0
        assert "Processed 1 row(s)" in capsys.readouterr().out

        exit_code = main(["--settings", str(settings_path), "companies", "list"])
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "Acme Corp" in out
        assert "greenhouse" in out

    def test_import_missing_file_returns_error(self, settings_path: Path, capsys):
        exit_code = main(["--settings", str(settings_path), "companies", "import", "does_not_exist.csv"])
        assert exit_code == 1
        assert "File not found" in capsys.readouterr().err

    def test_import_replace_deactivates_companies_not_in_new_file(self, settings_path: Path, tmp_path: Path, capsys):
        csv_path = tmp_path / "companies.csv"
        csv_path.write_text(COMPANIES_CSV, encoding="utf-8")
        main(["--settings", str(settings_path), "companies", "import", str(csv_path)])
        capsys.readouterr()

        other_csv = tmp_path / "other.csv"
        other_csv.write_text(
            "name,domain,ats_type,board_id,classification\nOther Co,other.example.com,ashby,other,product\n",
            encoding="utf-8",
        )
        exit_code = main(["--settings", str(settings_path), "companies", "import", str(other_csv), "--replace"])
        assert exit_code == 0
        assert "Deactivated 1" in capsys.readouterr().out

        main(["--settings", str(settings_path), "companies", "list", "--all"])
        assert "Acme Corp" in capsys.readouterr().out  # still present, just inactive

        main(["--settings", str(settings_path), "companies", "list"])
        assert "Acme Corp" not in capsys.readouterr().out  # excluded from active-only listing


class TestOverridesCommand:
    def test_set_job_override(self, settings_path: Path, capsys):
        exit_code = main(
            ["--settings", str(settings_path), "overrides", "set", "job", "5", "verdict", "strong_match", "--reason", "known team"]
        )
        assert exit_code == 0
        assert "Override set: job#5.verdict = strong_match" in capsys.readouterr().out

    def test_set_company_override(self, settings_path: Path, capsys):
        exit_code = main(
            ["--settings", str(settings_path), "overrides", "set", "company", "3", "classification", "consulting"]
        )
        assert exit_code == 0
        assert "Override set: company#3.classification = consulting" in capsys.readouterr().out


class TestReportAndAuditCommands:
    def test_report_with_empty_database(self, settings_path: Path, capsys):
        exit_code = main(["--settings", str(settings_path), "report"])
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "Markdown:" in out
        md_path = Path(out.splitlines()[0].split("Markdown: ", 1)[1])
        assert md_path.exists()
        assert md_path.name.endswith("PT.md")  # Seattle-time filename, not UTC

    def test_audit_with_empty_database(self, settings_path: Path, capsys):
        exit_code = main(["--settings", str(settings_path), "audit"])
        assert exit_code == 0
        assert "Filtered postings: 0" in capsys.readouterr().out


class TestCrawlCommand:
    @respx.mock
    def test_crawl_prints_stats(self, settings_path: Path, tmp_path: Path, capsys):
        csv_path = tmp_path / "companies.csv"
        csv_path.write_text(COMPANIES_CSV, encoding="utf-8")
        main(["--settings", str(settings_path), "companies", "import", str(csv_path)])
        capsys.readouterr()

        respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
            return_value=httpx.Response(
                200,
                json={
                    "jobs": [
                        {
                            "id": 1, "title": "Senior Backend Engineer",
                            "location": {"name": "Remote - US"},
                            "absolute_url": "https://acme.example.com/jobs/1",
                            "content": "<p>$180,000 - $220,000</p>",
                            "metadata": [], "departments": [],
                        }
                    ]
                },
            )
        )

        exit_code = main(["--settings", str(settings_path), "crawl"])
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "Companies attempted: 1" in out
        assert "Companies successfully crawled: 1" in out
        assert "Postings fetched (all departments): 1" in out
        assert "Skipped as outside the software-engineering job family: 0" in out


class TestEvaluateDryRun:
    def test_counts_what_would_be_sent_without_calling_the_api(self, settings_path, monkeypatch, capsys):
        # A key is set, so a real client would be built — the dry run must still make no call.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-unused")
        settings = load_settings(settings_path=settings_path, load_dotenv_file=False)
        with Database(settings.db_path) as db:
            company = Company(
                name="Acme", domain="acme.example.com", careers_url=None, ats_type=AtsType.GREENHOUSE,
                board_id="acme", classification=CompanyClassification.PRODUCT,
                classification_source=ClassificationSource.SEED,
            )
            company.id = db.upsert_company(company)
            db.upsert_job(normalize_posting(RawPosting(
                source_job_id="1", title="Senior Backend Engineer", location_raw="Remote - US",
                employment_type_raw="Full-time",
                description_html="<p>We build our own SaaS product. $180,000 - $220,000 per year.</p>",
                description_text=None, posting_url="https://acme.example.com/jobs/1", apply_url=None,
            ), company, settings))

        assert main(["--settings", str(settings_path), "evaluate", "--dry-run"]) == 0

        assert "Dry run, no API calls: 1 postings would be sent to the LLM" in capsys.readouterr().out
        with Database(settings.db_path) as db:
            assert db.latest_evaluation_run() is None


class TestRunBackup:
    """`run` — the daily command — ends with a verified database backup when a folder is set."""

    @pytest.fixture(autouse=True)
    def _no_llm(self, monkeypatch):
        # main() builds a real client from the environment; an empty key means none is built,
        # so `run` filters and reports without any LLM call.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")

    def test_run_ends_with_a_verified_backup(self, settings_path, tmp_path, monkeypatch, capsys):
        backups = tmp_path / "drive"
        monkeypatch.setenv("JOBSCAN_BACKUP_DIR", str(backups))

        assert main(["--settings", str(settings_path), "run"]) == 0

        assert (backups / "jobscan.db.gz").exists()
        assert "integrity check passed" in capsys.readouterr().out

    def test_other_commands_do_not_back_up(self, settings_path, tmp_path, monkeypatch):
        backups = tmp_path / "drive"
        monkeypatch.setenv("JOBSCAN_BACKUP_DIR", str(backups))
        csv_path = tmp_path / "companies.csv"
        csv_path.write_text(COMPANIES_CSV, encoding="utf-8")

        assert main(["--settings", str(settings_path), "companies", "import", str(csv_path)]) == 0

        assert not backups.exists()

    def test_a_failed_backup_is_reported_on_stdout_and_fails_the_run(self, settings_path, tmp_path, monkeypatch, capsys):
        from jobscan import backup
        monkeypatch.setenv("JOBSCAN_BACKUP_DIR", str(tmp_path / "drive"))
        monkeypatch.setattr(backup, "_integrity_problem", lambda path: "malformed")

        assert main(["--settings", str(settings_path), "run"]) == 1

        assert "Database backup FAILED" in capsys.readouterr().out

    def test_no_backup_folder_means_no_backup(self, settings_path, capsys):
        assert main(["--settings", str(settings_path), "run"]) == 0
        assert "Database backup" not in capsys.readouterr().out
