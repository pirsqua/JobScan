"""CLI-level tests: argument parsing, command dispatch, and printed output.

Deliberately avoids `evaluate`/`run` here — `main()` always builds a real AnthropicClient from
settings/env with no injection seam, so LLM-touching behavior is covered at the `evaluate_all()`
level in test_evaluate.py instead of faking a client through the CLI layer.
"""
from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx
import yaml

from jobscan.cli import main

COMPANIES_CSV = """name,domain,ats_type,board_id,classification
Acme Corp,acme.example.com,greenhouse,acme,product
"""


@pytest.fixture(autouse=True)
def _clean_jobscan_env(monkeypatch):
    # main() loads settings from the real environment/.env — strip anything that could make
    # these tests depend on the host machine's ambient configuration.
    for key in (
        "JOBSCAN_DB_PATH", "JOBSCAN_OUTPUT_DIR", "JOBSCAN_ANTHROPIC_MODEL",
        "JOBSCAN_LOG_LEVEL", "JOBSCAN_PROFILE_PATH",
    ):
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
