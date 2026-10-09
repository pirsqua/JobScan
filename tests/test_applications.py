from __future__ import annotations

import datetime as dt

from jobscan.applications import load_applications, normalize_url
from jobscan.db import Database
from jobscan.evaluate import evaluate_all
from jobscan.reports.csv_report import write_csv_report
from jobscan.reports.data import assemble_report_data
from jobscan.reports.json_report import build_json_report
from jobscan.reports.markdown_report import render_markdown

from tests.test_reports import seed_evaluated_job, strong_match_client


def write_applications(settings, *entries: str) -> None:
    settings.applications_path.write_text("applications:\n" + "".join(entries), encoding="utf-8")


def entry(url: str, applied_on: str = "2026-09-24", company: str = "Acme Corp", title: str = "Senior Backend Engineer") -> str:
    return f"  - company: {company}\n    title: {title}\n    url: {url}\n    applied_on: {applied_on}\n"


class TestLoading:
    def test_no_file_means_no_applications(self, settings):
        assert load_applications(settings.applications_path) == []

    def test_dates_are_parsed(self, settings):
        write_applications(settings, entry("https://acme.example.com/jobs/1"))
        (application,) = load_applications(settings.applications_path)
        assert application.applied_on == dt.date(2026, 9, 24)

    def test_url_variants_of_the_same_posting_match(self):
        # Observed: the candidate's links had a trailing slash (Reddit) or lacked one before the
        # query string (Posit) where the crawled posting_url didn't / did.
        assert normalize_url("https://job-boards.greenhouse.io/reddit/jobs/6469397/") == \
            normalize_url("https://job-boards.greenhouse.io/reddit/jobs/6469397")
        assert normalize_url("https://posit.co/job-detail?gh_jid=7984391003") == \
            normalize_url("https://posit.co/job-detail/?gh_jid=7984391003")
        assert normalize_url("https://x.example/jobs?id=1") != normalize_url("https://x.example/jobs?id=2")


class TestReport:
    def test_applied_role_is_starred_where_it_stands(self, db: Database, settings):
        seed_evaluated_job(db, settings)
        evaluate_all(db, settings, client=strong_match_client())
        write_applications(settings, entry("https://ACME.example.com/jobs/1/"))

        data = assemble_report_data(db, settings)

        assert data.best_bets[0].applied_on == dt.date(2026, 9, 24)
        (application,) = data.applications
        assert application.standing == "Best Bet"
        markdown = render_markdown(data)
        assert "## Applications" in markdown
        assert "★ Acme Corp — Senior Backend Engineer (applied 2026-09-24)" in markdown
        assert build_json_report(data)["best_bets"][0]["applied_on"] == "2026-09-24"

    def test_closed_and_untracked_postings_still_listed(self, db: Database, settings):
        company, _ = seed_evaluated_job(db, settings)
        db.close_missing_jobs(company.id, seen_source_job_ids=set())
        write_applications(
            settings,
            entry("https://acme.example.com/jobs/1"),
            entry("https://elsewhere.example.com/jobs/9", applied_on="2026-09-10", company="Other", title="Engineer"),
        )

        standings = [row.standing for row in assemble_report_data(db, settings).applications]

        assert standings[0] == "Not tracked (posting never crawled)"  # sorted by date: 09-10 first
        assert standings[1].startswith("Closed ")

    def test_csv_carries_applied_on(self, db: Database, settings, tmp_path):
        seed_evaluated_job(db, settings)
        evaluate_all(db, settings, client=strong_match_client())
        write_applications(settings, entry("https://acme.example.com/jobs/1"))

        path = tmp_path / "report.csv"
        write_csv_report(assemble_report_data(db, settings), path)

        assert "applied_on" in path.read_text(encoding="utf-8").splitlines()[0]
        assert "2026-09-24" in path.read_text(encoding="utf-8")
