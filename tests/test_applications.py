from __future__ import annotations

import datetime as dt

from jobscan.applications import load_applications, load_pauses, normalize_url
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


def write_pause(settings, company: str, until: str, reason: str = "") -> None:
    settings.applications_path.write_text(
        f"applications: []\npaused:\n  - company: {company}\n    until: {until}\n    reason: {reason}\n", encoding="utf-8"
    )


class TestLoading:
    def test_no_file_means_no_applications(self, settings):
        assert load_applications(settings.applications_path) == []

    def test_dates_are_parsed(self, settings):
        write_applications(settings, entry("https://acme.example.com/jobs/1"))
        (application,) = load_applications(settings.applications_path)
        assert application.applied_on == dt.date(2026, 9, 24)

    def test_pauses_are_optional(self, settings):
        write_applications(settings, entry("https://acme.example.com/jobs/1"))
        assert load_pauses(settings.applications_path) == []
        write_pause(settings, "Acme Corp", "2026-12-20", reason="3 applications per 60 days")
        (pause,) = load_pauses(settings.applications_path)
        assert pause.until == dt.date(2026, 12, 20) and pause.reason == "3 applications per 60 days"

    def test_url_variants_of_the_same_posting_match(self):
        # Observed: recorded links with a trailing slash, or without one before the query string,
        # where the crawled posting_url didn't / did.
        assert normalize_url("https://job-boards.greenhouse.io/acme/jobs/1234567/") == \
            normalize_url("https://job-boards.greenhouse.io/acme/jobs/1234567")
        assert normalize_url("https://acme.example.com/job-detail?gh_jid=7654321") == \
            normalize_url("https://acme.example.com/job-detail/?gh_jid=7654321")
        assert normalize_url("https://x.example/jobs?id=1") != normalize_url("https://x.example/jobs?id=2")


class TestReport:
    def test_applied_role_leaves_the_shortlist_and_shows_where_it_stands(self, db: Database, settings):
        # The shortlist holds only what still needs a decision; the Applications table says where
        # an applied posting stands.
        seed_evaluated_job(db, settings)
        evaluate_all(db, settings, client=strong_match_client())
        write_applications(settings, entry("https://ACME.example.com/jobs/1/"))

        data = assemble_report_data(db, settings)

        assert data.best_bets == []
        (application,) = data.applications
        assert application.standing == "Best Bet"
        markdown = render_markdown(data)
        assert "## Applications" in markdown
        assert "| 2026-09-24 | [Acme Corp — Senior Backend Engineer]" in markdown
        report = build_json_report(data)
        assert report["best_bets"] == [] and report["applications"][0]["standing"] == "Best Bet"
        assert report["all_evaluated"][0]["applied_on"] == "2026-09-24"

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

    def test_paused_company_is_held_out_of_the_shortlist_until_its_date(self, db: Database, settings):
        # Observed: an employer accepting only three applications per 60 days. Its roles stay crawled
        # and evaluated, but leave the shortlist until the window resets — and say so, rather than
        # silently vanishing.
        seed_evaluated_job(db, settings)
        evaluate_all(db, settings, client=strong_match_client())
        write_pause(settings, "acme corp", "2999-01-01", reason="3 applications per 60 days")

        data = assemble_report_data(db, settings)

        assert data.best_bets == []
        (paused,) = data.paused
        assert paused.held_back == 1
        assert ("- **acme corp** until 2999-01-01 — 3 applications per 60 days (1 shortlisted role held back)"
                in render_markdown(data))
        assert build_json_report(data)["paused"] == [
            {"company": "acme corp", "until": "2999-01-01", "reason": "3 applications per 60 days", "held_back": 1}
        ]

    def test_pause_ends_on_its_date(self, db: Database, settings):
        seed_evaluated_job(db, settings)
        evaluate_all(db, settings, client=strong_match_client())
        write_pause(settings, "Acme Corp", "2000-01-01")

        data = assemble_report_data(db, settings)

        assert len(data.best_bets) == 1 and data.paused == []
        assert "## Paused" not in render_markdown(data)

    def test_csv_carries_applied_on(self, db: Database, settings, tmp_path):
        seed_evaluated_job(db, settings)
        evaluate_all(db, settings, client=strong_match_client())
        write_applications(settings, entry("https://acme.example.com/jobs/1"))

        path = tmp_path / "report.csv"
        write_csv_report(assemble_report_data(db, settings), path)

        assert "applied_on" in path.read_text(encoding="utf-8").splitlines()[0]
        assert "2026-09-24" in path.read_text(encoding="utf-8")
