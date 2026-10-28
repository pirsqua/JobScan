from __future__ import annotations

import dataclasses
import datetime as dt

import pytest

from jobscan.applications import load_applications, load_pauses, normalize_url
from jobscan.db import Database
from jobscan.evaluate import evaluate_all
from jobscan.models import Verdict
from jobscan.reports import data as report_data
from jobscan.reports.csv_report import write_csv_report
from jobscan.reports.data import assemble_report_data
from jobscan.reports.json_report import build_json_report
from jobscan.reports.markdown_report import render_markdown
from jobscan.timeutil import SEATTLE_TZ

from tests.fakes import seed_company, seed_job, strong_match_client


def seed_acme_posting(db: Database, settings):
    """Acme Corp and its one posting, https://acme.example.com/jobs/1."""
    company = seed_company(db, "Acme Corp")
    return company, seed_job(db, settings, company)


def write_applications(settings, *entries: str) -> None:
    settings.applications_path.write_text("applications:\n" + "".join(entries), encoding="utf-8")


def entry(url: str, applied_on: str = "2026-09-24", company: str = "Acme Corp", title: str = "Senior Backend Engineer",
          status: str | None = None, status_on: str | None = None) -> str:
    lines = f"  - company: {company}\n    title: {title}\n    url: {url}\n    applied_on: {applied_on}\n"
    if status:
        lines += f"    status: {status}\n"
    if status_on:
        lines += f"    status_on: {status_on}\n"
    return lines


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

    def test_status_defaults_to_applied_and_is_read_when_given(self, settings):
        write_applications(
            settings,
            entry("https://acme.example.com/jobs/1"),
            entry("https://acme.example.com/jobs/2", status="Rejected", status_on="2026-10-02"),
        )
        waiting, passed = load_applications(settings.applications_path)
        assert (waiting.status, waiting.status_on) == ("applied", None)
        assert (passed.status, passed.status_on) == ("rejected", dt.date(2026, 10, 2))

    def test_an_unknown_status_is_an_error_naming_the_choices(self, settings):
        # A typo would otherwise count the application as still waiting for a reply.
        write_applications(settings, entry("https://acme.example.com/jobs/1", status="rejcted"))
        with pytest.raises(ValueError, match="unknown status 'rejcted'.*applied, screening, interviewing"):
            load_applications(settings.applications_path)

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
        seed_acme_posting(db, settings)
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

    def test_applications_show_their_status_and_a_summary(self, db: Database, settings, monkeypatch):
        monkeypatch.setattr(report_data, "now_seattle", lambda: dt.datetime(2026, 10, 24, 9, tzinfo=SEATTLE_TZ))
        write_applications(
            settings,
            entry("https://acme.example.com/jobs/1", applied_on="2026-10-01"),
            entry("https://acme.example.com/jobs/2", applied_on="2026-09-10", status="rejected", status_on="2026-09-23"),
            entry("https://acme.example.com/jobs/3", applied_on="2026-09-15", status="interviewing"),
        )

        data = assemble_report_data(db, settings)
        markdown = render_markdown(data)

        assert "**3** applications — 1 awaiting a reply, 1 interviewing, 1 rejected" in markdown
        assert "| Applied | Role | Status | Where it stands |" in markdown
        assert "| Rejected 2026-09-23 |" in markdown
        assert "| Interviewing |" in markdown
        assert "| No reply yet (23 days) |" in markdown
        rejected = next(a for a in build_json_report(data)["applications"] if a["status"] == "rejected")
        assert (rejected["status_on"], rejected["days_waiting"]) == ("2026-09-23", None)

    def test_closed_and_untracked_postings_still_listed(self, db: Database, settings):
        company, _ = seed_acme_posting(db, settings)
        db.close_missing_jobs(company.id, seen_source_job_ids=set())
        write_applications(
            settings,
            entry("https://acme.example.com/jobs/1"),
            entry("https://elsewhere.example.com/jobs/9", applied_on="2026-09-10", company="Other", title="Engineer"),
        )

        standings = [row.standing for row in assemble_report_data(db, settings).applications]

        assert standings[0] == "Not tracked (posting never crawled)"  # sorted by date: 09-10 first
        assert standings[1].startswith("Closed ")

    def test_standing_says_why_an_applied_role_is_off_the_shortlist(self, db: Database, settings):
        company, job = seed_acme_posting(db, settings)
        write_applications(settings, entry("https://acme.example.com/jobs/1"))

        def standing() -> str:
            (row,) = assemble_report_data(db, settings).applications
            return row.standing

        assert standing() == "Not evaluated yet"
        evaluate_all(db, settings, client=strong_match_client())
        db.save_evaluation(dataclasses.replace(
            db.get_evaluation_for_job(job.id), id=None, verdict=Verdict.REJECT, worth_applying=False,
            primary_rejection_reason="Requires production Rust."))
        assert standing() == "Rejected — Requires production Rust."
        seed_job(db, settings, company, location="Hybrid - Seattle, WA")
        assert standing() == "Filtered out (not_remote_explicit)"
        db.conn.execute("UPDATE companies SET active = 0 WHERE id = ?", (company.id,))
        assert standing() == "Company no longer searched"

    def test_paused_company_is_held_out_of_the_shortlist_until_its_date(self, db: Database, settings):
        # Observed: an employer accepting only three applications per 60 days. Its roles stay crawled
        # and evaluated, but leave the shortlist until the window resets — and say so, rather than
        # silently vanishing.
        seed_acme_posting(db, settings)
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
        seed_acme_posting(db, settings)
        evaluate_all(db, settings, client=strong_match_client())
        write_pause(settings, "Acme Corp", "2000-01-01")

        data = assemble_report_data(db, settings)

        assert len(data.best_bets) == 1 and data.paused == []
        assert "## Paused" not in render_markdown(data)

    def test_csv_carries_applied_on(self, db: Database, settings, tmp_path):
        seed_acme_posting(db, settings)
        evaluate_all(db, settings, client=strong_match_client())
        write_applications(settings, entry("https://acme.example.com/jobs/1"))

        path = tmp_path / "report.csv"
        write_csv_report(assemble_report_data(db, settings), path)

        assert "applied_on" in path.read_text(encoding="utf-8").splitlines()[0]
        assert "2026-09-24" in path.read_text(encoding="utf-8")
