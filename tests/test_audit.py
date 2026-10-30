from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from jobscan.audit import AuditRow, build_audit_rows, write_audit_csv, write_audit_markdown
from jobscan.db import Database
from jobscan.models import FilterLogEntry, Verdict
from tests.fakes import make_evaluation, seed_company, seed_job


class TestBuildAuditRows:
    def test_includes_factual_filter_entries(self, db: Database, settings):
        company = seed_company(db)
        job = seed_job(db, settings, company)
        db.record_filter_log(
            FilterLogEntry(
                job.id,
                "factual_filter",
                "salary_below_min",
                "published max $150,000 is below the $170,000 minimum",
                datetime.now(UTC),
            )
        )

        rows = build_audit_rows(db)

        assert len(rows) == 1
        assert rows[0].company == "Acme"
        assert rows[0].stage == "factual_filter"
        assert rows[0].reason == "salary_below_min"

    def test_includes_llm_reject_with_gaps_and_evidence(self, db: Database, settings):
        company = seed_company(db)
        job = seed_job(db, settings, company)
        db.save_evaluation(
            make_evaluation(
                job.id,
                Verdict.REJECT,
                primary_rejection_reason="Requires AWS",
                required_gaps=["No AWS"],
                evidence=["quote"],
            )
        )

        rows = build_audit_rows(db)

        assert len(rows) == 1
        assert rows[0].stage == "llm_reject"
        assert rows[0].reason == "Requires AWS"
        assert '"No AWS"' in rows[0].detail
        assert '"quote"' in rows[0].detail

    def test_includes_llm_borderline(self, db: Database, settings):
        company = seed_company(db)
        job = seed_job(db, settings, company)
        db.save_evaluation(make_evaluation(job.id, Verdict.BORDERLINE))

        rows = build_audit_rows(db)

        assert len(rows) == 1
        assert rows[0].stage == "llm_borderline"

    def test_reject_without_primary_reason_gets_placeholder(self, db: Database, settings):
        company = seed_company(db)
        job = seed_job(db, settings, company)
        db.save_evaluation(make_evaluation(job.id, Verdict.REJECT, primary_rejection_reason=None))

        rows = build_audit_rows(db)

        assert rows[0].reason == "(no primary rejection reason given)"

    def test_excludes_strong_and_plausible_matches(self, db: Database, settings):
        company = seed_company(db)
        job = seed_job(db, settings, company)
        db.save_evaluation(make_evaluation(job.id, Verdict.STRONG_MATCH))

        assert build_audit_rows(db) == []

    def test_combines_factual_and_llm_rows_across_jobs(self, db: Database, settings):
        company = seed_company(db)
        filtered_job = seed_job(db, settings, company, source_job_id="1", title="Contract Role")
        db.record_filter_log(
            FilterLogEntry(
                filtered_job.id, "factual_filter", "not_full_time", "contract", datetime.now(UTC)
            )
        )
        rejected_job = seed_job(
            db, settings, company, source_job_id="2", title="Go Backend Engineer"
        )
        db.save_evaluation(
            make_evaluation(rejected_job.id, Verdict.REJECT, primary_rejection_reason="Go required")
        )

        rows = build_audit_rows(db)

        assert {r.stage for r in rows} == {"factual_filter", "llm_reject"}
        assert len(rows) == 2


class TestWriteAuditCsv:
    def test_writes_header_and_rows(self, tmp_path: Path, db: Database, settings):
        company = seed_company(db)
        job = seed_job(db, settings, company)
        db.record_filter_log(
            FilterLogEntry(
                job.id, "factual_filter", "salary_below_min", "detail text", datetime.now(UTC)
            )
        )
        rows = build_audit_rows(db)

        path = tmp_path / "audit.csv"
        write_audit_csv(rows, path)

        content = path.read_text(encoding="utf-8")
        assert "company,title,posting_url,stage,reason,detail" in content
        assert "Acme" in content
        assert "salary_below_min" in content


class TestWriteAuditMarkdown:
    def test_empty_rows_shows_placeholder(self, tmp_path: Path):
        path = tmp_path / "audit.md"
        write_audit_markdown([], path)

        content = path.read_text(encoding="utf-8")
        assert "Total filtered postings: 0" in content
        assert "Nothing has been filtered yet" in content

    def test_renders_table_with_rows(self, tmp_path: Path, db: Database, settings):
        company = seed_company(db)
        job = seed_job(db, settings, company)
        db.record_filter_log(
            FilterLogEntry(
                job.id, "factual_filter", "salary_below_min", "detail text", datetime.now(UTC)
            )
        )
        rows = build_audit_rows(db)

        path = tmp_path / "audit.md"
        write_audit_markdown(rows, path)

        content = path.read_text(encoding="utf-8")
        assert "Total filtered postings: 1" in content
        assert "| Acme |" in content
        assert "[posting](" in content

    def test_long_detail_is_truncated(self, tmp_path: Path):
        row = AuditRow(
            company="Acme",
            title="Engineer",
            posting_url=None,
            stage="llm_reject",
            reason="gap",
            detail="x" * 300,
        )
        path = tmp_path / "audit.md"
        write_audit_markdown([row], path)

        content = path.read_text(encoding="utf-8")
        assert "x" * 200 + "…" in content
        assert "x" * 300 not in content

    def test_pipe_and_newline_in_detail_are_escaped(self, tmp_path: Path):
        row = AuditRow(
            company="Acme",
            title="Engineer",
            posting_url=None,
            stage="llm_reject",
            reason="gap",
            detail="line1|with pipe\nline2",
        )
        path = tmp_path / "audit.md"
        write_audit_markdown([row], path)

        content = path.read_text(encoding="utf-8")
        assert "line1\\|with pipe line2" in content

    def test_no_link_when_posting_url_missing(self, tmp_path: Path):
        row = AuditRow(
            company="Acme",
            title="Engineer",
            posting_url=None,
            stage="factual_filter",
            reason="no_salary_published",
            detail=None,
        )
        path = tmp_path / "audit.md"
        write_audit_markdown([row], path)

        content = path.read_text(encoding="utf-8")
        assert "| Acme | Engineer | factual_filter | no_salary_published |  |  |" in content
