from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from jobscan.audit import AuditRow, build_audit_rows, write_audit_csv, write_audit_markdown
from jobscan.db import Database
from jobscan.models import AtsType, Company, Evaluation, FilterLogEntry, RawPosting, Verdict
from jobscan.normalize import normalize_posting


def seed_company(db: Database, name: str = "Acme") -> Company:
    company = Company(
        name=name, domain=f"{name.lower()}.example.com", careers_url=None,
        ats_type=AtsType.GREENHOUSE, board_id=name.lower(),
    )
    company.id = db.upsert_company(company)
    return company


def seed_job(db: Database, settings, company: Company, source_job_id: str = "1", title: str = "Senior Backend Engineer"):
    raw = RawPosting(
        source_job_id=source_job_id, title=title, location_raw="Remote - US",
        employment_type_raw="Full-time", description_html="<p>We build our own SaaS product.</p>",
        description_text=None, posting_url=f"https://acme.example.com/jobs/{source_job_id}", apply_url=None,
    )
    job = normalize_posting(raw, company, settings)
    job_id, _, _ = db.upsert_job(job)
    return db.get_job(job_id)


def make_evaluation(
    job_id: int, verdict: Verdict, primary_rejection_reason: str | None = None,
    required_gaps: list[str] | None = None, evidence: list[str] | None = None,
) -> Evaluation:
    return Evaluation(
        job_id=job_id, description_hash="hash", verdict=verdict, confidence=0.5,
        compensation_assessment="ok", remote_verification="ok",
        required_matches=[], required_gaps=required_gaps or [], preferred_gaps=[],
        minor_caveats=[], evidence=evidence or [], credibility_assessment="ok",
        is_product_company=True, primary_rejection_reason=primary_rejection_reason,
        model_name="test-model", created_at=datetime.now(timezone.utc),
    )


class TestBuildAuditRows:
    def test_includes_factual_filter_entries(self, db: Database, settings):
        company = seed_company(db)
        job = seed_job(db, settings, company)
        db.record_filter_log(
            FilterLogEntry(job.id, "factual_filter", "salary_below_min", "published max $150,000 is below the $170,000 minimum", datetime.now(timezone.utc))
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
            make_evaluation(job.id, Verdict.REJECT, primary_rejection_reason="Requires AWS", required_gaps=["No AWS"], evidence=["quote"])
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
        db.record_filter_log(FilterLogEntry(filtered_job.id, "factual_filter", "not_full_time", "contract", datetime.now(timezone.utc)))
        rejected_job = seed_job(db, settings, company, source_job_id="2", title="Go Backend Engineer")
        db.save_evaluation(make_evaluation(rejected_job.id, Verdict.REJECT, primary_rejection_reason="Go required"))

        rows = build_audit_rows(db)

        assert {r.stage for r in rows} == {"factual_filter", "llm_reject"}
        assert len(rows) == 2


class TestWriteAuditCsv:
    def test_writes_header_and_rows(self, tmp_path: Path, db: Database, settings):
        company = seed_company(db)
        job = seed_job(db, settings, company)
        db.record_filter_log(FilterLogEntry(job.id, "factual_filter", "salary_below_min", "detail text", datetime.now(timezone.utc)))
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
        db.record_filter_log(FilterLogEntry(job.id, "factual_filter", "salary_below_min", "detail text", datetime.now(timezone.utc)))
        rows = build_audit_rows(db)

        path = tmp_path / "audit.md"
        write_audit_markdown(rows, path)

        content = path.read_text(encoding="utf-8")
        assert "Total filtered postings: 1" in content
        assert "| Acme |" in content
        assert "[posting](" in content

    def test_long_detail_is_truncated(self, tmp_path: Path):
        row = AuditRow(company="Acme", title="Engineer", posting_url=None, stage="llm_reject", reason="gap", detail="x" * 300)
        path = tmp_path / "audit.md"
        write_audit_markdown([row], path)

        content = path.read_text(encoding="utf-8")
        assert "x" * 200 + "…" in content
        assert "x" * 300 not in content

    def test_pipe_and_newline_in_detail_are_escaped(self, tmp_path: Path):
        row = AuditRow(company="Acme", title="Engineer", posting_url=None, stage="llm_reject", reason="gap", detail="line1|with pipe\nline2")
        path = tmp_path / "audit.md"
        write_audit_markdown([row], path)

        content = path.read_text(encoding="utf-8")
        assert "line1\\|with pipe line2" in content

    def test_no_link_when_posting_url_missing(self, tmp_path: Path):
        row = AuditRow(company="Acme", title="Engineer", posting_url=None, stage="factual_filter", reason="no_salary_published", detail=None)
        path = tmp_path / "audit.md"
        write_audit_markdown([row], path)

        content = path.read_text(encoding="utf-8")
        assert "| Acme | Engineer | factual_filter | no_salary_published |  |  |" in content
