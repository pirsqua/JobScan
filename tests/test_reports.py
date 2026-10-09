from __future__ import annotations

from jobscan.db import Database
from jobscan.evaluate import evaluate_all
from jobscan.llm.client import AnthropicClient
from jobscan.models import AtsType, ClassificationSource, Company, CompanyClassification, RawPosting
from jobscan.normalize import normalize_posting
from jobscan.reports.csv_report import write_csv_report
from jobscan.reports.data import assemble_report_data
from jobscan.reports.json_report import build_json_report
from jobscan.reports.markdown_report import render_markdown


def seed_evaluated_job(db: Database, settings):
    company = Company(
        name="Acme Corp", domain="acme.example.com", careers_url=None,
        ats_type=AtsType.GREENHOUSE, board_id="acme",
        classification=CompanyClassification.PRODUCT, classification_source=ClassificationSource.SEED,
    )
    company.id = db.upsert_company(company)
    raw = RawPosting(
        source_job_id="1",
        title="Senior Backend Engineer",
        location_raw="Remote - US",
        employment_type_raw="Full-time",
        description_html="<p>We build our own SaaS product. $180,000 - $220,000 per year.</p>",
        description_text=None,
        posting_url="https://acme.example.com/jobs/1",
        apply_url="https://acme.example.com/jobs/1/apply",
    )
    job = normalize_posting(raw, company, settings)
    db.upsert_job(job)
    return company, raw


def strong_match_client() -> AnthropicClient:
    from types import SimpleNamespace

    from jobscan.llm.schemas import JOB_EVALUATION_TOOL_NAME

    block = SimpleNamespace(
        type="tool_use",
        name=JOB_EVALUATION_TOOL_NAME,
        input={
            "verdict": "strong_match",
            "confidence": 0.9,
            "scope_fit": "at_level",
            "evidence_coverage_percent": 90,
            "specialist_tenure_assessment": {"classification": "not_applicable", "specialty": "", "explanation": ""},
            "requirement_evidence": [],
            "growth_dimensions": [],
            "hidden_staff_signals": [],
            "is_product_company": True,
            "compensation_assessment": "Well above $170,000.",
            "remote_employment_verification": "US remote.",
            "working_hours_quote": "",
            "working_hours_fit": "compatible",
            "required_matches": ["C#", "Azure"],
            "required_gaps": [],
            "preferred_only_gaps": [],
            "minor_caveats": [],
            "evidence": ["quote"],
            "credibility_assessment": "Great fit.",
            "why_this_is_or_is_not_gettable": "At-level with strong direct evidence.",
            "primary_rejection_reason": None,
            "worth_applying": True,
        },
    )
    response = SimpleNamespace(content=[block], usage=SimpleNamespace(input_tokens=100, output_tokens=50))

    class Messages:
        def create(self, **kwargs):
            return response

    class Sdk:
        messages = Messages()

    return AnthropicClient(api_key=None, model="test-model", client=Sdk())


class TestReports:
    def test_markdown_report_renders_without_evaluations(self, db: Database, settings):
        seed_evaluated_job(db, settings)
        data = assemble_report_data(db, settings)
        markdown = render_markdown(data)
        assert "Run statistics" in markdown
        assert "Best Bets" in markdown

    def test_cached_verdict_is_dropped_once_the_job_fails_factual_filters(self, db: Database, settings):
        # Evaluations are cached by description text, so a re-crawl that reveals a fact the
        # verdict never saw (here: the role is actually hybrid) leaves the old strong_match in
        # place. The report must re-check the facts rather than trust the cached verdict.
        company, raw = seed_evaluated_job(db, settings)
        evaluate_all(db, settings, client=strong_match_client())
        assert assemble_report_data(db, settings).best_bets

        raw.location_raw = "Hybrid - Seattle, WA"
        db.upsert_job(normalize_posting(raw, company, settings))
        assert assemble_report_data(db, settings).best_bets == []

    def test_markdown_report_lists_recommendation_after_evaluation(self, db: Database, settings):
        seed_evaluated_job(db, settings)
        eval_stats = evaluate_all(db, settings, client=strong_match_client())

        data = assemble_report_data(db, settings, evaluate_stats=eval_stats)
        markdown = render_markdown(data)
        assert "Acme Corp" in markdown
        assert "strong_match" in markdown

        json_report = build_json_report(data)
        assert json_report["best_bets"][0]["company"] == "Acme Corp"

        csv_path = settings.output_dir / "report.csv"
        write_csv_report(data, csv_path)
        assert csv_path.exists()
        assert "Acme Corp" in csv_path.read_text(encoding="utf-8")
