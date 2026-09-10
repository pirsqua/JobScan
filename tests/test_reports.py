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


class TestReports:
    def test_markdown_report_renders_without_evaluations(self, db: Database, settings):
        seed_evaluated_job(db, settings)
        data = assemble_report_data(db, settings)
        markdown = render_markdown(data)
        assert "Run statistics" in markdown
        assert "Recommended roles" in markdown

    def test_markdown_report_lists_recommendation_after_evaluation(self, db: Database, settings):
        from types import SimpleNamespace

        from jobscan.llm.schemas import JOB_EVALUATION_TOOL_NAME

        seed_evaluated_job(db, settings)

        block = SimpleNamespace(
            type="tool_use",
            name=JOB_EVALUATION_TOOL_NAME,
            input={
                "verdict": "strong_match",
                "confidence": 0.9,
                "is_product_company": True,
                "compensation_assessment": "Well above $170,000.",
                "remote_verification": "US remote.",
                "required_matches": ["C#", "Azure"],
                "required_gaps": [],
                "preferred_gaps": [],
                "minor_caveats": [],
                "evidence": ["quote"],
                "credibility_assessment": "Great fit.",
                "primary_rejection_reason": None,
            },
        )
        usage = SimpleNamespace(input_tokens=100, output_tokens=50)
        response = SimpleNamespace(content=[block], usage=usage)

        class Messages:
            def create(self, **kwargs):
                return response

        class Sdk:
            messages = Messages()

        client = AnthropicClient(api_key=None, model="test-model", client=Sdk())
        eval_stats = evaluate_all(db, settings, client=client)

        data = assemble_report_data(db, settings, evaluate_stats=eval_stats)
        markdown = render_markdown(data)
        assert "Acme Corp" in markdown
        assert "strong_match" in markdown

        json_report = build_json_report(data)
        assert json_report["recommended"][0]["company"] == "Acme Corp"

        csv_path = settings.output_dir / "report.csv"
        write_csv_report(data, csv_path)
        assert csv_path.exists()
        assert "Acme Corp" in csv_path.read_text(encoding="utf-8")
