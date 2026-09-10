from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from jobscan.db import Database
from jobscan.evaluate import evaluate_all
from jobscan.llm.client import AnthropicClient
from jobscan.llm.schemas import JOB_EVALUATION_TOOL_NAME
from jobscan.models import (
    AtsType,
    ClassificationSource,
    Company,
    CompanyClassification,
    ManualOverride,
    RawPosting,
    Verdict,
)
from jobscan.normalize import normalize_posting

VALID_EVAL_INPUT = {
    "verdict": "strong_match",
    "confidence": 0.8,
    "is_product_company": True,
    "compensation_assessment": "Comfortably above $170,000.",
    "remote_verification": "US remote, no restrictions noted.",
    "required_matches": ["C#", "Azure"],
    "required_gaps": [],
    "preferred_gaps": [],
    "minor_caveats": [],
    "evidence": ["quoted text"],
    "credibility_assessment": "Strong fit.",
    "primary_rejection_reason": None,
}


class FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.call_count = 0

    def create(self, **kwargs):
        self.call_count += 1
        if not self._responses:
            raise AssertionError("LLM was called more times than expected")
        return self._responses.pop(0)


class FakeSdkClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)


def tool_response(input_dict, input_tokens=100, output_tokens=50):
    block = SimpleNamespace(type="tool_use", name=JOB_EVALUATION_TOOL_NAME, input=input_dict)
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(content=[block], usage=usage)


def seed_company(db: Database, name="Acme", board_id="acme", classification=None) -> Company:
    company = Company(
        name=name, domain=f"{board_id}.example.com", careers_url=None,
        ats_type=AtsType.GREENHOUSE, board_id=board_id,
    )
    if classification is not None:
        company.classification = classification
        company.classification_source = ClassificationSource.SEED
    company.id = db.upsert_company(company)
    return company


def seed_job(db: Database, settings, company: Company, source_job_id="1", title="Senior Backend Engineer", description="We build our own SaaS product. $180,000 - $220,000 per year."):
    raw = RawPosting(
        source_job_id=source_job_id,
        title=title,
        location_raw="Remote - US",
        employment_type_raw="Full-time",
        description_html=f"<p>{description}</p>",
        description_text=None,
        posting_url=f"https://acme.example.com/jobs/{source_job_id}",
        apply_url=f"https://acme.example.com/jobs/{source_job_id}/apply",
    )
    job = normalize_posting(raw, company, settings)
    job_id, _, _ = db.upsert_job(job)
    return db.get_job(job_id)


class TestEvaluateAll:
    def test_factual_rejection_skips_llm(self, db: Database, settings):
        company = seed_company(db)
        seed_job(db, settings, company, description="No salary mentioned here at all.")
        client = AnthropicClient(api_key=None, model="test-model", client=FakeSdkClient([]))

        stats = evaluate_all(db, settings, client=client)

        assert stats.factual_rejected == 1
        assert stats.sent_to_llm == 0

    def test_llm_evaluation_happy_path(self, db: Database, settings):
        company = seed_company(db, classification=CompanyClassification.PRODUCT)
        seed_job(db, settings, company)
        client = AnthropicClient(api_key=None, model="test-model", client=FakeSdkClient([tool_response(VALID_EVAL_INPUT)]))

        stats = evaluate_all(db, settings, client=client)

        assert stats.sent_to_llm == 1
        assert stats.verdict_counts["strong_match"] == 1

    def test_second_run_uses_cache_not_new_llm_call(self, db: Database, settings):
        company = seed_company(db, classification=CompanyClassification.PRODUCT)
        seed_job(db, settings, company)
        sdk = FakeSdkClient([tool_response(VALID_EVAL_INPUT)])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        evaluate_all(db, settings, client=client)
        stats2 = evaluate_all(db, settings, client=client)

        assert sdk.messages.call_count == 1
        assert stats2.cache_hits == 1
        assert stats2.sent_to_llm == 0

    def test_duplicate_description_across_jobs_reuses_cache(self, db: Database, settings):
        company = seed_company(db, classification=CompanyClassification.PRODUCT)
        same_description = "We build our own SaaS product. $180,000 - $220,000 per year."
        seed_job(db, settings, company, source_job_id="1", description=same_description)
        seed_job(db, settings, company, source_job_id="2", title="Senior Backend Engineer II", description=same_description)

        sdk = FakeSdkClient([tool_response(VALID_EVAL_INPUT)])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        stats = evaluate_all(db, settings, client=client)

        assert sdk.messages.call_count == 1
        assert stats.sent_to_llm == 1
        assert stats.cache_hits == 1

    def test_no_api_key_marks_unverified_not_rejected(self, db: Database, settings):
        company = seed_company(db)
        seed_job(db, settings, company)
        assert settings.anthropic_api_key is None

        stats = evaluate_all(db, settings)

        assert stats.unverified == 1
        assert stats.factual_rejected == 0

    def test_job_level_manual_override_skips_llm(self, db: Database, settings):
        company = seed_company(db)
        job = seed_job(db, settings, company, description="No salary published here.")
        db.set_manual_override(
            ManualOverride("job", job.id, "verdict", "strong_match", "I know this role", datetime.now(timezone.utc))
        )
        client = AnthropicClient(api_key=None, model="test-model", client=FakeSdkClient([]))

        stats = evaluate_all(db, settings, client=client)

        assert stats.manual_overrides_applied == 1
        assert stats.factual_rejected == 0
        evaluation = db.get_evaluation_for_job(job.id)
        assert evaluation.verdict == Verdict.STRONG_MATCH

    def test_company_classification_override_rejects_without_llm_classification_call(self, db: Database, settings):
        company = seed_company(db)  # unknown by default
        seed_job(db, settings, company)
        db.set_manual_override(
            ManualOverride("company", company.id, "classification", "consulting", "known staffing shop", datetime.now(timezone.utc))
        )
        client = AnthropicClient(api_key=None, model="test-model", client=FakeSdkClient([]))

        stats = evaluate_all(db, settings, client=client)

        assert stats.factual_rejected == 1
        assert stats.companies_classified == 0
