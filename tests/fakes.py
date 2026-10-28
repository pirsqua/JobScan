"""Test doubles shared across the suite: a stand-in for the Anthropic SDK, the model's answers,
and a database seeded with companies, postings and evaluations."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from jobscan.db import Database
from jobscan.llm.client import AnthropicClient
from jobscan.models import (
    AtsType,
    ClassificationSource,
    Company,
    CompanyClassification,
    Evaluation,
    EvidenceClassification,
    JobPosting,
    RawPosting,
    RequirementEvidence,
    RequirementImportance,
    ScopeFit,
    SpecialistTenureAssessment,
    SpecialistTenureClassification,
    Verdict,
)
from jobscan.normalize import normalize_posting

EXAMPLE_PROFILE = Path(__file__).resolve().parent.parent / "config" / "candidate_profile.example.yaml"


def data_folder(path: Path) -> Path:
    """A personal data folder holding the committed example profile, never the owner's own."""
    path.mkdir(parents=True, exist_ok=True)
    shutil.copy(EXAMPLE_PROFILE, path / "candidate_profile.yaml")
    return path


# A complete strong-match evaluation as the model returns it. Vary it with dict(VALID_EVALUATION, ...).
VALID_EVALUATION = {
    "day_to_day_work": "Building backend services.",
    "work_fit": "in_lane",
    "requirement_evidence": [
        {
            "requirement": "5+ years backend engineering",
            "posting_evidence": "5+ years of backend engineering experience required.",
            "stated_as": "required",
            "importance": "central",
            "candidate_evidence": "~15 years of backend engineering experience.",
            "evidence_classification": "directly_demonstrated",
        }
    ],
    "specialist_tenure_assessment": {
        "classification": "not_applicable",
        "specialty": "",
        "explanation": "No specialized tenure requirement beyond general backend experience.",
    },
    "required_matches": ["C#", "Azure"],
    "required_gaps": [],
    "preferred_only_gaps": ["Kubernetes"],
    "growth_dimensions": [],
    "hidden_staff_signals": [],
    "is_product_company": True,
    "remote_employment_verification": "Explicitly US remote, no state restrictions mentioned.",
    "working_hours_quote": "",
    "working_hours_fit": "compatible",
    "compensation_assessment": "Range comfortably clears $170,000.",
    "evidence": ["\"We build our own SaaS platform using C# and Azure.\""],
    "minor_caveats": [],
    "scope_fit": "at_level",
    "evidence_coverage_percent": 85,
    "credibility_assessment": "Strong fit, worth applying.",
    "why_this_is_or_is_not_gettable": "At-level scope with strong direct evidence makes this a credible near-term interview.",
    "verdict": "strong_match",
    "confidence": 0.85,
    "worth_applying": True,
    "primary_rejection_reason": None,
}


def json_response(data: dict, input_tokens: int = 100, output_tokens: int = 50) -> SimpleNamespace:
    """An SDK message whose one text block is ``data`` as JSON, the way a structured output arrives."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=json.dumps(data))],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
        stop_reason="end_turn",
    )


def refusal_response(category: str = "cyber") -> SimpleNamespace:
    """The ordinary 200 the model answers with when it declines a request."""
    return SimpleNamespace(
        content=[], usage=SimpleNamespace(input_tokens=10, output_tokens=0), stop_reason="refusal",
        stop_details=SimpleNamespace(type="refusal", category=category, explanation=None),
    )


class FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("LLM was called more times than expected")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeSdkClient:
    """Stands in for ``anthropic.Anthropic``: hands out the queued responses in order, raising any
    queued exception, and records every call's arguments in ``messages.calls``."""

    def __init__(self, responses):
        self.messages = FakeMessages(responses)


def fake_client(*responses, model: str = "test-model") -> AnthropicClient:
    return AnthropicClient(api_key=None, model=model, client=FakeSdkClient(responses))


def strong_match_client() -> AnthropicClient:
    """Judges one posting a strong match."""
    return fake_client(json_response(VALID_EVALUATION))


def seed_company(
    db: Database, name: str = "Acme", classification: CompanyClassification = CompanyClassification.PRODUCT,
    board_id: str | None = None,
) -> Company:
    """A registry company, classified as a product company unless told otherwise — so evaluating
    its postings makes no classification call. Pass UNKNOWN to exercise that call."""
    board_id = board_id or name.lower().replace(" ", "")
    company = Company(
        name=name, domain=f"{board_id}.example.com", careers_url=None, ats_type=AtsType.GREENHOUSE, board_id=board_id,
    )
    if classification != CompanyClassification.UNKNOWN:
        company.classification, company.classification_source = classification, ClassificationSource.SEED
    company.id = db.upsert_company(company)
    return company


def seed_job(
    db: Database, settings, company: Company, source_job_id: str = "1", title: str = "Senior Backend Engineer",
    description: str = "We build our own SaaS product. $180,000 - $220,000 per year.",
    salary_min: int | None = None, salary_max: int | None = None, location: str = "Remote - US",
) -> JobPosting:
    """A crawled posting, normalized as the crawl would; seeding the same source_job_id again
    updates it. ``salary_min``/``salary_max`` append a published base-salary range."""
    salary_line = f"<p>Base salary range: ${salary_min:,} - ${salary_max:,} per year.</p>" if salary_min else ""
    raw = RawPosting(
        source_job_id=source_job_id, title=title, location_raw=location, employment_type_raw="Full-time",
        description_html=f"<p>{description}</p>{salary_line}", description_text=None,
        posting_url=f"https://acme.example.com/jobs/{source_job_id}",
        apply_url=f"https://acme.example.com/jobs/{source_job_id}/apply",
    )
    job_id, _, _ = db.upsert_job(normalize_posting(raw, company, settings))
    return db.get_job(job_id)


def make_evaluation(
    job_id: int, verdict: Verdict = Verdict.STRONG_MATCH, scope_fit: ScopeFit = ScopeFit.AT_LEVEL, *,
    central_directly_demonstrated: int = 0, **fields,
) -> Evaluation:
    """An evaluation to save straight to the database, no LLM involved. ``fields`` set any other
    Evaluation field; ``central_directly_demonstrated`` adds that many met central requirements."""
    requirement_evidence = [
        RequirementEvidence(
            requirement=f"central requirement {i}", importance=RequirementImportance.CENTRAL,
            evidence_classification=EvidenceClassification.DIRECTLY_DEMONSTRATED,
            candidate_evidence="x", posting_evidence="y",
        )
        for i in range(central_directly_demonstrated)
    ]
    values = dict(
        job_id=job_id, description_hash=f"hash-{job_id}", verdict=verdict, confidence=0.7,
        scope_fit=scope_fit, evidence_coverage_percent=50,
        specialist_tenure_assessment=SpecialistTenureAssessment(
            classification=SpecialistTenureClassification.NOT_APPLICABLE, specialty="", explanation="",
        ),
        requirement_evidence=requirement_evidence, growth_dimensions=[], hidden_staff_signals=[],
        compensation_assessment="ok", remote_employment_verification="ok", required_matches=[],
        required_gaps=[], preferred_only_gaps=[], minor_caveats=[], evidence=[], credibility_assessment="ok",
        why_this_is_or_is_not_gettable="ok", is_product_company=True, primary_rejection_reason=None,
        model_name="test-model", created_at=datetime.now(timezone.utc),
    )
    return Evaluation(**{**values, **fields})
