"""Canonical data models shared across the crawl, filter, evaluate and report stages.

Plain dataclasses are used (not an ORM) so the SQLite layer in ``db.py`` can stay a thin,
explicit mapping instead of hiding schema decisions behind a framework.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jobscan.config import Settings


class AtsType(str, Enum):
    GREENHOUSE = "greenhouse"
    ASHBY = "ashby"
    LEVER = "lever"


class CompanyClassification(str, Enum):
    PRODUCT = "product"
    CONSULTING = "consulting"
    UNKNOWN = "unknown"


class ClassificationSource(str, Enum):
    SEED = "seed"
    LLM = "llm"
    MANUAL = "manual"


class RemoteScope(str, Enum):
    REMOTE_US = "remote_us"
    REMOTE_US_RESTRICTED = "remote_us_restricted"
    HYBRID = "hybrid"
    ONSITE = "onsite"
    UNKNOWN = "unknown"


class EmploymentType(str, Enum):
    FULL_TIME = "full_time"
    CONTRACT = "contract"
    PART_TIME = "part_time"
    INTERN = "intern"
    UNKNOWN = "unknown"


class SalarySource(str, Enum):
    STRUCTURED = "structured"
    DESCRIPTION = "description"
    NONE = "none"


class JobStatus(str, Enum):
    ACTIVE = "active"
    CLOSED = "closed"


class Verdict(str, Enum):
    STRONG_MATCH = "strong_match"
    PLAUSIBLE_MATCH = "plausible_match"
    BORDERLINE = "borderline"
    REJECT = "reject"


class ScopeFit(str, Enum):
    """How this posting's scope compares to demonstrated experience — separate from verdict,
    which is about qualification match. A title saying "Senior" doesn't make a role at_level."""

    AT_LEVEL = "at_level"
    ONE_STEP_UP = "one_step_up"
    TWO_PLUS_STEPS_UP = "two_plus_steps_up"
    BELOW_LEVEL = "below_level"


class RequirementImportance(str, Enum):
    CENTRAL = "central"
    SECONDARY = "secondary"


class EvidenceClassification(str, Enum):
    DIRECTLY_DEMONSTRATED = "directly_demonstrated"
    CREDIBLY_TRANSFERABLE = "credibly_transferable"
    WEAKLY_INFERRED = "weakly_inferred"
    NOT_DEMONSTRATED = "not_demonstrated"


class SpecialistTenureClassification(str, Enum):
    MEETS = "meets"
    ADJACENT = "adjacent"
    INSUFFICIENT = "insufficient"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class RequirementEvidence:
    requirement: str
    importance: RequirementImportance
    evidence_classification: EvidenceClassification
    candidate_evidence: str
    posting_evidence: str


@dataclass
class SpecialistTenureAssessment:
    classification: SpecialistTenureClassification
    specialty: str
    explanation: str


@dataclass
class Company:
    """A row in the employer registry."""

    name: str
    domain: str | None
    careers_url: str | None
    ats_type: AtsType
    board_id: str
    id: int | None = None
    classification: CompanyClassification = CompanyClassification.UNKNOWN
    classification_source: ClassificationSource | None = None
    classification_confidence: float | None = None
    classification_evidence: str | None = None
    active: bool = True
    last_crawled_at: datetime | None = None
    discovery_source: str | None = None
    notes: str | None = None


@dataclass
class RawPosting:
    """What a source adapter hands back, before normalization."""

    source_job_id: str
    title: str
    location_raw: str | None
    employment_type_raw: str | None
    description_html: str | None
    description_text: str | None
    posting_url: str | None
    apply_url: str | None
    department: str | None = None
    published_at: str | None = None
    salary_min: float | None = None
    salary_max: float | None = None
    salary_currency: str | None = None
    salary_period: str | None = None
    salary_source: SalarySource = SalarySource.NONE
    remote_flag: bool | None = None  # explicit is-remote signal from the source API, if any


@dataclass
class JobPosting:
    """A normalized, persisted job posting."""

    company_id: int
    source: AtsType
    source_job_id: str
    title: str
    location_raw: str | None
    remote_scope: RemoteScope
    employment_type_raw: str | None
    employment_type: EmploymentType
    description_text: str
    description_html: str | None
    description_hash: str
    posting_url: str | None
    apply_url: str | None
    department: str | None
    published_at: datetime | None
    first_seen_at: datetime
    last_seen_at: datetime
    id: int | None = None
    salary_min: float | None = None
    salary_max: float | None = None
    salary_currency: str | None = None
    salary_period: str | None = None
    salary_source: SalarySource = SalarySource.NONE
    closed_at: datetime | None = None
    status: JobStatus = JobStatus.ACTIVE


@dataclass
class Evaluation:
    """The LLM's structured verdict on a single job description snapshot.

    ``verdict`` is qualification match; ``scope_fit`` is separately whether the role's actual
    scope (not its title) is at, above, or below demonstrated experience — a "Senior" title with
    matching keywords does not by itself make a role ``at_level``.
    """

    job_id: int
    description_hash: str
    verdict: Verdict
    confidence: float
    scope_fit: ScopeFit
    evidence_coverage_percent: int
    specialist_tenure_assessment: SpecialistTenureAssessment
    requirement_evidence: list[RequirementEvidence]
    growth_dimensions: list[str]
    hidden_staff_signals: list[str]
    compensation_assessment: str
    remote_employment_verification: str
    required_matches: list[str]
    required_gaps: list[str]
    preferred_only_gaps: list[str]
    minor_caveats: list[str]
    evidence: list[str]
    credibility_assessment: str
    why_this_is_or_is_not_gettable: str
    is_product_company: bool
    primary_rejection_reason: str | None
    model_name: str
    created_at: datetime
    id: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class ManualOverride:
    entity_type: str  # "job" | "company"
    entity_id: int
    field: str
    value: str
    reason: str | None
    created_at: datetime
    id: int | None = None


@dataclass
class FilterLogEntry:
    job_id: int
    stage: str  # "factual_filter" | "llm_error"
    reason: str
    detail: str | None
    created_at: datetime
    id: int | None = None


@dataclass
class CrawlRunStats:
    started_at: datetime
    companies_attempted: int = 0
    companies_succeeded: int = 0
    boards_failed: list[str] = field(default_factory=list)
    postings_fetched: int = 0
    postings_out_of_family: int = 0
    new_postings: int = 0
    changed_postings: int = 0
    closed_postings: int = 0
    finished_at: datetime | None = None
    id: int | None = None


@dataclass
class EvaluateStats:
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    jobs_considered: int = 0
    factual_rejected: int = 0
    companies_classified: int = 0
    sent_to_llm: int = 0
    cache_hits: int = 0
    manual_overrides_applied: int = 0
    verdict_counts: dict[str, int] = field(default_factory=dict)
    unverified: int = 0
    llm_errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    finished_at: datetime | None = None
    id: int | None = None

    def estimated_cost_usd(self, settings: "Settings") -> float | None:
        pricing = settings.pricing_for(settings.anthropic_model)
        if pricing is None:
            return None
        return (
            self.input_tokens / 1_000_000 * pricing.input_per_million
            + self.output_tokens / 1_000_000 * pricing.output_per_million
        )

    def _bump_verdict(self, verdict: str) -> None:
        self.verdict_counts[verdict] = self.verdict_counts.get(verdict, 0) + 1
