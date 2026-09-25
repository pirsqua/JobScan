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
    """Greenhouse/Ashby/Lever/Workday/Jobvite have real adapters (see jobscan.adapters.ADAPTERS).
    The rest are recorded-but-not-yet-crawlable — a registry curator can note that a company uses
    e.g. SmartRecruiters without the import failing, and jobscan.crawl already skips any company
    whose ats_type has no entry in ADAPTERS (reported as "no adapter registered" rather than
    attempted). Keep such companies active=false until a real adapter exists. Workday's board_id
    encodes "{tenant}/{cluster}/{site}" (see jobscan.adapters.workday) rather than a single token.

    ESRI is a one-off: a bespoke, single-company careers platform (not a multi-tenant ATS other
    registry companies could ever share) that still turned out to be crawlable — its own real
    search API. Deliberately NOT folded into CUSTOM, since CUSTOM has no adapter and ADAPTERS is
    keyed by ats_type; doing so would wrongly route every other "custom" company through Esri's
    API. A future one-off bespoke platform worth crawling gets its own value the same way."""

    GREENHOUSE = "greenhouse"
    ASHBY = "ashby"
    LEVER = "lever"
    WORKDAY = "workday"
    JOBVITE = "jobvite"
    ESRI = "esri"
    SMARTRECRUITERS = "smartrecruiters"
    CUSTOM = "custom"


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
    # Defaults True so evaluations cached before this field existed aren't retroactively treated
    # as "not worth applying" — only newly-evaluated postings get the stricter routing.
    worth_applying: bool = True
    # Separate from input_tokens (non-cached portion only) because prompt caching bills them at
    # different rates — see EvaluateStats.estimated_cost_usd.
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None


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
    # Populated once prompt caching is in use (system prompt + tool schema + candidate profile
    # are cached on every job-eval/company-eval call — see AnthropicClient.call_tool). Tracked
    # separately from input_tokens because Anthropic bills them at different rates: a cache write
    # costs more than a normal input token, a cache read costs much less.
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    # Cheap first-pass screen (jobscan.llm.triage) — tracked entirely separately from the fields
    # above because it typically runs a different, cheaper model than the main evaluation, so it
    # needs its own token counts and its own model name to look up its own pricing tier.
    triaged: int = 0
    triage_skipped: int = 0
    triage_input_tokens: int = 0
    triage_output_tokens: int = 0
    triage_cache_creation_input_tokens: int = 0
    triage_cache_read_input_tokens: int = 0
    triage_model: str | None = None
    finished_at: datetime | None = None
    id: int | None = None

    # Anthropic's standard ephemeral-cache multipliers of the base input price, applied to
    # whatever a model's own input_per_million is configured as — these ratios are the same
    # across models, only the base rate differs.
    _CACHE_WRITE_MULTIPLIER = 1.25
    _CACHE_READ_MULTIPLIER = 0.1

    @staticmethod
    def _tier_cost(
        input_tokens: int, output_tokens: int, cache_creation_input_tokens: int,
        cache_read_input_tokens: int, pricing: "ModelPricing",
    ) -> float:
        return (
            input_tokens / 1_000_000 * pricing.input_per_million
            + cache_creation_input_tokens / 1_000_000 * pricing.input_per_million * EvaluateStats._CACHE_WRITE_MULTIPLIER
            + cache_read_input_tokens / 1_000_000 * pricing.input_per_million * EvaluateStats._CACHE_READ_MULTIPLIER
            + output_tokens / 1_000_000 * pricing.output_per_million
        )

    def estimated_cost_usd(self, settings: "Settings") -> float | None:
        pricing = settings.pricing_for(settings.anthropic_model)
        if pricing is None:
            return None
        cost = self._tier_cost(
            self.input_tokens, self.output_tokens,
            self.cache_creation_input_tokens, self.cache_read_input_tokens, pricing,
        )
        if self.triage_model:
            triage_pricing = settings.pricing_for(self.triage_model)
            if triage_pricing is not None:
                cost += self._tier_cost(
                    self.triage_input_tokens, self.triage_output_tokens,
                    self.triage_cache_creation_input_tokens, self.triage_cache_read_input_tokens,
                    triage_pricing,
                )
        return cost

    def _bump_verdict(self, verdict: str) -> None:
        self.verdict_counts[verdict] = self.verdict_counts.get(verdict, 0) + 1
