"""Pydantic models for the two structured LLM calls.

These double as (a) the JSON Schema handed to Claude via forced tool-use, so the response is
structurally guaranteed, and (b) the validator applied to whatever comes back, so a malformed or
hallucinated response is rejected rather than silently cached.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Verdict = Literal["strong_match", "plausible_match", "borderline", "reject"]
CompanyClass = Literal["product", "consulting", "unknown"]
ScopeFit = Literal["at_level", "one_step_up", "two_plus_steps_up", "below_level"]
RequirementImportance = Literal["central", "secondary"]
RequirementStrength = Literal["required", "strongly_implied", "preferred", "trait_or_interest"]
EvidenceClassification = Literal["directly_demonstrated", "credibly_transferable", "weakly_inferred", "not_demonstrated"]
SpecialistTenureClassification = Literal["meets", "adjacent", "insufficient", "not_applicable"]

_LIST_FIELDS = (
    "required_matches", "required_gaps", "preferred_only_gaps", "minor_caveats", "evidence",
    "growth_dimensions", "hidden_staff_signals",
)


def _first_point(value: object) -> str | None:
    """First one or two points from a list-shaped field, joined — or the whole thing if the
    model sent a bare string instead of a list (this runs before _coerce_str_to_list normalizes
    that). None if there's nothing usable."""
    if isinstance(value, list) and value:
        return "; ".join(str(item) for item in value[:2])
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None

# extra="ignore" on every model below: observed live, the model occasionally adds one stray
# duplicate-ish field (e.g. "preferred_only_gaps_2", "specialist_tenure_assessment_dummy")
# alongside the correctly-named ones. Drop unrecognized keys rather than rejecting an otherwise-
# valid response over a single hallucinated extra — every field we actually read is still
# strictly type/enum/range-checked regardless.


class RequirementEvidenceItem(BaseModel):
    # Field order is generation order: quote the posting, classify how it frames the item, and only
    # then decide importance and evidence — so the wording is on the page before it's weighed.
    requirement: str
    posting_evidence: str = Field(description="The posting language this requirement is drawn from, quoted with its qualifiers (\"ideally\", \"helpful\", \"or a willingness to learn\", ...).")
    stated_as: RequirementStrength = Field(
        description="How the posting's own wording frames it: required (must/required/minimum/N+ "
        "years/\"you have\" in a requirements list), strongly_implied (unstated but plainly "
        "unavoidable to do the job), preferred (helpful, a plus, bonus, nice to have, preferred, "
        "ideally, familiarity with, exposure to, \"or willingness to learn\", and technologies "
        "named as examples — \"technologies like ...\", \"such as ...\"), or trait_or_interest "
        "(curiosity, high agency, eagerness to learn, interest in a topic)."
    )
    importance: RequirementImportance = Field(
        description="central only for required or strongly_implied items the hiring bar rests on; "
        "preferred and trait_or_interest items are always secondary."
    )
    candidate_evidence: str = Field(description="What in the candidate's background supports this classification, if anything.")
    evidence_classification: EvidenceClassification

    model_config = {"extra": "ignore"}

    @model_validator(mode="after")
    def _only_required_items_are_central(self) -> RequirementEvidenceItem:
        """Enforced rather than trusted: a nice-to-have the model also tagged central would
        otherwise drag down coverage and scope exactly as if it were a hard requirement."""
        if self.stated_as in ("preferred", "trait_or_interest") and self.importance == "central":
            self.importance = "secondary"
        return self


class SpecialistTenureAssessment(BaseModel):
    classification: SpecialistTenureClassification
    specialty: str = Field(description="The specialty in question, e.g. 'data engineering'. Empty string if not_applicable.")
    explanation: str

    model_config = {"extra": "ignore"}


class JobEvaluationResult(BaseModel):
    # Field order is generation order — the tool call is forced and there's no separate thinking
    # step, so this schema IS the model's reasoning sequence. Evidence and analysis come first and
    # the verdict near the end. Observed live with the verdict first: the model committed to
    # "reject" for Pilot, then wrote in primary_rejection_reason that it had no reason to reject
    # ("[correction: no healthcare issue here] ... This role should not be a reject"); and
    # rubric rules were rationalized past (preferred items listed as growth dimensions) to fit a
    # verdict already given.
    requirement_evidence: list[RequirementEvidenceItem] = Field(default_factory=list)
    specialist_tenure_assessment: SpecialistTenureAssessment
    required_matches: list[str] = Field(default_factory=list)
    required_gaps: list[str] = Field(
        default_factory=list,
        description="Unmet required or strongly implied qualifications only — never a preferred item.",
    )
    preferred_only_gaps: list[str] = Field(default_factory=list)
    growth_dimensions: list[str] = Field(
        default_factory=list,
        description="Material central, REQUIRED parts of the role not already demonstrated at "
        "roughly the required level. Never a preferred/helpful/learnable item, a new business "
        "domain the posting doesn't require prior experience in, working with ambiguity, "
        "tech-leading or owning delivery within the team (technical leadership is a match), a "
        "technology named only as an example (\"using technologies like AWS, MySQL and "
        "Kubernetes\" — equivalent Azure/SQL Server experience meets it), or an interest. Omit "
        "minor tool differences.",
    )
    hidden_staff_signals: list[str] = Field(
        default_factory=list,
        description="Signals that this 'Senior' posting actually carries staff-level or "
        "elite-startup scope (organization-wide influence, repeated zero-to-one ownership, "
        "extreme scale/reliability, sole technical authority, ...). Breadth of influence beyond "
        "one team — never ordinary senior expectations such as owning or driving one's own "
        "features, owning the team's goals or delivery, tech-leading or leading engineers on the "
        "team through a project, sharing practices through writing or tech talks, autonomy, "
        "comfort with ambiguity, prototyping, mentoring or ambitious wording. Empty when there are "
        "none — never list reasons something is NOT a signal.",
    )
    is_product_company: bool = Field(
        description="True if this specific posting is for building/operating the company's own "
        "product or platform, false if it is client-delivery/billable consulting work."
    )
    remote_employment_verification: str = Field(
        description="Assessment of whether the role is genuinely U.S. remote and open to a "
        "Washington State resident."
    )
    compensation_assessment: str = Field(
        description="Whether the published range passes the candidate's compensation rule — a "
        "pass/fail gate, not a score."
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Short direct quotes or close paraphrases from the posting supporting the verdict.",
    )
    minor_caveats: list[str] = Field(default_factory=list)
    scope_fit: ScopeFit = Field(
        description="Whether the role's actual scope is at, one step above, two-plus steps "
        "above, or below demonstrated experience — independent of title or keyword overlap."
    )
    evidence_coverage_percent: int = Field(
        ge=0, le=100,
        description="0 to 100: weighted coverage of important required qualifications by "
        "demonstrated or transferable evidence — not a mechanical keyword-overlap count.",
    )
    credibility_assessment: str = Field(
        description="One or two sentences on whether applying is professionally credible given fit."
    )
    why_this_is_or_is_not_gettable: str = Field(
        description="Concise, specific explanation of interview/offer plausibility given scope_fit "
        "and evidence_coverage_percent — not a restatement of the verdict."
    )
    verdict: Verdict = Field(description="Follows from the analysis above, per the verdict rules.")
    confidence: float = Field(ge=0.0, le=1.0, description="0.0 to 1.0.")
    worth_applying: bool = Field(
        description="A forced yes/no distillation of why_this_is_or_is_not_gettable, independent "
        "of the verdict label — would a real interview loop plausibly survive contact with this "
        "posting's central requirements? False when the posting REQUIRES a technology, domain, or "
        "scale the candidate has never demonstrably touched (e.g. required Rust/Kafka/"
        "Kubernetes-as-core-systems, a security clearance, formal people management, production "
        "AI/ML ownership) such that interviewers would almost certainly expose the mismatch — even "
        "when evidence_coverage_percent looks moderate from secondary-requirement overlap. Never "
        "False merely for a language or domain the posting calls helpful, preferred or learnable. "
        "True for genuine aspirational stretches where the gap is depth/scale/tenure in "
        "already-familiar territory, and always True for strong_match/plausible_match.",
    )
    primary_rejection_reason: str | None = Field(
        default=None,
        description="REQUIRED (non-null) whenever verdict is 'reject' or 'borderline' — always "
        "set it, even when the reason is already implied by required_gaps or growth_dimensions. "
        "Leave null only for strong_match/plausible_match.",
    )

    model_config = {"extra": "ignore"}

    @field_validator(*_LIST_FIELDS, mode="before")
    @classmethod
    def _coerce_str_to_list(cls, value: object) -> object:
        """The model occasionally returns one string (e.g. a single paragraph) for these fields
        instead of a list of short points, despite the enforced tool schema. Split it into lines
        rather than failing validation and discarding an otherwise-good evaluation."""
        if isinstance(value, str):
            return [line.strip("-•* \t") for line in value.strip().splitlines() if line.strip()]
        return value

    @model_validator(mode="before")
    @classmethod
    def _unwrap_single_nested_object(cls, data: object) -> object:
        """Observed live (Lumos, 2026-10-06): the model wrapped its whole answer in one extra
        object — {"evaluation": {...every field...}} — failing validation and losing a complete
        evaluation. Unwrap a lone nested object that carries the verdict."""
        if isinstance(data, dict) and len(data) == 1:
            (inner,) = data.values()
            if isinstance(inner, dict) and "verdict" in inner:
                return inner
        return data

    @model_validator(mode="before")
    @classmethod
    def _backfill_gettability_fields_for_clear_rejects(cls, data: object) -> object:
        """Observed live: on a clear-cut reject/borderline the model sometimes omits
        primary_rejection_reason, why_this_is_or_is_not_gettable (a required field with no
        default — missing it fails validation outright), or both — treating whichever
        gap-describing field it did fill in as self-explanatory. Synthesize a reason from
        required_gaps/growth_dimensions/credibility_assessment when primary_rejection_reason
        itself is missing (mirroring jobscan.llm.job_eval's post-validation fallback for the
        already-valid case), then backfill the gettability fields from that reason — only when a
        reason is actually derivable, so a genuinely malformed response still fails as before."""
        if not isinstance(data, dict) or data.get("verdict") not in ("reject", "borderline"):
            return data

        reason = data.get("primary_rejection_reason")
        if not reason:
            reason = (
                _first_point(data.get("required_gaps"))
                or _first_point(data.get("growth_dimensions"))
                or data.get("credibility_assessment")
            )
            if reason:
                data["primary_rejection_reason"] = reason

        if reason:
            if not data.get("why_this_is_or_is_not_gettable"):
                data["why_this_is_or_is_not_gettable"] = f"Not gettable as described: {reason}"
            if not data.get("credibility_assessment"):
                data["credibility_assessment"] = f"Not a credible application given: {reason}"
        return data


JOB_EVALUATION_TOOL_NAME = "submit_job_evaluation"

JOB_EVALUATION_TOOL_SCHEMA = {
    "name": JOB_EVALUATION_TOOL_NAME,
    "description": "Submit the structured evaluation of a single job posting against the candidate profile.",
    "input_schema": JobEvaluationResult.model_json_schema(),
}


TriageDisqualifier = Literal[
    "none", "staff_or_higher_title", "people_management", "not_remote", "excluded_industry",
    "not_engineering", "required_unfamiliar_language", "required_specialty",
]


class TriageResult(BaseModel):
    # Quote and category come before the decision, so the skip has to rest on something the
    # posting actually says, in a category the instructions allow; jobscan.llm.triage
    # .skip_is_substantiated checks both before any skip is honoured.
    disqualifier_quote: str = Field(
        description="When skipping: the posting's exact words that establish the disqualifier, "
        "copied character for character from the title or text — one phrase or sentence, no "
        "paraphrase or ellipsis. Empty string when not skipping."
    )
    disqualifier: TriageDisqualifier = Field(
        description="The one category from the instructions that justifies skipping, or none."
    )
    skip_full_evaluation: bool = Field(
        description="True only when confident a careful reviewer would also reject this posting "
        "outright. False whenever there's real overlap alongside gaps, or any genuine uncertainty "
        "— the full evaluation exists for exactly that judgment call."
    )
    reason: str = Field(description="One short sentence.")

    model_config = {"extra": "ignore"}


TRIAGE_TOOL_NAME = "submit_triage"

TRIAGE_TOOL_SCHEMA = {
    "name": TRIAGE_TOOL_NAME,
    "description": "Submit the fast lenient screening decision for whether a posting warrants a full evaluation.",
    "input_schema": TriageResult.model_json_schema(),
}


class CompanyClassificationResult(BaseModel):
    classification: CompanyClass
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str = Field(description="One or two sentences citing what indicates product vs. consulting.")

    model_config = {"extra": "ignore"}


COMPANY_CLASSIFICATION_TOOL_NAME = "submit_company_classification"

COMPANY_CLASSIFICATION_TOOL_SCHEMA = {
    "name": COMPANY_CLASSIFICATION_TOOL_NAME,
    "description": "Submit the classification of a company as a product company vs. a consulting/staffing/outsourcing firm.",
    "input_schema": CompanyClassificationResult.model_json_schema(),
}
