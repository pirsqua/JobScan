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
    requirement: str
    importance: RequirementImportance
    evidence_classification: EvidenceClassification
    candidate_evidence: str = Field(description="What in the candidate's background supports this classification, if anything.")
    posting_evidence: str = Field(description="The posting language this requirement is drawn from.")

    model_config = {"extra": "ignore"}


class SpecialistTenureAssessment(BaseModel):
    classification: SpecialistTenureClassification
    specialty: str = Field(description="The specialty in question, e.g. 'data engineering'. Empty string if not_applicable.")
    explanation: str

    model_config = {"extra": "ignore"}


class JobEvaluationResult(BaseModel):
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    scope_fit: ScopeFit = Field(
        description="Whether the role's actual scope is at, one step above, two-plus steps "
        "above, or below demonstrated experience — independent of title or keyword overlap."
    )
    evidence_coverage_percent: int = Field(
        ge=0, le=100,
        description="Weighted coverage of important required qualifications by demonstrated or "
        "transferable evidence — not a mechanical keyword-overlap count.",
    )
    specialist_tenure_assessment: SpecialistTenureAssessment
    requirement_evidence: list[RequirementEvidenceItem] = Field(default_factory=list)
    growth_dimensions: list[str] = Field(
        default_factory=list,
        description="Material central parts of the role not already demonstrated at roughly the "
        "required level. Omit minor tool differences.",
    )
    hidden_staff_signals: list[str] = Field(
        default_factory=list,
        description="Signals that this 'Senior' posting actually carries staff-level or "
        "elite-startup scope (organization-wide influence, repeated zero-to-one ownership, "
        "extreme scale/reliability, sole technical authority, ...).",
    )
    is_product_company: bool = Field(
        description="True if this specific posting is for building/operating the company's own "
        "product or platform, false if it is client-delivery/billable consulting work."
    )
    compensation_assessment: str = Field(
        description="Assessment of whether the published range makes a $170,000+ offer credible."
    )
    remote_employment_verification: str = Field(
        description="Assessment of whether the role is genuinely U.S. remote and open to a "
        "Washington State resident."
    )
    required_matches: list[str] = Field(default_factory=list)
    required_gaps: list[str] = Field(default_factory=list)
    preferred_only_gaps: list[str] = Field(default_factory=list)
    minor_caveats: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(
        default_factory=list,
        description="Short direct quotes or close paraphrases from the posting supporting the verdict.",
    )
    credibility_assessment: str = Field(
        description="One or two sentences on whether applying is professionally credible given fit."
    )
    why_this_is_or_is_not_gettable: str = Field(
        description="Concise, specific explanation of interview/offer plausibility given scope_fit "
        "and evidence_coverage_percent — not a restatement of the verdict."
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
