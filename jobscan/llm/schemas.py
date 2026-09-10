"""Pydantic models for the two structured LLM calls.

These double as (a) the JSON Schema handed to Claude via forced tool-use, so the response is
structurally guaranteed, and (b) the validator applied to whatever comes back, so a malformed or
hallucinated response is rejected rather than silently cached.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Verdict = Literal["strong_match", "plausible_match", "borderline", "reject"]
CompanyClass = Literal["product", "consulting", "unknown"]


class JobEvaluationResult(BaseModel):
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    is_product_company: bool = Field(
        description="True if this specific posting is for building/operating the company's own "
        "product or platform, false if it is client-delivery/billable consulting work."
    )
    compensation_assessment: str = Field(
        description="Assessment of whether the published range makes a $170,000+ offer credible."
    )
    remote_verification: str = Field(
        description="Assessment of whether the role is genuinely U.S. remote and open to a "
        "Washington State resident."
    )
    required_matches: list[str] = Field(default_factory=list)
    required_gaps: list[str] = Field(default_factory=list)
    preferred_gaps: list[str] = Field(default_factory=list)
    minor_caveats: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(
        default_factory=list,
        description="Short direct quotes or close paraphrases from the posting supporting the verdict.",
    )
    credibility_assessment: str = Field(
        description="One or two sentences on whether applying is professionally credible given fit."
    )
    primary_rejection_reason: str | None = Field(
        default=None, description="Set only when verdict is 'reject' or 'borderline'."
    )

    model_config = {"extra": "forbid"}


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

    model_config = {"extra": "forbid"}


COMPANY_CLASSIFICATION_TOOL_NAME = "submit_company_classification"

COMPANY_CLASSIFICATION_TOOL_SCHEMA = {
    "name": COMPANY_CLASSIFICATION_TOOL_NAME,
    "description": "Submit the classification of a company as a product company vs. a consulting/staffing/outsourcing firm.",
    "input_schema": CompanyClassificationResult.model_json_schema(),
}
