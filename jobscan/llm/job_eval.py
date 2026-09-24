"""Single structured LLM call that screens one job posting against the candidate profile."""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import ValidationError

from jobscan.llm.client import AnthropicClient, LlmCallError
from jobscan.llm.prompts import (
    JOB_EVAL_SYSTEM,
    build_candidate_profile_block,
    build_job_eval_user_message,
    render_profile_text,
)
from jobscan.llm.schemas import JOB_EVALUATION_TOOL_NAME, JOB_EVALUATION_TOOL_SCHEMA, JobEvaluationResult
from jobscan.models import (
    Company,
    Evaluation,
    EvidenceClassification,
    JobPosting,
    RequirementEvidence,
    RequirementImportance,
    ScopeFit,
    SpecialistTenureAssessment,
    SpecialistTenureClassification,
    Verdict,
)


def evaluate_job(
    client: AnthropicClient,
    profile: dict,
    company: Company,
    job: JobPosting,
) -> Evaluation:
    """Raises LlmCallError on any failure (API error, malformed/invalid response) — callers
    should catch this, log it to the filter/audit trail, and leave the job unverified rather
    than caching a bad result."""
    assert job.id is not None

    profile_text = render_profile_text(profile)
    profile_block = build_candidate_profile_block(profile_text)
    user_message = build_job_eval_user_message(company, job)

    result = client.call_tool(
        system=JOB_EVAL_SYSTEM,
        user_message=user_message,
        tool_schema=JOB_EVALUATION_TOOL_SCHEMA,
        tool_name=JOB_EVALUATION_TOOL_NAME,
        # Observed live: the default 2048 was silently truncating ~99% of real calls
        # (stop_reason="max_tokens") — requirement_evidence in particular is expensive (a list of
        # structured objects) and was missing from 93% of historical evaluations as a result,
        # masked as "no findings" by its empty-list default rather than a visible failure.
        max_tokens=8192,
        # The candidate profile (~1,900 tokens) is identical for every posting in a run —
        # cached alongside the always-cached system prompt/tool schema.
        cacheable_prefix=profile_block,
    )

    try:
        parsed = JobEvaluationResult.model_validate(result.input)
    except ValidationError as exc:
        raise LlmCallError(f"job evaluation response failed schema validation: {exc}") from exc

    specialist_tenure = SpecialistTenureAssessment(
        classification=SpecialistTenureClassification(parsed.specialist_tenure_assessment.classification),
        specialty=parsed.specialist_tenure_assessment.specialty,
        explanation=parsed.specialist_tenure_assessment.explanation,
    )
    requirement_evidence = [
        RequirementEvidence(
            requirement=item.requirement,
            importance=RequirementImportance(item.importance),
            evidence_classification=EvidenceClassification(item.evidence_classification),
            candidate_evidence=item.candidate_evidence,
            posting_evidence=item.posting_evidence,
        )
        for item in parsed.requirement_evidence
    ]

    primary_rejection_reason = parsed.primary_rejection_reason
    if not primary_rejection_reason and parsed.verdict in ("reject", "borderline"):
        # Observed live: the model frequently leaves this specific field blank on reject/
        # borderline verdicts (70% of rejects in one real run) despite the prompt requiring it,
        # even though the reasoning is right there in required_gaps/growth_dimensions/
        # credibility_assessment. Synthesize from what it did fill in rather than showing
        # "(not specified)" in every report and audit listing.
        if parsed.required_gaps:
            primary_rejection_reason = "; ".join(parsed.required_gaps[:2])
        elif parsed.growth_dimensions:
            primary_rejection_reason = "; ".join(parsed.growth_dimensions[:2])
        else:
            primary_rejection_reason = parsed.credibility_assessment

    return Evaluation(
        job_id=job.id,
        description_hash=job.description_hash,
        verdict=Verdict(parsed.verdict),
        confidence=parsed.confidence,
        scope_fit=ScopeFit(parsed.scope_fit),
        evidence_coverage_percent=parsed.evidence_coverage_percent,
        specialist_tenure_assessment=specialist_tenure,
        requirement_evidence=requirement_evidence,
        growth_dimensions=parsed.growth_dimensions,
        hidden_staff_signals=parsed.hidden_staff_signals,
        compensation_assessment=parsed.compensation_assessment,
        remote_employment_verification=parsed.remote_employment_verification,
        required_matches=parsed.required_matches,
        required_gaps=parsed.required_gaps,
        preferred_only_gaps=parsed.preferred_only_gaps,
        minor_caveats=parsed.minor_caveats,
        evidence=parsed.evidence,
        credibility_assessment=parsed.credibility_assessment,
        why_this_is_or_is_not_gettable=parsed.why_this_is_or_is_not_gettable,
        is_product_company=parsed.is_product_company,
        primary_rejection_reason=primary_rejection_reason,
        worth_applying=parsed.worth_applying,
        model_name=client.model,
        created_at=datetime.now(timezone.utc),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cache_creation_input_tokens=result.cache_creation_input_tokens,
        cache_read_input_tokens=result.cache_read_input_tokens,
    )
