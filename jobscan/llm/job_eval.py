"""Single structured LLM call that screens one job posting against the candidate profile."""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import ValidationError

from jobscan.llm.client import AnthropicClient, LlmCallError
from jobscan.llm.prompts import JOB_EVAL_SYSTEM, build_job_eval_user_message, render_profile_text
from jobscan.llm.schemas import JOB_EVALUATION_TOOL_NAME, JOB_EVALUATION_TOOL_SCHEMA, JobEvaluationResult
from jobscan.models import Company, Evaluation, JobPosting, Verdict


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
    user_message = build_job_eval_user_message(profile_text, company, job)

    result = client.call_tool(
        system=JOB_EVAL_SYSTEM,
        user_message=user_message,
        tool_schema=JOB_EVALUATION_TOOL_SCHEMA,
        tool_name=JOB_EVALUATION_TOOL_NAME,
    )

    try:
        parsed = JobEvaluationResult.model_validate(result.input)
    except ValidationError as exc:
        raise LlmCallError(f"job evaluation response failed schema validation: {exc}") from exc

    return Evaluation(
        job_id=job.id,
        description_hash=job.description_hash,
        verdict=Verdict(parsed.verdict),
        confidence=parsed.confidence,
        compensation_assessment=parsed.compensation_assessment,
        remote_verification=parsed.remote_verification,
        required_matches=parsed.required_matches,
        required_gaps=parsed.required_gaps,
        preferred_gaps=parsed.preferred_gaps,
        minor_caveats=parsed.minor_caveats,
        evidence=parsed.evidence,
        credibility_assessment=parsed.credibility_assessment,
        is_product_company=parsed.is_product_company,
        primary_rejection_reason=parsed.primary_rejection_reason,
        model_name=client.model,
        created_at=datetime.now(timezone.utc),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )
