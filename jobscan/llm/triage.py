"""Cheap first-pass screen: is a posting worth the full, expensive job_eval.py evaluation at all?

Wired into evaluate_all() when settings.triage_model is configured. Validated against 692
historical evaluations before enabling: 0 false negatives on the 3 known genuine matches, ~89% of
real rejects correctly flagged skippable — see the settings.yaml comment on triage_model. A false
negative here silently destroys a real opportunity, which is a far worse failure mode than the
wasted cost of a false positive reaching the full evaluation anyway — the prompt is written to be
lenient accordingly, and callers should treat a failed triage call as "don't skip", not as
evidence the posting is a safe reject.
"""
from __future__ import annotations

from pydantic import ValidationError

from jobscan.llm.client import AnthropicClient, LlmCallError
from jobscan.llm.prompts import TRIAGE_SYSTEM, build_candidate_profile_block, build_triage_user_message
from jobscan.llm.schemas import TRIAGE_TOOL_NAME, TRIAGE_TOOL_SCHEMA, TriageResult
from jobscan.models import Company, JobPosting


def triage_job(
    client: AnthropicClient, profile_text: str, company: Company, job: JobPosting
) -> tuple[TriageResult, int, int, int, int]:
    """Returns (result, input_tokens, output_tokens, cache_creation_input_tokens,
    cache_read_input_tokens). Raises LlmCallError on any failure."""
    profile_block = build_candidate_profile_block(profile_text)
    user_message = build_triage_user_message(company, job)

    result = client.call_tool(
        system=TRIAGE_SYSTEM,
        user_message=user_message,
        tool_schema=TRIAGE_TOOL_SCHEMA,
        tool_name=TRIAGE_TOOL_NAME,
        max_tokens=300,
        cacheable_prefix=profile_block,
    )

    try:
        parsed = TriageResult.model_validate(result.input)
    except ValidationError as exc:
        raise LlmCallError(f"triage response failed schema validation: {exc}") from exc
    return (
        parsed, result.input_tokens, result.output_tokens,
        result.cache_creation_input_tokens, result.cache_read_input_tokens,
    )
