"""Single structured LLM call, cached indefinitely per company, that classifies a company as a
product company vs. consulting/staffing/outsourcing shop."""
from __future__ import annotations

import httpx
from bs4 import BeautifulSoup
from pydantic import ValidationError

from jobscan.llm.client import AnthropicClient, LlmCallError
from jobscan.llm.prompts import COMPANY_EVAL_SYSTEM, build_company_eval_user_message
from jobscan.llm.schemas import (
    COMPANY_CLASSIFICATION_TOOL_NAME,
    COMPANY_CLASSIFICATION_TOOL_SCHEMA,
    CompanyClassificationResult,
)

MAX_HOMEPAGE_CHARS = 3000
MAX_JOB_TEXT_CHARS = 2000


def fetch_homepage_text(http_client: httpx.Client, domain: str | None) -> str:
    """Best-effort fetch of visible homepage text, used as "About" context. Never raises —
    classification can proceed on the job description alone if the homepage can't be fetched."""
    if not domain:
        return ""
    url = domain if domain.startswith("http") else f"https://{domain}"
    try:
        response = http_client.get(url, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPError:
        return ""
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = " ".join(text.split())
    return text[:MAX_HOMEPAGE_CHARS]


def classify_company(
    client: AnthropicClient,
    company_name: str,
    domain: str | None,
    homepage_text: str,
    sample_job_text: str,
) -> tuple[CompanyClassificationResult, int, int, int, int]:
    """Returns (result, input_tokens, output_tokens, cache_creation_input_tokens,
    cache_read_input_tokens)."""
    user_message = build_company_eval_user_message(
        company_name, domain, homepage_text, sample_job_text[:MAX_JOB_TEXT_CHARS]
    )
    result = client.call_tool(
        system=COMPANY_EVAL_SYSTEM,
        user_message=user_message,
        tool_schema=COMPANY_CLASSIFICATION_TOOL_SCHEMA,
        tool_name=COMPANY_CLASSIFICATION_TOOL_NAME,
    )
    try:
        parsed = CompanyClassificationResult.model_validate(result.input)
    except ValidationError as exc:
        raise LlmCallError(f"company classification response failed schema validation: {exc}") from exc
    return (
        parsed, result.input_tokens, result.output_tokens,
        result.cache_creation_input_tokens, result.cache_read_input_tokens,
    )
