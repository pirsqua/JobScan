"""Evaluation orchestration: safe factual filters, then (for unknown companies) one cached
company-classification call, then one cached-by-description-hash job evaluation call.

Manual overrides always win: a job-level "verdict" override or a company-level "classification"
override short-circuits the automated pipeline for that entity.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import httpx

from jobscan.config import Settings
from jobscan.db import Database
from jobscan.filters import FilterReason, apply_factual_filters
from jobscan.llm.client import AnthropicClient, LlmCallError
from jobscan.llm.company_eval import classify_company, fetch_homepage_text
from jobscan.llm.job_eval import evaluate_job
from jobscan.llm.prompts import (
    evaluation_rubric_version,
    load_profile,
    render_profile_text,
    triage_rubric_version,
)
from jobscan.llm.schemas import TriageResult
from jobscan.llm.triage import skip_is_substantiated, triage_job
from jobscan.logging_setup import get_logger, log_extra
from jobscan.models import (
    ClassificationSource,
    Company,
    CompanyClassification,
    Evaluation,
    EvaluateStats,
    FilterLogEntry,
    JobPosting,
    ManualOverride,
    ScopeFit,
    SpecialistTenureAssessment,
    SpecialistTenureClassification,
    Verdict,
)

logger = get_logger("evaluate")


def _now() -> datetime:
    return datetime.now(timezone.utc)


# Placeholder confidence for a triage-skip pseudo-evaluation: the triage prompt only skips when it
# professes confidence the posting is a clear reject, but it doesn't produce its own confidence
# score, so this reflects that framing rather than a real measured value.
_TRIAGE_SKIP_CONFIDENCE = 0.9


def _build_triage_skip_evaluation(
    job: JobPosting, company: Company, triage: TriageResult, triage_model: str, rubric: str
) -> Evaluation:
    """A lightweight stand-in Evaluation for a posting the cheap triage screened out before the
    expensive full evaluation ran — cached the same way a real evaluation would be (by job_id +
    description_hash), so it isn't re-triaged every run, and clearly distinguishable via
    model_name from a real full evaluation."""
    reason = triage.reason
    return Evaluation(
        job_id=job.id,
        description_hash=job.description_hash,
        verdict=Verdict.REJECT,
        confidence=_TRIAGE_SKIP_CONFIDENCE,
        scope_fit=ScopeFit.NOT_ASSESSED,
        evidence_coverage_percent=0,
        specialist_tenure_assessment=SpecialistTenureAssessment(
            classification=SpecialistTenureClassification.NOT_APPLICABLE,
            specialty="",
            explanation="Screened out by the cheap triage pass before a full evaluation ran.",
        ),
        requirement_evidence=[],
        growth_dimensions=[],
        hidden_staff_signals=[],
        compensation_assessment="not independently assessed (screened out by triage)",
        remote_employment_verification="not independently assessed (screened out by triage)",
        required_matches=[],
        required_gaps=[reason],
        preferred_only_gaps=[],
        minor_caveats=[],
        # The category and the posting's own words the skip was substantiated by — so an audit of
        # "why was this screened out?" always has the quote, not just the model's summary.
        evidence=[f'{triage.disqualifier}: "{triage.disqualifier_quote}"'],
        credibility_assessment=reason,
        why_this_is_or_is_not_gettable=f"Not gettable as described: {reason}",
        is_product_company=company.classification == CompanyClassification.PRODUCT,
        primary_rejection_reason=reason,
        worth_applying=False,
        model_name=f"triage:{triage_model}",
        created_at=_now(),
        rubric_version=rubric,
    )


def evaluate_all(
    db: Database,
    settings: Settings,
    client: AnthropicClient | None = None,
    triage_client: AnthropicClient | None = None,
    limit: int | None = None,
    refresh_stale: bool = False,
) -> EvaluateStats:
    """``limit``, if set, caps the number of real LLM calls (job evaluations + company
    classifications + triage screens) made in this run — factual filtering and cache hits are
    free and unaffected. Useful for bounding spend on a single invocation.

    ``refresh_stale`` re-screens postings whose cached verdict was made under a different rubric
    (the prompt, tool schema or candidate profile behind that kind of verdict — see
    ``evaluation_rubric_version`` / ``triage_rubric_version``) instead of reusing it.
    Off by default because it spends money: a rubric change otherwise only reaches new or changed
    postings. Without an API key it just counts them, as unverified, for a cost estimate.

    ``triage_client``, if not overridden, is built from ``settings.triage_model`` when set — a
    cheap first-pass screen (jobscan.llm.triage) that runs before the expensive full evaluation.
    Leave triage_model unset in settings to disable triage entirely."""
    stats = EvaluateStats()
    run_id = db.start_evaluation_run(stats)
    profile = load_profile(settings.profile_path)
    profile_text = render_profile_text(profile)
    full_rubric = evaluation_rubric_version(profile_text)
    triage_rubric = triage_rubric_version(profile_text)

    if client is None and settings.anthropic_api_key:
        client = AnthropicClient(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
            max_retries=settings.anthropic_max_retries,
            timeout_seconds=settings.anthropic_timeout_seconds,
        )
    if client is None:
        logger.warning("no ANTHROPIC_API_KEY configured — postings will be filtered but not LLM-evaluated")

    if triage_client is None and settings.anthropic_api_key and settings.triage_model:
        triage_client = AnthropicClient(
            api_key=settings.anthropic_api_key,
            model=settings.triage_model,
            max_retries=settings.anthropic_max_retries,
            timeout_seconds=settings.anthropic_timeout_seconds,
        )
    if triage_client is not None:
        stats.triage_model = triage_client.model

    headers = {"User-Agent": settings.http_user_agent}
    with httpx.Client(timeout=settings.http_timeout_seconds, headers=headers) as http_client:
        jobs = db.get_active_jobs()
        for job in jobs:
            if limit is not None and (stats.sent_to_llm + stats.companies_classified + stats.triaged) >= limit:
                break

            stats.jobs_considered += 1
            company = db.get_company(job.company_id)
            if company is None:
                continue

            job_overrides = db.get_manual_overrides("job", job.id)
            if "verdict" in job_overrides:
                stats.manual_overrides_applied += 1
                evaluation = Evaluation(
                    job_id=job.id,
                    description_hash=job.description_hash,
                    verdict=Verdict(job_overrides["verdict"]),
                    confidence=1.0,
                    scope_fit=ScopeFit.AT_LEVEL,
                    evidence_coverage_percent=100,
                    specialist_tenure_assessment=SpecialistTenureAssessment(
                        classification=SpecialistTenureClassification.NOT_APPLICABLE,
                        specialty="",
                        explanation="Manually overridden by candidate; not LLM-assessed.",
                    ),
                    requirement_evidence=[],
                    growth_dimensions=[],
                    hidden_staff_signals=[],
                    compensation_assessment="manual override",
                    remote_employment_verification="manual override",
                    required_matches=[],
                    required_gaps=[],
                    preferred_only_gaps=[],
                    minor_caveats=[],
                    evidence=["manually overridden"],
                    credibility_assessment="Manually overridden by candidate.",
                    why_this_is_or_is_not_gettable="Manually overridden by candidate.",
                    is_product_company=company.classification == CompanyClassification.PRODUCT,
                    primary_rejection_reason=None,
                    worth_applying=True,
                    model_name="manual_override",
                    created_at=_now(),
                )
                db.save_evaluation(evaluation)
                stats._bump_verdict(evaluation.verdict.value)
                continue

            company_overrides = db.get_manual_overrides("company", company.id)
            if "classification" in company_overrides:
                company = replace(company, classification=CompanyClassification(company_overrides["classification"]))

            filter_result = apply_factual_filters(job, company, settings.min_base_salary)
            if not filter_result.passed:
                if not db.has_filter_log(job.id, filter_result.reason):
                    db.record_filter_log(
                        FilterLogEntry(job.id, "factual_filter", filter_result.reason, filter_result.detail, _now())
                    )
                stats.factual_rejected += 1
                continue

            if company.classification == CompanyClassification.UNKNOWN and "classification" not in company_overrides:
                if client is not None:
                    homepage_text = fetch_homepage_text(http_client, company.domain)
                    try:
                        result, in_tok, out_tok, cache_write_tok, cache_read_tok = classify_company(
                            client, company.name, company.domain, homepage_text, job.description_text
                        )
                        stats.input_tokens += in_tok
                        stats.output_tokens += out_tok
                        stats.cache_creation_input_tokens += cache_write_tok
                        stats.cache_read_input_tokens += cache_read_tok
                        stats.companies_classified += 1
                        db.set_company_classification(
                            company.id,
                            CompanyClassification(result.classification),
                            ClassificationSource.LLM,
                            result.confidence,
                            result.evidence,
                        )
                        company = replace(company, classification=CompanyClassification(result.classification))
                    except LlmCallError as exc:
                        logger.warning(
                            "company classification failed",
                            extra=log_extra(company=company.name, error=str(exc)),
                        )

            if company.classification == CompanyClassification.CONSULTING:
                if not db.has_filter_log(job.id, FilterReason.CONSULTING_EMPLOYER):
                    db.record_filter_log(
                        FilterLogEntry(
                            job.id, "factual_filter", FilterReason.CONSULTING_EMPLOYER,
                            f"{company.name} classified as consulting after review", _now(),
                        )
                    )
                stats.factual_rejected += 1
                continue

            cached = db.get_evaluation_by_hash(
                job.description_hash, rubric_versions=(full_rubric, triage_rubric) if refresh_stale else None
            )
            if cached is not None:
                stats.cache_hits += 1
                if cached.job_id != job.id:
                    cached = replace(cached, id=None, job_id=job.id)
                    db.save_evaluation(cached)
                stats._bump_verdict(cached.verdict.value)
                continue

            if client is None:
                stats.unverified += 1
                continue

            if triage_client is not None:
                try:
                    triage_result, t_in, t_out, t_cache_w, t_cache_r = triage_job(
                        triage_client, profile_text, company, job
                    )
                except LlmCallError as exc:
                    # A failed triage call is not evidence of a safe reject — proceed to the full
                    # evaluation rather than risk silently dropping a real opportunity.
                    logger.warning(
                        "triage call failed — proceeding to full evaluation",
                        extra=log_extra(job_id=job.id, title=job.title, error=str(exc)),
                    )
                else:
                    stats.triaged += 1
                    stats.triage_input_tokens += t_in
                    stats.triage_output_tokens += t_out
                    stats.triage_cache_creation_input_tokens += t_cache_w
                    stats.triage_cache_read_input_tokens += t_cache_r
                    if triage_result.skip_full_evaluation and not skip_is_substantiated(triage_result, job):
                        stats.triage_overruled += 1
                        logger.info(
                            "triage skip not substantiated by the posting — running the full evaluation",
                            extra=log_extra(
                                job_id=job.id, title=job.title, disqualifier=triage_result.disqualifier,
                                quote=triage_result.disqualifier_quote, reason=triage_result.reason,
                            ),
                        )
                    elif triage_result.skip_full_evaluation:
                        stats.triage_skipped += 1
                        skip_evaluation = _build_triage_skip_evaluation(
                            job, company, triage_result, triage_client.model, triage_rubric
                        )
                        db.save_evaluation(skip_evaluation)
                        stats._bump_verdict(skip_evaluation.verdict.value)
                        continue

            try:
                evaluation = replace(evaluate_job(client, profile, company, job), rubric_version=full_rubric)
            except LlmCallError as exc:
                logger.warning("job evaluation failed", extra=log_extra(job_id=job.id, title=job.title, error=str(exc)))
                db.record_filter_log(FilterLogEntry(job.id, "llm_error", "llm_call_failed", str(exc), _now()))
                stats.unverified += 1
                stats.llm_errors += 1
                continue

            db.save_evaluation(evaluation)
            stats.sent_to_llm += 1
            stats.input_tokens += evaluation.input_tokens or 0
            stats.output_tokens += evaluation.output_tokens or 0
            stats.cache_creation_input_tokens += evaluation.cache_creation_input_tokens or 0
            stats.cache_read_input_tokens += evaluation.cache_read_input_tokens or 0
            stats._bump_verdict(evaluation.verdict.value)

    stats.finished_at = _now()
    db.finish_evaluation_run(run_id, stats)
    return stats
