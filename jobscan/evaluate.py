"""Evaluation orchestration: safe factual filters, then (for unknown companies) one cached
company-classification call, then one cached-by-description-hash job evaluation call.

Manual overrides always win: a job-level "verdict" override or a company-level "classification"
override short-circuits the automated pipeline for that entity.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

from jobscan.config import Settings
from jobscan.db import Database
from jobscan.filters import FilterReason, apply_factual_filters
from jobscan.http_client import make_client
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


# A borderline judged not worth applying is the one verdict that sits right on the shortlist line,
# and the model can't be made to sample deterministically (temperature is deprecated for it).
# Observed: the same LaunchDarkly posting, same rubric, came back not-worth-applying twice and
# worth-applying once in three runs — so one unlucky sample silently dropped it. Such a near-miss
# gets one more evaluation and keeps the more favourable; a clear reject (a failed hard filter, a
# missing required skill) doesn't.
_VERDICT_RANK = {Verdict.REJECT: 0, Verdict.BORDERLINE: 1, Verdict.PLAUSIBLE_MATCH: 2, Verdict.STRONG_MATCH: 3}


def _is_near_miss(evaluation: Evaluation) -> bool:
    return evaluation.verdict == Verdict.BORDERLINE and not evaluation.worth_applying


def _favourability(evaluation: Evaluation) -> tuple:
    return (
        evaluation.verdict != Verdict.REJECT and evaluation.worth_applying,
        _VERDICT_RANK[evaluation.verdict],
        evaluation.evidence_coverage_percent,
    )


def _same_verdict(current: Evaluation | None, cached: Evaluation) -> bool:
    """Whether ``current`` already is ``cached`` or a copy of it — copies keep the original's
    timestamp, so a verdict shared by postings with the same text isn't copied again every run."""
    return current is not None and current.created_at == cached.created_at and current.model_name == cached.model_name


def _count_tokens(stats: EvaluateStats, evaluation: Evaluation) -> None:
    stats.input_tokens += evaluation.input_tokens or 0
    stats.output_tokens += evaluation.output_tokens or 0
    stats.cache_creation_input_tokens += evaluation.cache_creation_input_tokens or 0
    stats.cache_read_input_tokens += evaluation.cache_read_input_tokens or 0


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


def _manual_override_evaluation(job: JobPosting, company: Company, verdict: Verdict) -> Evaluation:
    """The candidate's own verdict on a posting, stored like an evaluation so the report treats
    it uniformly; it bypasses the factual filters and every LLM call."""
    return Evaluation(
        job_id=job.id,
        description_hash=job.description_hash,
        verdict=verdict,
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


@dataclass
class _Screening:
    """What the LLM calls for one posting produced. Made in a worker thread, recorded on the main
    thread — the Database is one SQLite connection."""

    job: JobPosting
    company: Company
    triage: TriageResult | None = None
    triage_tokens: tuple[int, int, int, int] = (0, 0, 0, 0)
    triage_error: str | None = None
    triage_overruled: bool = False
    # A substantiated triage skip, stored in place of a full evaluation.
    skip: Evaluation | None = None
    # Every full evaluation made (two for a near-miss's second look) and the one kept.
    evaluations: list[Evaluation] = field(default_factory=list)
    chosen: Evaluation | None = None
    error: str | None = None
    second_look_error: str | None = None


def _screen(
    job: JobPosting, company: Company, profile: dict, profile_text: str, client: AnthropicClient,
    triage_client: AnthropicClient | None, settings: Settings, full_rubric: str, triage_rubric: str,
) -> _Screening:
    """The LLM calls for one posting: triage, then — unless triage substantiates a skip — the full
    evaluation, and a second look for a near-miss. Touches no database, so it's safe to run in a
    worker thread."""
    outcome = _Screening(job, company)
    if triage_client is not None:
        try:
            result, t_in, t_out, t_cache_w, t_cache_r = triage_job(triage_client, profile_text, company, job)
        except LlmCallError as exc:
            # A failed triage call is not evidence of a safe reject — go on to the full evaluation
            # rather than risk silently dropping a real opportunity.
            outcome.triage_error = str(exc)
        else:
            outcome.triage, outcome.triage_tokens = result, (t_in, t_out, t_cache_w, t_cache_r)
            if result.skip_full_evaluation:
                if skip_is_substantiated(
                    result, job, settings.candidate_state, settings.candidate_state_name,
                    settings.candidate_time_zone, settings.candidate_excluded_time_zones,
                ):
                    outcome.skip = _build_triage_skip_evaluation(job, company, result, triage_client.model, triage_rubric)
                    return outcome
                outcome.triage_overruled = True

    try:
        evaluation = replace(evaluate_job(client, profile, company, job), rubric_version=full_rubric)
    except LlmCallError as exc:
        outcome.error = str(exc)
        return outcome
    outcome.evaluations.append(evaluation)
    if _is_near_miss(evaluation):
        try:
            second = replace(evaluate_job(client, profile, company, job), rubric_version=full_rubric)
        except LlmCallError as exc:
            outcome.second_look_error = str(exc)
        else:
            outcome.evaluations.append(second)
            if _favourability(second) > _favourability(evaluation):
                evaluation = second
    outcome.chosen = evaluation
    return outcome


def evaluate_all(
    db: Database,
    settings: Settings,
    client: AnthropicClient | None = None,
    triage_client: AnthropicClient | None = None,
    limit: int | None = None,
    refresh_stale: bool = False,
    dry_run: bool = False,
) -> EvaluateStats:
    """``limit``, if set, caps the number of real LLM calls (job evaluations + company
    classifications + triage screens) made in this run — factual filtering and cache hits are
    free and unaffected. Useful for bounding spend on a single invocation. With a limit, postings
    are screened one at a time, so the cap is exact; without one, ``settings.evaluate_workers``
    make their LLM calls in parallel (all database work stays on this thread).

    ``refresh_stale`` re-screens postings whose cached verdict was made under a different rubric
    (the prompt, schema, candidate profile or model behind that kind of verdict — see
    ``evaluation_rubric_version`` / ``triage_rubric_version``) instead of reusing it.
    Off by default because it spends money: a rubric change otherwise only reaches new or changed
    postings. Without an API key it just counts them, as unverified, for a cost estimate.

    ``dry_run`` makes no LLM calls whatever the configuration: postings that would go to the LLM
    are counted as unverified, and the run isn't logged, so a later report still shows the last
    real run. Factual filtering is real and is recorded as usual.

    ``triage_client``, if not overridden, is built from ``settings.triage_model`` when set — a
    cheap first-pass screen (jobscan.llm.triage) that runs before the expensive full evaluation.
    Leave triage_model unset in settings to disable triage entirely."""
    stats = EvaluateStats()
    run_id = None if dry_run else db.start_evaluation_run(stats)
    profile = load_profile(settings.profile_path)
    profile_text = render_profile_text(profile)
    full_rubric = evaluation_rubric_version(profile_text, settings.anthropic_model, settings.anthropic_thinking)
    triage_rubric = triage_rubric_version(profile_text, settings.triage_model or "", settings.triage_thinking)

    if dry_run:
        client = triage_client = None
    if client is None and settings.anthropic_api_key and not dry_run:
        client = AnthropicClient(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
            max_retries=settings.anthropic_max_retries,
            timeout_seconds=settings.anthropic_timeout_seconds,
            thinking=settings.anthropic_thinking,
        )
    if client is None and not dry_run:
        logger.warning("no ANTHROPIC_API_KEY configured — postings will be filtered but not LLM-evaluated")

    if triage_client is None and settings.anthropic_api_key and settings.triage_model and not dry_run:
        triage_client = AnthropicClient(
            api_key=settings.anthropic_api_key,
            model=settings.triage_model,
            max_retries=settings.anthropic_max_retries,
            timeout_seconds=settings.anthropic_timeout_seconds,
            thinking=settings.triage_thinking,
        )
    if triage_client is not None:
        stats.triage_model = triage_client.model

    workers = 1 if limit is not None else max(1, settings.evaluate_workers)
    # Postings waiting for their LLM calls, and later postings with the same description text,
    # which take a copy of that verdict when it lands — so a role posted in several locations still
    # costs one evaluation (as the cache gives it when screening runs one at a time).
    pending: list[tuple[JobPosting, Company]] = []
    followers: dict[str, list[JobPosting]] = {}

    def screen(job: JobPosting, company: Company) -> _Screening:
        assert client is not None
        return _screen(job, company, profile, profile_text, client, triage_client, settings, full_rubric, triage_rubric)

    def record(outcome: _Screening) -> None:
        job = outcome.job
        if outcome.triage_error is not None:
            logger.warning(
                "triage call failed — proceeding to full evaluation",
                extra=log_extra(job_id=job.id, title=job.title, error=outcome.triage_error),
            )
        if outcome.triage is not None:
            stats.triaged += 1
            t_in, t_out, t_cache_w, t_cache_r = outcome.triage_tokens
            stats.triage_input_tokens += t_in
            stats.triage_output_tokens += t_out
            stats.triage_cache_creation_input_tokens += t_cache_w
            stats.triage_cache_read_input_tokens += t_cache_r
        if outcome.triage_overruled:
            stats.triage_overruled += 1
            logger.info(
                "triage skip not substantiated by the posting — running the full evaluation",
                extra=log_extra(
                    job_id=job.id, title=job.title, disqualifier=outcome.triage.disqualifier,
                    quote=outcome.triage.disqualifier_quote, reason=outcome.triage.reason,
                ),
            )

        saved: Evaluation | None
        if outcome.skip is not None:
            stats.triage_skipped += 1
            saved = outcome.skip
        else:
            for evaluation in outcome.evaluations:
                _count_tokens(stats, evaluation)
            if outcome.second_look_error is not None:
                logger.warning("second look failed — keeping the first evaluation",
                               extra=log_extra(job_id=job.id, title=job.title, error=outcome.second_look_error))
            if len(outcome.evaluations) > 1:
                stats.second_looks += 1
                if outcome.chosen is outcome.evaluations[1]:
                    stats.second_looks_kept += 1
            saved = outcome.chosen
            if saved is None:
                logger.warning("job evaluation failed", extra=log_extra(job_id=job.id, title=job.title, error=outcome.error))
                db.record_filter_log(FilterLogEntry(job.id, "llm_error", "llm_call_failed", outcome.error, _now()))
                stats.unverified += 1
                stats.llm_errors += 1
            else:
                stats.sent_to_llm += 1
        if saved is not None:
            db.save_evaluation(saved)
            stats._bump_new_verdict(saved.verdict.value)

        for follower in followers.pop(job.description_hash, []):
            if saved is None:
                stats.unverified += 1
            else:
                stats.cache_hits += 1
                db.save_evaluation(replace(saved, id=None, job_id=follower.id))
                stats._bump_verdict(saved.verdict.value)

    with make_client(settings) as http_client:
        jobs = db.get_active_jobs()
        for job in jobs:
            if limit is not None and (stats.sent_to_llm + stats.companies_classified + stats.triaged) >= limit:
                break

            company = db.get_company(job.company_id)
            if company is None or not company.active:
                # A deactivated company's postings keep their history but leave the pool: nothing
                # re-crawls its board to close them, so they'd otherwise be re-checked forever.
                continue
            stats.jobs_considered += 1

            job_overrides = db.get_manual_overrides("job", job.id)
            if "verdict" in job_overrides:
                stats.manual_overrides_applied += 1
                evaluation = _manual_override_evaluation(job, company, Verdict(job_overrides["verdict"]))
                current = db.get_evaluation_for_job(job.id)
                # Saved once, not on every run: evaluations keep their history.
                if not (current is not None and current.model_name == "manual_override"
                        and current.verdict == evaluation.verdict and current.description_hash == job.description_hash):
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
                    stats.newly_filtered += 1
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
                    stats.newly_filtered += 1
                stats.factual_rejected += 1
                continue

            cached = db.get_evaluation_by_hash(
                job.description_hash, rubric_versions=(full_rubric, triage_rubric) if refresh_stale else None
            )
            if cached is not None:
                stats.cache_hits += 1
                # A verdict made for another posting with the same text, or an older verdict of this
                # posting whose rubric is current again (a reverted change): copy it forward so it's
                # the posting's newest — what the report reads.
                if not _same_verdict(db.get_evaluation_for_job(job.id), cached):
                    db.save_evaluation(replace(cached, id=None, job_id=job.id))
                stats._bump_verdict(cached.verdict.value)
                continue

            if client is None:
                stats.unverified += 1
                continue

            if job.description_hash in followers:
                followers[job.description_hash].append(job)
            elif workers == 1:
                followers[job.description_hash] = []
                record(screen(job, company))
            else:
                followers[job.description_hash] = []
                pending.append((job, company))

    if pending:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(screen, job, company) for job, company in pending]
            for future in as_completed(futures):
                record(future.result())

    stats.finished_at = _now()
    if run_id is not None:
        db.finish_evaluation_run(run_id, stats)
    return stats
