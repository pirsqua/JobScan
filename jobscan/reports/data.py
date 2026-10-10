"""Assembles the data reports are rendered from: pulls together jobs, companies, evaluations and
the most recent crawl_run row into one structure the markdown/csv/json renderers all share.

Grouping and ranking implement the scope-calibration ranking order: verdict alone isn't enough —
a "plausible_match" that is actually a hidden-staff-level stretch must not outrank a genuinely
at-level "plausible_match", and compensation is a clamped tiebreaker, not an unbounded score.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from jobscan.applications import Application, load_applications, normalize_url
from jobscan.config import Settings
from jobscan.db import Database
from jobscan.filters import apply_factual_filters
from jobscan.models import (
    Company,
    CrawlRunStats,
    EvaluateStats,
    Evaluation,
    EvidenceClassification,
    JobPosting,
    RequirementImportance,
    ScopeFit,
    Verdict,
)
from jobscan.parsing import annualize
from jobscan.timeutil import now_seattle


@dataclass
class JobReportRow:
    job: JobPosting
    company: Company
    evaluation: Evaluation | None
    applied_on: dt.date | None = None


@dataclass
class ApplicationRow:
    """One of the candidate's applications and where that posting stands now."""

    application: Application
    job: JobPosting | None
    standing: str


@dataclass
class ReportData:
    generated_at: str
    crawl_stats: CrawlRunStats | None
    evaluate_stats: EvaluateStats | None
    estimated_cost_usd: float | None
    best_bets: list[JobReportRow]
    growth_bets: list[JobReportRow]
    attractive_stretches: list[JobReportRow]
    all_evaluated: list[JobReportRow]
    unverified_count: int
    applications: list[ApplicationRow] = field(default_factory=list)


# The spec's six ranking groups, most to least preferred. Any (verdict, scope_fit) combination
# the prompt's own verdict definitions rule out (e.g. strong_match + two_plus_steps_up) still
# gets a group here defensively, so a schema-contract slip surfaces in a sane place in the
# ordering rather than crashing or silently sorting first.
_RANKING_GROUPS: list[tuple[Verdict, ScopeFit]] = [
    (Verdict.STRONG_MATCH, ScopeFit.AT_LEVEL),
    (Verdict.PLAUSIBLE_MATCH, ScopeFit.AT_LEVEL),
    (Verdict.PLAUSIBLE_MATCH, ScopeFit.ONE_STEP_UP),
    (Verdict.BORDERLINE, ScopeFit.ONE_STEP_UP),
    (Verdict.BORDERLINE, ScopeFit.TWO_PLUS_STEPS_UP),
]


def _ranking_group(evaluation: Evaluation) -> int:
    try:
        return _RANKING_GROUPS.index((evaluation.verdict, evaluation.scope_fit)) + 1
    except ValueError:
        return 6


def _still_passes_factual_filters(row: JobReportRow, settings: Settings) -> bool:
    """Evaluations are cached by description text, so a verdict can outlive the facts it was made
    under (a posting reclassified as hybrid, a salary edited below the minimum, a company later
    confirmed as consulting). Re-check those facts at report time rather than trust the verdict —
    except for the candidate's own manual verdict overrides, which deliberately bypass them."""
    if row.evaluation is not None and row.evaluation.model_name == "manual_override":
        return True
    return apply_factual_filters(row.job, row.company, settings.min_base_salary).passed


def _report_section(evaluation: Evaluation | None) -> str:
    """Which of the four report sections a row belongs in. Decoupled from _ranking_group: e.g. a
    borderline verdict lands in "attractive_stretches" regardless of exactly which scope_fit it
    carries, since that's the section's whole purpose — group number only orders rows within it.

    worth_applying is the gate for that section specifically: verdict=borderline alone doesn't
    mean "genuine stretch" — the model can also reach for it on a role that's really just not
    attainable (central requirements in a technology/domain/scale never demonstrated) without
    formally calling it a reject. Observed live: a role requiring Rust/Kafka/Kubernetes-as-core
    systems the candidate has never touched landed in Attractive Stretches purely because the
    verdict happened to be "borderline" rather than "reject" — worth_applying=False routes those
    to rejected_or_unverified instead, regardless of the nominal verdict/scope_fit label."""
    if evaluation is None or evaluation.verdict == Verdict.REJECT:
        return "rejected_or_unverified"
    # Enforced here rather than trusted to the verdict: Eastern/Central hours, required or merely
    # preferred, fail the candidate's hard filter — observed live, a posting "ideally" wanting
    # Eastern hours came out a strong-match Best Bet.
    if evaluation.working_hours_fit is not None and evaluation.working_hours_fit.disqualifies:
        return "rejected_or_unverified"
    if evaluation.verdict in (Verdict.STRONG_MATCH, Verdict.PLAUSIBLE_MATCH):
        if evaluation.scope_fit == ScopeFit.AT_LEVEL:
            return "best_bets"
        if evaluation.scope_fit == ScopeFit.ONE_STEP_UP and evaluation.verdict == Verdict.PLAUSIBLE_MATCH:
            return "growth_bets"
        if evaluation.scope_fit == ScopeFit.BELOW_LEVEL:
            # Not a Best Bet (it underuses the candidate), but never dropped: below-Senior roles
            # with passing pay and backend work are wanted. These once vanished from every
            # section, unlisted even among the rejects.
            return "attractive_stretches"
    if not evaluation.worth_applying:
        return "rejected_or_unverified"
    # Anything else attractive-but-unconfident (borderline at any scope_fit, or a
    # verdict/scope_fit combination the prompt's own rules shouldn't produce) is a stretch.
    return "attractive_stretches"


def _central_directly_demonstrated_count(evaluation: Evaluation) -> int:
    return sum(
        1
        for item in evaluation.requirement_evidence
        if item.importance == RequirementImportance.CENTRAL
        and item.evidence_classification == EvidenceClassification.DIRECTLY_DEMONSTRATED
    )


def _clamped_compensation_score(job: JobPosting, settings: Settings) -> float:
    """Compensation is primarily a hard gate (jobscan.filters); here it only breaks ties between
    similarly-credible roles, clamped so an elite outlier salary can't out-rank a role the
    candidate is more credibly positioned to actually win."""
    candidates = [
        v for v in (annualize(job.salary_min, job.salary_period), annualize(job.salary_max, job.salary_period))
        if v is not None
    ]
    if not candidates:
        return 0.0
    midpoint = sum(candidates) / len(candidates)
    clamp_ceiling = settings.min_base_salary * 1.2
    return min(midpoint, clamp_ceiling)


def _freshness_epoch(job: JobPosting) -> float:
    moment = job.published_at or job.first_seen_at
    return moment.timestamp() if moment else 0.0


def _rank_key(row: JobReportRow, settings: Settings) -> tuple:
    ev = row.evaluation
    assert ev is not None
    return (
        _ranking_group(ev),
        -ev.evidence_coverage_percent,
        -_central_directly_demonstrated_count(ev),
        len(ev.required_gaps),
        -_clamped_compensation_score(row.job, settings),
        -_freshness_epoch(row.job),
    )


_SECTION_LABELS = {
    "best_bets": "Best Bet", "growth_bets": "Growth Bet", "attractive_stretches": "Attractive Stretch",
}


def _application_standing(db: Database, job: JobPosting | None, settings: Settings) -> str:
    if job is None:
        return "Not tracked (posting never crawled)"
    if job.closed_at is not None:
        return f"Closed {job.closed_at.date().isoformat()}"
    company = db.get_company(job.company_id)
    if company is None or not company.active:
        return "Company no longer searched"
    facts = apply_factual_filters(job, company, settings.min_base_salary)
    if not facts.passed:
        return f"Filtered out ({facts.reason})"
    evaluation = db.get_evaluation_for_job(job.id)
    if evaluation is None:
        return "Not evaluated yet"
    section = _report_section(evaluation)
    if section in _SECTION_LABELS:
        return _SECTION_LABELS[section]
    return f"Rejected — {declined_reason(evaluation)}"


def is_declined(evaluation: Evaluation) -> bool:
    """Evaluated but in no shortlist section — rejected, judged not worth applying, or failing a
    rule code enforces over the verdict (see _report_section)."""
    return _report_section(evaluation) == "rejected_or_unverified"


def declined_reason(evaluation: Evaluation) -> str:
    if evaluation.verdict != Verdict.REJECT and evaluation.working_hours_fit is not None and evaluation.working_hours_fit.disqualifies:
        return f'expects Eastern/Central Time hours: "{evaluation.working_hours_quote}"'
    return evaluation.primary_rejection_reason or "no reason recorded"


def _application_rows(db: Database, applications: list[Application], settings: Settings) -> list[ApplicationRow]:
    job_ids = {normalize_url(url): job_id for job_id, url in db.list_job_posting_urls()}
    rows = []
    for application in sorted(applications, key=lambda a: a.applied_on):
        job_id = job_ids.get(normalize_url(application.url))
        job = db.get_job(job_id) if job_id is not None else None
        rows.append(ApplicationRow(application, job, _application_standing(db, job, settings)))
    return rows


def assemble_report_data(
    db: Database,
    settings: Settings,
    crawl_stats: CrawlRunStats | None = None,
    evaluate_stats: EvaluateStats | None = None,
) -> ReportData:
    applications = load_applications(settings.applications_path)
    applied_on = {normalize_url(a.url): a.applied_on for a in applications}
    rows: list[JobReportRow] = []
    for job in db.get_active_jobs():
        company = db.get_company(job.company_id)
        if company is None:
            continue
        evaluation = db.get_evaluation_for_job(job.id)
        rows.append(JobReportRow(
            job=job, company=company, evaluation=evaluation,
            applied_on=applied_on.get(normalize_url(job.posting_url)) if job.posting_url else None,
        ))

    sections: dict[str, list[JobReportRow]] = {"best_bets": [], "growth_bets": [], "attractive_stretches": []}
    for row in rows:
        section = _report_section(row.evaluation)
        if section in sections and _still_passes_factual_filters(row, settings):
            sections[section].append(row)

    for key in sections:
        sections[key].sort(key=lambda r: _rank_key(r, settings))

    evaluated = [r for r in rows if r.evaluation is not None]
    unverified_count = sum(1 for r in rows if r.evaluation is None)

    estimated_cost = evaluate_stats.estimated_cost_usd(settings) if evaluate_stats else None

    return ReportData(
        generated_at=now_seattle().isoformat(),
        crawl_stats=crawl_stats,
        evaluate_stats=evaluate_stats,
        estimated_cost_usd=estimated_cost,
        best_bets=sections["best_bets"],
        growth_bets=sections["growth_bets"],
        attractive_stretches=sections["attractive_stretches"],
        all_evaluated=evaluated,
        unverified_count=unverified_count,
        applications=_application_rows(db, applications, settings),
    )
