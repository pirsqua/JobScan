"""Assembles the data reports are rendered from: pulls together jobs, companies, evaluations and
the most recent crawl_run row into one structure the markdown/csv/json renderers all share.

Grouping and ranking implement the scope-calibration ranking order: verdict alone isn't enough —
a "plausible_match" that is actually a hidden-staff-level stretch must not outrank a genuinely
at-level "plausible_match", and compensation is a clamped tiebreaker, not an unbounded score.
"""
from __future__ import annotations

from dataclasses import dataclass

from jobscan.config import Settings
from jobscan.db import Database
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


def _report_section(evaluation: Evaluation | None) -> str:
    """Which of the four report sections a row belongs in. Decoupled from _ranking_group: e.g. a
    borderline verdict lands in "attractive_stretches" regardless of exactly which scope_fit it
    carries, since that's the section's whole purpose — group number only orders rows within it."""
    if evaluation is None or evaluation.verdict == Verdict.REJECT:
        return "rejected_or_unverified"
    if evaluation.verdict in (Verdict.STRONG_MATCH, Verdict.PLAUSIBLE_MATCH):
        if evaluation.scope_fit == ScopeFit.AT_LEVEL:
            return "best_bets"
        if evaluation.scope_fit == ScopeFit.ONE_STEP_UP and evaluation.verdict == Verdict.PLAUSIBLE_MATCH:
            return "growth_bets"
        if evaluation.scope_fit == ScopeFit.BELOW_LEVEL:
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


def assemble_report_data(
    db: Database,
    settings: Settings,
    crawl_stats: CrawlRunStats | None = None,
    evaluate_stats: EvaluateStats | None = None,
) -> ReportData:
    rows: list[JobReportRow] = []
    for job in db.get_active_jobs():
        company = db.get_company(job.company_id)
        if company is None:
            continue
        evaluation = db.get_evaluation_for_job(job.id)
        rows.append(JobReportRow(job=job, company=company, evaluation=evaluation))

    sections: dict[str, list[JobReportRow]] = {"best_bets": [], "growth_bets": [], "attractive_stretches": []}
    for row in rows:
        section = _report_section(row.evaluation)
        if section in sections:
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
    )
