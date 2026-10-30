"""Assembles the data reports are rendered from: pulls together jobs, companies, evaluations and
the most recent crawl_run row into one structure the markdown/csv/json renderers all share.

Grouping and ranking implement the scope-calibration ranking order: verdict alone isn't enough —
a "plausible_match" that is actually a hidden-staff-level stretch must not outrank a genuinely
at-level "plausible_match", and compensation is a clamped tiebreaker, not an unbounded score.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field, replace

from jobscan.applications import Application, Pause, load_applications, load_pauses, normalize_url
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
    RequirementStrength,
    ScopeFit,
    SpecialistTenureClassification,
    Verdict,
    WorkFit,
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
    # Days since applying, while the status is still "applied" (no reply yet); None otherwise.
    days_waiting: int | None = None


@dataclass
class PausedRow:
    """A company paused in the applications file, and how many of its shortlisted roles it holds back."""

    pause: Pause
    held_back: int


@dataclass
class PoolStats:
    """The current state of every active posting at an active company, computed at report time —
    as opposed to a run's own work (CrawlRunStats, EvaluateStats.new_verdict_counts). The three
    groups partition the pool: factually rejected, evaluated, awaiting evaluation."""

    active_postings: int = 0
    factually_rejected: int = 0
    evaluated: int = 0
    awaiting_evaluation: int = 0
    verdict_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class ReportData:
    generated_at: str
    crawl_stats: CrawlRunStats | None
    evaluate_stats: EvaluateStats | None
    estimated_cost_usd: float | None
    best_bets: list[JobReportRow]
    good_fits: list[JobReportRow]
    growth_bets: list[JobReportRow]
    attractive_stretches: list[JobReportRow]
    all_evaluated: list[JobReportRow]
    pool: PoolStats
    applications: list[ApplicationRow] = field(default_factory=list)
    paused: list[PausedRow] = field(default_factory=list)


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


def settle_scope(evaluation: Evaluation) -> Evaluation:
    """Draw the at_level / one_step_up line from the evaluation's own structured requirement
    evidence instead of its one-word label, which is sampled and flips. Observed: Samsara's
    Software Engineer II, evaluated on consecutive days with the same evidence — no central
    requirement undemonstrated, specialist tenure not applicable, no staff signals — was labelled
    one_step_up, then at_level: Growth Bet one day, Best Bet the next.

    Raised to at_level only on clean evidence: no central requirement not_demonstrated, specialist
    tenure meets or not applicable, no hidden staff signals. Lowered to one_step_up when a central
    requirement is not_demonstrated or specialist tenure is insufficient (a strong_match then caps
    at plausible_match — strong_match requires at_level). The staff-signal list is loose (it has
    held "we do not have a QA team"), so it can block a raise but never forces a drop. Other
    labels, and evaluations without central requirement evidence, are left as the model gave them."""
    if evaluation.scope_fit not in (ScopeFit.AT_LEVEL, ScopeFit.ONE_STEP_UP):
        return evaluation
    central = [
        item
        for item in evaluation.requirement_evidence
        if item.importance == RequirementImportance.CENTRAL
    ]
    if not central:
        return evaluation
    tenure = evaluation.specialist_tenure_assessment.classification
    gap = any(
        item.evidence_classification == EvidenceClassification.NOT_DEMONSTRATED for item in central
    )
    if gap or tenure == SpecialistTenureClassification.INSUFFICIENT:
        verdict = (
            Verdict.PLAUSIBLE_MATCH
            if evaluation.verdict == Verdict.STRONG_MATCH
            else evaluation.verdict
        )
        return replace(evaluation, scope_fit=ScopeFit.ONE_STEP_UP, verdict=verdict)
    clean_tenure = tenure in (
        SpecialistTenureClassification.MEETS,
        SpecialistTenureClassification.NOT_APPLICABLE,
    )
    if clean_tenure and not evaluation.hidden_staff_signals:
        return replace(evaluation, scope_fit=ScopeFit.AT_LEVEL)
    return evaluation


def _missing_requirements(evaluation: Evaluation) -> int:
    """Required qualifications the candidate essentially lacks: stated as required or strongly
    implied, with evidence only weakly inferred or not demonstrated — whatever importance the model
    gave them. Observed: Affirm's "experience shipping web apps using declarative UI frameworks like
    React or Vue", filed as a secondary, weakly inferred requirement of a plausible_match at level,
    made a Best Bet of a frontend stretch."""
    return sum(
        1
        for item in evaluation.requirement_evidence
        if item.stated_as in (RequirementStrength.REQUIRED, RequirementStrength.STRONGLY_IMPLIED)
        and item.evidence_classification
        in (EvidenceClassification.WEAKLY_INFERRED, EvidenceClassification.NOT_DEMONSTRATED)
    )


def _central_transferable_count(evaluation: Evaluation) -> int:
    return sum(
        1
        for item in evaluation.requirement_evidence
        if item.importance == RequirementImportance.CENTRAL
        and item.evidence_classification == EvidenceClassification.CREDIBLY_TRANSFERABLE
    )


def _section_from_verdict(evaluation: Evaluation | None) -> str:
    """Which of the five report sections a row belongs in, before the work-fit cap. Decoupled from _ranking_group: e.g. a
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
    # Enforced here rather than trusted to the verdict: hours that fail the profile's working-hours
    # filter fail a hard filter — observed live, a posting "ideally" wanting hours the filter rules
    # out came out a strong-match Best Bet.
    if evaluation.working_hours_fit is not None and evaluation.working_hours_fit.disqualifies:
        return "rejected_or_unverified"
    # Likewise more frontend than the profile accepts: without professional React or Angular
    # experience, the backend half fitting doesn't win the role (Sonnet 5.5 made a Growth Bet of a
    # Senior Full-Stack role whose work was half React features).
    if evaluation.frontend_fit is not None and evaluation.frontend_fit.disqualifies:
        return "rejected_or_unverified"
    if evaluation.verdict in (Verdict.STRONG_MATCH, Verdict.PLAUSIBLE_MATCH):
        # The tiers split on what the candidate lacks, read from the requirement-by-requirement
        # evidence, not the one-word verdict alone: Sonnet 5.5 called far more postings plausible
        # matches than Sonnet 5 had (Best Bets went from 23 to 61, too many to choose from).
        missing = _missing_requirements(evaluation)
        # A below-Senior title tiers like an at-level one: with pay passing the candidate's floor
        # and the work in lane, it is among the most gettable applications (a mid-level band can
        # reach well past a weaker company's Senior one). It once sat in Attractive Stretches as
        # "underusing" the candidate, out of sight of the sections applied from; the report flags
        # the level instead.
        if evaluation.scope_fit in (ScopeFit.AT_LEVEL, ScopeFit.BELOW_LEVEL):
            if missing == 0:
                return "best_bets" if evaluation.verdict == Verdict.STRONG_MATCH else "good_fits"
            return "growth_bets" if missing == 1 else "attractive_stretches"
        if (
            evaluation.scope_fit == ScopeFit.ONE_STEP_UP
            and evaluation.verdict == Verdict.PLAUSIBLE_MATCH
        ):
            return "growth_bets" if missing <= 1 else "attractive_stretches"
    if not evaluation.worth_applying:
        return "rejected_or_unverified"
    # Anything else attractive-but-unconfident (borderline at any scope_fit, or a
    # verdict/scope_fit combination the prompt's own rules shouldn't produce) is a stretch.
    return "attractive_stretches"


_SHORTLIST = ("best_bets", "good_fits", "growth_bets", "attractive_stretches")
# Day-to-day work outside the candidate's lane caps the tier, however clean the qualifications list
# reads. Observed: GitHub's npm team came out a plausible match because its list accepted C#.
_WORK_FIT_CAP = {WorkFit.PARTLY_OUTSIDE: "growth_bets", WorkFit.OUTSIDE: "attractive_stretches"}


def _report_section(evaluation: Evaluation | None) -> str:
    """The report section a row belongs in: the verdict and evidence decide it, then the role's
    day-to-day work (``work_fit``) can only push it down."""
    section = _section_from_verdict(evaluation)
    cap = (
        _WORK_FIT_CAP.get(evaluation.work_fit)
        if evaluation is not None and evaluation.work_fit
        else None
    )
    if cap is not None and section in _SHORTLIST:
        return max(section, cap, key=_SHORTLIST.index)
    return section


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
        v
        for v in (
            annualize(job.salary_min, job.salary_period),
            annualize(job.salary_max, job.salary_period),
        )
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
        _missing_requirements(ev),
        _central_transferable_count(ev),
        -ev.evidence_coverage_percent,
        -_central_directly_demonstrated_count(ev),
        len(ev.required_gaps),
        -_clamped_compensation_score(row.job, settings),
        -_freshness_epoch(row.job),
    )


_SECTION_LABELS = {
    "best_bets": "Best Bet",
    "good_fits": "Good Fit",
    "growth_bets": "Growth Bet",
    "attractive_stretches": "Attractive Stretch",
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
    evaluation = settle_scope(evaluation)
    section = _report_section(evaluation)
    if section in _SECTION_LABELS:
        return _SECTION_LABELS[section]
    return f"Rejected — {declined_reason(evaluation)}"


def is_declined(evaluation: Evaluation) -> bool:
    """Evaluated but in no shortlist section — rejected, judged not worth applying, or failing a
    rule code enforces over the verdict (see _report_section)."""
    return _report_section(evaluation) == "rejected_or_unverified"


def declined_reason(evaluation: Evaluation) -> str:
    if (
        evaluation.verdict != Verdict.REJECT
        and evaluation.working_hours_fit is not None
        and evaluation.working_hours_fit.disqualifies
    ):
        return f'working hours fail the hours filter: "{evaluation.working_hours_quote}"'
    if (
        evaluation.verdict != Verdict.REJECT
        and evaluation.frontend_fit is not None
        and evaluation.frontend_fit.disqualifies
    ):
        return f"more frontend work than your profile accepts: {evaluation.day_to_day_work}"
    return evaluation.primary_rejection_reason or "no reason recorded"


def _application_rows(
    db: Database, applications: list[Application], settings: Settings, today: dt.date
) -> list[ApplicationRow]:
    job_ids = {normalize_url(url): job_id for job_id, url in db.list_job_posting_urls()}
    rows = []
    for application in sorted(applications, key=lambda a: a.applied_on):
        job_id = job_ids.get(normalize_url(application.url))
        job = db.get_job(job_id) if job_id is not None else None
        waiting = (today - application.applied_on).days if application.status == "applied" else None
        rows.append(
            ApplicationRow(application, job, _application_standing(db, job, settings), waiting)
        )
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
        if company is None or not company.active:
            continue
        evaluation = db.get_evaluation_for_job(job.id)
        if evaluation is not None:
            evaluation = settle_scope(evaluation)
        rows.append(
            JobReportRow(
                job=job,
                company=company,
                evaluation=evaluation,
                applied_on=applied_on.get(normalize_url(job.posting_url))
                if job.posting_url
                else None,
            )
        )

    # A posting already applied to leaves the shortlist — the Applications table shows where it
    # stands — so the sections hold only what still needs a decision. So does one at a company
    # paused until a later date (e.g. a limit on how often you can apply); it is still crawled and
    # evaluated, and returns on that date.
    pauses = load_pauses(settings.applications_path)
    today = now_seattle().date()
    held_back: dict[str, int] = {}
    sections: dict[str, list[JobReportRow]] = {
        "best_bets": [],
        "good_fits": [],
        "growth_bets": [],
        "attractive_stretches": [],
    }
    for row in rows:
        section = _report_section(row.evaluation)
        if (
            section in sections
            and row.applied_on is None
            and _still_passes_factual_filters(row, settings)
        ):
            pause = next((p for p in pauses if p.holds(row.company.name, today)), None)
            if pause is not None:
                held_back[pause.company] = held_back.get(pause.company, 0) + 1
                continue
            sections[section].append(row)

    for section_rows in sections.values():
        section_rows.sort(key=lambda r: _rank_key(r, settings))

    evaluated = [r for r in rows if r.evaluation is not None]
    pool = PoolStats(active_postings=len(rows))
    for row in rows:
        if not _still_passes_factual_filters(row, settings):
            pool.factually_rejected += 1
        elif row.evaluation is None:
            pool.awaiting_evaluation += 1
        else:
            pool.evaluated += 1
            verdict = row.evaluation.verdict.value
            pool.verdict_counts[verdict] = pool.verdict_counts.get(verdict, 0) + 1

    estimated_cost = evaluate_stats.estimated_cost_usd(settings) if evaluate_stats else None

    return ReportData(
        generated_at=now_seattle().isoformat(),
        crawl_stats=crawl_stats,
        evaluate_stats=evaluate_stats,
        estimated_cost_usd=estimated_cost,
        best_bets=sections["best_bets"],
        good_fits=sections["good_fits"],
        growth_bets=sections["growth_bets"],
        attractive_stretches=sections["attractive_stretches"],
        all_evaluated=evaluated,
        pool=pool,
        applications=_application_rows(db, applications, settings, today),
        paused=[PausedRow(p, held_back.get(p.company, 0)) for p in pauses if today < p.until],
    )
