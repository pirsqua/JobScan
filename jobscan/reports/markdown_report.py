"""Markdown report: run statistics, the candidate's applications and where each stands, Best Bets,
Good Fits, Growth Bets, Attractive Stretches, and a summary of what was rejected or left unverified.
Roles already applied to are listed under Applications, not in the shortlist, and starred (★)
wherever else they appear."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from jobscan.models import Evaluation, JobPosting, ScopeFit, Verdict, WorkFit
from jobscan.reports.data import JobReportRow, ReportData, declined_reason, is_declined
from jobscan.timeutil import SEATTLE_TZ


def _freshness(job: JobPosting) -> str:
    if job.published_at:
        return f"published {job.published_at.date().isoformat()}; first seen {job.first_seen_at.date().isoformat()}"
    return f"first seen {job.first_seen_at.date().isoformat()}"


def _salary_line(job: JobPosting) -> str:
    if job.salary_min is None and job.salary_max is None:
        return "not published"
    lo = f"${job.salary_min:,.0f}" if job.salary_min is not None else "?"
    hi = f"${job.salary_max:,.0f}" if job.salary_max is not None else "?"
    return f"{lo} - {hi} / {job.salary_period or 'year'} (source: {job.salary_source.value})"


_VERDICT_ORDER = ("strong_match", "plausible_match", "borderline", "reject")


def _verdicts(counts: dict[str, int]) -> str:
    parts = [f"{counts[v]} {v.replace('_', ' ')}" for v in _VERDICT_ORDER if counts.get(v)]
    return ", ".join(parts) or "none"


def _seattle_time(when: datetime | None) -> str:
    return when.astimezone(SEATTLE_TZ).strftime("%Y-%m-%d %H:%M PT") if when else "unknown time"


def _render_stats(data: ReportData) -> list[str]:
    """Two blocks that must not be mixed: what the latest run did, and the state of the whole pool.
    Evaluation re-checks every active posting each run, so its pool-wide counts (all factual
    rejections, every cached verdict) once sat beside "0 new postings" as if they were the run's."""
    cs, es = data.crawl_stats, data.evaluate_stats
    lines = ["## Run statistics", "", "### Latest run", ""]
    if cs:
        failed = f" — failed: {'; '.join(cs.boards_failed)}" if cs.boards_failed else ""
        lines += [
            f"- Crawl ({_seattle_time(cs.started_at)}): **{cs.companies_succeeded}** of {cs.companies_attempted} "
            f"companies crawled{failed}",
            f"- Postings fetched: {cs.postings_fetched} across all departments, {cs.postings_out_of_family} of them "
            "outside software engineering (never stored)",
            f"- **{cs.new_postings}** new, **{cs.changed_postings}** changed, **{cs.closed_postings}** closed",
        ]
    else:
        lines.append("- No crawl on record.")
    if es:
        lines.append(f"- Evaluation ({_seattle_time(es.started_at)}): **{es.newly_filtered}** newly rejected by factual filters")
        if es.triage_model:
            lines.append(f"- Triage: **{es.triaged}** screened, {es.triage_skipped} skipped before a full evaluation")
        lines.append(f"- Full evaluations: **{es.sent_to_llm}**")
        if es.second_looks:
            lines.append(f"- Near-miss second looks: {es.second_looks} ({es.second_looks_kept} came out better)")
        if es.companies_classified:
            lines.append(f"- Companies newly classified: {es.companies_classified}")
        if es.llm_errors:
            lines.append(f"- LLM errors (left awaiting evaluation): {es.llm_errors}")
        lines.append(f"- New verdicts: {_verdicts(es.new_verdict_counts)}")
        if data.estimated_cost_usd is not None:
            lines.append(f"- Cost: **${data.estimated_cost_usd:,.2f}**")
    else:
        lines.append("- No evaluation on record.")

    pool = data.pool
    lines += [
        "",
        "### Current pool",
        "",
        "_Every active posting at an active company, as of this report._",
        "",
        f"- Active postings: **{pool.active_postings}**",
        f"- Rejected by factual filters (salary, remote, employment type, consulting): **{pool.factually_rejected}**",
        f"- Evaluated: **{pool.evaluated}** — {_verdicts(pool.verdict_counts)}",
        f"- Awaiting evaluation: **{pool.awaiting_evaluation}**",
        f"- Shortlist (not yet applied to): **{len(data.best_bets)}** Best Bets, **{len(data.good_fits)}** "
        f"Good Fits, **{len(data.growth_bets)}** Growth Bets, **{len(data.attractive_stretches)}** "
        "Attractive Stretches",
    ]
    return lines


def _role(row: JobReportRow) -> str:
    """'Company — Title', starred with the date when the candidate has already applied."""
    name = f"{row.company.name} — {row.job.title}"
    return f"★ {name} (applied {row.applied_on.isoformat()})" if row.applied_on else name


_WORK_FIT_NOTES = {
    WorkFit.PARTLY_OUTSIDE: " — partly outside your lane",
    WorkFit.OUTSIDE: " — outside your lane",
}


def _work_line(ev: Evaluation) -> list[str]:
    """What the team's engineers do day to day, whenever the evaluation says."""
    if not ev.day_to_day_work:
        return []
    return [f"- Day to day: {ev.day_to_day_work}{_WORK_FIT_NOTES.get(ev.work_fit, '') if ev.work_fit else ''}"]


def _hours_line(ev: Evaluation) -> list[str]:
    """The posting's own words on working hours, whenever it states any."""
    if not ev.working_hours_quote:
        return []
    note = " — fails the working-hours filter" if ev.working_hours_fit and ev.working_hours_fit.disqualifies else ""
    return [f'- Working hours: "{ev.working_hours_quote}"{note}']


def _render_applications(data: ReportData) -> list[str]:
    lines = ["", "## Applications", ""]
    if not data.applications:
        lines.append("_None recorded — add them to your applications file (`JOBSCAN_APPLICATIONS_PATH`)._")
        return lines
    lines += ["| Applied | Role | Where it stands |", "|---|---|---|"]
    for row in data.applications:
        app = row.application
        standing = row.standing.replace("|", "\\|")
        lines.append(f"| {app.applied_on.isoformat()} | [{app.company} — {app.title}]({app.url}) | {standing} |")
    return lines


def _render_shortlist(title: str, intro: str, rows: list[JobReportRow], empty_note: str) -> list[str]:
    lines = ["", f"## {title}", "", f"_{intro}_", ""]
    if not rows:
        lines.append(f"_{empty_note}_")
        return lines

    for row in rows:
        job, company, ev = row.job, row.company, row.evaluation
        assert ev is not None
        lines += [
            f"### {_role(row)} ({ev.verdict.value}, {ev.scope_fit.value}, "
            f"{ev.evidence_coverage_percent}% coverage, confidence {ev.confidence:.2f})",
            "",
            f"- Salary: {_salary_line(job)}",
            f"- Apply: {job.apply_url or job.posting_url or 'n/a'}",
            f"- Freshness: {_freshness(job)}",
        ]
        lines += _work_line(ev)
        lines += _hours_line(ev)
        if ev.required_matches:
            lines.append(f"- Exact matches: {'; '.join(ev.required_matches)}")
        if ev.growth_dimensions:
            lines.append(f"- Growth dimension to note: {'; '.join(ev.growth_dimensions)}")
        if ev.minor_caveats:
            lines.append(f"- Minor caveats: {'; '.join(ev.minor_caveats)}")
        lines.append(f"- Compensation assessment: {ev.compensation_assessment}")
        lines.append(f"- Why this is gettable: {ev.why_this_is_or_is_not_gettable}")
        lines.append(f"- Credibility: {ev.credibility_assessment}")
        lines.append("")

    return lines


def _render_attractive_stretches(rows: list[JobReportRow]) -> list[str]:
    lines = [
        "", "## Attractive Stretches", "",
        "_Strong technical overlap, but two or more required qualifications you don't have, two or "
        "more unproven scope dimensions, or a hidden staff-level expectation behind an "
        "ordinary-looking Senior title — or a good fit that sits below your level. Worth a look, "
        "not a first pick._", "",
    ]
    if not rows:
        lines.append("_None._")
        return lines

    for row in rows:
        job, company, ev = row.job, row.company, row.evaluation
        assert ev is not None
        stretch = ev.growth_dimensions or [ev.primary_rejection_reason or ""]
        # A match that would otherwise be a Best or Growth Bet lands here for its level or its work.
        if ev.scope_fit == ScopeFit.BELOW_LEVEL:
            stretch = ["below the candidate's level"] + stretch
        if ev.work_fit == WorkFit.OUTSIDE:
            stretch = ["the day-to-day work is outside your lane"] + stretch
        stretch_text = "; ".join(item for item in stretch if item) or "(not specified)"
        lines += [
            f"### {_role(row)} ({ev.verdict.value}, {ev.scope_fit.value})",
            "",
            f"- Salary: {_salary_line(job)}",
            f"- Apply: {job.apply_url or job.posting_url or 'n/a'}",
            f"- Freshness: {_freshness(job)}",
            f"- Strong matching areas: {'; '.join(ev.required_matches) if ev.required_matches else '(none noted)'}",
            f"- What makes this a stretch: {stretch_text}",
        ]
        lines += _work_line(ev)
        lines += _hours_line(ev)
        if ev.hidden_staff_signals:
            lines.append(f"- Hidden staff-level signals: {'; '.join(ev.hidden_staff_signals)}")
        lines.append(f"- What would make this credible: {ev.why_this_is_or_is_not_gettable}")
        lines.append(f"- Evidence: {'; '.join(ev.evidence) if ev.evidence else '(none captured)'}")
        lines.append("")
    return lines


def _render_rejected_or_unverified(data: ReportData) -> list[str]:
    lines = ["", "## Rejected or Unverified", ""]
    # Every evaluated posting in no shortlist section: verdict=reject outright, but also e.g. a
    # borderline judged not worth applying, or hours that fail the hours filter — so nothing is
    # silently dropped from every section (see reports.data._report_section).
    llm_declined = [r for r in data.all_evaluated if r.evaluation and is_declined(r.evaluation)]

    lines += [
        f"- Rejected by factual filters: **{data.pool.factually_rejected}**",
        f"- Awaiting evaluation: **{data.pool.awaiting_evaluation}**",
    ]
    lines.append(
        f"- Rejected or not realistically worth applying per the LLM: **{len(llm_declined)}** "
        "(listed below; run `python -m jobscan audit` for the complete factual-filter and "
        "unverified listing with reasons)"
    )
    lines.append("")

    if not llm_declined:
        lines.append("_No LLM rejections to list._")
        return lines

    lines.append("| Company | Title | Verdict | Primary rejection reason |")
    lines.append("|---|---|---|---|")
    for row in llm_declined:
        ev: Evaluation = row.evaluation  # type: ignore[assignment]
        reason = declined_reason(ev).replace("|", "\\|")
        if ev.verdict == Verdict.REJECT:
            verdict_label = "reject"
        elif ev.working_hours_fit and ev.working_hours_fit.disqualifies:
            verdict_label = f"{ev.verdict.value}, but hours fail"
        elif not ev.worth_applying:
            verdict_label = f"{ev.verdict.value}, not worth applying"
        else:
            verdict_label = f"{ev.verdict.value}, {ev.scope_fit.value}"
        title = f"★ {row.job.title}" if row.applied_on else row.job.title
        lines.append(f"| {row.company.name} | {title} | {verdict_label} | {reason} |")
    return lines


def render_markdown(data: ReportData) -> str:
    lines = [f"# JobScan report — {data.generated_at}", ""]
    lines += _render_stats(data)
    lines += _render_applications(data)
    lines += _render_shortlist(
        "Best Bets",
        "Strong matches at your level, with every required qualification demonstrated or credibly "
        "transferable. Apply to these first.",
        data.best_bets, "None right now.",
    )
    lines += _render_shortlist(
        "Good Fits",
        "Plausible matches at your level, with nothing required missing.",
        data.good_fits, "None right now.",
    )
    lines += _render_shortlist(
        "Growth Bets",
        "One step up with no more than one material growth dimension, or at your level but missing "
        "one required qualification.",
        data.growth_bets, "None right now.",
    )
    lines += _render_attractive_stretches(data.attractive_stretches)
    lines += _render_rejected_or_unverified(data)
    return "\n".join(lines) + "\n"


def write_markdown_report(data: ReportData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(data), encoding="utf-8")
