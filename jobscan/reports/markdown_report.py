"""Markdown report: run statistics, the candidate's applications and where each stands, Best Bets,
Growth Bets, Attractive Stretches, and a summary of what was rejected or left unverified. Roles
already applied to are starred (★) wherever they appear."""
from __future__ import annotations

from pathlib import Path

from jobscan.models import Evaluation, JobPosting, ScopeFit, Verdict, WorkingHoursFit
from jobscan.reports.data import JobReportRow, ReportData, declined_reason, is_declined


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


def _render_stats(data: ReportData) -> list[str]:
    lines = ["## Run statistics", ""]
    cs = data.crawl_stats
    es = data.evaluate_stats
    if cs:
        lines += [
            f"- Companies attempted: **{cs.companies_attempted}**",
            f"- Companies successfully crawled: **{cs.companies_succeeded}**",
            f"- Failed boards: **{len(cs.boards_failed)}**" + (f" — {', '.join(cs.boards_failed)}" if cs.boards_failed else ""),
            f"- Total postings fetched this crawl (all departments): **{cs.postings_fetched}**",
            f"- Skipped as outside the software-engineering job family (never stored): **{cs.postings_out_of_family}**",
            f"- New postings: **{cs.new_postings}**",
            f"- Changed postings: **{cs.changed_postings}**",
            f"- Closed postings: **{cs.closed_postings}**",
        ]
    else:
        lines.append("- No crawl was run in this invocation (reporting on existing database state).")

    if es:
        rejected_n = es.verdict_counts.get("reject", 0)
        lines += [
            f"- Postings sent to the LLM this run: **{es.sent_to_llm}** (cache hits reused: {es.cache_hits})",
            f"- Rejected by safe factual filters: **{es.factual_rejected}**",
            f"- Companies newly classified: **{es.companies_classified}**",
            f"- Manual overrides applied: **{es.manual_overrides_applied}**",
            f"- Best Bets: **{len(data.best_bets)}**",
            f"- Growth Bets: **{len(data.growth_bets)}**",
            f"- Attractive Stretches: **{len(data.attractive_stretches)}**",
            f"- Rejected by LLM: **{rejected_n}**",
            f"- Unverified (no evaluation obtained, e.g. missing API key or LLM error): **{es.unverified}**",
        ]
        if data.estimated_cost_usd is not None:
            lines.append(
                f"- Estimated LLM cost this run: **${data.estimated_cost_usd:,.4f}** "
                f"({es.input_tokens:,} input / {es.output_tokens:,} output tokens)"
            )
        else:
            lines.append(
                f"- LLM tokens used this run: {es.input_tokens:,} input / {es.output_tokens:,} output "
                "(cost n/a — model not in pricing table)"
            )
    else:
        lines += [
            "- No evaluation was run in this invocation (reporting on existing database state).",
            f"- Best Bets: **{len(data.best_bets)}**  Growth Bets: **{len(data.growth_bets)}**  "
            f"Attractive Stretches: **{len(data.attractive_stretches)}**",
            f"- Postings with no evaluation on record: **{data.unverified_count}**",
        ]

    return lines


def _role(row: JobReportRow) -> str:
    """'Company — Title', starred with the date when the candidate has already applied."""
    name = f"{row.company.name} — {row.job.title}"
    return f"★ {name} (applied {row.applied_on.isoformat()})" if row.applied_on else name


def _hours_line(ev: Evaluation) -> list[str]:
    """The posting's own words on working hours, whenever it states any."""
    if not ev.working_hours_quote:
        return []
    note = " — expects Eastern/Central hours" if ev.working_hours_fit == WorkingHoursFit.EASTERN_OR_CENTRAL_PREFERRED else ""
    return [f'- Working hours: "{ev.working_hours_quote}"{note}']


def _render_applications(data: ReportData) -> list[str]:
    lines = ["", "## Applications", ""]
    if not data.applications:
        lines.append("_None recorded — add them to `data/applications.yaml`._")
        return lines
    lines += ["| Applied | Role | Where it stands |", "|---|---|---|"]
    for row in data.applications:
        app = row.application
        standing = row.standing.replace("|", "\\|")
        lines.append(f"| {app.applied_on.isoformat()} | [{app.company} — {app.title}]({app.url}) | {standing} |")
    return lines


def _render_best_or_growth(title: str, rows: list[JobReportRow], empty_note: str) -> list[str]:
    lines = ["", f"## {title}", ""]
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
        "_Strong technical overlap, but two or more unproven scope dimensions — or a hidden "
        "staff-level/elite-startup expectation behind an ordinary-looking Senior title — or a good "
        "fit that sits below your level or expects Eastern/Central hours. Worth a look, not a Best "
        "Bet._", "",
    ]
    if not rows:
        lines.append("_None._")
        return lines

    for row in rows:
        job, company, ev = row.job, row.company, row.evaluation
        assert ev is not None
        stretch = ev.growth_dimensions or [ev.primary_rejection_reason or ""]
        # Roles that would otherwise be a Best or Growth Bet land here for their level or hours.
        if ev.scope_fit == ScopeFit.BELOW_LEVEL:
            stretch = ["below the candidate's level"] + stretch
        if ev.working_hours_fit == WorkingHoursFit.EASTERN_OR_CENTRAL_PREFERRED:
            stretch = ["expects Eastern/Central Time hours"] + stretch
        stretch_text = "; ".join(item for item in stretch if item) or "(not specified)"
        lines += [
            f"### {_role(row)} ({ev.verdict.value}, {ev.scope_fit.value})",
            "",
            f"- Strong matching areas: {'; '.join(ev.required_matches) if ev.required_matches else '(none noted)'}",
            f"- What makes this a stretch: {stretch_text}",
        ]
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
    # borderline judged not worth applying, or required Eastern/Central hours — so nothing is
    # silently dropped from every section (see reports.data._report_section).
    llm_declined = [r for r in data.all_evaluated if r.evaluation and is_declined(r.evaluation)]

    es = data.evaluate_stats
    summary = [f"- Unverified (no evaluation on record): **{data.unverified_count}**"]
    if es:
        summary.insert(0, f"- Rejected by safe factual filters this run: **{es.factual_rejected}**")
    lines += summary
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
        elif ev.working_hours_fit == WorkingHoursFit.EASTERN_OR_CENTRAL_REQUIRED:
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
    lines += _render_best_or_growth(
        "Best Bets", data.best_bets,
        "None yet — at-level strong/plausible matches will show up here.",
    )
    lines += _render_best_or_growth(
        "Growth Bets", data.growth_bets,
        "None yet — genuine one-step-up roles with no more than one material growth dimension will show up here.",
    )
    lines += _render_attractive_stretches(data.attractive_stretches)
    lines += _render_rejected_or_unverified(data)
    return "\n".join(lines) + "\n"


def write_markdown_report(data: ReportData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(data), encoding="utf-8")
