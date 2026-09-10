"""Markdown report: run statistics, recommended roles, and the attractive-rejection log."""
from __future__ import annotations

from pathlib import Path

from jobscan.models import JobPosting
from jobscan.reports.data import JobReportRow, ReportData


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
        recommended_n = es.verdict_counts.get("strong_match", 0) + es.verdict_counts.get("plausible_match", 0)
        rejected_n = es.verdict_counts.get("reject", 0)
        borderline_n = es.verdict_counts.get("borderline", 0)
        lines += [
            f"- Postings sent to the LLM this run: **{es.sent_to_llm}** (cache hits reused: {es.cache_hits})",
            f"- Rejected by safe factual filters: **{es.factual_rejected}**",
            f"- Companies newly classified: **{es.companies_classified}**",
            f"- Manual overrides applied: **{es.manual_overrides_applied}**",
            f"- Recommended (strong/plausible match): **{recommended_n}**",
            f"- Borderline: **{borderline_n}**",
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
        lines.append("- No evaluation was run in this invocation (reporting on existing database state).")
        lines.append(f"- Postings with no evaluation on record: **{data.unverified_count}**")

    return lines


def _render_recommended(rows: list[JobReportRow]) -> list[str]:
    lines = ["", "## Recommended roles", ""]
    if not rows:
        lines.append("_None yet._")
        return lines

    for row in rows:
        job, company, ev = row.job, row.company, row.evaluation
        lines += [
            f"### {company.name} — {job.title} ({ev.verdict.value}, confidence {ev.confidence:.2f})",
            "",
            f"- Salary: {_salary_line(job)}",
            f"- Apply: {job.apply_url or job.posting_url or 'n/a'}",
            f"- Freshness: {_freshness(job)}",
        ]
        if ev.required_matches:
            lines.append(f"- Exact matches: {'; '.join(ev.required_matches)}")
        if ev.minor_caveats:
            lines.append(f"- Minor caveats: {'; '.join(ev.minor_caveats)}")
        lines.append(f"- Compensation assessment: {ev.compensation_assessment}")
        lines.append(f"- Credibility: {ev.credibility_assessment}")
        lines.append("")

    return lines


def _render_rejections(rows: list[JobReportRow]) -> list[str]:
    lines = ["", "## Attractive rejection log", "", "_Close calls worth a second look._", ""]
    if not rows:
        lines.append("_None._")
        return lines

    for row in rows:
        job, company, ev = row.job, row.company, row.evaluation
        lines += [
            f"### {company.name} — {job.title} ({ev.verdict.value})",
            "",
            f"- Strong matching areas: {'; '.join(ev.required_matches) if ev.required_matches else '(none noted)'}",
            f"- Material requirement causing rejection: {ev.primary_rejection_reason or '(not specified)'}",
            f"- Evidence: {'; '.join(ev.evidence) if ev.evidence else '(none captured)'}",
            "",
        ]
    return lines


def render_markdown(data: ReportData) -> str:
    lines = [f"# JobScan report — {data.generated_at}", ""]
    lines += _render_stats(data)
    lines += _render_recommended(data.recommended)
    lines += _render_rejections(data.attractive_rejections)
    return "\n".join(lines) + "\n"


def write_markdown_report(data: ReportData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(data), encoding="utf-8")
