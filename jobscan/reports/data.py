"""Assembles the data reports are rendered from: pulls together jobs, companies, evaluations and
the most recent crawl_run row into one structure the markdown/csv/json renderers all share."""
from __future__ import annotations

from dataclasses import dataclass

from jobscan.config import Settings
from jobscan.db import Database
from jobscan.evaluate import EvaluateStats
from jobscan.models import Company, CrawlRunStats, Evaluation, JobPosting, Verdict


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
    recommended: list[JobReportRow]
    attractive_rejections: list[JobReportRow]
    all_evaluated: list[JobReportRow]
    unverified_count: int


def assemble_report_data(
    db: Database,
    settings: Settings,
    crawl_stats: CrawlRunStats | None = None,
    evaluate_stats: EvaluateStats | None = None,
) -> ReportData:
    from datetime import datetime, timezone

    rows: list[JobReportRow] = []
    for job in db.get_active_jobs():
        company = db.get_company(job.company_id)
        if company is None:
            continue
        evaluation = db.get_evaluation_for_job(job.id)
        rows.append(JobReportRow(job=job, company=company, evaluation=evaluation))

    recommended = sorted(
        (r for r in rows if r.evaluation and r.evaluation.verdict in (Verdict.STRONG_MATCH, Verdict.PLAUSIBLE_MATCH)),
        key=lambda r: (r.evaluation.verdict != Verdict.STRONG_MATCH, -r.evaluation.confidence),
    )

    attractive_rejections = sorted(
        (
            r
            for r in rows
            if r.evaluation
            and (
                r.evaluation.verdict == Verdict.BORDERLINE
                or (r.evaluation.verdict == Verdict.REJECT and len(r.evaluation.required_matches) >= 1)
            )
        ),
        key=lambda r: -len(r.evaluation.required_matches),
    )

    evaluated = [r for r in rows if r.evaluation is not None]
    unverified_count = sum(1 for r in rows if r.evaluation is None)

    estimated_cost = evaluate_stats.estimated_cost_usd(settings) if evaluate_stats else None

    return ReportData(
        generated_at=datetime.now(timezone.utc).isoformat(),
        crawl_stats=crawl_stats,
        evaluate_stats=evaluate_stats,
        estimated_cost_usd=estimated_cost,
        recommended=recommended,
        attractive_rejections=attractive_rejections,
        all_evaluated=evaluated,
        unverified_count=unverified_count,
    )
