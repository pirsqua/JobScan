"""Flat CSV export of every evaluated job — one row per posting."""
from __future__ import annotations

import csv
from pathlib import Path

from jobscan.reports.data import ReportData

FIELDS = [
    "company", "title", "applied_on", "verdict", "scope_fit", "evidence_coverage_percent", "confidence",
    "salary_min", "salary_max", "salary_period", "salary_source", "location_raw", "remote_scope",
    "employment_type", "working_hours_fit", "working_hours_quote", "posting_url", "apply_url", "published_at", "first_seen_at",
    "required_matches", "required_gaps", "preferred_only_gaps", "minor_caveats",
    "growth_dimensions", "hidden_staff_signals", "specialist_tenure_classification",
    "primary_rejection_reason", "worth_applying", "is_product_company", "model_name",
]


def write_csv_report(data: ReportData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in data.all_evaluated:
            job, company, ev = row.job, row.company, row.evaluation
            writer.writerow(
                {
                    "company": company.name,
                    "title": job.title,
                    "applied_on": row.applied_on.isoformat() if row.applied_on else "",
                    "verdict": ev.verdict.value,
                    "scope_fit": ev.scope_fit.value,
                    "evidence_coverage_percent": ev.evidence_coverage_percent,
                    "confidence": ev.confidence,
                    "salary_min": job.salary_min,
                    "salary_max": job.salary_max,
                    "salary_period": job.salary_period,
                    "salary_source": job.salary_source.value,
                    "location_raw": job.location_raw,
                    "remote_scope": job.remote_scope.value,
                    "employment_type": job.employment_type.value,
                    "working_hours_fit": ev.working_hours_fit.value if ev.working_hours_fit else "",
                    "working_hours_quote": ev.working_hours_quote,
                    "posting_url": job.posting_url,
                    "apply_url": job.apply_url,
                    "published_at": job.published_at.isoformat() if job.published_at else "",
                    "first_seen_at": job.first_seen_at.isoformat() if job.first_seen_at else "",
                    "required_matches": "; ".join(ev.required_matches),
                    "required_gaps": "; ".join(ev.required_gaps),
                    "preferred_only_gaps": "; ".join(ev.preferred_only_gaps),
                    "minor_caveats": "; ".join(ev.minor_caveats),
                    "growth_dimensions": "; ".join(ev.growth_dimensions),
                    "hidden_staff_signals": "; ".join(ev.hidden_staff_signals),
                    "specialist_tenure_classification": ev.specialist_tenure_assessment.classification.value,
                    "primary_rejection_reason": ev.primary_rejection_reason or "",
                    "worth_applying": ev.worth_applying,
                    "is_product_company": ev.is_product_company,
                    "model_name": ev.model_name,
                }
            )
