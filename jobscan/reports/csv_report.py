"""Flat CSV export of every evaluated job — one row per posting."""
from __future__ import annotations

import csv
from pathlib import Path

from jobscan.reports.data import ReportData

FIELDS = [
    "company", "title", "verdict", "confidence", "salary_min", "salary_max", "salary_period",
    "salary_source", "location_raw", "remote_scope", "employment_type", "posting_url",
    "apply_url", "published_at", "first_seen_at", "required_matches", "required_gaps",
    "preferred_gaps", "minor_caveats", "primary_rejection_reason", "is_product_company",
    "model_name",
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
                    "verdict": ev.verdict.value,
                    "confidence": ev.confidence,
                    "salary_min": job.salary_min,
                    "salary_max": job.salary_max,
                    "salary_period": job.salary_period,
                    "salary_source": job.salary_source.value,
                    "location_raw": job.location_raw,
                    "remote_scope": job.remote_scope.value,
                    "employment_type": job.employment_type.value,
                    "posting_url": job.posting_url,
                    "apply_url": job.apply_url,
                    "published_at": job.published_at.isoformat() if job.published_at else "",
                    "first_seen_at": job.first_seen_at.isoformat() if job.first_seen_at else "",
                    "required_matches": "; ".join(ev.required_matches),
                    "required_gaps": "; ".join(ev.required_gaps),
                    "preferred_gaps": "; ".join(ev.preferred_gaps),
                    "minor_caveats": "; ".join(ev.minor_caveats),
                    "primary_rejection_reason": ev.primary_rejection_reason or "",
                    "is_product_company": ev.is_product_company,
                    "model_name": ev.model_name,
                }
            )
