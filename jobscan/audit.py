"""Audit report: every posting the pipeline filtered out (factually or by the LLM) and why, so
false negatives can be reviewed by hand rather than trusted blindly."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from jobscan.db import Database
from jobscan.models import Verdict


@dataclass
class AuditRow:
    company: str
    title: str
    posting_url: str | None
    stage: str  # "factual_filter" | "llm_error" | "llm_reject" | "llm_borderline"
    reason: str
    detail: str | None


def build_audit_rows(db: Database) -> list[AuditRow]:
    rows: list[AuditRow] = []

    for entry in db.all_filter_log_entries():
        job = db.get_job(entry["job_id"])
        if job is None:
            continue
        company = db.get_company(job.company_id)
        rows.append(
            AuditRow(
                company=company.name if company else "unknown",
                title=job.title,
                posting_url=job.posting_url,
                stage=entry["stage"],
                reason=entry["reason"],
                detail=entry["detail"],
            )
        )

    for job in db.get_active_jobs():
        evaluation = db.get_evaluation_for_job(job.id)
        if evaluation is None or evaluation.verdict not in (Verdict.REJECT, Verdict.BORDERLINE):
            continue
        company = db.get_company(job.company_id)
        rows.append(
            AuditRow(
                company=company.name if company else "unknown",
                title=job.title,
                posting_url=job.posting_url,
                stage="llm_reject" if evaluation.verdict == Verdict.REJECT else "llm_borderline",
                reason=evaluation.primary_rejection_reason or "(no primary rejection reason given)",
                detail=json.dumps(
                    {
                        "scope_fit": evaluation.scope_fit.value,
                        "growth_dimensions": evaluation.growth_dimensions,
                        "required_gaps": evaluation.required_gaps,
                        "evidence": evaluation.evidence,
                    }
                ),
            )
        )

    return rows


def write_audit_csv(rows: list[AuditRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["company", "title", "posting_url", "stage", "reason", "detail"])
        for row in rows:
            writer.writerow([row.company, row.title, row.posting_url, row.stage, row.reason, row.detail])


def write_audit_markdown(rows: list[AuditRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Audit report: every filtered posting", "", f"Total filtered postings: {len(rows)}", ""]
    if not rows:
        lines.append("_Nothing has been filtered yet — run `crawl` and `evaluate` first._")
    else:
        lines.append("| Company | Title | Stage | Reason | Detail | Link |")
        lines.append("|---|---|---|---|---|---|")
        for row in rows:
            link = f"[posting]({row.posting_url})" if row.posting_url else ""
            detail = (row.detail or "").replace("|", "\\|").replace("\n", " ")
            if len(detail) > 200:
                detail = detail[:200] + "…"
            lines.append(f"| {row.company} | {row.title} | {row.stage} | {row.reason} | {detail} | {link} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
