"""Full JSON export: run stats plus every evaluated job and its evaluation, for downstream
tooling or manual inspection."""
from __future__ import annotations

import json
from pathlib import Path

from jobscan.reports.data import ReportData


def _job_dict(row) -> dict:
    job, company, ev = row.job, row.company, row.evaluation
    d = {
        "company": company.name,
        "company_domain": company.domain,
        "title": job.title,
        "location_raw": job.location_raw,
        "remote_scope": job.remote_scope.value,
        "employment_type": job.employment_type.value,
        "salary_min": job.salary_min,
        "salary_max": job.salary_max,
        "salary_period": job.salary_period,
        "salary_source": job.salary_source.value,
        "posting_url": job.posting_url,
        "apply_url": job.apply_url,
        "published_at": job.published_at.isoformat() if job.published_at else None,
        "first_seen_at": job.first_seen_at.isoformat() if job.first_seen_at else None,
        "last_seen_at": job.last_seen_at.isoformat() if job.last_seen_at else None,
        "description_hash": job.description_hash,
    }
    if ev is not None:
        d["evaluation"] = {
            "verdict": ev.verdict.value,
            "confidence": ev.confidence,
            "scope_fit": ev.scope_fit.value,
            "evidence_coverage_percent": ev.evidence_coverage_percent,
            "specialist_tenure_assessment": {
                "classification": ev.specialist_tenure_assessment.classification.value,
                "specialty": ev.specialist_tenure_assessment.specialty,
                "explanation": ev.specialist_tenure_assessment.explanation,
            },
            "requirement_evidence": [
                {
                    "requirement": item.requirement,
                    "importance": item.importance.value,
                    "evidence_classification": item.evidence_classification.value,
                    "candidate_evidence": item.candidate_evidence,
                    "posting_evidence": item.posting_evidence,
                }
                for item in ev.requirement_evidence
            ],
            "growth_dimensions": ev.growth_dimensions,
            "hidden_staff_signals": ev.hidden_staff_signals,
            "compensation_assessment": ev.compensation_assessment,
            "remote_employment_verification": ev.remote_employment_verification,
            "required_matches": ev.required_matches,
            "required_gaps": ev.required_gaps,
            "preferred_only_gaps": ev.preferred_only_gaps,
            "minor_caveats": ev.minor_caveats,
            "evidence": ev.evidence,
            "credibility_assessment": ev.credibility_assessment,
            "why_this_is_or_is_not_gettable": ev.why_this_is_or_is_not_gettable,
            "is_product_company": ev.is_product_company,
            "primary_rejection_reason": ev.primary_rejection_reason,
            "model_name": ev.model_name,
        }
    else:
        d["evaluation"] = None
    return d


def build_json_report(data: ReportData) -> dict:
    return {
        "generated_at": data.generated_at,
        "crawl_stats": vars(data.crawl_stats) if data.crawl_stats else None,
        "evaluate_stats": {
            "jobs_considered": data.evaluate_stats.jobs_considered,
            "factual_rejected": data.evaluate_stats.factual_rejected,
            "companies_classified": data.evaluate_stats.companies_classified,
            "sent_to_llm": data.evaluate_stats.sent_to_llm,
            "cache_hits": data.evaluate_stats.cache_hits,
            "manual_overrides_applied": data.evaluate_stats.manual_overrides_applied,
            "verdict_counts": data.evaluate_stats.verdict_counts,
            "unverified": data.evaluate_stats.unverified,
            "input_tokens": data.evaluate_stats.input_tokens,
            "output_tokens": data.evaluate_stats.output_tokens,
        }
        if data.evaluate_stats
        else None,
        "estimated_cost_usd": data.estimated_cost_usd,
        "best_bets": [_job_dict(r) for r in data.best_bets],
        "growth_bets": [_job_dict(r) for r in data.growth_bets],
        "attractive_stretches": [_job_dict(r) for r in data.attractive_stretches],
        "all_evaluated": [_job_dict(r) for r in data.all_evaluated],
    }


def write_json_report(data: ReportData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_json_report(data), indent=2, default=str), encoding="utf-8")
