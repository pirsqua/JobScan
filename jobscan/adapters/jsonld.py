"""Shared helpers for reading a schema.org JobPosting JSON-LD block — a de facto standard for
Google-for-Jobs indexing that multiple, unrelated ATS platforms embed verbatim on their detail
pages (confirmed live so far: Jobvite, JazzHR/ApplyToJob, iCIMS), not something specific to any one.
"""
from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from jobscan.adapters.schemas import JobPostingJsonLd
from jobscan.models import SalarySource

_JSON_LD_RE = re.compile(r'<script type="application/ld\+json">(.*?)</script>', re.DOTALL)


def extract_job_posting_json_ld(html: str) -> JobPostingJsonLd | None:
    for block in _JSON_LD_RE.findall(html):
        try:
            payload = json.loads(block, strict=False)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or payload.get("@type") != "JobPosting":
            continue
        try:
            return JobPostingJsonLd.model_validate(payload)
        except ValidationError:
            continue
    return None


def salary_fields(detail: JobPostingJsonLd) -> dict[str, Any]:
    """``RawPosting`` salary keyword arguments from the block's structured ``baseSalary`` — empty
    (leaving RawPosting's no-salary defaults) when the company doesn't publish one."""
    value = detail.baseSalary.value if detail.baseSalary else None
    if value is None:
        return {}
    salary_min, salary_max = to_float(value.minValue), to_float(value.maxValue)
    if not (salary_min or salary_max):
        return {}
    return {
        "salary_min": salary_min,
        "salary_max": salary_max,
        "salary_currency": detail.baseSalary.currency or None,
        "salary_period": "hour" if "HOUR" in (value.unitText or "").upper() else "year",
        "salary_source": SalarySource.STRUCTURED,
    }


def to_float(value: str | float | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
