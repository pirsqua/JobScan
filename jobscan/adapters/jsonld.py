"""Shared helpers for reading a schema.org JobPosting JSON-LD block — a de facto standard for
Google-for-Jobs indexing that multiple, unrelated ATS platforms embed verbatim on their detail
pages (confirmed live so far: Jobvite, JazzHR/ApplyToJob), not something specific to either one.
"""
from __future__ import annotations

import json
import re

from pydantic import ValidationError

from jobscan.adapters.schemas import JobPostingJsonLd

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


def to_float(value: str | float | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
