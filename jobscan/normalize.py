"""Turn a source adapter's RawPosting into a normalized JobPosting ready for persistence."""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone

from bs4 import BeautifulSoup

from jobscan.config import Settings
from jobscan.models import Company, JobPosting, JobStatus, RawPosting
from jobscan.parsing import normalize_employment_type, normalize_location, resolve_salary

_WHITESPACE_RE = re.compile(r"\s+")


def html_to_text(html: str | None) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator=" ")


def compute_description_hash(description_text: str) -> str:
    normalized = _WHITESPACE_RE.sub(" ", description_text).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _parse_published_at(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        cleaned = raw.replace("Z", "+00:00")
        return datetime.fromisoformat(cleaned)
    except ValueError:
        return None


def normalize_posting(
    raw: RawPosting,
    company: Company,
    settings: Settings,
    now: datetime | None = None,
) -> JobPosting:
    now = now or datetime.now(timezone.utc)

    description_text = raw.description_text or html_to_text(raw.description_html)
    description_text = description_text.strip()
    description_hash = compute_description_hash(description_text or raw.title)

    remote_scope, _note = normalize_location(
        raw.location_raw,
        description_text,
        raw.workplace_type,
        settings.candidate_state,
        settings.candidate_state_name,
    )
    employment_type = normalize_employment_type(raw.employment_type_raw)

    salary = resolve_salary(
        raw.salary_min, raw.salary_max, raw.salary_currency, raw.salary_period, description_text
    )

    assert company.id is not None
    return JobPosting(
        company_id=company.id,
        source=company.ats_type,
        source_job_id=raw.source_job_id,
        title=raw.title.strip(),
        location_raw=raw.location_raw,
        remote_scope=remote_scope,
        workplace_type=raw.workplace_type,
        employment_type_raw=raw.employment_type_raw,
        employment_type=employment_type,
        description_text=description_text,
        description_html=raw.description_html,
        description_hash=description_hash,
        posting_url=raw.posting_url,
        apply_url=raw.apply_url or raw.posting_url,
        department=raw.department,
        published_at=_parse_published_at(raw.published_at),
        first_seen_at=now,
        last_seen_at=now,
        salary_min=salary.salary_min,
        salary_max=salary.salary_max,
        salary_currency=salary.salary_currency,
        salary_period=salary.salary_period,
        salary_source=salary.salary_source,
        status=JobStatus.ACTIVE,
    )
