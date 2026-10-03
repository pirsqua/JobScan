"""Greenhouse Job Board API adapter.

Public, unauthenticated endpoint: ``https://boards-api.greenhouse.io/v1/boards/{board}/jobs``
with ``content=true`` returns every open posting (including the full HTML description) for a
board token in a single response — Greenhouse does not paginate this endpoint.

Salary is rarely a first-class field; state pay-transparency laws mean it is usually embedded in
the metadata (as a custom field) or in the description body itself, so both are surfaced to the
downstream text-based salary parser.
"""
from __future__ import annotations

import re
from typing import Iterable

import httpx
from pydantic import ValidationError

from jobscan.adapters.base import AdapterError, SourceAdapter
from jobscan.adapters.schemas import GreenhouseJob, GreenhouseJobsResponse
from jobscan.models import AtsType, RawPosting, SalarySource

BASE_URL = "https://boards-api.greenhouse.io/v1/boards/{board}/jobs"

_SALARY_FIELD_RE = re.compile(r"salary|compensation|pay\s*range", re.IGNORECASE)
_EMPLOYMENT_FIELD_RE = re.compile(r"employment\s*type|job\s*type", re.IGNORECASE)


class GreenhouseAdapter(SourceAdapter):
    ats_type = AtsType.GREENHOUSE

    def fetch_postings(self, board_id: str) -> Iterable[RawPosting]:
        url = BASE_URL.format(board=board_id)
        try:
            response = self.client.get(url, params={"content": "true"})
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:
            raise AdapterError(f"greenhouse board '{board_id}': request failed: {exc}") from exc
        except ValueError as exc:
            raise AdapterError(f"greenhouse board '{board_id}': invalid JSON: {exc}") from exc

        try:
            parsed = GreenhouseJobsResponse.model_validate(payload)
        except ValidationError as exc:
            raise AdapterError(f"greenhouse board '{board_id}': unexpected response shape: {exc}") from exc

        return [self._parse_job(job) for job in parsed.jobs]

    def _parse_job(self, job: GreenhouseJob) -> RawPosting:
        salary_hint = None
        employment_type_raw = None
        for field in job.metadata or []:
            name = field.name or ""
            if field.value is None:
                continue
            if _SALARY_FIELD_RE.search(name):
                salary_hint = str(field.value)
            elif _EMPLOYMENT_FIELD_RE.search(name):
                employment_type_raw = str(field.value)

        location = job.location.name if job.location else None
        department = job.departments[0].name if job.departments else None

        content = job.content or ""
        if salary_hint:
            content = f"{content}\n<p>{salary_hint}</p>"

        return RawPosting(
            source_job_id=str(job.id),
            title=job.title.strip(),
            location_raw=location,
            employment_type_raw=employment_type_raw,
            description_html=content,
            description_text=None,
            posting_url=job.absolute_url,
            apply_url=job.absolute_url,
            department=department,
            published_at=job.first_published or job.updated_at,
            salary_source=SalarySource.NONE,
        )
