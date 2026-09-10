"""Lever Postings API adapter.

Public, unauthenticated endpoint: ``https://api.lever.co/v0/postings/{company}?mode=json``.
The endpoint accepts ``skip``/``limit`` offset pagination; a single call is usually enough for a
typical company's board, but this adapter pages through it defensively (looping until a short
page comes back) so boards with large numbers of postings are still fetched completely.

Some Lever boards publish structured pay-transparency data via a ``salaryRange`` object; where
present it's used directly, otherwise the description text is handed to the regex-based parser
downstream.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import httpx
from pydantic import ValidationError

from jobscan.adapters.base import AdapterError, SourceAdapter
from jobscan.adapters.schemas import LeverCategories, LeverPosting, LeverPostingsAdapter
from jobscan.models import AtsType, RawPosting, SalarySource

BASE_URL = "https://api.lever.co/v0/postings/{company}"
PAGE_SIZE = 100


class LeverAdapter(SourceAdapter):
    ats_type = AtsType.LEVER

    def fetch_postings(self, board_id: str) -> Iterable[RawPosting]:
        url = BASE_URL.format(company=board_id)
        postings: list[RawPosting] = []
        skip = 0

        while True:
            try:
                response = self.client.get(
                    url, params={"mode": "json", "skip": skip, "limit": PAGE_SIZE}
                )
                response.raise_for_status()
                payload = response.json()
            except httpx.HTTPError as exc:
                raise AdapterError(f"lever board '{board_id}': request failed: {exc}") from exc
            except ValueError as exc:
                raise AdapterError(f"lever board '{board_id}': invalid JSON: {exc}") from exc

            try:
                page = LeverPostingsAdapter.validate_python(payload)
            except ValidationError as exc:
                raise AdapterError(f"lever board '{board_id}': unexpected response shape: {exc}") from exc

            postings.extend(self._parse_job(job) for job in page)

            if len(page) < PAGE_SIZE:
                break
            skip += PAGE_SIZE

        return postings

    def _parse_job(self, job: LeverPosting) -> RawPosting:
        categories = job.categories or LeverCategories()
        location = categories.location
        workplace_type = job.workplaceType

        description_html = job.description or ""
        for section in job.lists or []:
            if section.content:
                description_html += f"<h4>{section.text or ''}</h4>{section.content}"

        salary_min = salary_max = salary_currency = salary_period = None
        if job.salaryRange and (job.salaryRange.min or job.salaryRange.max):
            salary_min = job.salaryRange.min
            salary_max = job.salaryRange.max
            salary_currency = job.salaryRange.currency
            interval = job.salaryRange.interval or ""
            salary_period = "hour" if "hour" in interval else "year"

        remote_flag = workplace_type.lower() == "remote" if workplace_type else None

        published_at = None
        if job.createdAt is not None:
            published_at = datetime.fromtimestamp(job.createdAt / 1000, tz=timezone.utc).isoformat()

        return RawPosting(
            source_job_id=job.id,
            title=job.text.strip(),
            location_raw=location or workplace_type,
            employment_type_raw=categories.commitment,
            description_html=description_html,
            description_text=job.descriptionPlain,
            posting_url=job.hostedUrl,
            apply_url=job.applyUrl or job.hostedUrl,
            department=categories.team or categories.department,
            published_at=published_at,
            salary_min=salary_min,
            salary_max=salary_max,
            salary_currency=salary_currency,
            salary_period=salary_period,
            salary_source=SalarySource.STRUCTURED if (salary_min or salary_max) else SalarySource.NONE,
            remote_flag=remote_flag,
        )
