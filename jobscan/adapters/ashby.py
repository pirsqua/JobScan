"""Ashby public Job Board API adapter.

Public, unauthenticated endpoint:
``https://api.ashbyhq.com/posting-api/job-board/{board_name}?includeCompensation=true``.
Ashby returns every open posting for a board in one response (no pagination parameters are
published for this endpoint), including a compensation summary string when the company has
opted in to publishing it.
"""
from __future__ import annotations

from typing import Iterable

import httpx
from pydantic import ValidationError

from jobscan.adapters.base import AdapterError, SourceAdapter
from jobscan.adapters.schemas import AshbyJob, AshbyJobBoardResponse
from jobscan.models import AtsType, RawPosting, SalarySource

BASE_URL = "https://api.ashbyhq.com/posting-api/job-board/{board}"


class AshbyAdapter(SourceAdapter):
    ats_type = AtsType.ASHBY

    def fetch_postings(self, board_id: str) -> Iterable[RawPosting]:
        url = BASE_URL.format(board=board_id)
        try:
            response = self.client.get(url, params={"includeCompensation": "true"})
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:
            raise AdapterError(f"ashby board '{board_id}': request failed: {exc}") from exc
        except ValueError as exc:
            raise AdapterError(f"ashby board '{board_id}': invalid JSON: {exc}") from exc

        try:
            parsed = AshbyJobBoardResponse.model_validate(payload)
        except ValidationError as exc:
            raise AdapterError(f"ashby board '{board_id}': unexpected response shape: {exc}") from exc

        return [self._parse_job(job) for job in parsed.jobs]

    def _parse_job(self, job: AshbyJob) -> RawPosting:
        salary_hint = None
        if job.compensation:
            salary_hint = (
                job.compensation.compensationTierSummary
                or job.compensation.scrapeableCompensationSalarySummary
            )

        description_html = job.descriptionHtml
        description_text = job.descriptionPlain
        if salary_hint:
            suffix = f"\nCompensation: {salary_hint}"
            if description_text:
                description_text += suffix
            elif description_html:
                description_html += f"<p>Compensation: {salary_hint}</p>"

        return RawPosting(
            source_job_id=job.id,
            title=job.title.strip(),
            location_raw=job.location,
            employment_type_raw=job.employmentType,
            description_html=description_html,
            description_text=description_text,
            posting_url=job.jobUrl,
            apply_url=job.applyUrl or job.jobUrl,
            department=job.department or job.team,
            published_at=job.publishedAt,
            salary_source=SalarySource.NONE,
            remote_flag=job.isRemote,
        )
