"""DirectEmployers career-microsite adapter.

DirectEmployers Association builds careers sites for its member employers (Alaska Airlines'
careers.alaskaair.com among them). Each site reads one shared search API,
``GET https://prod-search-api.jobsyn.org/api/v1/solr/search?page={n}``, which picks the employer
from the request's ``x-origin`` header — the site's own host — and returns full job records,
description included, ten per page with a ``has_more_pages`` flag. Confirmed live on
careers.alaskaair.com with no bot-gating.

``board_id`` is the microsite's host. ``https://{host}/{guid}/job/`` redirects to the job's
canonical page, so it serves as the posting URL.
"""
from __future__ import annotations

from typing import Iterable

import httpx
from pydantic import ValidationError

from jobscan.adapters.base import AdapterError, SourceAdapter
from jobscan.adapters.schemas import DirectEmployersJob, DirectEmployersSearchResponse
from jobscan.models import AtsType, RawPosting

SEARCH_URL = "https://prod-search-api.jobsyn.org/api/v1/solr/search"
MAX_PAGES = 500


class DirectEmployersAdapter(SourceAdapter):
    ats_type = AtsType.DIRECTEMPLOYERS

    def fetch_postings(self, board_id: str) -> Iterable[RawPosting]:
        postings: list[RawPosting] = []
        for page_number in range(1, MAX_PAGES + 1):
            try:
                response = self.client.get(SEARCH_URL, params={"page": page_number}, headers={"x-origin": board_id})
                response.raise_for_status()
                page = DirectEmployersSearchResponse.model_validate(response.json())
            except httpx.HTTPError as exc:
                raise AdapterError(f"directemployers board '{board_id}': request failed: {exc}") from exc
            except ValidationError as exc:  # before ValueError, which it subclasses
                raise AdapterError(f"directemployers board '{board_id}': unexpected response shape: {exc}") from exc
            except ValueError as exc:
                raise AdapterError(f"directemployers board '{board_id}': invalid JSON: {exc}") from exc
            postings.extend(self._parse_job(board_id, job) for job in page.jobs)
            if not page.pagination.has_more_pages or not page.jobs:
                return postings
        raise AdapterError(f"directemployers board '{board_id}': more than {MAX_PAGES} pages — paging did not end")

    def _parse_job(self, host: str, job: DirectEmployersJob) -> RawPosting:
        posting_url = f"https://{host}/{job.guid}/job/"
        return RawPosting(
            source_job_id=job.guid,
            title=job.title_exact.strip(),
            location_raw=job.location_exact,
            employment_type_raw=job.job_type,
            description_html=job.description,
            description_text=None,
            posting_url=posting_url,
            apply_url=posting_url,
            published_at=job.date_new,
        )
