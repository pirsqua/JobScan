"""Workday CXS (Candidate Experience Search) adapter.

Public, unauthenticated JSON API used by nearly every Workday-hosted careers site:
``POST https://{tenant}.{cluster}.myworkdayjobs.com/wday/cxs/{tenant}/{site}/jobs`` lists
postings (paginated, 20 per page — larger limits are rejected with HTTP 400), and
``GET .../wday/cxs/{tenant}/{site}{externalPath}`` returns one posting's full detail
(description, location, employment type). {tenant}, {cluster} (e.g. "wd1", "wd5", "wd12"), and
{site} vary per company and aren't derivable from the domain name, so a registry entry encodes
all three in board_id as "{tenant}/{cluster}/{site}".

Unlike Greenhouse/Ashby/Lever (one call returns every full posting), the list endpoint here
returns only a title and a path — full detail costs one extra request per posting. Companies'
Workday boards typically span every department, so the job-family title gate is applied here,
before the detail fetch, rather than only afterward in the crawl orchestrator — otherwise a board
with a handful of engineering roles among hundreds of postings would cost hundreds of unnecessary
detail requests. The remaining detail fetches are independent of each other and run through a
small thread pool.

board_id may also carry the tenant's own search facets as a query string, sent as the list
request's ``appliedFacets`` so the server does the narrowing — e.g.
"motorolasolutions/wd5/Careers?locationCountry=bc33aa3152ec42d4995f4791a106ed09". The facet
*name* varies by tenant ("locationCountry", "Country", ...; each list response's ``facets`` array
shows which exist), but the United States value id above is Workday-global — confirmed identical
across six registry tenants.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Iterable
from urllib.parse import parse_qsl

import httpx
from pydantic import ValidationError

from jobscan.adapters.base import AdapterError, SourceAdapter
from jobscan.adapters.schemas import WorkdayJobBrief, WorkdayJobDetailResponse, WorkdayJobsListResponse
from jobscan.job_family import is_engineering_title
from jobscan.logging_setup import get_logger, log_extra
from jobscan.models import AtsType, RawPosting, SalarySource

PAGE_SIZE = 20
# Observed live (Motorola Solutions): past the real end of results, this API doesn't return a
# short/empty page to signal "done" — it silently wraps around and re-serves page 1 forever, with
# `total` still reported normally. A pure "stop on a short page" loop never terminates against
# that. MAX_PAGES is a hard backstop in case `total` itself is ever wrong for some board — chosen
# well above any registry company's actual board size (the largest observed so far is ~900).
MAX_PAGES = 300
DETAIL_CONCURRENCY = 8

logger = get_logger("adapters.workday")


class WorkdayAdapter(SourceAdapter):
    ats_type = AtsType.WORKDAY

    def fetch_postings(self, board_id: str) -> Iterable[RawPosting]:
        tenant, cluster, site, facets = _parse_board_id(board_id)
        list_url = f"https://{tenant}.{cluster}.myworkdayjobs.com/wday/cxs/{tenant}/{site}/jobs"
        detail_base = f"https://{tenant}.{cluster}.myworkdayjobs.com/wday/cxs/{tenant}/{site}"

        briefs: list[WorkdayJobBrief] = []
        offset = 0
        total: int | None = None
        for _ in range(MAX_PAGES):
            try:
                response = self.client.post(
                    list_url,
                    json={"appliedFacets": facets, "limit": PAGE_SIZE, "offset": offset, "searchText": ""},
                )
                response.raise_for_status()
                payload = response.json()
            except httpx.HTTPError as exc:
                raise AdapterError(f"workday board '{board_id}': request failed: {exc}") from exc
            except ValueError as exc:
                raise AdapterError(f"workday board '{board_id}': invalid JSON: {exc}") from exc

            try:
                page = WorkdayJobsListResponse.model_validate(payload)
            except ValidationError as exc:
                raise AdapterError(f"workday board '{board_id}': unexpected response shape: {exc}") from exc

            if not page.jobPostings:
                break
            briefs.extend(page.jobPostings)
            # Observed live: `total` can be inconsistently 0 on an otherwise-normal page (a
            # flaky read, not the real count) — only lock it in once a positive value shows up,
            # so a bad early read doesn't cause the loop to stop before it's actually done.
            if total is None and page.total > 0:
                total = page.total
            offset += PAGE_SIZE
            if len(page.jobPostings) < PAGE_SIZE or (total is not None and offset >= total):
                break
        else:
            logger.warning(
                "workday pagination hit MAX_PAGES without reaching total — stopping early",
                extra=log_extra(board_id=board_id, fetched=len(briefs), reported_total=total),
            )

        matched = [b for b in briefs if b.externalPath and is_engineering_title(b.title)]
        with ThreadPoolExecutor(max_workers=DETAIL_CONCURRENCY) as pool:
            results = pool.map(lambda brief: self._fetch_detail(detail_base, brief, board_id), matched)
            return [posting for posting in results if posting is not None]

    def _fetch_detail(self, detail_base: str, brief: WorkdayJobBrief, board_id: str) -> RawPosting | None:
        url = f"{detail_base}{brief.externalPath}"
        try:
            response = self.client.get(url)
            response.raise_for_status()
            detail = WorkdayJobDetailResponse.model_validate(response.json())
        except (httpx.HTTPError, ValueError, ValidationError) as exc:
            # One bad requisition (pulled mid-crawl, malformed detail page, ...) shouldn't drop
            # the rest of an otherwise-healthy board — log it and move on.
            logger.warning(
                "workday job detail fetch failed",
                extra=log_extra(board_id=board_id, external_path=brief.externalPath, error=str(exc)),
            )
            return None

        info = detail.jobPostingInfo
        return RawPosting(
            source_job_id=info.jobReqId or brief.externalPath,
            title=(info.title or brief.title).strip(),
            location_raw=info.location or brief.locationsText,
            employment_type_raw=info.timeType,
            description_html=info.jobDescription,
            description_text=None,
            posting_url=info.externalUrl,
            apply_url=info.externalUrl,
            salary_source=SalarySource.NONE,
        )


def _parse_board_id(board_id: str) -> tuple[str, str, str, dict[str, list[str]]]:
    path, _, query = board_id.partition("?")
    parts = path.split("/")
    if len(parts) != 3 or not all(parts):
        raise AdapterError(
            f"workday board_id '{board_id}' must be '<tenant>/<cluster>/<site>' (e.g. 'acme/wd1/Acme')"
        )
    facets: dict[str, list[str]] = {}
    for name, value in parse_qsl(query):
        facets.setdefault(name, []).append(value)
    return parts[0], parts[1], parts[2], facets
