"""Rippling ATS adapter (Rippling's built-in job board product, not to be confused with the
Rippling HR platform generally).

A board lives at ``https://ats.rippling.com/{slug}/jobs`` — a Next.js app. Its job list and job
detail data load from ``https://ats.rippling.com/_next/data/{buildId}/en-GB/{slug}/jobs...json``
— the same JSON Next.js uses client-side for page navigation, not a separate "real" API, but
genuinely public and unauthenticated (confirmed live: a plain scripted request with no browser,
no cookies, gets a normal 200 with real data). ``{buildId}`` changes on every Rippling deploy and
a stale one 404s, so it's read fresh each crawl from the ``__NEXT_DATA__`` script tag Next.js
embeds in the plain HTML board page — the framework's own standard way of telling the client
which build is live, not a scraping trick specific to this adapter.

Confirmed live (Netwrix): no bot-gating — no captcha, no block on a plain scripted request.

Like Workday, the list only returns a title — full detail costs one extra request per posting —
so the job-family title gate runs before the detail fetch.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

import httpx
from pydantic import ValidationError

from jobscan.adapters.base import AdapterError, SourceAdapter
from jobscan.adapters.schemas import (
    RipplingJobBrief,
    RipplingJobPost,
    RipplingJobPostsPage,
    RipplingPayRange,
)
from jobscan.job_family import is_engineering_title
from jobscan.logging_setup import get_logger, log_extra
from jobscan.models import AtsType, RawPosting, SalarySource, WorkplaceType
from jobscan.parsing import parse_workplace_type

BASE_URL = "https://ats.rippling.com"
_NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', re.DOTALL
)
MAX_PAGES = 200

logger = get_logger("adapters.rippling")


class RipplingAdapter(SourceAdapter):
    ats_type = AtsType.RIPPLING

    def fetch_postings(self, board_id: str) -> Iterable[RawPosting]:
        build_id = self._get_build_id(board_id)
        data_base = f"{BASE_URL}/_next/data/{build_id}/en-GB/{board_id}"

        briefs = []
        for page_num in range(MAX_PAGES):
            try:
                response = self.client.get(
                    f"{data_base}/jobs.json", params={"jobBoardSlug": board_id, "page": page_num}
                )
                response.raise_for_status()
                payload = response.json()
            except httpx.HTTPError as exc:
                raise AdapterError(f"rippling board '{board_id}': request failed: {exc}") from exc
            except ValueError as exc:
                raise AdapterError(f"rippling board '{board_id}': invalid JSON: {exc}") from exc

            raw_page = _find_job_posts_data(payload)
            if raw_page is None:
                raise AdapterError(f"rippling board '{board_id}': no job-posts data in response")
            try:
                parsed_page = RipplingJobPostsPage.model_validate(raw_page)
            except ValidationError as exc:
                raise AdapterError(f"rippling board '{board_id}': unexpected response shape: {exc}") from exc

            if not parsed_page.items:
                break
            briefs.extend(parsed_page.items)
            if page_num + 1 >= parsed_page.totalPages:
                break
        else:
            logger.warning(
                "rippling pagination hit MAX_PAGES without reaching totalPages",
                extra=log_extra(board_id=board_id, fetched=len(briefs)),
            )

        postings = []
        seen: set[str] = set()
        for brief in briefs:
            # Observed live (InRule): the list repeats a posting once per work location.
            if brief.id in seen or not is_engineering_title(brief.name):
                continue
            seen.add(brief.id)
            posting = self._fetch_detail(data_base, brief, board_id)
            if posting is not None:
                postings.append(posting)
        return postings

    def _get_build_id(self, board_id: str) -> str:
        try:
            response = self.client.get(f"{BASE_URL}/{board_id}/jobs")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise AdapterError(f"rippling board '{board_id}': request failed: {exc}") from exc

        match = _NEXT_DATA_RE.search(response.text)
        if not match:
            raise AdapterError(f"rippling board '{board_id}': __NEXT_DATA__ not found on board page")
        try:
            next_data = json.loads(match.group(1))
        except ValueError as exc:
            raise AdapterError(f"rippling board '{board_id}': invalid __NEXT_DATA__ JSON: {exc}") from exc

        build_id = next_data.get("buildId")
        if not build_id:
            raise AdapterError(f"rippling board '{board_id}': __NEXT_DATA__ has no buildId")
        return build_id

    def _fetch_detail(self, data_base: str, brief: RipplingJobBrief, board_id: str) -> RawPosting | None:
        job_id = brief.id
        try:
            response = self.client.get(
                f"{data_base}/jobs/{job_id}.json", params={"jobBoardSlug": board_id, "jobId": job_id}
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning(
                "rippling job detail fetch failed", extra=log_extra(board_id=board_id, job_id=job_id, error=str(exc))
            )
            self.incomplete = True
            return None

        raw_post = (payload.get("pageProps") or {}).get("apiData", {}).get("jobPost")
        if raw_post is None:
            logger.warning("rippling job detail missing jobPost", extra=log_extra(board_id=board_id, job_id=job_id))
            self.incomplete = True
            return None
        try:
            post = RipplingJobPost.model_validate(raw_post)
        except ValidationError as exc:
            logger.warning(
                "rippling job detail unexpected shape",
                extra=log_extra(board_id=board_id, job_id=job_id, error=str(exc)),
            )
            self.incomplete = True
            return None

        pay = _pay_range(post.payRangeDetails)
        return RawPosting(
            source_job_id=post.uuid or job_id,
            title=(post.name or brief.name).strip(),
            location_raw="; ".join(post.workLocations) or None,
            employment_type_raw=(post.employmentType.id if post.employmentType else None),
            description_html="".join(post.description.values()) or None,
            description_text=None,
            posting_url=post.url,
            apply_url=post.url,
            department=post.department.name if post.department else None,
            published_at=post.createdOn,
            salary_min=pay[0] if pay else None,
            salary_max=pay[1] if pay else None,
            salary_currency=pay[2] if pay else None,
            salary_period=pay[3] if pay else None,
            salary_source=SalarySource.STRUCTURED if pay else SalarySource.NONE,
            workplace_type=_workplace_type(brief),
        )


def _pay_range(ranges: list[RipplingPayRange]) -> tuple[float | None, float | None, str, str] | None:
    """(min, max, currency, period) from Rippling's per-location pay bands, or None. The remote band
    is the one a remote hire gets, so it wins when there is one; among the rest, the highest-paying
    band — like the description parser, ambiguity resolves toward keeping a posting the salary
    filter would otherwise reject. Observed live: InRule publishes its bands only here, so without
    reading them every posting failed as "no salary published"."""
    usable = []
    for band in ranges:
        if band.rangeStart is None and band.rangeEnd is None:
            continue
        frequency = (band.frequency or "YEAR").upper()
        scale, period = {"YEAR": (1, "year"), "MONTH": (12, "year"), "HOUR": (1, "hour")}.get(frequency, (None, None))
        if scale is None:
            continue
        low = band.rangeStart * scale if band.rangeStart is not None else None
        high = band.rangeEnd * scale if band.rangeEnd is not None else None
        usable.append((band.isRemote, high if high is not None else low, (low, high, (band.currency or "USD").upper(), period)))
    if not usable:
        return None
    return max(usable, key=lambda item: (item[0], item[1] or 0))[2]


def _workplace_type(brief: RipplingJobBrief) -> WorkplaceType | None:
    """A posting can list several locations with different arrangements; any remote option makes
    the role remote-eligible (the location string then decides which country)."""
    types = {parse_workplace_type(loc.workplaceType) for loc in brief.locations} - {None}
    for candidate in (WorkplaceType.REMOTE, WorkplaceType.HYBRID, WorkplaceType.ONSITE):
        if candidate in types:
            return candidate
    return None


def _find_job_posts_data(payload: Any) -> dict | None:
    queries = ((payload.get("pageProps") or {}).get("dehydratedState") or {}).get("queries") or []
    for query in queries:
        key = query.get("queryKey") or []
        if "job-posts" in key:
            return (query.get("state") or {}).get("data")
    return None
