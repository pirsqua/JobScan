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
from typing import Any, Iterable

import httpx
from pydantic import ValidationError

from jobscan.adapters.base import AdapterError, SourceAdapter
from jobscan.adapters.schemas import RipplingJobPost, RipplingJobPostsPage
from jobscan.job_family import is_engineering_title
from jobscan.logging_setup import get_logger, log_extra
from jobscan.models import AtsType, RawPosting, SalarySource

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
        for brief in briefs:
            if not is_engineering_title(brief.name):
                continue
            posting = self._fetch_detail(data_base, brief.id, brief.name, board_id)
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

    def _fetch_detail(self, data_base: str, job_id: str, brief_name: str, board_id: str) -> RawPosting | None:
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
            return None

        raw_post = (payload.get("pageProps") or {}).get("apiData", {}).get("jobPost")
        if raw_post is None:
            logger.warning("rippling job detail missing jobPost", extra=log_extra(board_id=board_id, job_id=job_id))
            return None
        try:
            post = RipplingJobPost.model_validate(raw_post)
        except ValidationError as exc:
            logger.warning(
                "rippling job detail unexpected shape",
                extra=log_extra(board_id=board_id, job_id=job_id, error=str(exc)),
            )
            return None

        return RawPosting(
            source_job_id=post.uuid or job_id,
            title=(post.name or brief_name).strip(),
            location_raw="; ".join(post.workLocations) or None,
            employment_type_raw=(post.employmentType.id if post.employmentType else None),
            description_html="".join(post.description.values()) or None,
            description_text=None,
            posting_url=post.url,
            apply_url=post.url,
            department=post.department.name if post.department else None,
            published_at=post.createdOn,
            salary_source=SalarySource.NONE,
        )


def _find_job_posts_data(payload: Any) -> dict | None:
    queries = ((payload.get("pageProps") or {}).get("dehydratedState") or {}).get("queries") or []
    for query in queries:
        key = query.get("queryKey") or []
        if "job-posts" in key:
            return (query.get("state") or {}).get("data")
    return None
