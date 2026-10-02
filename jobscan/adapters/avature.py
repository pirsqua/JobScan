"""Avature careers-portal adapter (server-rendered HTML — no JSON API).

Avature is a widely-used enterprise ATS. Several registry companies' Avature-hosted boards put
their job search behind active reCAPTCHA gating (confirmed live via network capture) — those
stay off-limits, consistent with this project's policy of never attempting to defeat deliberate
anti-bot measures. But not every Avature tenant does this: Siemens Digital Industries Software's
global board (hosted at jobs.siemens.com) serves the same job-search and job-detail pages as
plain, unauthenticated, non-JS HTML with no captcha anywhere in the flow — confirmed live by
submitting a real search and inspecting every request made. This adapter is for tenants like
that one; a future tenant that turns out to be captcha-gated will simply fail this adapter's
requests (plain 200s with no job markup, or a captcha challenge) and surface as a failed board.

List: ``GET https://{board_id}/SearchJobs/?listFilterMode=1&folderRecordsPerPage={PAGE_SIZE}&
folderOffset={offset}``. The server ignores any requested page size and always returns
``PAGE_SIZE`` results per page (confirmed live) — pagination is purely via ``folderOffset``,
ending at the first page with fewer than ``PAGE_SIZE`` results (confirmed live: no Workday-style
wraparound past the real end — the count just drops to 0).

Detail: ``GET https://{board_id}/JobDetail/{id}``. Two HTML sections: a structured label/value
metadata block (identified by its "Job ID" field, rather than by a CSS class that may not be
portal-specific) and one or more rich-text sections making up the actual description.

``board_id`` encodes "{host}/{locale}/{portal path}" (e.g. "jobs.siemens.com/en_US/externaljobs")
— unlike Workday, a single host+path prefix is enough to build every URL this adapter needs.

Siemens' global board alone has thousands of postings, and this tenant's server hard-codes
``PAGE_SIZE`` at 6 regardless of what's requested (confirmed live) — sequential one-at-a-time
pagination means hundreds of round trips. Both the list pagination and the per-posting detail
fetches are independent requests (each page/detail only needs its own offset/id, not any other
page's result), so both run through a small thread pool instead of one request at a time — the
same thing a browser does when it opens several connections to the same host concurrently, not
an attempt to defeat any rate limit.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Iterable

import httpx
from bs4 import BeautifulSoup

from jobscan.adapters.base import AdapterError, SourceAdapter
from jobscan.job_family import is_engineering_title
from jobscan.logging_setup import get_logger, log_extra
from jobscan.models import AtsType, RawPosting, SalarySource

PAGE_SIZE = 6
# Confirmed live (Siemens DISW): the real end of the board is a page with 0 results, not a
# wraparound back to page 1 — but MAX_PAGES is still a hard backstop in case some other tenant's
# board behaves like Workday's observed wraparound bug, so a bad signal can't hang the crawl.
MAX_PAGES = 1000
# How many list pages (or detail pages) are requested at once. Modest and polite — this is a
# public job-search page meant to serve many simultaneous browsing candidates, not a rate-limit
# bypass — but still enough to turn an hundreds-of-requests, one-at-a-time crawl (the real cost of
# this tenant's hard-coded page size of 6) into a few dozen rounds instead.
CONCURRENCY = 10

logger = get_logger("adapters.avature")


class AvatureAdapter(SourceAdapter):
    ats_type = AtsType.AVATURE

    def fetch_postings(self, board_id: str) -> Iterable[RawPosting]:
        base_url = f"https://{board_id}"
        list_url = f"{base_url}/SearchJobs/"

        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            briefs = self._fetch_all_briefs(pool, list_url, board_id)

            matched = [b for b in briefs if is_engineering_title(b[0])]
            results = pool.map(
                lambda brief: self._fetch_detail(base_url, brief[1], brief[0], brief[2], board_id), matched
            )
            return [posting for posting in results if posting is not None]

    def _fetch_all_briefs(
        self, pool: ThreadPoolExecutor, list_url: str, board_id: str
    ) -> list[tuple[str, str, str | None]]:
        briefs: list[tuple[str, str, str | None]] = []
        offset = 0
        for _ in range(0, MAX_PAGES, CONCURRENCY):
            batch_offsets = [offset + i * PAGE_SIZE for i in range(CONCURRENCY)]
            pages = list(pool.map(lambda o: self._fetch_list_page(list_url, o, board_id), batch_offsets))

            reached_end = False
            for page in pages:
                if not page:
                    reached_end = True
                    break
                briefs.extend(page)
                if len(page) < PAGE_SIZE:
                    reached_end = True
                    break
            if reached_end:
                return briefs
            offset += PAGE_SIZE * CONCURRENCY
        else:
            logger.warning(
                "avature pagination hit MAX_PAGES without reaching the end",
                extra=log_extra(board_id=board_id, fetched=len(briefs)),
            )
        return briefs

    def _fetch_list_page(self, list_url: str, offset: int, board_id: str) -> list[tuple[str, str, str | None]]:
        try:
            response = self.client.get(
                list_url,
                params={"listFilterMode": "1", "folderRecordsPerPage": PAGE_SIZE, "folderOffset": offset},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise AdapterError(f"avature board '{board_id}': request failed: {exc}") from exc
        return _parse_list_page(response.text)

    def _fetch_detail(
        self, base_url: str, detail_url: str, brief_title: str, brief_location: str | None, board_id: str
    ) -> RawPosting | None:
        try:
            response = self.client.get(detail_url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            # One bad requisition shouldn't drop the rest of an otherwise-healthy board.
            logger.warning(
                "avature job detail fetch failed",
                extra=log_extra(board_id=board_id, detail_url=detail_url, error=str(exc)),
            )
            return None

        fields, description_html = _parse_detail_page(response.text)
        job_id = fields.get("job id") or detail_url.rstrip("/").rsplit("/", 1)[-1]

        return RawPosting(
            source_job_id=job_id,
            title=brief_title.strip(),
            location_raw=fields.get("location(s)") or brief_location,
            employment_type_raw=fields.get("job type"),
            description_html=description_html or None,
            description_text=None,
            posting_url=detail_url,
            apply_url=f"{base_url}/ApplicationMethods?folderId={job_id}",
            department=fields.get("field of work"),
            published_at=_parse_posted_since(fields.get("posted since")),
            salary_source=SalarySource.NONE,
        )


def _parse_list_page(html: str) -> list[tuple[str, str, str | None]]:
    soup = BeautifulSoup(html, "html.parser")
    briefs: list[tuple[str, str, str | None]] = []
    for article in soup.select("article.article--result"):
        link = article.select_one(".article__header__text__title a")
        if link is None or not link.get("href"):
            continue
        title = link.get_text(strip=True)
        location_el = article.select_one(".list-item-location")
        location_text = location_el.get_text(strip=True) if location_el else None
        briefs.append((title, link["href"], location_text))
    return briefs


def _parse_detail_page(html: str) -> tuple[dict[str, str], str]:
    soup = BeautifulSoup(html, "html.parser")
    fields: dict[str, str] = {}
    description_parts: list[str] = []

    for article in soup.select("article.article--details"):
        field_divs = article.select(".article__content__view__field")
        labels = [
            label_el.get_text(strip=True).lower()
            for label_el in (d.select_one(".article__content__view__field__label") for d in field_divs)
            if label_el is not None
        ]
        if "job id" in labels:
            for div in field_divs:
                label_el = div.select_one(".article__content__view__field__label")
                value_el = div.select_one(".article__content__view__field__value")
                if label_el is None or value_el is None:
                    continue
                label = label_el.get_text(strip=True).lower()
                items = value_el.select("li")
                value = "; ".join(li.get_text(strip=True) for li in items) if items else value_el.get_text(" ", strip=True)
                if value:
                    fields[label] = value
        else:
            content = article.select_one(".article__content__view")
            if content is not None:
                description_parts.append(str(content))

    return fields, "".join(description_parts)


def _parse_posted_since(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%d-%b-%Y").date().isoformat()
    except ValueError:
        return None
