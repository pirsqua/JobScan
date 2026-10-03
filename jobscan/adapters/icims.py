"""iCIMS classic job-portal adapter.

Server-rendered HTML: ``https://{portal}.icims.com/jobs/search?ss=1&in_iframe=1&pr={page}`` lists
open postings (``pr`` is a 0-indexed page number; the footer reads "Page X of N"), each row
carrying the title, a location string — several locations pipe-joined, e.g. "US-Remote-Remote |
US-CA-San Francisco" — and a link to ``/jobs/{id}/{slug}/job``. ``in_iframe=1`` asks for the bare
listing the portal otherwise wraps in an iframe. Each detail page embeds a schema.org JobPosting
JSON-LD block (full description, employment type, ``jobLocationType: TELECOMMUTE`` on remote
roles), read with the shared ``jobscan.adapters.jsonld`` helper. Confirmed live with no
bot-gating on Yelp (``uscareers-yelp``) and RealPage (``careers-realpagepms``).

``board_id`` is the portal's subdomain, which varies per company and often per region
(``uscareers-yelp`` vs ``cancareers-yelp``) — point it at the U.S. portal. Some iCIMS customers'
classic portals list nothing at all (GitHub, DocuSign: their jobs are served through iCIMS's
newer career-site front end instead, which this adapter doesn't read).

As with Jobvite/JazzHR, the list page doesn't carry the full posting, so the job-family title
gate runs before the per-posting detail fetch.
"""
from __future__ import annotations

import re
from typing import Iterable, NamedTuple

import httpx
from bs4 import BeautifulSoup

from jobscan.adapters.base import AdapterError, SourceAdapter
from jobscan.adapters.jsonld import extract_job_posting_json_ld, salary_fields
from jobscan.job_family import is_engineering_title
from jobscan.logging_setup import get_logger, log_extra
from jobscan.models import AtsType, RawPosting
from jobscan.parsing import parse_workplace_type

BASE_URL = "https://{portal}.icims.com"
# Safety cap on pagination; a real portal reports its page count and the loop stops there.
MAX_PAGES = 50
_PAGE_COUNT_RE = re.compile(r"Page\s*\d+\s*of\s*(\d+)")
_JOB_ID_RE = re.compile(r"/jobs/(\d+)/")

logger = get_logger("adapters.icims")


class _Brief(NamedTuple):
    job_id: str
    title: str
    detail_url: str
    location: str | None


class ICIMSAdapter(SourceAdapter):
    ats_type = AtsType.ICIMS

    def fetch_postings(self, board_id: str) -> Iterable[RawPosting]:
        search_url = f"{BASE_URL.format(portal=board_id)}/jobs/search"
        briefs: dict[str, _Brief] = {}
        for page in range(MAX_PAGES):
            try:
                response = self.client.get(search_url, params={"ss": 1, "in_iframe": 1, "pr": page})
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise AdapterError(f"icims portal '{board_id}': request failed: {exc}") from exc

            page_briefs, page_count = _parse_list_page(response.text)
            new = [brief for brief in page_briefs if brief.job_id not in briefs]
            briefs.update((brief.job_id, brief) for brief in new)
            if not new or page_count is None or page + 1 >= page_count:
                break

        postings = []
        for brief in briefs.values():
            if not is_engineering_title(brief.title):
                continue
            posting = self._fetch_detail(brief, board_id)
            if posting is not None:
                postings.append(posting)
        return postings

    def _fetch_detail(self, brief: _Brief, board_id: str) -> RawPosting | None:
        try:
            response = self.client.get(brief.detail_url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning(
                "icims job detail fetch failed",
                extra=log_extra(board_id=board_id, detail_url=brief.detail_url, error=str(exc)),
            )
            return None

        detail = extract_job_posting_json_ld(response.text)
        if detail is None:
            logger.warning(
                "icims job detail missing a JobPosting JSON-LD block",
                extra=log_extra(board_id=board_id, detail_url=brief.detail_url),
            )
            return None

        # The listing's href asks for the bare iframe body; the canonical page drops that query.
        posting_url = brief.detail_url.split("?", 1)[0]
        return RawPosting(
            source_job_id=brief.job_id,
            title=(detail.title or brief.title).strip(),
            location_raw=brief.location,
            employment_type_raw=detail.employmentType,
            description_html=detail.description,
            description_text=None,
            posting_url=posting_url,
            apply_url=posting_url,
            published_at=detail.datePosted,
            workplace_type=parse_workplace_type(detail.jobLocationType),
            **salary_fields(detail),
        )


def _parse_list_page(html: str) -> tuple[list[_Brief], int | None]:
    soup = BeautifulSoup(html, "html.parser")
    briefs = []
    for row in soup.select("div.row"):
        link = row.select_one(".title a.iCIMS_Anchor")
        match = _JOB_ID_RE.search(link.get("href", "")) if link else None
        if match is None:
            continue
        heading = link.select_one("h3")
        location = row.select_one(".header.left span:not(.field-label)")
        briefs.append(
            _Brief(
                job_id=match.group(1),
                title=(heading or link).get_text(" ", strip=True),
                detail_url=link["href"],
                location=(location.get_text(" ", strip=True) or None) if location else None,
            )
        )
    count = _PAGE_COUNT_RE.search(soup.get_text(" "))
    return briefs, int(count.group(1)) if count else None
