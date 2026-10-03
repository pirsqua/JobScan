"""Jobvite public careers-site adapter.

Server-rendered HTML: ``https://jobs.jobvite.com/{company}`` lists every open posting grouped by
department in a single page (no pagination observed on the boards checked so far), each linking
to ``https://jobs.jobvite.com/{company}/job/{id}``. Both pages are plain HTML — no JS rendering
needed. The detail page embeds a standard schema.org ``JobPosting`` JSON-LD block (a de facto
standard many ATS platforms emit for Google-for-Jobs indexing, not Jobvite-specific) with the
full description, employment type, and — when the company publishes it — a base salary range.

Like Workday, the list page doesn't carry the full posting, so the job-family title gate runs
before the per-posting detail fetch rather than only afterward in the crawl orchestrator —
otherwise a board with a handful of engineering roles among many departments would cost one
wasted detail request per non-engineering posting.
"""
from __future__ import annotations

import re
from typing import Iterable

import httpx
from bs4 import BeautifulSoup

from jobscan.adapters.base import AdapterError, SourceAdapter
from jobscan.adapters.jsonld import extract_job_posting_json_ld, salary_fields
from jobscan.job_family import is_engineering_title
from jobscan.logging_setup import get_logger, log_extra
from jobscan.models import AtsType, RawPosting
from jobscan.parsing import parse_workplace_type

BASE_URL = "https://jobs.jobvite.com/{company}"
_WHITESPACE_RE = re.compile(r"\s+")

logger = get_logger("adapters.jobvite")


class JobviteAdapter(SourceAdapter):
    ats_type = AtsType.JOBVITE

    def fetch_postings(self, board_id: str) -> Iterable[RawPosting]:
        list_url = BASE_URL.format(company=board_id)
        try:
            response = self.client.get(list_url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise AdapterError(f"jobvite board '{board_id}': request failed: {exc}") from exc

        briefs = _parse_list_page(response.text, board_id)

        postings = []
        for brief_title, path, location_text in briefs:
            if not is_engineering_title(brief_title):
                continue
            posting = self._fetch_detail(path, brief_title, location_text, board_id)
            if posting is not None:
                postings.append(posting)
        return postings

    def _fetch_detail(
        self, path: str, brief_title: str, location_text: str | None, board_id: str
    ) -> RawPosting | None:
        url = f"https://jobs.jobvite.com{path}"
        try:
            response = self.client.get(url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning(
                "jobvite job detail fetch failed", extra=log_extra(board_id=board_id, path=path, error=str(exc))
            )
            return None

        detail = extract_job_posting_json_ld(response.text)
        if detail is None:
            logger.warning(
                "jobvite job detail missing a JobPosting JSON-LD block",
                extra=log_extra(board_id=board_id, path=path),
            )
            return None

        source_job_id = path.rstrip("/").rsplit("/", 1)[-1]
        return RawPosting(
            source_job_id=source_job_id,
            title=(detail.title or brief_title).strip(),
            location_raw=location_text,
            employment_type_raw=detail.employmentType,
            description_html=detail.description,
            description_text=None,
            posting_url=url,
            apply_url=f"{url}/apply",
            published_at=detail.datePosted,
            workplace_type=parse_workplace_type(detail.jobLocationType),
            **salary_fields(detail),
        )


def _parse_list_page(html: str, board_id: str) -> list[tuple[str, str, str | None]]:
    soup = BeautifulSoup(html, "html.parser")
    prefix = f"/{board_id}/job/"
    briefs: list[tuple[str, str, str | None]] = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if prefix not in href:
            continue
        title = link.get_text(strip=True)
        location_text = None
        row = link.find_parent("tr")
        if row is not None:
            location_cell = row.find("td", class_="jv-job-list-location")
            if location_cell is not None:
                # get_text(strip=True) only trims each text node's own ends, not the newlines and
                # indentation the source HTML leaves between nodes (e.g. "Stockholm,\n    Sweden")
                # — collapse everything to single spaces.
                raw_text = location_cell.get_text(" ", strip=True)
                location_text = _WHITESPACE_RE.sub(" ", raw_text).strip() or None
        briefs.append((title, href, location_text))
    return briefs
