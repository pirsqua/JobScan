"""JazzHR ("ApplyToJob") public careers-board adapter.

Server-rendered HTML: ``https://{company}.applytojob.com/apply/jobs/`` lists every open posting
in one table, grouped by department (no pagination observed on the one board checked so far,
mirroring Jobvite's experience), each linking to a detail page at a bare ``/apply/jobs/details/
{token}`` or ``/apply/{token}/{slug}`` style URL (the href on the list page is followed directly
rather than reconstructed, since JazzHR doesn't use a predictable token format). Both pages are
plain HTML — no JS rendering needed, confirmed live with no bot-gating.

The detail page embeds the same schema.org ``JobPosting`` JSON-LD block Jobvite's adapter reads
(a de facto standard, not specific to either platform) — full description, employment type, and,
when the company publishes it, a structured base salary range. Reuses the shared
``jobscan.adapters.jsonld`` helper rather than duplicating that extraction logic.

Like Jobvite, the list page doesn't carry the full posting, so the job-family title gate runs
before the per-posting detail fetch.
"""
from __future__ import annotations

from typing import Iterable
from urllib.parse import urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup

from jobscan.adapters.base import AdapterError, SourceAdapter
from jobscan.adapters.jsonld import extract_job_posting_json_ld, to_float
from jobscan.job_family import is_engineering_title
from jobscan.logging_setup import get_logger, log_extra
from jobscan.models import AtsType, RawPosting, SalarySource

BASE_URL = "https://{company}.applytojob.com"

logger = get_logger("adapters.jazzhr")


class JazzHRAdapter(SourceAdapter):
    ats_type = AtsType.JAZZHR

    def fetch_postings(self, board_id: str) -> Iterable[RawPosting]:
        base_url = BASE_URL.format(company=board_id)
        list_url = f"{base_url}/apply/jobs/"
        try:
            response = self.client.get(list_url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise AdapterError(f"jazzhr board '{board_id}': request failed: {exc}") from exc

        briefs = _parse_list_page(response.text, base_url)

        postings = []
        for brief_title, detail_url, location_text in briefs:
            if not is_engineering_title(brief_title):
                continue
            posting = self._fetch_detail(detail_url, brief_title, location_text, board_id)
            if posting is not None:
                postings.append(posting)
        return postings

    def _fetch_detail(
        self, detail_url: str, brief_title: str, location_text: str | None, board_id: str
    ) -> RawPosting | None:
        try:
            response = self.client.get(detail_url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning(
                "jazzhr job detail fetch failed",
                extra=log_extra(board_id=board_id, detail_url=detail_url, error=str(exc)),
            )
            return None

        detail = extract_job_posting_json_ld(response.text)
        if detail is None:
            logger.warning(
                "jazzhr job detail missing a JobPosting JSON-LD block",
                extra=log_extra(board_id=board_id, detail_url=detail_url),
            )
            return None

        salary_min = salary_max = salary_currency = salary_period = None
        salary_source = SalarySource.NONE
        if detail.baseSalary and detail.baseSalary.value:
            salary_min = to_float(detail.baseSalary.value.minValue)
            salary_max = to_float(detail.baseSalary.value.maxValue)
            if salary_min or salary_max:
                salary_currency = detail.baseSalary.currency or None
                unit = (detail.baseSalary.value.unitText or "").upper()
                salary_period = "hour" if "HOUR" in unit else "year"
                salary_source = SalarySource.STRUCTURED

        return RawPosting(
            source_job_id=urlsplit(detail_url).path.rstrip("/").rsplit("/", 1)[-1],
            title=(detail.title or brief_title).strip(),
            location_raw=location_text,
            employment_type_raw=detail.employmentType,
            description_html=detail.description,
            description_text=None,
            posting_url=detail_url,
            apply_url=detail_url,
            published_at=detail.datePosted,
            salary_min=salary_min,
            salary_max=salary_max,
            salary_currency=salary_currency,
            salary_period=salary_period,
            salary_source=salary_source,
        )


def _parse_list_page(html: str, base_url: str) -> list[tuple[str, str, str | None]]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("#jobs_table")
    if table is None:
        return []

    briefs: list[tuple[str, str, str | None]] = []
    for row in table.select("tr"):
        link = row.select_one("a.job_title_link")
        if link is None or not link.get("href"):
            continue
        title = link.get_text(strip=True)
        cells = row.find_all("td")
        location_text = cells[1].get_text(strip=True) if len(cells) > 1 else None
        briefs.append((title, urljoin(base_url, link["href"]), location_text or None))
    return briefs
