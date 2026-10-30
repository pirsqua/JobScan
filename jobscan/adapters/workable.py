"""Workable hosted job board adapter (``apply.workable.com/{account}``).

Public, unauthenticated JSON endpoints behind Workable's own hosted board:
``POST https://apply.workable.com/api/v3/accounts/{account}/jobs`` lists published jobs ten at a
time (the response's ``nextPage`` token, sent back as ``token``, fetches the next page), and
``GET https://apply.workable.com/api/v2/accounts/{account}/jobs/{shortcode}`` returns one job's
description, requirements, benefits, workplace type and — when the company publishes it —
structured salary (``salary_from``/``salary_to``/``salary_frequency``). Confirmed live on FMX
(account "fmx") with no bot-gating; an unknown account returns 404.

Like SmartRecruiters, the list carries only titles and locations, so the job-family title gate
runs before the per-job detail fetch, and those fetches run through a small thread pool.
"""
from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

import httpx
from pydantic import ValidationError

from jobscan.adapters.base import AdapterError, SourceAdapter
from jobscan.adapters.schemas import (
    WorkableJobDetail,
    WorkableJobsPage,
    WorkableJobSummary,
    WorkableLocation,
)
from jobscan.job_family import is_engineering_title
from jobscan.logging_setup import get_logger, log_extra
from jobscan.models import AtsType, RawPosting, SalarySource, WorkplaceType
from jobscan.parsing import parse_workplace_type

LIST_URL = "https://apply.workable.com/api/v3/accounts/{account}/jobs"
DETAIL_URL = "https://apply.workable.com/api/v2/accounts/{account}/jobs/{shortcode}"
POSTING_URL = "https://apply.workable.com/{account}/j/{shortcode}/"
# The hosted board's own list request; empty filters mean every published job.
LIST_BODY = {"query": "", "location": [], "department": [], "worktype": [], "remote": []}
MAX_PAGES = 100
DETAIL_CONCURRENCY = 8

# Workable's job "type" codes, spelled the way normalize_employment_type reads them.
_EMPLOYMENT_TYPES = {
    "full": "Full-time", "part": "Part-time", "contract": "Contract", "temporary": "Temporary",
    "internship": "Internship",
}

logger = get_logger("adapters.workable")


class WorkableAdapter(SourceAdapter):
    ats_type = AtsType.WORKABLE

    def fetch_postings(self, board_id: str) -> Iterable[RawPosting]:
        summaries: list[WorkableJobSummary] = []
        token: str | None = None
        for _ in range(MAX_PAGES):
            body = {**LIST_BODY, "token": token} if token else LIST_BODY
            try:
                response = self.client.post(LIST_URL.format(account=board_id), json=body)
                response.raise_for_status()
                page = WorkableJobsPage.model_validate(response.json())
            except httpx.HTTPError as exc:
                raise AdapterError(f"workable board '{board_id}': request failed: {exc}") from exc
            except ValidationError as exc:  # before ValueError, which it subclasses
                raise AdapterError(f"workable board '{board_id}': unexpected response shape: {exc}") from exc
            except ValueError as exc:
                raise AdapterError(f"workable board '{board_id}': invalid JSON: {exc}") from exc
            summaries.extend(page.results)
            token = page.nextPage
            if not token or not page.results:
                break
        else:
            raise AdapterError(f"workable board '{board_id}': more than {MAX_PAGES} pages — paging did not end")

        matched = [s for s in summaries if is_engineering_title(s.title)]
        with ThreadPoolExecutor(max_workers=DETAIL_CONCURRENCY) as pool:
            results = pool.map(lambda summary: self._fetch_detail(board_id, summary), matched)
            return [posting for posting in results if posting is not None]

    def _fetch_detail(self, account: str, summary: WorkableJobSummary) -> RawPosting | None:
        try:
            response = self.client.get(DETAIL_URL.format(account=account, shortcode=summary.shortcode))
            response.raise_for_status()
            detail = WorkableJobDetail.model_validate(response.json())
        except (httpx.HTTPError, ValueError, ValidationError) as exc:
            # One bad job shouldn't drop the rest of an otherwise-healthy board.
            logger.warning(
                "workable job detail fetch failed",
                extra=log_extra(board_id=account, posting_id=summary.shortcode, error=str(exc)),
            )
            self.incomplete = True
            return None

        description_html = "".join(
            section for section in (detail.description, detail.requirements, detail.benefits) if section
        )

        salary_min = salary_max = salary_period = None
        if detail.salary_from or detail.salary_to:
            salary_min, salary_max = detail.salary_from, detail.salary_to
            frequency = (detail.salary_frequency or "year").lower()
            if "hour" in frequency:
                salary_period = "hour"
            elif "month" in frequency:
                # The salary parser's periods are year and hour only.
                salary_min = salary_min * 12 if salary_min else None
                salary_max = salary_max * 12 if salary_max else None
                salary_period = "year"
            else:
                salary_period = "year"

        posting_url = POSTING_URL.format(account=account, shortcode=detail.shortcode)
        return RawPosting(
            source_job_id=detail.shortcode,
            title=(detail.title or summary.title).strip(),
            location_raw=_location_text(detail),
            employment_type_raw=_EMPLOYMENT_TYPES.get((detail.type or "").lower(), detail.type),
            description_html=description_html or None,
            description_text=None,
            posting_url=posting_url,
            apply_url=f"{posting_url}apply/",
            department=", ".join(detail.department) or None,
            published_at=detail.published,
            salary_min=salary_min,
            salary_max=salary_max,
            salary_currency=detail.salary_currency_iso_code if salary_period else None,
            salary_period=salary_period,
            salary_source=SalarySource.STRUCTURED if salary_period else SalarySource.NONE,
            workplace_type=parse_workplace_type(detail.workplace) or (WorkplaceType.REMOTE if detail.remote else None),
        )


def _location_text(detail: WorkableJobDetail) -> str | None:
    """Every visible location as "City, Region, Country", prefixed "Remote - " for a remote job so
    the location field states it the way other boards do."""
    places = [loc for loc in (detail.locations or ([detail.location] if detail.location else [])) if not loc.hidden]
    text = "; ".join(filter(None, (_place(loc) for loc in places))) or None
    if detail.workplace == "remote" or (detail.workplace is None and detail.remote):
        return f"Remote - {text}" if text else "Remote"
    return text


def _place(location: WorkableLocation) -> str:
    return ", ".join(part for part in (location.city, location.region, location.country) if part)
