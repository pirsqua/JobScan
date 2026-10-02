"""SmartRecruiters Posting API adapter.

Public, unauthenticated REST API: ``GET https://api.smartrecruiters.com/v1/companies/{company}/
postings`` lists postings (paginated via ``offset``/``limit``), and ``GET .../postings/{id}``
returns one posting's full detail — including, on many US postings, structured pay-transparency
data (``compensation: {min, max, currency, period}``) used directly where present.

Like Workday, the list endpoint only returns a title — full detail (and compensation) costs one
extra request per posting — so the job-family title gate runs here, before the detail fetch,
rather than only afterward in the crawl orchestrator.
"""
from __future__ import annotations

from typing import Iterable

import httpx
from pydantic import ValidationError

from jobscan.adapters.base import AdapterError, SourceAdapter
from jobscan.adapters.schemas import (
    SmartRecruitersPostingBrief,
    SmartRecruitersPostingDetail,
    SmartRecruitersPostingsListResponse,
)
from jobscan.job_family import is_engineering_title
from jobscan.logging_setup import get_logger, log_extra
from jobscan.models import AtsType, RawPosting, SalarySource

BASE_URL = "https://api.smartrecruiters.com/v1/companies/{company}/postings"
PAGE_SIZE = 100

logger = get_logger("adapters.smartrecruiters")


class SmartRecruitersAdapter(SourceAdapter):
    ats_type = AtsType.SMARTRECRUITERS

    def fetch_postings(self, board_id: str) -> Iterable[RawPosting]:
        list_url = BASE_URL.format(company=board_id)
        briefs: list[SmartRecruitersPostingBrief] = []
        offset = 0

        while True:
            try:
                response = self.client.get(list_url, params={"limit": PAGE_SIZE, "offset": offset})
                response.raise_for_status()
                payload = response.json()
            except httpx.HTTPError as exc:
                raise AdapterError(f"smartrecruiters board '{board_id}': request failed: {exc}") from exc
            except ValueError as exc:
                raise AdapterError(f"smartrecruiters board '{board_id}': invalid JSON: {exc}") from exc

            try:
                page = SmartRecruitersPostingsListResponse.model_validate(payload)
            except ValidationError as exc:
                raise AdapterError(f"smartrecruiters board '{board_id}': unexpected response shape: {exc}") from exc

            briefs.extend(page.content)
            if len(page.content) < PAGE_SIZE:
                break
            offset += PAGE_SIZE

        postings = []
        for brief in briefs:
            if not is_engineering_title(brief.name):
                continue
            posting = self._fetch_detail(list_url, brief, board_id)
            if posting is not None:
                postings.append(posting)
        return postings

    def _fetch_detail(
        self, list_url: str, brief: SmartRecruitersPostingBrief, board_id: str
    ) -> RawPosting | None:
        try:
            response = self.client.get(f"{list_url}/{brief.id}")
            response.raise_for_status()
            detail = SmartRecruitersPostingDetail.model_validate(response.json())
        except (httpx.HTTPError, ValueError, ValidationError) as exc:
            # One bad requisition shouldn't drop the rest of an otherwise-healthy board.
            logger.warning(
                "smartrecruiters job detail fetch failed",
                extra=log_extra(board_id=board_id, posting_id=brief.id, error=str(exc)),
            )
            return None

        sections = detail.jobAd.sections if detail.jobAd else None
        description_html = ""
        for section in (
            (sections.companyDescription if sections else None),
            (sections.jobDescription if sections else None),
            (sections.qualifications if sections else None),
            (sections.additionalInformation if sections else None),
        ):
            if section and section.text:
                description_html += f"<h4>{section.title or ''}</h4>{section.text}"

        location = detail.location
        employment_type = detail.typeOfEmployment
        department = detail.department or detail.function

        salary_min = salary_max = salary_currency = salary_period = None
        comp = detail.compensation
        if comp and (comp.min or comp.max):
            salary_min = comp.min
            salary_max = comp.max
            salary_currency = comp.currency
            salary_period = "hour" if "HOUR" in (comp.period or "").upper() else "year"

        return RawPosting(
            source_job_id=detail.id,
            title=(detail.name or brief.name).strip(),
            location_raw=location.fullLocation if location else None,
            employment_type_raw=employment_type.label if employment_type else None,
            description_html=description_html or None,
            description_text=None,
            posting_url=detail.postingUrl,
            apply_url=detail.applyUrl or detail.postingUrl,
            department=department.label if department else None,
            published_at=detail.releasedDate,
            salary_min=salary_min,
            salary_max=salary_max,
            salary_currency=salary_currency,
            salary_period=salary_period,
            salary_source=SalarySource.STRUCTURED if (salary_min or salary_max) else SalarySource.NONE,
            remote_flag=location.remote if location else None,
        )
