"""Oracle Recruiting Cloud (Fusion HCM Candidate Experience) adapter.

The candidate-experience site reads two public REST resources on the tenant's Fusion host:
``recruitingCEJobRequisitions`` (``finder=findReqs;siteNumber={site},limit,offset`` — titles,
location and workplace type, ``TotalJobsCount`` overall) and ``recruitingCEJobRequisitionDetails``
(``finder=ById;Id="{id}",siteNumber={site}`` — the full description, responsibilities and
qualifications). Confirmed live on Akamai (host fa-extu-saasfaprod1.fa.ocs.oraclecloud.com, site
CX_1) with no bot-gating — a check on 2026-10-06 had found its branded jobs page blocked; on
2026-10-12 both the page and these resources answered plain requests.

``board_id`` is "{fusion host}/{site number}". The list only carries titles, so the job-family
title gate runs before the per-job detail fetch.
"""
from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

import httpx
from pydantic import ValidationError

from jobscan.adapters.base import AdapterError, SourceAdapter
from jobscan.adapters.schemas import (
    OracleDetailResponse,
    OracleRequisitionBrief,
    OracleSearchResponse,
)
from jobscan.job_family import is_engineering_title
from jobscan.logging_setup import get_logger, log_extra
from jobscan.models import AtsType, RawPosting
from jobscan.parsing import parse_workplace_type

API = "https://{host}/hcmRestApi/resources/latest"
PAGE_SIZE = 100
MAX_PAGES = 100
DETAIL_CONCURRENCY = 8

logger = get_logger("adapters.oracle")


class OracleAdapter(SourceAdapter):
    ats_type = AtsType.ORACLE

    def fetch_postings(self, board_id: str) -> Iterable[RawPosting]:
        host, _, site = board_id.partition("/")
        if not host or not site:
            raise AdapterError(f"oracle board '{board_id}': expected '{{fusion host}}/{{site number}}'")
        api = API.format(host=host)
        briefs: list[OracleRequisitionBrief] = []
        for page_number in range(MAX_PAGES):
            finder = f"findReqs;siteNumber={site},limit={PAGE_SIZE},offset={page_number * PAGE_SIZE},sortBy=POSTING_DATES_DESC"
            try:
                response = self.client.get(
                    f"{api}/recruitingCEJobRequisitions",
                    params={"onlyData": "true", "expand": "requisitionList", "finder": finder},
                )
                response.raise_for_status()
                result = OracleSearchResponse.model_validate(response.json())
            except httpx.HTTPError as exc:
                raise AdapterError(f"oracle board '{board_id}': request failed: {exc}") from exc
            except ValidationError as exc:  # before ValueError, which it subclasses
                raise AdapterError(f"oracle board '{board_id}': unexpected response shape: {exc}") from exc
            except ValueError as exc:
                raise AdapterError(f"oracle board '{board_id}': invalid JSON: {exc}") from exc
            page = result.items[0] if result.items else None
            if page is None or not page.requisitionList:
                break
            briefs.extend(page.requisitionList)
            if len(briefs) >= page.TotalJobsCount:
                break
        else:
            raise AdapterError(f"oracle board '{board_id}': more than {MAX_PAGES} pages — paging did not end")

        matched = [b for b in briefs if is_engineering_title(b.Title)]
        with ThreadPoolExecutor(max_workers=DETAIL_CONCURRENCY) as pool:
            results = pool.map(lambda brief: self._fetch_detail(api, host, site, brief, board_id), matched)
            return [posting for posting in results if posting is not None]

    def _fetch_detail(
        self, api: str, host: str, site: str, brief: OracleRequisitionBrief, board_id: str
    ) -> RawPosting | None:
        try:
            response = self.client.get(
                f"{api}/recruitingCEJobRequisitionDetails",
                params={"onlyData": "true", "finder": f'ById;Id="{brief.Id}",siteNumber={site}'},
            )
            response.raise_for_status()
            detail = OracleDetailResponse.model_validate(response.json()).items[0]
        except (httpx.HTTPError, ValueError, ValidationError, IndexError) as exc:
            logger.warning(
                "oracle job detail fetch failed",
                extra=log_extra(board_id=board_id, requisition_id=brief.Id, error=str(exc)),
            )
            self.incomplete = True
            return None

        description_html = "".join(
            section for section in (
                detail.ExternalDescriptionStr, detail.ExternalResponsibilitiesStr, detail.ExternalQualificationsStr,
            ) if section
        )
        posting_url = f"https://{host}/hcmUI/CandidateExperience/en/sites/{site}/job/{detail.Id}"
        return RawPosting(
            source_job_id=detail.Id,
            title=(detail.Title or brief.Title).strip(),
            location_raw=detail.PrimaryLocation,
            employment_type_raw=detail.JobSchedule or detail.JobType,
            description_html=description_html or None,
            description_text=None,
            posting_url=posting_url,
            apply_url=posting_url,
            department=detail.Department,
            published_at=detail.ExternalPostedStartDate,
            workplace_type=parse_workplace_type(detail.WorkplaceType),
        )
