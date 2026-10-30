"""Pinpoint adapter.

Public, unauthenticated feed: ``GET https://{company}.pinpointhq.com/postings.json`` returns every
open posting in one response — description, responsibilities, skills, benefits, workplace type
("remote"/"hybrid"/"onsite"), employment type, and structured pay when the company makes it
visible (``compensation_visible``). Confirmed live on Infor (``infor``), whose branded
careers.infor.com serves the same feed.

``board_id`` is the company's pinpointhq.com subdomain.
"""
from __future__ import annotations

from collections.abc import Iterable

import httpx
from pydantic import ValidationError

from jobscan.adapters.base import AdapterError, SourceAdapter
from jobscan.adapters.schemas import PinpointPosting, PinpointPostingsResponse
from jobscan.models import AtsType, RawPosting, SalarySource
from jobscan.parsing import parse_workplace_type

FEED_URL = "https://{company}.pinpointhq.com/postings.json"


class PinpointAdapter(SourceAdapter):
    ats_type = AtsType.PINPOINT

    def fetch_postings(self, board_id: str) -> Iterable[RawPosting]:
        try:
            response = self.client.get(FEED_URL.format(company=board_id))
            response.raise_for_status()
            feed = PinpointPostingsResponse.model_validate(response.json())
        except httpx.HTTPError as exc:
            raise AdapterError(f"pinpoint board '{board_id}': request failed: {exc}") from exc
        except ValidationError as exc:  # before ValueError, which it subclasses
            raise AdapterError(f"pinpoint board '{board_id}': unexpected response shape: {exc}") from exc
        except ValueError as exc:
            raise AdapterError(f"pinpoint board '{board_id}': invalid JSON: {exc}") from exc
        return [self._parse_posting(posting) for posting in feed.data]

    def _parse_posting(self, posting: PinpointPosting) -> RawPosting:
        description_html = "".join(
            section for section in (
                posting.description, posting.key_responsibilities, posting.skills_knowledge_expertise, posting.benefits,
            ) if section
        )
        salary_min = salary_max = salary_period = None
        if posting.compensation_visible and (posting.compensation_minimum or posting.compensation_maximum):
            salary_min, salary_max = posting.compensation_minimum, posting.compensation_maximum
            frequency = (posting.compensation_frequency or "year").lower()
            if "hour" in frequency:
                salary_period = "hour"
            elif "month" in frequency:
                # The salary parser's periods are year and hour only.
                salary_min = salary_min * 12 if salary_min else None
                salary_max = salary_max * 12 if salary_max else None
                salary_period = "year"
            else:
                salary_period = "year"

        location = posting.location
        place = ", ".join(dict.fromkeys(part for part in (
            location.city or location.name, location.province,
        ) if part)) if location else None
        department = posting.job.department.name if posting.job and posting.job.department else None
        return RawPosting(
            source_job_id=posting.id,
            title=posting.title.strip(),
            location_raw=place or None,
            employment_type_raw=posting.employment_type_text,
            description_html=description_html or None,
            description_text=None,
            posting_url=posting.url,
            apply_url=posting.url,
            department=department,
            salary_min=salary_min,
            salary_max=salary_max,
            salary_currency=posting.compensation_currency if salary_period else None,
            salary_period=salary_period,
            salary_source=SalarySource.STRUCTURED if salary_period else SalarySource.NONE,
            workplace_type=parse_workplace_type(posting.workplace_type),
        )
