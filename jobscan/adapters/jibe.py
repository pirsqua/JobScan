"""iCIMS career-site adapter (the product formerly called Jibe).

Many iCIMS customers serve jobs through this career-site front end rather than iCIMS's classic
portal (jobscan.adapters.icims), whose listing is then empty — GitHub, DocuSign, Schneider Electric.
The site's own JSON endpoint, ``GET https://{host}/api/jobs?page={n}&limit=100``, returns full job
records — description, responsibilities, qualifications, location, employment type — ``totalCount``
at a time, so no per-job detail fetch is needed. Confirmed live with no bot-gating on
www.github.careers, careers.docusign.com and careers.se.com.

``board_id`` is the site's host, optionally followed by the endpoint's own filters as a query
string, forwarded on every request — e.g. "careers.se.com?country=United States" (confirmed live:
3,667 worldwide postings -> 823 U.S.).

Remote work shows only in the location name ("US Remote"), so the location field carries it.
Pay is usually in the description text, which the salary parser reads downstream; the
salary_min_value/salary_max_value fields are used when a site fills them.
"""
from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from urllib.parse import parse_qsl

import httpx
from pydantic import ValidationError

from jobscan.adapters.base import AdapterError, SourceAdapter
from jobscan.adapters.schemas import JibeJobData, JibeJobsPage
from jobscan.models import AtsType, RawPosting, SalarySource

PAGE_SIZE = 100
MAX_PAGES = 200

_EMPLOYMENT_TYPES = {
    "FULL_TIME": "Full-time", "PART_TIME": "Part-time", "CONTRACTOR": "Contract", "TEMPORARY": "Temporary",
    "INTERN": "Internship",
}


class JibeAdapter(SourceAdapter):
    ats_type = AtsType.JIBE

    def fetch_postings(self, board_id: str) -> Iterable[RawPosting]:
        host, _, query = board_id.partition("?")
        filters = parse_qsl(query)
        postings: list[RawPosting] = []
        for page_number in range(1, MAX_PAGES + 1):
            try:
                params = [("page", str(page_number)), ("limit", str(PAGE_SIZE))] + filters
                response = self.client.get(f"https://{host}/api/jobs", params=params)
                response.raise_for_status()
                page = JibeJobsPage.model_validate(response.json())
            except httpx.HTTPError as exc:
                raise AdapterError(f"jibe board '{board_id}': request failed: {exc}") from exc
            except ValidationError as exc:  # before ValueError, which it subclasses
                raise AdapterError(f"jibe board '{board_id}': unexpected response shape: {exc}") from exc
            except ValueError as exc:
                raise AdapterError(f"jibe board '{board_id}': invalid JSON: {exc}") from exc
            postings.extend(self._parse_job(host, job.data) for job in page.jobs)
            if not page.jobs or len(postings) >= page.totalCount:
                return postings
        raise AdapterError(f"jibe board '{board_id}': more than {MAX_PAGES} pages — paging did not end")

    def _parse_job(self, host: str, job: JibeJobData) -> RawPosting:
        description_html = "".join(
            section for section in (job.description, job.responsibilities, job.qualifications) if section
        )
        salary_min, salary_max = _salary(job.salary_min_value), _salary(job.salary_max_value)
        salary_period = None
        if salary_min or salary_max:
            salary_period = "year" if (salary_max or salary_min) >= 1000 else "hour"
        return RawPosting(
            source_job_id=job.req_id,
            title=job.title.strip(),
            location_raw=_location_text(job),
            employment_type_raw=_EMPLOYMENT_TYPES.get(job.employment_type or "", job.employment_type),
            description_html=description_html or None,
            description_text=None,
            # Redirects to each site's canonical job page (/careers-home/jobs/... on some sites).
            posting_url=f"https://{host}/jobs/{job.req_id}?lang=en-us",
            apply_url=job.apply_url,
            department=job.department,
            published_at=_iso_date(job.posted_date),
            salary_min=salary_min,
            salary_max=salary_max,
            salary_currency="USD" if salary_period else None,
            salary_period=salary_period,
            salary_source=SalarySource.STRUCTURED if salary_period else SalarySource.NONE,
        )


def _location_text(job: JibeJobData) -> str | None:
    """Both location fields when they differ: the name carries "Remote" ("US Remote"), the full
    location the place ("United States")."""
    parts = [part for part in (job.location_name, job.full_location) if part]
    if len(parts) == 2 and parts[0] != parts[1]:
        return f"{parts[0]} ({parts[1]})"
    return parts[0] if parts else None


def _salary(value: str | float | None) -> float | None:
    try:
        amount = float(value) if value is not None else None
    except ValueError:  # "None" and other placeholders arrive as strings
        return None
    return amount if amount and amount > 0 else None


def _iso_date(raw: str | None) -> str | None:
    """"2026-10-12T22:32:00+0000" or "October 12, 2026" (sites differ) as ISO 8601."""
    if not raw:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%B %d, %Y"):
        try:
            return datetime.strptime(raw, fmt).isoformat()
        except ValueError:
            continue
    return None
