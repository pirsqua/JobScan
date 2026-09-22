"""Pydantic models for the raw JSON each ATS API returns.

These validate the external data at the boundary — before anything in this codebase reads a
field off it — so an unexpected shape (a renamed field, a missing list) surfaces as a clear
AdapterError instead of a KeyError/TypeError deep in normalization. ``extra="ignore"`` because
these are third-party APIs that add fields over time; we only assert the shape of what we read.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, TypeAdapter

# ---------------------------------------------------------------------------
# Greenhouse: GET https://boards-api.greenhouse.io/v1/boards/{board}/jobs?content=true
# ---------------------------------------------------------------------------


class GreenhouseLocation(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str | None = None


class GreenhouseDepartment(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str | None = None


class GreenhouseMetadataField(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str | None = None
    # Greenhouse custom-field values vary by field type (text, number, multi-select list, ...).
    # We only ever read fields whose *name* matches a salary/employment-type pattern and stringify
    # the value, so there's no real shape to validate here — Any is the honest type.
    value: Any = None


class GreenhouseJob(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: int
    title: str = ""
    location: GreenhouseLocation | None = None
    absolute_url: str | None = None
    content: str | None = None
    # Greenhouse sends an explicit JSON null (not just an omitted key) for these on some boards
    # when there's nothing to report — Optional, defaulting to None, handles both cases.
    metadata: list[GreenhouseMetadataField] | None = None
    departments: list[GreenhouseDepartment] | None = None
    updated_at: str | None = None
    first_published: str | None = None


class GreenhouseJobsResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    jobs: list[GreenhouseJob]


# ---------------------------------------------------------------------------
# Ashby: GET https://api.ashbyhq.com/posting-api/job-board/{board}?includeCompensation=true
# ---------------------------------------------------------------------------


class AshbyCompensation(BaseModel):
    model_config = ConfigDict(extra="ignore")
    compensationTierSummary: str | None = None
    scrapeableCompensationSalarySummary: str | None = None


class AshbyJob(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    title: str = ""
    location: str | None = None
    isRemote: bool | None = None
    employmentType: str | None = None
    descriptionHtml: str | None = None
    descriptionPlain: str | None = None
    jobUrl: str | None = None
    applyUrl: str | None = None
    department: str | None = None
    team: str | None = None
    publishedAt: str | None = None
    compensation: AshbyCompensation | None = None


class AshbyJobBoardResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    jobs: list[AshbyJob]


# ---------------------------------------------------------------------------
# Lever: GET https://api.lever.co/v0/postings/{company}?mode=json  (top-level JSON array)
# ---------------------------------------------------------------------------


class LeverCategories(BaseModel):
    model_config = ConfigDict(extra="ignore")
    location: str | None = None
    commitment: str | None = None
    team: str | None = None
    department: str | None = None


class LeverSalaryRange(BaseModel):
    model_config = ConfigDict(extra="ignore")
    min: float | None = None
    max: float | None = None
    currency: str | None = None
    interval: str | None = None


class LeverListSection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    text: str | None = None
    content: str | None = None


class LeverPosting(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    text: str = ""
    categories: LeverCategories | None = None
    workplaceType: str | None = None
    description: str | None = None
    descriptionPlain: str | None = None
    lists: list[LeverListSection] | None = None
    hostedUrl: str | None = None
    applyUrl: str | None = None
    salaryRange: LeverSalaryRange | None = None
    createdAt: float | None = None


LeverPostingsAdapter = TypeAdapter(list[LeverPosting])


# ---------------------------------------------------------------------------
# Workday: POST https://{tenant}.{cluster}.myworkdayjobs.com/wday/cxs/{tenant}/{site}/jobs
#          GET  https://{tenant}.{cluster}.myworkdayjobs.com/wday/cxs/{tenant}/{site}{externalPath}
# ---------------------------------------------------------------------------


class WorkdayJobBrief(BaseModel):
    model_config = ConfigDict(extra="ignore")
    title: str = ""
    externalPath: str | None = None
    locationsText: str | None = None


class WorkdayJobsListResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    total: int = 0
    jobPostings: list[WorkdayJobBrief] = []


class WorkdayJobPostingInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")
    title: str = ""
    jobDescription: str | None = None
    location: str | None = None
    timeType: str | None = None
    jobReqId: str | None = None
    externalUrl: str | None = None


class WorkdayJobDetailResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    jobPostingInfo: WorkdayJobPostingInfo


# ---------------------------------------------------------------------------
# Jobvite: GET https://jobs.jobvite.com/{company} (HTML list)
#          GET https://jobs.jobvite.com/{company}/job/{id} (HTML detail, with an embedded
#          schema.org JobPosting JSON-LD block carrying the full description/salary/type)
# ---------------------------------------------------------------------------


class JsonLdQuantitativeValue(BaseModel):
    model_config = ConfigDict(extra="ignore")
    # minValue/maxValue come back as an empty string, not an absent field, when a company hasn't
    # published a range — str | float so both the empty-string and populated cases validate.
    minValue: str | float | None = None
    maxValue: str | float | None = None
    unitText: str | None = None


class JsonLdBaseSalary(BaseModel):
    model_config = ConfigDict(extra="ignore")
    currency: str | None = None
    value: JsonLdQuantitativeValue | None = None


# ---------------------------------------------------------------------------
# Esri: POST https://esearchapi.esri.com/search
# ---------------------------------------------------------------------------


class EsriSearchHitDoc(BaseModel):
    model_config = ConfigDict(extra="ignore")
    displayurl: str | None = None
    # Includes a " Job | Esri Career Opportunity" SEO suffix; metaFields.JobTitle is the clean
    # title and is preferred where present.
    title: str = ""
    metaFields: dict[str, Any] = {}


class EsriSearchHit(BaseModel):
    model_config = ConfigDict(extra="ignore")
    doc: EsriSearchHitDoc


class EsriSearchInner(BaseModel):
    model_config = ConfigDict(extra="ignore")
    count: int = 0
    hits: list[EsriSearchHit] = []


class EsriSearchResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    search: EsriSearchInner


class JobPostingJsonLd(BaseModel):
    """schema.org JobPosting structured data embedded on each Jobvite detail page for SEO/
    Google-for-Jobs indexing — a de facto standard, not Jobvite-specific, but this is the only
    adapter that currently relies on it."""

    model_config = ConfigDict(extra="ignore")
    title: str = ""
    description: str | None = None
    datePosted: str | None = None
    employmentType: str | None = None
    baseSalary: JsonLdBaseSalary | None = None
