"""Safe, deterministic factual filters.

Deliberately small: each check is a fact the source data already states (a salary ceiling, an
explicit "hybrid" location, a raw employment-type field, a company already confirmed as a
staffing shop). Nothing here tries to judge qualification language — that ambiguity is always
routed to the LLM instead of guessed at with more regexes.

Whether a posting is a software-engineering role at all is decided earlier, at crawl time (see
jobscan.job_family) — a non-engineering posting never reaches this stage because it's never
persisted in the first place.
"""
from __future__ import annotations

from dataclasses import dataclass

from jobscan.models import Company, CompanyClassification, EmploymentType, JobPosting, RemoteScope
from jobscan.parsing import annualize


class FilterReason:
    NO_SALARY_PUBLISHED = "no_salary_published"
    SALARY_BELOW_MIN = "salary_below_min"
    NOT_REMOTE_EXPLICIT = "not_remote_explicit"
    STATE_EXCLUDED = "state_excluded"
    NOT_FULL_TIME = "not_full_time"
    CONSULTING_EMPLOYER = "consulting_employer"


@dataclass
class FilterResult:
    passed: bool
    reason: str | None = None
    detail: str | None = None


def apply_factual_filters(job: JobPosting, company: Company, min_base_salary: int) -> FilterResult:
    """Returns the first deterministic rejection reason, or passed=True if none apply."""

    if company.classification == CompanyClassification.CONSULTING:
        return FilterResult(
            False,
            FilterReason.CONSULTING_EMPLOYER,
            f"{company.name} is classified as consulting/staffing "
            f"(source={company.classification_source.value if company.classification_source else 'unknown'})",
        )

    if job.remote_scope in (RemoteScope.HYBRID, RemoteScope.ONSITE):
        return FilterResult(
            False,
            FilterReason.NOT_REMOTE_EXPLICIT,
            f"remote_scope={job.remote_scope.value}, location_raw={job.location_raw!r}",
        )

    if job.remote_scope == RemoteScope.REMOTE_US_RESTRICTED:
        return FilterResult(
            False,
            FilterReason.STATE_EXCLUDED,
            f"location_raw={job.location_raw!r} is not confirmed open to a Washington-based remote employee",
        )

    if job.employment_type in (EmploymentType.CONTRACT, EmploymentType.PART_TIME, EmploymentType.INTERN):
        return FilterResult(
            False,
            FilterReason.NOT_FULL_TIME,
            f"employment_type={job.employment_type.value} (raw={job.employment_type_raw!r})",
        )

    annual_min = annualize(job.salary_min, job.salary_period)
    annual_max = annualize(job.salary_max, job.salary_period)

    if annual_min is None and annual_max is None:
        return FilterResult(False, FilterReason.NO_SALARY_PUBLISHED, "no salary range found in structured fields or description text")

    ceiling = annual_max if annual_max is not None else annual_min
    if ceiling is not None and ceiling < min_base_salary:
        return FilterResult(
            False,
            FilterReason.SALARY_BELOW_MIN,
            f"published max ${ceiling:,.0f} is below the ${min_base_salary:,} minimum",
        )

    return FilterResult(True)
