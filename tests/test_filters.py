from __future__ import annotations

from datetime import datetime, timezone

import pytest

from jobscan.filters import FilterReason, apply_factual_filters
from jobscan.models import (
    AtsType,
    ClassificationSource,
    Company,
    CompanyClassification,
    EmploymentType,
    JobPosting,
    JobStatus,
    RemoteScope,
    SalarySource,
)

NOW = datetime.now(timezone.utc)


def make_company(**overrides) -> Company:
    defaults = dict(
        id=1,
        name="Acme Corp",
        domain="acme.example.com",
        careers_url="https://acme.example.com/careers",
        ats_type=AtsType.GREENHOUSE,
        board_id="acme",
    )
    defaults.update(overrides)
    return Company(**defaults)


def make_job(**overrides) -> JobPosting:
    defaults = dict(
        id=1,
        company_id=1,
        source=AtsType.GREENHOUSE,
        source_job_id="123",
        title="Senior Backend Engineer",
        location_raw="Remote - US",
        remote_scope=RemoteScope.REMOTE_US,
        employment_type_raw="Full-time",
        employment_type=EmploymentType.FULL_TIME,
        description_text="We build a SaaS product for backend engineers.",
        description_html=None,
        description_hash="abc123",
        posting_url="https://acme.example.com/jobs/123",
        apply_url="https://acme.example.com/jobs/123/apply",
        department="Engineering",
        published_at=NOW,
        first_seen_at=NOW,
        last_seen_at=NOW,
        salary_min=180000,
        salary_max=220000,
        salary_currency="USD",
        salary_period="year",
        salary_source=SalarySource.STRUCTURED,
        status=JobStatus.ACTIVE,
    )
    defaults.update(overrides)
    return JobPosting(**defaults)


class TestFactualFilters:
    def test_healthy_posting_passes(self):
        result = apply_factual_filters(make_job(), make_company(), min_base_salary=170000)
        assert result.passed

    def test_salary_max_below_threshold_rejects(self):
        job = make_job(salary_min=140000, salary_max=160000)
        result = apply_factual_filters(job, make_company(), min_base_salary=170000)
        assert not result.passed
        assert result.reason == FilterReason.SALARY_BELOW_MIN

    def test_salary_max_at_threshold_passes(self):
        job = make_job(salary_min=140000, salary_max=170000)
        result = apply_factual_filters(job, make_company(), min_base_salary=170000)
        assert result.passed

    def test_no_salary_rejects(self):
        job = make_job(salary_min=None, salary_max=None)
        result = apply_factual_filters(job, make_company(), min_base_salary=170000)
        assert not result.passed
        assert result.reason == FilterReason.NO_SALARY_PUBLISHED

    def test_hybrid_rejects(self):
        job = make_job(remote_scope=RemoteScope.HYBRID)
        result = apply_factual_filters(job, make_company(), min_base_salary=170000)
        assert not result.passed
        assert result.reason == FilterReason.NOT_REMOTE_EXPLICIT

    def test_onsite_rejects(self):
        job = make_job(remote_scope=RemoteScope.ONSITE)
        result = apply_factual_filters(job, make_company(), min_base_salary=170000)
        assert not result.passed
        assert result.reason == FilterReason.NOT_REMOTE_EXPLICIT

    def test_state_restricted_rejects(self):
        job = make_job(remote_scope=RemoteScope.REMOTE_US_RESTRICTED)
        result = apply_factual_filters(job, make_company(), min_base_salary=170000)
        assert not result.passed
        assert result.reason == FilterReason.STATE_EXCLUDED

    def test_remote_us_unknown_scope_passes_through_to_llm(self):
        job = make_job(remote_scope=RemoteScope.UNKNOWN)
        result = apply_factual_filters(job, make_company(), min_base_salary=170000)
        assert result.passed

    def test_contract_rejects(self):
        job = make_job(employment_type=EmploymentType.CONTRACT, employment_type_raw="Contract")
        result = apply_factual_filters(job, make_company(), min_base_salary=170000)
        assert not result.passed
        assert result.reason == FilterReason.NOT_FULL_TIME

    def test_part_time_rejects(self):
        job = make_job(employment_type=EmploymentType.PART_TIME)
        result = apply_factual_filters(job, make_company(), min_base_salary=170000)
        assert not result.passed
        assert result.reason == FilterReason.NOT_FULL_TIME

    def test_confirmed_consulting_employer_rejects(self):
        company = make_company(
            classification=CompanyClassification.CONSULTING,
            classification_source=ClassificationSource.LLM,
        )
        result = apply_factual_filters(make_job(), company, min_base_salary=170000)
        assert not result.passed
        assert result.reason == FilterReason.CONSULTING_EMPLOYER

    def test_unknown_classification_does_not_reject(self):
        company = make_company(classification=CompanyClassification.UNKNOWN)
        result = apply_factual_filters(make_job(), company, min_base_salary=170000)
        assert result.passed

    def test_consulting_check_runs_before_salary_check(self):
        company = make_company(
            classification=CompanyClassification.CONSULTING,
            classification_source=ClassificationSource.MANUAL,
        )
        job = make_job(salary_min=None, salary_max=None)
        result = apply_factual_filters(job, company, min_base_salary=170000)
        assert result.reason == FilterReason.CONSULTING_EMPLOYER
