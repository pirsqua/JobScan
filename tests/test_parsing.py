from __future__ import annotations

import pytest

from jobscan.models import EmploymentType, RemoteScope, SalarySource
from jobscan.parsing import (
    annualize,
    normalize_employment_type,
    normalize_location,
    parse_salary_from_text,
    resolve_salary,
)


class TestSalaryParsing:
    def test_dollar_range_with_commas(self):
        result = parse_salary_from_text("The base salary range for this role is $170,000 - $210,000 per year.")
        assert result.salary_min == 170000
        assert result.salary_max == 210000
        assert result.salary_period == "year"
        assert result.salary_source == SalarySource.DESCRIPTION

    def test_k_suffix_range(self):
        result = parse_salary_from_text("Compensation: $170K-$210K USD annually")
        assert result.salary_min == 170000
        assert result.salary_max == 210000
        assert result.salary_currency == "USD"

    def test_em_dash_and_to_variants(self):
        assert parse_salary_from_text("Salary range: $150,000 to $190,000").salary_min == 150000
        assert parse_salary_from_text("Pay range $150,000—$190,000").salary_max == 190000

    def test_hourly_range_detected(self):
        result = parse_salary_from_text("This contract role pays $85/hr - $100/hr")
        assert result.salary_period == "hour"
        assert result.salary_min == 85
        assert result.salary_max == 100

    def test_single_value_salary(self):
        result = parse_salary_from_text("Starting salary is $190,000 depending on experience.")
        assert result.salary_min == 190000
        assert result.salary_max == 190000

    def test_no_salary_present(self):
        result = parse_salary_from_text("We are looking for a great engineer to join our team.")
        assert result.salary_min is None
        assert result.salary_max is None
        assert result.salary_source == SalarySource.NONE

    def test_none_text(self):
        result = parse_salary_from_text(None)
        assert result.salary_source == SalarySource.NONE

    def test_prefers_salary_context_over_unrelated_dollar_amounts(self):
        text = (
            "We raised $150,000,000 in our Series C. "
            "The base salary range for this role is $175,000 - $205,000."
        )
        result = parse_salary_from_text(text)
        assert result.salary_min == 175000
        assert result.salary_max == 205000

    def test_ignores_implausible_amounts(self):
        # $1,500,000 is out of the plausible annual-salary band and should be skipped.
        result = parse_salary_from_text("Our company is valued at $1,500,000 - $2,500,000 today.")
        assert result.salary_min is None

    def test_structured_takes_precedence_over_text(self):
        result = resolve_salary(150000, 190000, "USD", "year", "This role pays $999,000 - $1,000,000 (typo in description).")
        assert result.salary_min == 150000
        assert result.salary_max == 190000
        assert result.salary_source == SalarySource.STRUCTURED

    def test_structured_single_value_fills_max(self):
        result = resolve_salary(180000, None, "USD", "year", None)
        assert result.salary_min == 180000
        assert result.salary_max == 180000


class TestAttainabilityRule:
    """The $170,000 rule: reject only when the published MAX is below the minimum; the range
    does not need to start at $170,000."""

    def test_max_below_threshold_fails(self):
        assert annualize(160000, "year") < 170000

    def test_max_at_threshold_passes(self):
        assert annualize(170000, "year") >= 170000

    def test_range_straddling_threshold_has_max_above(self):
        # e.g. $150,000 - $185,000: max is above 170k, so this should NOT be a safe rejection.
        assert annualize(185000, "year") >= 170000

    def test_hourly_annualizes_above_threshold(self):
        # $85/hr * 2080 = $176,800/yr
        assert annualize(85, "hour") == pytest.approx(176800)

    def test_hourly_annualizes_below_threshold(self):
        assert annualize(70, "hour") == pytest.approx(145600)

    def test_none_passes_through(self):
        assert annualize(None, "year") is None


class TestLocationParsing:
    def test_explicit_remote_us(self):
        scope, note = normalize_location("Remote - United States", None, None)
        assert scope == RemoteScope.REMOTE_US

    def test_remote_flag_true_with_bare_remote_location(self):
        scope, _ = normalize_location("Remote", None, True)
        assert scope == RemoteScope.REMOTE_US

    def test_hybrid_explicit(self):
        scope, note = normalize_location("Hybrid - Seattle, WA", None, None)
        assert scope == RemoteScope.HYBRID
        assert note is not None

    def test_onsite_specific_city_not_remote(self):
        scope, _ = normalize_location("San Francisco, CA", None, False)
        assert scope == RemoteScope.ONSITE

    def test_state_explicitly_excluded(self):
        scope, note = normalize_location(
            "Remote (US)", "This role is remote, except for candidates based in Washington.", None
        )
        assert scope == RemoteScope.REMOTE_US_RESTRICTED
        assert "Washington" in note

    def test_remote_tied_to_ambiguous_us_hub_city_is_deferred_not_rejected(self):
        # A remote posting naming a specific U.S. hub city (common when a company lists its HQ
        # alongside nationwide remote eligibility) is a judgment call, not a safe rejection.
        scope, note = normalize_location("Remote - San Francisco Bay Area", None, True)
        assert scope == RemoteScope.UNKNOWN
        assert "ambiguous" in note

    def test_remote_tied_to_non_us_location_is_restricted(self):
        scope, note = normalize_location("Remote - Canada", None, True)
        assert scope == RemoteScope.REMOTE_US_RESTRICTED
        assert "non-U.S." in note

    def test_remote_tied_to_uk_city_is_restricted(self):
        scope, _ = normalize_location("London, UK (Remote)", None, True)
        assert scope == RemoteScope.REMOTE_US_RESTRICTED

    def test_unknown_when_no_signal(self):
        scope, note = normalize_location("", None, None)
        assert scope == RemoteScope.UNKNOWN


class TestEmploymentTypeParsing:
    def test_full_time(self):
        assert normalize_employment_type("Full-time") == EmploymentType.FULL_TIME

    def test_contract(self):
        assert normalize_employment_type("Contract") == EmploymentType.CONTRACT

    def test_contractor_variant(self):
        assert normalize_employment_type("Contractor") == EmploymentType.CONTRACT

    def test_part_time(self):
        assert normalize_employment_type("Part-Time") == EmploymentType.PART_TIME

    def test_intern(self):
        assert normalize_employment_type("Internship") == EmploymentType.INTERN

    def test_unknown_when_blank(self):
        assert normalize_employment_type(None) == EmploymentType.UNKNOWN

    def test_unknown_when_unrecognized(self):
        assert normalize_employment_type("Freelance Gig") == EmploymentType.UNKNOWN
