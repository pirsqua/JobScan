from __future__ import annotations

import pytest

from jobscan.models import EmploymentType, RemoteScope, SalarySource, WorkplaceType
from jobscan.parsing import (
    annualize,
    normalize_employment_type,
    normalize_location,
    parse_salary_from_text,
    parse_workplace_type,
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

    def test_between_and_range(self):
        # Observed live (Yelp): only the low end was read, so a $112K-$269K band looked like a
        # flat $112K offer and was wrongly rejected as below the salary floor.
        result = parse_salary_from_text(
            "we expect the compensation range for this role to be between $112,000 and $269,000."
        )
        assert (result.salary_min, result.salary_max) == (112000, 269000)

    def test_and_without_between_is_not_a_range(self):
        result = parse_salary_from_text("Base salary is $120,000 and a $40,000 signing bonus.")
        assert (result.salary_min, result.salary_max) == (120000, 120000)

    def test_currency_code_before_each_bound(self):
        # Observed live (RealPage): "USD" between the dash and the second "$" hid the maximum.
        result = parse_salary_from_text("Pay Range USD $125,700.00 - USD $213,900.00 /Yr.")
        assert (result.salary_min, result.salary_max) == (125700, 213900)
        assert result.salary_currency == "USD"
        assert result.salary_period == "year"

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

    def test_distant_hourly_disclaimer_does_not_misclassify_a_real_annual_range(self):
        # Observed live (Sentry): a generic pay-transparency disclaimer mentioning "hourly" can
        # sit near-ish an actual annual range in the same sentence, causing the range to be
        # evaluated against hourly bounds ($15-$300) and discarded as implausible. Real text:
        text = (
            "This role focuses on cross-cutting projects that span multiple engineers and teams. "
            "The base salary range (or hourly wage range, if applicable) that the company "
            "reasonably expects to pay for this position is $155,000 to $400,000."
        )
        result = parse_salary_from_text(text)
        assert result.salary_min == 155000
        assert result.salary_max == 400000
        assert result.salary_period == "year"

    def test_hourly_indicator_immediately_adjacent_still_detected(self):
        # The fix must not lose genuine proximity-based hourly detection.
        text = "We raised $150,000,000 in funding. This contract role pays $85/hr - $100/hr."
        result = parse_salary_from_text(text)
        assert result.salary_period == "hour"
        assert result.salary_min == 85

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

    @pytest.mark.parametrize("location", ["Remote (U.S.)", "U.S. Remote", "Remote, U.S", "Remote - North America"])
    def test_us_spellings_with_periods_and_north_america_are_remote_us(self, location):
        # "Remote (U.S.)" used to fall through to UNKNOWN: a trailing \b after "U.S." never
        # matches before ")" or end-of-string.
        scope, _ = normalize_location(location, None, None)
        assert scope == RemoteScope.REMOTE_US

    def test_structured_remote_with_bare_remote_location(self):
        scope, _ = normalize_location("Remote", None, WorkplaceType.REMOTE)
        assert scope == RemoteScope.REMOTE_US

    def test_hybrid_explicit(self):
        scope, note = normalize_location("Hybrid - Seattle, WA", None, None)
        assert scope == RemoteScope.HYBRID
        assert note is not None

    def test_structured_hybrid_wins_over_office_location_and_remote_text(self):
        # Observed live (Plaid on Ashby): workplaceType "Hybrid", location "San Francisco HQ" —
        # the location string alone looks like an ambiguous hub city, the text may mention remote
        # in passing, but the source's own field settles it.
        scope, _ = normalize_location("San Francisco HQ", "Flexible, remote-friendly culture.", WorkplaceType.HYBRID)
        assert scope == RemoteScope.HYBRID

    def test_structured_onsite(self):
        scope, _ = normalize_location("San Francisco, CA", "We love remote tools.", WorkplaceType.ONSITE)
        assert scope == RemoteScope.ONSITE

    def test_no_remote_mention_anywhere_is_onsite(self):
        # Observed live (Fivetran, Greenhouse): a city location and a full description that never
        # mentions remote work — not a remote posting, so it shouldn't reach the LLM at all.
        scope, note = normalize_location(
            "Oakland, California, United States", "Build metadata services in Java and Python.", None
        )
        assert scope == RemoteScope.ONSITE
        assert "never mentions remote" in note

    def test_empty_posting_with_no_remote_signal_is_onsite(self):
        scope, _ = normalize_location("", None, None)
        assert scope == RemoteScope.ONSITE

    def test_city_location_with_remote_option_in_text_is_deferred(self):
        # Observed live (SeatGeek): "New York, New York" location, but the text offers "as many
        # days a week in the office as you'd like or 100% remotely" — genuine remote eligibility
        # only the LLM can confirm, so it must NOT be classified on-site from the city alone.
        scope, _ = normalize_location(
            "New York, New York", "Work as many days in the office as you'd like or 100% remotely.", None
        )
        assert scope == RemoteScope.UNKNOWN

    def test_state_explicitly_excluded(self):
        scope, note = normalize_location(
            "Remote (US)", "This role is remote, except for candidates based in Washington.", None
        )
        assert scope == RemoteScope.REMOTE_US_RESTRICTED
        assert "Washington" in note

    def test_remote_tied_to_ambiguous_us_hub_city_is_deferred_not_rejected(self):
        # A remote posting naming a specific U.S. hub city (common when a company lists its HQ
        # alongside nationwide remote eligibility) is a judgment call, not a safe rejection.
        scope, note = normalize_location("Remote - San Francisco Bay Area", None, WorkplaceType.REMOTE)
        assert scope == RemoteScope.UNKNOWN
        assert "ambiguous" in note

    def test_remote_tied_to_non_us_location_is_restricted(self):
        scope, note = normalize_location("Remote - Canada", None, WorkplaceType.REMOTE)
        assert scope == RemoteScope.REMOTE_US_RESTRICTED
        assert "non-U.S." in note

    def test_remote_tied_to_uk_city_is_restricted(self):
        scope, _ = normalize_location("London, UK (Remote)", None, WorkplaceType.REMOTE)
        assert scope == RemoteScope.REMOTE_US_RESTRICTED


class TestWorkplaceTypeParsing:
    @pytest.mark.parametrize("label, expected", [
        # Every vocabulary observed live across the registry's ATSs.
        ("Remote", WorkplaceType.REMOTE), ("Hybrid", WorkplaceType.HYBRID), ("OnSite", WorkplaceType.ONSITE),  # Ashby
        ("remote", WorkplaceType.REMOTE), ("onsite", WorkplaceType.ONSITE), ("unspecified", None),  # Lever
        ("REMOTE", WorkplaceType.REMOTE), ("ON_SITE", WorkplaceType.ONSITE), ("HYBRID", WorkplaceType.HYBRID),  # Rippling
        ("Remote only", WorkplaceType.REMOTE), ("Hybrid (Remote/Office)", WorkplaceType.HYBRID),  # Avature
        ("Office/Site only", WorkplaceType.ONSITE),
        ("TELECOMMUTE", WorkplaceType.REMOTE),  # schema.org JSON-LD
        (None, None), ("", None),
    ])
    def test_labels(self, label, expected):
        assert parse_workplace_type(label) == expected


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
