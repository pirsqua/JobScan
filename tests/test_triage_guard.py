"""skip_is_substantiated: a triage skip is honoured only with an allowed category and the posting's
own words for it. Skills are the triage model's judgment; titles and location wording are checked
here. Most cases are the live misfires that shaped it."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from jobscan.llm.schemas import TriageResult
from jobscan.llm.triage import skip_is_substantiated
from jobscan.models import AtsType, EmploymentType, JobPosting, RemoteScope

NOW = datetime.now(timezone.utc)


def job(title="Senior Software Engineer", description="", location="Remote - US") -> JobPosting:
    return JobPosting(
        company_id=1, source=AtsType.GREENHOUSE, source_job_id="1", title=title, location_raw=location,
        remote_scope=RemoteScope.REMOTE_US, employment_type_raw="Full-time", employment_type=EmploymentType.FULL_TIME,
        description_text=description, description_html=None, description_hash="h", posting_url=None, apply_url=None,
        department=None, published_at=None, first_seen_at=NOW, last_seen_at=NOW,
    )


def skip(disqualifier: str, quote: str, hard: bool = True) -> TriageResult:
    return TriageResult(disqualifier_quote=quote, quote_is_hard_requirement=hard, disqualifier=disqualifier,
                        skip_full_evaluation=True, reason="r")


class TestQuoteMustBeInThePosting:
    def test_not_skipping_is_never_a_skip(self):
        result = TriageResult(disqualifier_quote="", quote_is_hard_requirement=False, disqualifier="none", skip_full_evaluation=False, reason="r")
        assert not skip_is_substantiated(result, job())

    def test_skip_without_a_category(self):
        result = TriageResult(disqualifier_quote="x", quote_is_hard_requirement=True, disqualifier="none", skip_full_evaluation=True, reason="r")
        assert not skip_is_substantiated(result, job(description="x"))

    def test_quote_not_in_the_posting(self):
        # Observed live (Samsara SWE II): "max $191,000 is well below the $170,000 minimum" — words
        # and arithmetic the posting never contained.
        posting = job(description="Base salary $113,645 - $191,000.")
        assert not skip_is_substantiated(skip("required_specialty", "salary well below the minimum"), posting)

    def test_quote_matching_tolerates_case_whitespace_and_typographic_punctuation(self):
        posting = job(description="You’ve  shipped\nproduction   machine learning systems — at scale.")
        assert skip_is_substantiated(skip("required_specialty", "you've shipped production machine learning systems - at scale"), posting)

    def test_quote_matching_tolerates_non_breaking_hyphens(self):
        posting = job(description="We are a Rust‑first team building the platform.")
        assert skip_is_substantiated(skip("required_unfamiliar_language", "a Rust-first team"), posting)


class TestSkillsAreTheTriageModelsCall:
    # Parsing skill sentences with patterns proved too ambiguous; with a real quote, a skills skip
    # stands or falls on the triage model's own reading.
    @pytest.mark.parametrize("disqualifier, quote", [
        ("required_unfamiliar_language", "5+ years of Ruby on Rails programming"),
        ("required_specialty", "2+ years shipping production LLM-backed features"),
    ])
    def test_skill_skip_with_a_real_quote_is_honoured(self, disqualifier, quote):
        assert skip_is_substantiated(skip(disqualifier, quote), job(description=f"Requirements: {quote}."))


class TestTitles:
    @pytest.mark.parametrize("title", ["Staff Software Engineer", "Principal Engineer, Platform", "Senior Staff Backend Engineer"])
    def test_staff_or_higher_title(self, title):
        assert skip_is_substantiated(skip("staff_or_higher_title", title), job(title=title))

    def test_staff_title_is_decided_by_the_title_whatever_line_is_quoted(self):
        # Observed live: 66 Staff-titled postings overruled because the model quoted the
        # description ("We're hiring a Staff Software Engineer") rather than the title field.
        posting = job(title="Staff Backend Engineer - Grafana Enterprise", description="Build observability.")
        assert skip_is_substantiated(skip("staff_or_higher_title", "We're hiring a Staff Software Engineer"), posting)

    def test_staff_quote_that_is_not_the_title(self):
        # Observed live (Affirm "Software Engineer II, Fullstack (Card Acquisition)") described as
        # "a Senior/Staff-level leadership role".
        posting = job(title="Software Engineer II, Fullstack (Card Acquisition)", description="Work with staff engineers.")
        assert not skip_is_substantiated(skip("staff_or_higher_title", "staff engineers"), posting)

    def test_dual_level_title(self):
        title = "Senior/Staff Software Engineer - Platform"
        assert not skip_is_substantiated(skip("staff_or_higher_title", title), job(title=title))

    def test_people_management_on_a_manager_title(self):
        quote = "Lead and develop a team of backend engineers, providing coaching, feedback, and career growth opportunities"
        posting = job(title="Engineering Manager, Data Platform", description=quote)
        assert skip_is_substantiated(skip("people_management", quote), posting)

    def test_people_management_needs_a_manager_title(self):
        # Observed live (Affirm PBA - Growth, a Senior role): tech-leading claimed as management.
        quote = "leading engineers on your team through ambiguity"
        assert not skip_is_substantiated(skip("people_management", quote), job(description=quote))


class TestHardRequirement:
    def test_a_skill_skip_the_model_itself_calls_not_a_hard_requirement_is_overruled(self):
        # GitLab: skipped quoting "Exposure to, or strong interest in, ..." — the very example its
        # instructions forbid.
        quote = "Exposure to, or strong interest in, building on top of large language models"
        assert not skip_is_substantiated(skip("required_specialty", quote, hard=False), job(description=quote))

    def test_the_answer_only_gates_skill_categories(self):
        quote = "50% Telecommuting permitted."
        assert skip_is_substantiated(skip("not_remote", quote, hard=False), job(description=quote))


class TestNotEngineering:
    def test_a_title_naming_software_engineering_is_overruled(self):
        quote = "reporting to our co-founder, working alongside our first AI enablement engineer"  # Tailscale
        posting = job(title="Software Engineer, AI Enablement", description=quote)
        assert not skip_is_substantiated(skip("not_engineering", quote), posting)

    def test_an_engineering_sounding_title_for_another_profession_is_honoured(self):
        quote = "Partner with account executives to run technical demos"
        posting = job(title="Senior Solutions Engineer", description=quote)
        assert skip_is_substantiated(skip("not_engineering", quote), posting)


class TestLocation:
    def test_not_remote(self):
        posting = job(description="40 hrs/week. 50% Telecommuting permitted. Multiple Positions Available.")
        assert skip_is_substantiated(skip("not_remote", "50% Telecommuting permitted"), posting)

    def test_location_requirement_that_really_excludes_is_honoured(self):
        quote = "To be eligible for this role, you must be based in the NYC area."  # Cockroach Labs
        assert skip_is_substantiated(skip("not_remote", quote), job(description=quote))

    @pytest.mark.parametrize("quote", [
        # SeatGeek: a remote offer quoted as "not remote" because it mentions the office.
        "Flexible work environment, allowing you to work as many days a week in the office as you'd like or 100% remotely",
        # Discord: binds only people near San Francisco.
        "Candidates in the San Francisco area are required to be in the office 2 days a week",
        "Remote-first company with an optional office in Austin",
    ])
    def test_location_quote_that_offers_remote_work_or_binds_only_locals(self, quote):
        assert not skip_is_substantiated(skip("not_remote", quote), job(description=quote))

    def test_not_remote_quote_must_be_about_location(self):
        posting = job(description="Own the team's quarterly goals.")
        assert not skip_is_substantiated(skip("not_remote", "Own the team's quarterly goals"), posting)

    def test_exclusion_that_names_washington_dc_not_the_state_is_overruled(self):
        # Samsara: skipped as not_remote on Washington, D.C. — the same misreading a regex once made.
        quote = (
            "This is a remote position open to candidates residing in the US except the San Francisco Bay "
            "Metro Area, NYC Metro Area, and Washington, D.C. Metro Area."
        )
        assert not skip_is_substantiated(skip("not_remote", quote), job(description=quote))

    def test_exclusion_that_names_the_candidates_state_is_honoured(self):
        quote = "This role will be remote, but is not eligible to be hired in CA, CT, NJ, NY, PA, WA."  # Twilio
        assert skip_is_substantiated(skip("not_remote", quote), job(description=quote))

    def test_time_zone_quote_naming_the_candidates_own_zone_is_overruled(self):
        quote = "Remote based permanently in PST (Pacific Standard Time)."  # Scribe
        assert not skip_is_substantiated(skip("not_remote", quote), job(description=quote))

    @pytest.mark.parametrize("quote", [
        "This role can be based remotely in the U.S. or Canada, limited to Eastern or Central time zones only.",  # MongoDB
        "Note this role is remote for candidates residing in US within EST or CST time zone",  # Genesys
    ])
    def test_time_zone_quote_that_leaves_the_candidate_out_is_honoured(self, quote):
        assert skip_is_substantiated(skip("not_remote", quote), job(description=quote))

    @pytest.mark.parametrize("quote", [
        "This is a remote opportunity within Canada and the US, ideally with the ability to work within Eastern Time hours.",  # 1Password
        "Core hours 9-5 EST preferred.",
        "Fully remote, working Central Time business hours.",
    ])
    def test_eastern_or_central_hours_disqualify_even_as_a_preference(self, quote):
        assert skip_is_substantiated(skip("not_remote", quote), job(description=quote))

    @pytest.mark.parametrize("quote", [
        "Preferred locations: New York or San Francisco.",
        "Mountain Time hours preferred.",
        "Overlap with both Pacific and Eastern time zones is required.",
    ])
    def test_other_preferences_and_the_candidates_own_zone_are_not_disqualifiers(self, quote):
        assert not skip_is_substantiated(skip("not_remote", quote), job(description=quote))

    def test_a_state_list_with_ct_is_not_central_time(self):
        quote = "This role will be remote, but is not eligible to be hired in CA, CT, NJ, NY, PA."
        assert not skip_is_substantiated(skip("not_remote", quote), job(description=quote))

    def test_the_candidates_state_is_a_parameter(self):
        quote = "This role will be remote, but is not eligible to be hired in CA, CT, NJ, NY, PA, WA."
        assert not skip_is_substantiated(skip("not_remote", quote), job(description=quote), "OR", "Oregon")
