from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from jobscan.llm.client import AnthropicClient, LlmCallError
from jobscan.llm.company_eval import classify_company
from jobscan.llm.job_eval import evaluate_job
from jobscan.llm.prompts import load_profile, render_profile_text
from jobscan.llm.schemas import (
    COMPANY_CLASSIFICATION_TOOL_NAME,
    JOB_EVALUATION_TOOL_NAME,
    JOB_EVALUATION_TOOL_SCHEMA,
    TRIAGE_TOOL_NAME,
)
from jobscan.llm.triage import triage_job
from jobscan.models import (
    AtsType,
    Company,
    EmploymentType,
    EvidenceClassification,
    JobPosting,
    RemoteScope,
    RequirementImportance,
    RequirementStrength,
    SalarySource,
    ScopeFit,
    SpecialistTenureClassification,
    Verdict,
)

NOW = datetime.now(timezone.utc)


class FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("no more fake responses queued")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeSdkClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)


def tool_response(tool_name, input_dict, input_tokens=120, output_tokens=80):
    block = SimpleNamespace(type="tool_use", name=tool_name, input=input_dict)
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(content=[block], usage=usage)


def make_job() -> JobPosting:
    return JobPosting(
        id=1,
        company_id=1,
        source=AtsType.GREENHOUSE,
        source_job_id="1",
        title="Senior Backend Engineer",
        location_raw="Remote - US",
        remote_scope=RemoteScope.REMOTE_US,
        employment_type_raw="Full-time",
        employment_type=EmploymentType.FULL_TIME,
        description_text="We build our own SaaS platform using C# and Azure.",
        description_html=None,
        description_hash="hash1",
        posting_url="https://acme.example.com/1",
        apply_url="https://acme.example.com/1/apply",
        department="Engineering",
        published_at=NOW,
        first_seen_at=NOW,
        last_seen_at=NOW,
        salary_min=180000,
        salary_max=220000,
        salary_currency="USD",
        salary_period="year",
        salary_source=SalarySource.STRUCTURED,
    )


def make_company() -> Company:
    return Company(
        id=1, name="Acme Corp", domain="acme.example.com", careers_url=None,
        ats_type=AtsType.GREENHOUSE, board_id="acme",
    )


VALID_EVAL_INPUT = {
    "verdict": "strong_match",
    "confidence": 0.85,
    "scope_fit": "at_level",
    "evidence_coverage_percent": 85,
    "specialist_tenure_assessment": {
        "classification": "not_applicable",
        "specialty": "",
        "explanation": "No specialized tenure requirement beyond general backend experience.",
    },
    "requirement_evidence": [
        {
            "requirement": "5+ years backend engineering",
            "stated_as": "required",
            "importance": "central",
            "evidence_classification": "directly_demonstrated",
            "candidate_evidence": "~15 years of backend engineering experience.",
            "posting_evidence": "5+ years of backend engineering experience required.",
        }
    ],
    "growth_dimensions": [],
    "hidden_staff_signals": [],
    "is_product_company": True,
    "compensation_assessment": "Range comfortably clears $170,000.",
    "remote_employment_verification": "Explicitly US remote, no state restrictions mentioned.",
    "required_matches": ["C#", "Azure"],
    "required_gaps": [],
    "preferred_only_gaps": ["Kubernetes"],
    "minor_caveats": [],
    "evidence": ["\"We build our own SaaS platform using C# and Azure.\""],
    "credibility_assessment": "Strong fit, worth applying.",
    "why_this_is_or_is_not_gettable": "At-level scope with strong direct evidence makes this a credible near-term interview.",
    "worth_applying": True,
    "primary_rejection_reason": None,
}


@pytest.fixture()
def profile(settings):
    return load_profile(settings.profile_path)


class TestJobEvaluation:
    def test_requests_enough_max_tokens_to_avoid_truncation(self, profile):
        # Observed live: the old default of 2048 silently truncated ~99% of real calls
        # (stop_reason="max_tokens"), with requirement_evidence in particular going missing 93%
        # of the time — masked as "no findings" by its empty-list default, not a visible error.
        sdk = FakeSdkClient([tool_response(JOB_EVALUATION_TOOL_NAME, VALID_EVAL_INPUT)])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        evaluate_job(client, profile, make_company(), make_job())

        assert sdk.messages.calls[0]["max_tokens"] >= 8192

    def test_tool_is_sent_in_strict_mode_with_a_closed_all_required_schema(self, profile):
        # Strict tool use makes the API guarantee a schema-valid input — without it the model once
        # wrapped a whole evaluation in an extra {"evaluation": {...}} object (Lumos).
        sdk = FakeSdkClient([tool_response(JOB_EVALUATION_TOOL_NAME, VALID_EVAL_INPUT)])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        evaluate_job(client, profile, make_company(), make_job())

        tool = sdk.messages.calls[0]["tools"][0]
        assert tool["strict"] is True
        schema = tool["input_schema"]
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
        item = schema["$defs"]["RequirementEvidenceItem"]
        assert item["additionalProperties"] is False and set(item["required"]) == set(item["properties"])
        assert "minimum" not in schema["properties"]["confidence"]  # unsupported in strict mode
        # The fingerprinted schema itself is untouched — strictness is transport, not rubric.
        assert "strict" not in JOB_EVALUATION_TOOL_SCHEMA

    def test_worth_applying_false_is_passed_through(self, profile):
        # A borderline verdict the model judged not realistically attainable (central
        # requirements in an unfamiliar core technology, e.g. Rust/Kafka) — worth_applying=False
        # is what routes this out of Attractive Stretches in the report, so it must survive
        # unchanged from the raw model response through to the Evaluation the report reads.
        loose_input = dict(VALID_EVAL_INPUT, verdict="borderline", scope_fit="two_plus_steps_up", worth_applying=False)
        sdk = FakeSdkClient([tool_response(JOB_EVALUATION_TOOL_NAME, loose_input)])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        evaluation = evaluate_job(client, profile, make_company(), make_job())

        assert evaluation.worth_applying is False

    def test_worth_applying_missing_raises(self, profile):
        bad_input = dict(VALID_EVAL_INPUT)
        del bad_input["worth_applying"]
        sdk = FakeSdkClient([tool_response(JOB_EVALUATION_TOOL_NAME, bad_input)])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        with pytest.raises(LlmCallError):
            evaluate_job(client, profile, make_company(), make_job())

    def test_cache_tokens_are_captured_separately_from_input_tokens(self, profile):
        # cache_creation/cache_read tokens are billed at different rates than plain input tokens
        # (see EvaluateStats.estimated_cost_usd) and must survive from the raw API usage object
        # through to the stored Evaluation, not be dropped or folded into input_tokens.
        block = SimpleNamespace(type="tool_use", name=JOB_EVALUATION_TOOL_NAME, input=VALID_EVAL_INPUT)
        usage = SimpleNamespace(
            input_tokens=500, output_tokens=300,
            cache_creation_input_tokens=7743, cache_read_input_tokens=0,
        )
        sdk = FakeSdkClient([SimpleNamespace(content=[block], usage=usage)])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        evaluation = evaluate_job(client, profile, make_company(), make_job())

        assert evaluation.input_tokens == 500
        assert evaluation.cache_creation_input_tokens == 7743
        assert evaluation.cache_read_input_tokens == 0

    def test_happy_path_returns_evaluation(self, profile):
        sdk = FakeSdkClient([tool_response(JOB_EVALUATION_TOOL_NAME, VALID_EVAL_INPUT)])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        evaluation = evaluate_job(client, profile, make_company(), make_job())

        assert evaluation.worth_applying is True
        assert evaluation.verdict == Verdict.STRONG_MATCH
        assert evaluation.confidence == pytest.approx(0.85)
        assert evaluation.scope_fit == ScopeFit.AT_LEVEL
        assert evaluation.evidence_coverage_percent == 85
        assert evaluation.specialist_tenure_assessment.classification == SpecialistTenureClassification.NOT_APPLICABLE
        assert len(evaluation.requirement_evidence) == 1
        assert evaluation.requirement_evidence[0].importance == RequirementImportance.CENTRAL
        assert evaluation.requirement_evidence[0].evidence_classification == EvidenceClassification.DIRECTLY_DEMONSTRATED
        assert evaluation.required_matches == ["C#", "Azure"]
        assert evaluation.preferred_only_gaps == ["Kubernetes"]
        assert evaluation.remote_employment_verification == "Explicitly US remote, no state restrictions mentioned."
        assert evaluation.why_this_is_or_is_not_gettable
        assert evaluation.model_name == "test-model"
        assert evaluation.input_tokens == 120
        assert evaluation.output_tokens == 80

    def test_evidence_as_bare_string_is_coerced_to_list(self, profile):
        # Observed live: the model sometimes returns one paragraph instead of a list of points
        # for list-typed fields despite the enforced tool schema.
        loose_input = dict(VALID_EVAL_INPUT, evidence="\"We build our own SaaS platform using C# and Azure.\"")
        sdk = FakeSdkClient([tool_response(JOB_EVALUATION_TOOL_NAME, loose_input)])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        evaluation = evaluate_job(client, profile, make_company(), make_job())

        assert evaluation.evidence == ['"We build our own SaaS platform using C# and Azure."']

    def test_multiline_string_field_splits_into_multiple_points(self, profile):
        loose_input = dict(VALID_EVAL_INPUT, required_gaps="- No AWS experience\n- No Kotlin experience\n")
        sdk = FakeSdkClient([tool_response(JOB_EVALUATION_TOOL_NAME, loose_input)])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        evaluation = evaluate_job(client, profile, make_company(), make_job())

        assert evaluation.required_gaps == ["No AWS experience", "No Kotlin experience"]

    def test_reject_missing_rejection_reason_is_synthesized_from_required_gaps(self, profile):
        # Observed live: 37 of 53 real reject verdicts in one run had no primary_rejection_reason
        # at all, despite the prompt requiring it, even though required_gaps was populated.
        loose_input = dict(
            VALID_EVAL_INPUT, verdict="reject", primary_rejection_reason=None,
            required_gaps=["5+ years Ruby on Rails", "Deep AWS expertise"],
        )
        sdk = FakeSdkClient([tool_response(JOB_EVALUATION_TOOL_NAME, loose_input)])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        evaluation = evaluate_job(client, profile, make_company(), make_job())

        assert evaluation.primary_rejection_reason == "5+ years Ruby on Rails; Deep AWS expertise"

    def test_reject_missing_rejection_reason_falls_back_to_credibility_assessment(self, profile):
        loose_input = dict(
            VALID_EVAL_INPUT, verdict="reject", primary_rejection_reason=None,
            required_gaps=[], growth_dimensions=[],
            credibility_assessment="Not a fit given the domain mismatch.",
        )
        sdk = FakeSdkClient([tool_response(JOB_EVALUATION_TOOL_NAME, loose_input)])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        evaluation = evaluate_job(client, profile, make_company(), make_job())

        assert evaluation.primary_rejection_reason == "Not a fit given the domain mismatch."

    def test_strong_match_missing_rejection_reason_stays_none(self, profile):
        # No synthesis for a positive verdict — a missing reason there is correctly meaningless.
        loose_input = dict(VALID_EVAL_INPUT, verdict="strong_match", primary_rejection_reason=None)
        sdk = FakeSdkClient([tool_response(JOB_EVALUATION_TOOL_NAME, loose_input)])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        evaluation = evaluate_job(client, profile, make_company(), make_job())

        assert evaluation.primary_rejection_reason is None

    def test_stray_extra_field_is_ignored_not_rejected(self, profile):
        # Observed live: 10 of 11 failures in a 436-call run were the model adding one stray
        # duplicate-ish field (e.g. "preferred_only_gaps_2") alongside the correctly-named ones.
        loose_input = dict(VALID_EVAL_INPUT, preferred_only_gaps_2=[], specialist_tenure_assessment_dummy="x")
        sdk = FakeSdkClient([tool_response(JOB_EVALUATION_TOOL_NAME, loose_input)])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        evaluation = evaluate_job(client, profile, make_company(), make_job())

        assert evaluation.verdict == Verdict.STRONG_MATCH
        assert evaluation.preferred_only_gaps == ["Kubernetes"]

    def test_reject_missing_gettability_field_is_backfilled_from_rejection_reason(self, profile):
        # Observed live: on a clear-cut reject the model sometimes omits
        # why_this_is_or_is_not_gettable, treating primary_rejection_reason as sufficient on its
        # own — 9 of 56 real evaluations failed this way before the backfill was added.
        loose_input = dict(VALID_EVAL_INPUT, verdict="reject", primary_rejection_reason="Requires 5+ years Ruby on Rails.")
        del loose_input["why_this_is_or_is_not_gettable"]
        sdk = FakeSdkClient([tool_response(JOB_EVALUATION_TOOL_NAME, loose_input)])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        evaluation = evaluate_job(client, profile, make_company(), make_job())

        assert evaluation.verdict == Verdict.REJECT
        assert "Requires 5+ years Ruby on Rails." in evaluation.why_this_is_or_is_not_gettable

    def test_reject_missing_credibility_assessment_is_also_backfilled(self, profile):
        loose_input = dict(VALID_EVAL_INPUT, verdict="reject", primary_rejection_reason="Requires 5+ years Ruby on Rails.")
        del loose_input["why_this_is_or_is_not_gettable"]
        del loose_input["credibility_assessment"]
        sdk = FakeSdkClient([tool_response(JOB_EVALUATION_TOOL_NAME, loose_input)])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        evaluation = evaluate_job(client, profile, make_company(), make_job())

        assert "Requires 5+ years Ruby on Rails." in evaluation.credibility_assessment

    def test_missing_gettability_field_without_rejection_reason_still_raises(self, profile):
        # Not a reject/borderline verdict, so the backfill never engages regardless of what else
        # is missing — this should still be a genuine failure, not silently papered over.
        bad_input = dict(VALID_EVAL_INPUT)
        del bad_input["why_this_is_or_is_not_gettable"]
        bad_input["primary_rejection_reason"] = None
        sdk = FakeSdkClient([tool_response(JOB_EVALUATION_TOOL_NAME, bad_input)])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        with pytest.raises(LlmCallError):
            evaluate_job(client, profile, make_company(), make_job())

    def test_reject_missing_both_reason_and_gettability_field_is_synthesized(self, profile):
        # Observed live (Airbnb, Staff Workday Integration Engineer): the model omitted BOTH
        # primary_rejection_reason and the required why_this_is_or_is_not_gettable field on a
        # reject verdict, which used to fail validation outright since the old backfill only
        # covered "reason present, gettability field missing". required_gaps was still populated,
        # so a reason is derivable and the evaluation should be recovered rather than discarded.
        loose_input = dict(
            VALID_EVAL_INPUT, verdict="reject", primary_rejection_reason=None,
            required_gaps=["Deep Workday Studio/XML configuration expertise", "5+ years HR-systems integration"],
        )
        del loose_input["why_this_is_or_is_not_gettable"]
        sdk = FakeSdkClient([tool_response(JOB_EVALUATION_TOOL_NAME, loose_input)])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        evaluation = evaluate_job(client, profile, make_company(), make_job())

        assert evaluation.primary_rejection_reason == "Deep Workday Studio/XML configuration expertise; 5+ years HR-systems integration"
        assert "Deep Workday Studio/XML configuration expertise" in evaluation.why_this_is_or_is_not_gettable

    def test_reject_with_nothing_at_all_to_synthesize_from_still_raises(self, profile):
        # required_gaps, growth_dimensions, and credibility_assessment are all empty/blank too —
        # a genuinely unsalvageable response should still fail rather than backfilling nonsense.
        loose_input = dict(
            VALID_EVAL_INPUT, verdict="reject", primary_rejection_reason=None,
            required_gaps=[], growth_dimensions=[], credibility_assessment="",
        )
        del loose_input["why_this_is_or_is_not_gettable"]
        sdk = FakeSdkClient([tool_response(JOB_EVALUATION_TOOL_NAME, loose_input)])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        with pytest.raises(LlmCallError):
            evaluate_job(client, profile, make_company(), make_job())

    def test_missing_required_field_raises(self, profile):
        bad_input = dict(VALID_EVAL_INPUT)
        del bad_input["verdict"]
        sdk = FakeSdkClient([tool_response(JOB_EVALUATION_TOOL_NAME, bad_input)])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        with pytest.raises(LlmCallError):
            evaluate_job(client, profile, make_company(), make_job())

    def test_invalid_verdict_enum_raises(self, profile):
        bad_input = dict(VALID_EVAL_INPUT, verdict="definitely_yes")
        sdk = FakeSdkClient([tool_response(JOB_EVALUATION_TOOL_NAME, bad_input)])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        with pytest.raises(LlmCallError):
            evaluate_job(client, profile, make_company(), make_job())

    def test_confidence_out_of_range_raises(self, profile):
        bad_input = dict(VALID_EVAL_INPUT, confidence=1.5)
        sdk = FakeSdkClient([tool_response(JOB_EVALUATION_TOOL_NAME, bad_input)])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        with pytest.raises(LlmCallError):
            evaluate_job(client, profile, make_company(), make_job())

    def test_invalid_scope_fit_enum_raises(self, profile):
        bad_input = dict(VALID_EVAL_INPUT, scope_fit="basically_the_same")
        sdk = FakeSdkClient([tool_response(JOB_EVALUATION_TOOL_NAME, bad_input)])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        with pytest.raises(LlmCallError):
            evaluate_job(client, profile, make_company(), make_job())

    def test_evidence_coverage_percent_out_of_range_raises(self, profile):
        bad_input = dict(VALID_EVAL_INPUT, evidence_coverage_percent=150)
        sdk = FakeSdkClient([tool_response(JOB_EVALUATION_TOOL_NAME, bad_input)])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        with pytest.raises(LlmCallError):
            evaluate_job(client, profile, make_company(), make_job())

    def test_invalid_specialist_tenure_classification_raises(self, profile):
        bad_input = dict(VALID_EVAL_INPUT)
        bad_input["specialist_tenure_assessment"] = dict(bad_input["specialist_tenure_assessment"], classification="mostly")
        sdk = FakeSdkClient([tool_response(JOB_EVALUATION_TOOL_NAME, bad_input)])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        with pytest.raises(LlmCallError):
            evaluate_job(client, profile, make_company(), make_job())

    def test_schema_puts_the_analysis_before_the_verdict(self):
        # The forced tool call has no separate thinking step, so property order is reasoning order:
        # with the verdict first, the model committed to "reject" for Pilot and only then found it
        # had no reason to. Requirement wording must also precede its importance.
        properties = list(JOB_EVALUATION_TOOL_SCHEMA["input_schema"]["properties"])
        for analysis_field in ("requirement_evidence", "growth_dimensions", "hidden_staff_signals", "compensation_assessment", "scope_fit"):
            assert properties.index(analysis_field) < properties.index("verdict")
        item_properties = list(JOB_EVALUATION_TOOL_SCHEMA["input_schema"]["$defs"]["RequirementEvidenceItem"]["properties"])
        assert item_properties.index("posting_evidence") < item_properties.index("stated_as") < item_properties.index("importance")

    def test_answer_wrapped_in_one_extra_object_is_unwrapped(self, profile):
        # Observed live (Lumos): {"evaluation": {...every field...}} failed validation outright.
        sdk = FakeSdkClient([tool_response(JOB_EVALUATION_TOOL_NAME, {"evaluation": VALID_EVAL_INPUT})])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        evaluation = evaluate_job(client, profile, make_company(), make_job())

        assert evaluation.verdict == Verdict.STRONG_MATCH

    def test_preferred_item_tagged_central_is_demoted_to_secondary(self, profile):
        # Observed live (Posit): "Proficiency in either Go or Typescript ... are helpful for this
        # role" was scored as the role's central requirement, sinking an otherwise strong fit.
        item = dict(
            VALID_EVAL_INPUT["requirement_evidence"][0], requirement="Go or TypeScript",
            posting_evidence="Proficiency in either Go or Typescript ... are helpful for this role.",
            stated_as="preferred", importance="central", evidence_classification="not_demonstrated",
        )
        sdk = FakeSdkClient([tool_response(JOB_EVALUATION_TOOL_NAME, dict(VALID_EVAL_INPUT, requirement_evidence=[item]))])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        evaluation = evaluate_job(client, profile, make_company(), make_job())

        assert evaluation.requirement_evidence[0].stated_as == RequirementStrength.PREFERRED
        assert evaluation.requirement_evidence[0].importance == RequirementImportance.SECONDARY

    def test_required_item_tagged_central_stays_central(self, profile):
        sdk = FakeSdkClient([tool_response(JOB_EVALUATION_TOOL_NAME, VALID_EVAL_INPUT)])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        evaluation = evaluate_job(client, profile, make_company(), make_job())

        assert evaluation.requirement_evidence[0].importance == RequirementImportance.CENTRAL

    def test_requirement_missing_its_wording_classification_raises(self, profile):
        item = {k: v for k, v in VALID_EVAL_INPUT["requirement_evidence"][0].items() if k != "stated_as"}
        sdk = FakeSdkClient([tool_response(JOB_EVALUATION_TOOL_NAME, dict(VALID_EVAL_INPUT, requirement_evidence=[item]))])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        with pytest.raises(LlmCallError):
            evaluate_job(client, profile, make_company(), make_job())

    def test_invalid_requirement_evidence_importance_raises(self, profile):
        bad_input = dict(VALID_EVAL_INPUT)
        bad_input["requirement_evidence"] = [dict(bad_input["requirement_evidence"][0], importance="mostly_central")]
        sdk = FakeSdkClient([tool_response(JOB_EVALUATION_TOOL_NAME, bad_input)])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        with pytest.raises(LlmCallError):
            evaluate_job(client, profile, make_company(), make_job())

    def test_missing_specialist_tenure_assessment_raises(self, profile):
        bad_input = dict(VALID_EVAL_INPUT)
        del bad_input["specialist_tenure_assessment"]
        sdk = FakeSdkClient([tool_response(JOB_EVALUATION_TOOL_NAME, bad_input)])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        with pytest.raises(LlmCallError):
            evaluate_job(client, profile, make_company(), make_job())

    def test_growth_dimensions_as_bare_string_is_coerced_to_list(self, profile):
        loose_input = dict(VALID_EVAL_INPUT, growth_dimensions="Deep AWS distributed-systems ownership")
        sdk = FakeSdkClient([tool_response(JOB_EVALUATION_TOOL_NAME, loose_input)])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        evaluation = evaluate_job(client, profile, make_company(), make_job())

        assert evaluation.growth_dimensions == ["Deep AWS distributed-systems ownership"]

    def test_missing_tool_use_block_raises(self, profile):
        text_only_response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="I refuse to use tools.")],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )
        sdk = FakeSdkClient([text_only_response])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        with pytest.raises(LlmCallError):
            evaluate_job(client, profile, make_company(), make_job())

    def test_no_api_key_and_no_client_raises(self):
        with pytest.raises(LlmCallError):
            AnthropicClient(api_key=None, model="test-model")


class TestCompanyClassification:
    def test_happy_path(self):
        sdk = FakeSdkClient(
            [tool_response(COMPANY_CLASSIFICATION_TOOL_NAME, {
                "classification": "product",
                "confidence": 0.9,
                "evidence": "Homepage describes their own SaaS platform for developers.",
            })]
        )
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)
        result, in_tok, out_tok, cache_write_tok, cache_read_tok = classify_company(
            client, "Acme Corp", "acme.example.com", "We build tools for developers.", "Join our engineering team."
        )
        assert result.classification == "product"
        assert in_tok == 120
        assert cache_write_tok == 0
        assert cache_read_tok == 0

    def test_invalid_classification_enum_raises(self):
        sdk = FakeSdkClient(
            [tool_response(COMPANY_CLASSIFICATION_TOOL_NAME, {
                "classification": "nonprofit",
                "confidence": 0.5,
                "evidence": "unclear",
            })]
        )
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)
        with pytest.raises(LlmCallError):
            classify_company(client, "Acme Corp", "acme.example.com", "", "")


class TestTriage:
    def test_happy_path_skip_true(self, profile):
        sdk = FakeSdkClient(
            [tool_response(TRIAGE_TOOL_NAME, {"disqualifier_quote": "8+ years of Rust", "disqualifier": "required_unfamiliar_language", "skip_full_evaluation": True, "reason": "Requires 8+ years Rust."})]
        )
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        result, in_tok, out_tok, cache_write_tok, cache_read_tok = triage_job(
            client, render_profile_text(profile), make_company(), make_job()
        )

        assert result.skip_full_evaluation is True
        assert result.reason == "Requires 8+ years Rust."
        assert in_tok == 120
        assert cache_write_tok == 0
        assert cache_read_tok == 0

    def test_happy_path_skip_false(self, profile):
        sdk = FakeSdkClient(
            [tool_response(TRIAGE_TOOL_NAME, {"disqualifier_quote": "", "disqualifier": "none", "skip_full_evaluation": False, "reason": "Real backend overlap."})]
        )
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        result, *_ = triage_job(client, render_profile_text(profile), make_company(), make_job())

        assert result.skip_full_evaluation is False

    def test_uses_a_small_max_tokens(self, profile):
        # The whole point is a cheap, short-output call — it should not request anywhere near the
        # 8192 the full evaluation needs.
        sdk = FakeSdkClient(
            [tool_response(TRIAGE_TOOL_NAME, {"disqualifier_quote": "", "disqualifier": "none", "skip_full_evaluation": False, "reason": "ok"})]
        )
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        triage_job(client, render_profile_text(profile), make_company(), make_job())

        assert sdk.messages.calls[0]["max_tokens"] <= 1024

    def test_truncated_response_raises_a_clear_error(self, profile):
        # Observed live at max_tokens=300: a long quote crowded out `reason`; say so plainly
        # instead of surfacing it as a confusing "field required" validation error.
        response = tool_response(TRIAGE_TOOL_NAME, {"disqualifier_quote": "x" * 50})
        response.stop_reason = "max_tokens"
        client = AnthropicClient(api_key=None, model="test-model", client=FakeSdkClient([response]))

        with pytest.raises(LlmCallError, match="max_tokens"):
            triage_job(client, render_profile_text(profile), make_company(), make_job())

    def test_missing_required_field_raises(self, profile):
        sdk = FakeSdkClient([tool_response(TRIAGE_TOOL_NAME, {"reason": "missing the boolean"})])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        with pytest.raises(LlmCallError):
            triage_job(client, render_profile_text(profile), make_company(), make_job())
