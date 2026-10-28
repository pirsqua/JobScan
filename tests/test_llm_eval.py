from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest
import respx

from jobscan.llm.client import AnthropicClient, LlmCallError
from jobscan.llm.company_eval import classify_company, fetch_homepage_text
from jobscan.llm.job_eval import evaluate_job
from jobscan.llm.prompts import load_profile, render_profile_text
from jobscan.llm.schemas import JOB_EVALUATION_SCHEMA
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
    WorkFit,
)

from tests.fakes import VALID_EVALUATION, FakeSdkClient, json_response, refusal_response

NOW = datetime.now(timezone.utc)


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


@pytest.fixture()
def profile(settings):
    return load_profile(settings.profile_path)


class TestJobEvaluation:
    def test_requests_enough_max_tokens_to_avoid_truncation(self, profile):
        # Observed live: the old default of 2048 silently truncated ~99% of real calls
        # (stop_reason="max_tokens"), with requirement_evidence in particular going missing 93%
        # of the time — masked as "no findings" by its empty-list default, not a visible error.
        sdk = FakeSdkClient([json_response(VALID_EVALUATION)])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        evaluate_job(client, profile, make_company(), make_job())

        assert sdk.messages.calls[0]["max_tokens"] >= 8192

    def test_asks_for_structured_output_with_a_closed_all_required_schema(self, profile):
        # Structured outputs make the API guarantee a schema-valid answer — without an enforced
        # schema the model once wrapped a whole evaluation in an extra {"evaluation": {...}} object
        # (Lumos). No tool call: Sonnet 5.5 rejects a forced one.
        sdk = FakeSdkClient([json_response(VALID_EVALUATION)])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        evaluate_job(client, profile, make_company(), make_job())

        call = sdk.messages.calls[0]
        assert "tools" not in call and "tool_choice" not in call
        output_format = call["output_config"]["format"]
        assert output_format["type"] == "json_schema"
        schema = output_format["schema"]
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
        item = schema["$defs"]["RequirementEvidenceItem"]
        assert item["additionalProperties"] is False and set(item["required"]) == set(item["properties"])
        assert "minimum" not in schema["properties"]["confidence"]  # unsupported by structured outputs
        # The fingerprinted schema itself is untouched — strictness is transport, not rubric.
        assert "additionalProperties" not in JOB_EVALUATION_SCHEMA
        assert list(schema["properties"]) == list(JOB_EVALUATION_SCHEMA["properties"])  # order kept

    def test_the_thinking_setting_is_sent_only_when_configured(self, profile):
        sdk = FakeSdkClient([json_response(VALID_EVALUATION), json_response(VALID_EVALUATION)])

        evaluate_job(AnthropicClient(api_key=None, model="m", client=sdk, thinking="between_tools"),
                     profile, make_company(), make_job())
        evaluate_job(AnthropicClient(api_key=None, model="m", client=sdk), profile, make_company(), make_job())

        assert sdk.messages.calls[0]["thinking"] == {"type": "between_tools"}
        assert "thinking" not in sdk.messages.calls[1]

    def test_thinking_blocks_before_the_answer_are_skipped(self, profile):
        response = json_response(VALID_EVALUATION)
        response.content.insert(0, SimpleNamespace(type="thinking", thinking="", signature="sig"))
        client = AnthropicClient(api_key=None, model="test-model", client=FakeSdkClient([response]))

        evaluation = evaluate_job(client, profile, make_company(), make_job())

        assert evaluation.verdict == Verdict.STRONG_MATCH

    def test_a_refusal_is_an_error_never_a_verdict(self, profile):
        # Sonnet 5.5 declines some requests (e.g. "cyber") with a normal 200 — a posting for an
        # offensive-security role must be left awaiting evaluation, not cached as a reject.
        client = AnthropicClient(api_key=None, model="test-model", client=FakeSdkClient([refusal_response("cyber")]))

        with pytest.raises(LlmCallError, match="declined.*cyber"):
            evaluate_job(client, profile, make_company(), make_job())

    def test_worth_applying_false_is_passed_through(self, profile):
        # A borderline verdict the model judged not realistically attainable (central
        # requirements in an unfamiliar core technology, e.g. Rust/Kafka) — worth_applying=False
        # is what routes this out of Attractive Stretches in the report, so it must survive
        # unchanged from the raw model response through to the Evaluation the report reads.
        loose_input = dict(VALID_EVALUATION, verdict="borderline", scope_fit="two_plus_steps_up", worth_applying=False)
        sdk = FakeSdkClient([json_response(loose_input)])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        evaluation = evaluate_job(client, profile, make_company(), make_job())

        assert evaluation.worth_applying is False

    def test_cache_tokens_are_captured_separately_from_input_tokens(self, profile):
        # cache_creation/cache_read tokens are billed at different rates than plain input tokens
        # (see EvaluateStats.estimated_cost_usd) and must survive from the raw API usage object
        # through to the stored Evaluation, not be dropped or folded into input_tokens.
        block = SimpleNamespace(type="text", text=json.dumps(VALID_EVALUATION))
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
        sdk = FakeSdkClient([json_response(VALID_EVALUATION)])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        evaluation = evaluate_job(client, profile, make_company(), make_job())

        assert evaluation.worth_applying is True
        assert evaluation.verdict == Verdict.STRONG_MATCH
        assert evaluation.work_fit == WorkFit.IN_LANE
        assert evaluation.day_to_day_work == "Building backend services."
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
        assert evaluation.input_tokens == 100
        assert evaluation.output_tokens == 50

    @pytest.mark.parametrize("filled, reason", [
        ({"required_gaps": ["5+ years Ruby on Rails", "Deep AWS expertise", "Kafka"]}, "5+ years Ruby on Rails; Deep AWS expertise"),
        ({"growth_dimensions": ["Owning a payments ledger"]}, "Owning a payments ledger"),
        ({}, "Not a fit given the domain mismatch."),
    ])
    def test_a_reject_without_a_reason_takes_one_from_the_gaps_it_found(self, profile, filled, reason):
        # Observed live: 37 of 53 real reject verdicts in one run left primary_rejection_reason null
        # despite the prompt requiring it, even though required_gaps was populated.
        answer = {
            **VALID_EVALUATION, "verdict": "reject", "primary_rejection_reason": None, "required_gaps": [],
            "growth_dimensions": [], "credibility_assessment": "Not a fit given the domain mismatch.", **filled,
        }
        client = AnthropicClient(api_key=None, model="test-model", client=FakeSdkClient([json_response(answer)]))

        evaluation = evaluate_job(client, profile, make_company(), make_job())

        assert evaluation.primary_rejection_reason == reason

    def test_strong_match_missing_rejection_reason_stays_none(self, profile):
        # No synthesis for a positive verdict — a missing reason there is correctly meaningless.
        loose_input = dict(VALID_EVALUATION, verdict="strong_match", primary_rejection_reason=None)
        sdk = FakeSdkClient([json_response(loose_input)])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        evaluation = evaluate_job(client, profile, make_company(), make_job())

        assert evaluation.primary_rejection_reason is None

    @pytest.mark.parametrize("bad_input", [
        pytest.param({k: v for k, v in VALID_EVALUATION.items() if k != "verdict"}, id="missing-field"),
        pytest.param(dict(VALID_EVALUATION, verdict="definitely_yes"), id="unknown-verdict"),
        # Structured outputs don't support minimum/maximum, so ranges are checked only here.
        pytest.param(dict(VALID_EVALUATION, confidence=1.5), id="confidence-over-1"),
        pytest.param(dict(VALID_EVALUATION, evidence_coverage_percent=150), id="coverage-over-100"),
    ])
    def test_an_answer_that_fails_validation_raises(self, profile, bad_input):
        client = AnthropicClient(api_key=None, model="test-model", client=FakeSdkClient([json_response(bad_input)]))

        with pytest.raises(LlmCallError, match="failed schema validation"):
            evaluate_job(client, profile, make_company(), make_job())

    def test_schema_puts_the_analysis_before_the_verdict(self):
        # With thinking off the model writes the JSON in schema order, so property order is reasoning order:
        # with the verdict first, the model committed to "reject" for Pilot and only then found it
        # had no reason to. Requirement wording must also precede its importance.
        properties = list(JOB_EVALUATION_SCHEMA["properties"])
        # What the job's work is comes before any requirement (observed: GitHub's npm team judged a
        # plausible match off a qualifications list that accepted C#).
        assert properties[:3] == ["day_to_day_work", "work_fit", "requirement_evidence"]
        for analysis_field in ("requirement_evidence", "growth_dimensions", "hidden_staff_signals", "compensation_assessment", "scope_fit"):
            assert properties.index(analysis_field) < properties.index("verdict")
        item_properties = list(JOB_EVALUATION_SCHEMA["$defs"]["RequirementEvidenceItem"]["properties"])
        assert item_properties.index("posting_evidence") < item_properties.index("stated_as") < item_properties.index("importance")

    def test_preferred_item_tagged_central_is_demoted_to_secondary(self, profile):
        # Observed live (Posit): "Proficiency in either Go or Typescript ... are helpful for this
        # role" was scored as the role's central requirement, sinking an otherwise strong fit.
        item = dict(
            VALID_EVALUATION["requirement_evidence"][0], requirement="Go or TypeScript",
            posting_evidence="Proficiency in either Go or Typescript ... are helpful for this role.",
            stated_as="preferred", importance="central", evidence_classification="not_demonstrated",
        )
        sdk = FakeSdkClient([json_response(dict(VALID_EVALUATION, requirement_evidence=[item]))])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        evaluation = evaluate_job(client, profile, make_company(), make_job())

        assert evaluation.requirement_evidence[0].stated_as == RequirementStrength.PREFERRED
        assert evaluation.requirement_evidence[0].importance == RequirementImportance.SECONDARY

    def test_required_item_tagged_central_stays_central(self, profile):
        sdk = FakeSdkClient([json_response(VALID_EVALUATION)])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        evaluation = evaluate_job(client, profile, make_company(), make_job())

        assert evaluation.requirement_evidence[0].importance == RequirementImportance.CENTRAL

    def test_an_answer_that_is_not_json_raises(self, profile):
        prose_response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="This role looks like a strong match.")],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )
        sdk = FakeSdkClient([prose_response])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        with pytest.raises(LlmCallError):
            evaluate_job(client, profile, make_company(), make_job())

    def test_no_api_key_and_no_client_raises(self):
        with pytest.raises(LlmCallError):
            AnthropicClient(api_key=None, model="test-model")


class TestCompanyClassification:
    def test_happy_path(self):
        sdk = FakeSdkClient(
            [json_response({
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
        assert in_tok == 100
        assert cache_write_tok == 0
        assert cache_read_tok == 0

    def test_invalid_classification_enum_raises(self):
        sdk = FakeSdkClient(
            [json_response({
                "classification": "nonprofit",
                "confidence": 0.5,
                "evidence": "unclear",
            })]
        )
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)
        with pytest.raises(LlmCallError):
            classify_company(client, "Acme Corp", "acme.example.com", "", "")

    @respx.mock
    def test_homepage_text_is_the_visible_copy_without_scripts_navigation_or_footer(self):
        respx.get("https://acme.example.com/").mock(return_value=httpx.Response(200, text=(
            "<nav>Pricing Login</nav><script>track()</script><main>We  build\n tools for developers.</main><footer>©</footer>"
        )))
        with httpx.Client() as client:
            assert fetch_homepage_text(client, "acme.example.com") == "We build tools for developers."
            assert fetch_homepage_text(client, None) == ""


class TestTriage:
    def test_happy_path_skip_true(self, profile):
        sdk = FakeSdkClient(
            [json_response({"disqualifier_quote": "8+ years of Rust", "quote_is_hard_requirement": True, "disqualifier": "required_unfamiliar_language", "skip_full_evaluation": True, "reason": "Requires 8+ years Rust."})]
        )
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        result, in_tok, out_tok, cache_write_tok, cache_read_tok = triage_job(
            client, render_profile_text(profile), make_company(), make_job()
        )

        assert result.skip_full_evaluation is True
        assert result.reason == "Requires 8+ years Rust."
        assert in_tok == 100
        assert cache_write_tok == 0
        assert cache_read_tok == 0

    def test_happy_path_skip_false(self, profile):
        sdk = FakeSdkClient(
            [json_response({"disqualifier_quote": "", "quote_is_hard_requirement": False, "disqualifier": "none", "skip_full_evaluation": False, "reason": "Real backend overlap."})]
        )
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        result, *_ = triage_job(client, render_profile_text(profile), make_company(), make_job())

        assert result.skip_full_evaluation is False

    def test_uses_a_small_max_tokens(self, profile):
        # The whole point is a cheap, short-output call — it should not request anywhere near the
        # 8192 the full evaluation needs.
        sdk = FakeSdkClient(
            [json_response({"disqualifier_quote": "", "quote_is_hard_requirement": False, "disqualifier": "none", "skip_full_evaluation": False, "reason": "ok"})]
        )
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        triage_job(client, render_profile_text(profile), make_company(), make_job())

        assert sdk.messages.calls[0]["max_tokens"] <= 1024

    def test_truncated_response_raises_a_clear_error(self, profile):
        # Observed live at max_tokens=300: a long quote crowded out `reason`; say so plainly
        # instead of surfacing it as a confusing "field required" validation error.
        response = json_response({"disqualifier_quote": "x" * 50})
        response.stop_reason = "max_tokens"
        client = AnthropicClient(api_key=None, model="test-model", client=FakeSdkClient([response]))

        with pytest.raises(LlmCallError, match="max_tokens"):
            triage_job(client, render_profile_text(profile), make_company(), make_job())

    def test_missing_required_field_raises(self, profile):
        sdk = FakeSdkClient([json_response({"reason": "missing the boolean"})])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        with pytest.raises(LlmCallError):
            triage_job(client, render_profile_text(profile), make_company(), make_job())
