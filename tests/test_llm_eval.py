from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from jobscan.llm.client import AnthropicClient, LlmCallError
from jobscan.llm.company_eval import classify_company
from jobscan.llm.job_eval import evaluate_job
from jobscan.llm.prompts import load_profile
from jobscan.llm.schemas import COMPANY_CLASSIFICATION_TOOL_NAME, JOB_EVALUATION_TOOL_NAME
from jobscan.models import AtsType, Company, EmploymentType, JobPosting, RemoteScope, SalarySource, Verdict

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
    "is_product_company": True,
    "compensation_assessment": "Range comfortably clears $170,000.",
    "remote_verification": "Explicitly US remote, no state restrictions mentioned.",
    "required_matches": ["C#", "Azure"],
    "required_gaps": [],
    "preferred_gaps": ["Kubernetes"],
    "minor_caveats": [],
    "evidence": ["\"We build our own SaaS platform using C# and Azure.\""],
    "credibility_assessment": "Strong fit, worth applying.",
    "primary_rejection_reason": None,
}


@pytest.fixture()
def profile(settings):
    return load_profile(settings.profile_path)


class TestJobEvaluation:
    def test_happy_path_returns_evaluation(self, profile):
        sdk = FakeSdkClient([tool_response(JOB_EVALUATION_TOOL_NAME, VALID_EVAL_INPUT)])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        evaluation = evaluate_job(client, profile, make_company(), make_job())

        assert evaluation.verdict == Verdict.STRONG_MATCH
        assert evaluation.confidence == pytest.approx(0.85)
        assert evaluation.required_matches == ["C#", "Azure"]
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
        result, in_tok, out_tok = classify_company(client, "Acme Corp", "acme.example.com", "We build tools for developers.", "Join our engineering team.")
        assert result.classification == "product"
        assert in_tok == 120

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
