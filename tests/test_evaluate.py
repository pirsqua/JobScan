from __future__ import annotations

import dataclasses
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from jobscan.db import Database
from jobscan.evaluate import evaluate_all
from jobscan.llm.client import AnthropicClient
from jobscan.llm import prompts
from jobscan.llm.prompts import (
    evaluation_rubric_version,
    load_profile,
    render_profile_text,
    triage_rubric_version,
)
from jobscan.llm.schemas import JOB_EVALUATION_TOOL_NAME, TRIAGE_TOOL_NAME
from jobscan.models import (
    AtsType,
    ClassificationSource,
    Company,
    CompanyClassification,
    EvaluateStats,
    ManualOverride,
    RawPosting,
    ScopeFit,
    Verdict,
)
from jobscan.normalize import normalize_posting

VALID_EVAL_INPUT = {
    "verdict": "strong_match",
    "confidence": 0.8,
    "scope_fit": "at_level",
    "evidence_coverage_percent": 85,
    "specialist_tenure_assessment": {
        "classification": "not_applicable",
        "specialty": "",
        "explanation": "No specialized tenure requirement beyond general backend experience.",
    },
    "requirement_evidence": [],
    "growth_dimensions": [],
    "hidden_staff_signals": [],
    "is_product_company": True,
    "compensation_assessment": "Comfortably above $170,000.",
    "remote_employment_verification": "US remote, no restrictions noted.",
    "required_matches": ["C#", "Azure"],
    "required_gaps": [],
    "preferred_only_gaps": [],
    "minor_caveats": [],
    "evidence": ["quoted text"],
    "credibility_assessment": "Strong fit.",
    "why_this_is_or_is_not_gettable": "At-level scope, strong direct evidence, credible near-term interview.",
    "primary_rejection_reason": None,
    "worth_applying": True,
}


class FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.call_count = 0

    def create(self, **kwargs):
        self.call_count += 1
        if not self._responses:
            raise AssertionError("LLM was called more times than expected")
        return self._responses.pop(0)


class FakeSdkClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)


def tool_response(input_dict, input_tokens=100, output_tokens=50):
    block = SimpleNamespace(type="tool_use", name=JOB_EVALUATION_TOOL_NAME, input=input_dict)
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(content=[block], usage=usage)


def triage_tool_response(
    skip_full_evaluation: bool, reason="test reason", input_tokens=30, output_tokens=10,
    disqualifier=None, quote=None,
):
    """A skip defaults to one the substantiation check accepts: an allowed category, quoting words
    that really are in seed_job's default description."""
    if disqualifier is None:
        disqualifier = "not_engineering" if skip_full_evaluation else "none"
    if quote is None:
        quote = "We build our own SaaS product." if skip_full_evaluation else ""
    block = SimpleNamespace(
        type="tool_use", name=TRIAGE_TOOL_NAME,
        input={"disqualifier_quote": quote, "disqualifier": disqualifier,
               "skip_full_evaluation": skip_full_evaluation, "reason": reason},
    )
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(content=[block], usage=usage)


def seed_company(db: Database, name="Acme", board_id="acme", classification=None) -> Company:
    company = Company(
        name=name, domain=f"{board_id}.example.com", careers_url=None,
        ats_type=AtsType.GREENHOUSE, board_id=board_id,
    )
    if classification is not None:
        company.classification = classification
        company.classification_source = ClassificationSource.SEED
    company.id = db.upsert_company(company)
    return company


def seed_job(db: Database, settings, company: Company, source_job_id="1", title="Senior Backend Engineer", description="We build our own SaaS product. $180,000 - $220,000 per year."):
    raw = RawPosting(
        source_job_id=source_job_id,
        title=title,
        location_raw="Remote - US",
        employment_type_raw="Full-time",
        description_html=f"<p>{description}</p>",
        description_text=None,
        posting_url=f"https://acme.example.com/jobs/{source_job_id}",
        apply_url=f"https://acme.example.com/jobs/{source_job_id}/apply",
    )
    job = normalize_posting(raw, company, settings)
    job_id, _, _ = db.upsert_job(job)
    return db.get_job(job_id)


class TestEvaluateAll:
    def test_factual_rejection_skips_llm(self, db: Database, settings):
        company = seed_company(db)
        seed_job(db, settings, company, description="No salary mentioned here at all.")
        client = AnthropicClient(api_key=None, model="test-model", client=FakeSdkClient([]))

        stats = evaluate_all(db, settings, client=client)

        assert stats.factual_rejected == 1
        assert stats.sent_to_llm == 0

    def test_llm_evaluation_happy_path(self, db: Database, settings):
        company = seed_company(db, classification=CompanyClassification.PRODUCT)
        seed_job(db, settings, company)
        client = AnthropicClient(api_key=None, model="test-model", client=FakeSdkClient([tool_response(VALID_EVAL_INPUT)]))

        stats = evaluate_all(db, settings, client=client)

        assert stats.sent_to_llm == 1
        assert stats.verdict_counts["strong_match"] == 1

    def test_second_run_uses_cache_not_new_llm_call(self, db: Database, settings):
        company = seed_company(db, classification=CompanyClassification.PRODUCT)
        seed_job(db, settings, company)
        sdk = FakeSdkClient([tool_response(VALID_EVAL_INPUT)])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        evaluate_all(db, settings, client=client)
        stats2 = evaluate_all(db, settings, client=client)

        assert sdk.messages.call_count == 1
        assert stats2.cache_hits == 1
        assert stats2.sent_to_llm == 0

    def test_duplicate_description_across_jobs_reuses_cache(self, db: Database, settings):
        company = seed_company(db, classification=CompanyClassification.PRODUCT)
        same_description = "We build our own SaaS product. $180,000 - $220,000 per year."
        seed_job(db, settings, company, source_job_id="1", description=same_description)
        seed_job(db, settings, company, source_job_id="2", title="Senior Backend Engineer II", description=same_description)

        sdk = FakeSdkClient([tool_response(VALID_EVAL_INPUT)])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        stats = evaluate_all(db, settings, client=client)

        assert sdk.messages.call_count == 1
        assert stats.sent_to_llm == 1
        assert stats.cache_hits == 1

    def test_no_api_key_marks_unverified_not_rejected(self, db: Database, settings):
        company = seed_company(db)
        seed_job(db, settings, company)
        assert settings.anthropic_api_key is None

        stats = evaluate_all(db, settings)

        assert stats.unverified == 1
        assert stats.factual_rejected == 0

    def test_job_level_manual_override_skips_llm(self, db: Database, settings):
        company = seed_company(db)
        job = seed_job(db, settings, company, description="No salary published here.")
        db.set_manual_override(
            ManualOverride("job", job.id, "verdict", "strong_match", "I know this role", datetime.now(timezone.utc))
        )
        client = AnthropicClient(api_key=None, model="test-model", client=FakeSdkClient([]))

        stats = evaluate_all(db, settings, client=client)

        assert stats.manual_overrides_applied == 1
        assert stats.factual_rejected == 0
        evaluation = db.get_evaluation_for_job(job.id)
        assert evaluation.verdict == Verdict.STRONG_MATCH

    def test_company_classification_override_rejects_without_llm_classification_call(self, db: Database, settings):
        company = seed_company(db)  # unknown by default
        seed_job(db, settings, company)
        db.set_manual_override(
            ManualOverride("company", company.id, "classification", "consulting", "known staffing shop", datetime.now(timezone.utc))
        )
        client = AnthropicClient(api_key=None, model="test-model", client=FakeSdkClient([]))

        stats = evaluate_all(db, settings, client=client)

        assert stats.factual_rejected == 1
        assert stats.companies_classified == 0

    def test_run_stats_are_persisted_and_recoverable(self, db: Database, settings):
        # A standalone `report` invocation (run separately from `evaluate`) needs to recover the
        # most recent run's stats from the database rather than showing nothing.
        company = seed_company(db, classification=CompanyClassification.PRODUCT)
        seed_job(db, settings, company)
        client = AnthropicClient(api_key=None, model="test-model", client=FakeSdkClient([tool_response(VALID_EVAL_INPUT)]))

        stats = evaluate_all(db, settings, client=client)

        recovered = db.latest_evaluation_run()
        assert recovered is not None
        assert recovered.sent_to_llm == stats.sent_to_llm == 1
        assert recovered.jobs_considered == stats.jobs_considered
        assert recovered.verdict_counts == stats.verdict_counts
        assert recovered.input_tokens == stats.input_tokens
        assert recovered.finished_at is not None

    def test_limit_caps_real_llm_calls_not_free_work(self, db: Database, settings):
        company = seed_company(db, classification=CompanyClassification.PRODUCT)
        for i in range(3):
            seed_job(
                db, settings, company, source_job_id=str(i), title=f"Senior Backend Engineer {i}",
                description=f"We build our own SaaS product, role variant {i}. $180,000 - $220,000 per year.",
            )
        sdk = FakeSdkClient([tool_response(VALID_EVAL_INPUT), tool_response(VALID_EVAL_INPUT)])
        client = AnthropicClient(api_key=None, model="test-model", client=sdk)

        stats = evaluate_all(db, settings, client=client, limit=2)

        assert stats.sent_to_llm == 2
        assert sdk.messages.call_count == 2


class TestEstimatedCost:
    def test_applies_cache_write_and_read_multipliers(self, settings):
        # $3/MTok input, $15/MTok output (from the settings fixture). Cache writes cost 1.25x
        # base input, cache reads 0.1x — Anthropic's standard ephemeral-cache multipliers.
        stats = EvaluateStats(
            input_tokens=1000, cache_creation_input_tokens=1000, cache_read_input_tokens=1000,
            output_tokens=1000,
        )
        cost = stats.estimated_cost_usd(settings)
        expected = (1000 * 3.0 + 1000 * 3.0 * 1.25 + 1000 * 3.0 * 0.1) / 1_000_000 + 1000 * 15.0 / 1_000_000
        assert cost == pytest.approx(expected)

    def test_zero_cache_tokens_matches_plain_input_output_cost(self, settings):
        stats = EvaluateStats(input_tokens=2000, output_tokens=500)
        cost = stats.estimated_cost_usd(settings)
        assert cost == pytest.approx(2000 * 3.0 / 1_000_000 + 500 * 15.0 / 1_000_000)

    def test_unrecognized_model_returns_none(self, settings):
        stats = EvaluateStats(input_tokens=1000, output_tokens=100)
        unpriced_settings = dataclasses.replace(settings, anthropic_model="some-unpriced-model")
        assert stats.estimated_cost_usd(unpriced_settings) is None

    def test_triage_tokens_priced_at_the_triage_models_own_rate(self, settings):
        # Triage typically runs a cheaper model than the main evaluation ($0.8/$4 vs $3/$15 in
        # this fixture) — its tokens must use ITS pricing, not the main model's, or a run using
        # both would silently misreport cost.
        stats = EvaluateStats(
            input_tokens=1000, output_tokens=200,
            triage_input_tokens=5000, triage_output_tokens=500, triage_model="claude-haiku-test",
        )
        cost = stats.estimated_cost_usd(settings)
        expected = (1000 * 3.0 + 200 * 15.0) / 1_000_000 + (5000 * 0.8 + 500 * 4.0) / 1_000_000
        assert cost == pytest.approx(expected)

    def test_unpriced_triage_model_is_silently_excluded_not_fatal(self, settings):
        # The main run's cost estimate shouldn't disappear just because the triage model has no
        # pricing entry — that would make estimated_cost_usd return None for the whole run over a
        # config gap in a secondary, optional feature.
        stats = EvaluateStats(
            input_tokens=1000, output_tokens=200,
            triage_input_tokens=5000, triage_output_tokens=500, triage_model="some-unpriced-triage-model",
        )
        cost = stats.estimated_cost_usd(settings)
        assert cost == pytest.approx((1000 * 3.0 + 200 * 15.0) / 1_000_000)


class TestTriageIntegration:
    def test_skip_avoids_the_full_evaluation_call(self, db: Database, settings):
        company = seed_company(db, classification=CompanyClassification.PRODUCT)
        seed_job(db, settings, company)

        full_eval_sdk = FakeSdkClient([])  # no responses queued — an unexpected call raises
        full_eval_client = AnthropicClient(api_key=None, model="test-model", client=full_eval_sdk)
        triage_sdk = FakeSdkClient([triage_tool_response(skip_full_evaluation=True, reason="Requires 8+ years Rust.")])
        triage_client = AnthropicClient(api_key=None, model="test-haiku-model", client=triage_sdk)

        stats = evaluate_all(db, settings, client=full_eval_client, triage_client=triage_client)

        assert full_eval_sdk.messages.call_count == 0
        assert stats.triaged == 1
        assert stats.triage_skipped == 1
        assert stats.sent_to_llm == 0
        assert stats.verdict_counts == {"reject": 1}

        saved = db.get_evaluation_by_hash(db.get_active_jobs()[0].description_hash)
        assert saved.worth_applying is False
        assert saved.verdict == Verdict.REJECT
        assert saved.primary_rejection_reason == "Requires 8+ years Rust."
        assert saved.model_name == "triage:test-haiku-model"
        assert saved.evidence == ['not_engineering: "We build our own SaaS product."']

    def test_skip_the_posting_does_not_support_goes_to_the_full_evaluation(self, db: Database, settings):
        # Observed live: Haiku skipped a Software Engineer II as "overqualified" despite being told
        # level is not a skip reason. A skip now needs an allowed category and a real quote.
        company = seed_company(db, classification=CompanyClassification.PRODUCT)
        seed_job(db, settings, company)
        full_eval_sdk = FakeSdkClient([tool_response(VALID_EVAL_INPUT)])
        triage_sdk = FakeSdkClient([triage_tool_response(
            skip_full_evaluation=True, disqualifier="required_specialty", quote="would be overqualified",
        )])

        stats = evaluate_all(
            db, settings,
            client=AnthropicClient(api_key=None, model="test-model", client=full_eval_sdk),
            triage_client=AnthropicClient(api_key=None, model="test-haiku-model", client=triage_sdk),
        )

        assert full_eval_sdk.messages.call_count == 1
        assert (stats.triage_overruled, stats.triage_skipped, stats.sent_to_llm) == (1, 0, 1)

    def test_triage_screen_out_is_marked_not_assessed(self, db: Database, settings):
        company = seed_company(db, classification=CompanyClassification.PRODUCT)
        job = seed_job(db, settings, company)
        triage_client = AnthropicClient(
            api_key=None, model="test-haiku-model", client=FakeSdkClient([triage_tool_response(skip_full_evaluation=True)])
        )

        evaluate_all(db, settings, client=AnthropicClient(api_key=None, model="test-model", client=FakeSdkClient([])),
                     triage_client=triage_client)

        assert db.get_evaluation_for_job(job.id).scope_fit == ScopeFit.NOT_ASSESSED

    def test_no_skip_proceeds_to_the_full_evaluation(self, db: Database, settings):
        company = seed_company(db, classification=CompanyClassification.PRODUCT)
        seed_job(db, settings, company)

        full_eval_sdk = FakeSdkClient([tool_response(VALID_EVAL_INPUT)])
        full_eval_client = AnthropicClient(api_key=None, model="test-model", client=full_eval_sdk)
        triage_sdk = FakeSdkClient([triage_tool_response(skip_full_evaluation=False, reason="Real overlap.")])
        triage_client = AnthropicClient(api_key=None, model="test-haiku-model", client=triage_sdk)

        stats = evaluate_all(db, settings, client=full_eval_client, triage_client=triage_client)

        assert full_eval_sdk.messages.call_count == 1
        assert stats.triaged == 1
        assert stats.triage_skipped == 0
        assert stats.sent_to_llm == 1

    def test_failed_triage_call_fails_safe_to_full_evaluation(self, db: Database, settings):
        # A triage error must never be silently treated as "safe to skip" — it should behave as
        # if triage were never configured for that posting.
        company = seed_company(db, classification=CompanyClassification.PRODUCT)
        seed_job(db, settings, company)

        full_eval_sdk = FakeSdkClient([tool_response(VALID_EVAL_INPUT)])
        full_eval_client = AnthropicClient(api_key=None, model="test-model", client=full_eval_sdk)
        triage_sdk = FakeSdkClient([SimpleNamespace(content=[], usage=SimpleNamespace(input_tokens=0, output_tokens=0))])
        triage_client = AnthropicClient(api_key=None, model="test-haiku-model", client=triage_sdk)

        stats = evaluate_all(db, settings, client=full_eval_client, triage_client=triage_client)

        assert full_eval_sdk.messages.call_count == 1
        assert stats.sent_to_llm == 1
        assert stats.triage_skipped == 0

    def test_no_triage_client_skips_triage_entirely(self, db: Database, settings):
        # No triage_model configured (or no triage_client override) — behavior must be identical
        # to before triage existed.
        company = seed_company(db, classification=CompanyClassification.PRODUCT)
        seed_job(db, settings, company)

        full_eval_sdk = FakeSdkClient([tool_response(VALID_EVAL_INPUT)])
        full_eval_client = AnthropicClient(api_key=None, model="test-model", client=full_eval_sdk)

        stats = evaluate_all(db, settings, client=full_eval_client)

        assert stats.triaged == 0
        assert stats.triage_model is None
        assert stats.sent_to_llm == 1

    def test_limit_counts_triage_calls(self, db: Database, settings):
        company = seed_company(db, classification=CompanyClassification.PRODUCT)
        for i in range(2):
            seed_job(db, settings, company, source_job_id=str(i), title=f"Senior Backend Engineer {i}")

        full_eval_sdk = FakeSdkClient([])
        full_eval_client = AnthropicClient(api_key=None, model="test-model", client=full_eval_sdk)
        triage_sdk = FakeSdkClient([triage_tool_response(skip_full_evaluation=True)])
        triage_client = AnthropicClient(api_key=None, model="test-haiku-model", client=triage_sdk)

        stats = evaluate_all(db, settings, client=full_eval_client, triage_client=triage_client, limit=1)

        assert stats.triaged == 1
        assert full_eval_sdk.messages.call_count == 0

    def test_triage_stats_are_persisted_and_recoverable(self, db: Database, settings):
        company = seed_company(db, classification=CompanyClassification.PRODUCT)
        seed_job(db, settings, company)

        full_eval_client = AnthropicClient(api_key=None, model="test-model", client=FakeSdkClient([]))
        triage_client = AnthropicClient(
            api_key=None, model="test-haiku-model",
            client=FakeSdkClient([triage_tool_response(skip_full_evaluation=True, reason="No overlap.")]),
        )

        stats = evaluate_all(db, settings, client=full_eval_client, triage_client=triage_client)
        recovered = db.latest_evaluation_run()

        assert recovered is not None
        assert recovered.triaged == stats.triaged == 1
        assert recovered.triage_skipped == stats.triage_skipped == 1
        assert recovered.triage_model == stats.triage_model == "test-haiku-model"
        assert recovered.triage_input_tokens == stats.triage_input_tokens


def profile_text(settings) -> str:
    return render_profile_text(load_profile(settings.profile_path))


class TestRubricVersion:
    """The cache is keyed by description hash, so without a rubric stamp a fix to the prompts or
    profile would never reach a posting already judged (observed live: Posit's wrongful
    "not worth applying" would have stood forever after the rubric was corrected)."""

    def _evaluate_once(self, db, settings, input_dict=VALID_EVAL_INPUT):
        company = seed_company(db, classification=CompanyClassification.PRODUCT)
        job = seed_job(db, settings, company)
        client = AnthropicClient(api_key=None, model="test-model", client=FakeSdkClient([tool_response(input_dict)]))
        evaluate_all(db, settings, client=client)
        return job

    def _age(self, db, job):
        db.conn.execute("UPDATE evaluations SET rubric_version = 'older-rubric' WHERE job_id = ?", (job.id,))

    def test_full_evaluation_records_the_current_rubric(self, db: Database, settings):
        job = self._evaluate_once(db, settings)

        assert db.get_evaluation_for_job(job.id).rubric_version == evaluation_rubric_version(profile_text(settings))

    def test_triage_skip_records_the_current_rubric(self, db: Database, settings):
        company = seed_company(db, classification=CompanyClassification.PRODUCT)
        job = seed_job(db, settings, company)
        triage_client = AnthropicClient(
            api_key=None, model="test-haiku-model", client=FakeSdkClient([triage_tool_response(skip_full_evaluation=True)])
        )
        full_client = AnthropicClient(api_key=None, model="test-model", client=FakeSdkClient([]))

        evaluate_all(db, settings, client=full_client, triage_client=triage_client)

        assert db.get_evaluation_for_job(job.id).rubric_version == triage_rubric_version(profile_text(settings))

    def test_rubric_versions_change_with_the_profile_and_differ_by_kind(self):
        assert evaluation_rubric_version("profile A") != evaluation_rubric_version("profile B")
        assert evaluation_rubric_version("profile A") == evaluation_rubric_version("profile A")
        assert triage_rubric_version("profile A") != evaluation_rubric_version("profile A")

    def test_a_triage_prompt_change_leaves_full_evaluations_current(self, db: Database, settings, monkeypatch):
        # Observed live: a triage-only fix mid-refresh would otherwise have re-billed every full
        # evaluation just made, though none of them ever saw the triage prompt.
        self._evaluate_once(db, settings)
        monkeypatch.setattr(prompts, "TRIAGE_SYSTEM", prompts.TRIAGE_SYSTEM + " (revised)")
        sdk = FakeSdkClient([])

        stats = evaluate_all(
            db, settings, client=AnthropicClient(api_key=None, model="test-model", client=sdk), refresh_stale=True
        )

        assert sdk.messages.call_count == 0
        assert stats.cache_hits == 1

    def test_a_triage_prompt_change_makes_triage_screen_outs_stale(self, db: Database, settings, monkeypatch):
        company = seed_company(db, classification=CompanyClassification.PRODUCT)
        seed_job(db, settings, company)
        triage_client = AnthropicClient(
            api_key=None, model="test-haiku-model", client=FakeSdkClient([triage_tool_response(skip_full_evaluation=True)])
        )
        evaluate_all(db, settings, client=AnthropicClient(api_key=None, model="test-model", client=FakeSdkClient([])),
                     triage_client=triage_client)
        monkeypatch.setattr(prompts, "TRIAGE_SYSTEM", prompts.TRIAGE_SYSTEM + " (revised)")

        stats = evaluate_all(db, settings, refresh_stale=True)

        assert stats.unverified == 1
        assert stats.cache_hits == 0

    def test_verdict_from_an_older_rubric_is_reused_by_default(self, db: Database, settings):
        job = self._evaluate_once(db, settings)
        self._age(db, job)
        sdk = FakeSdkClient([])

        stats = evaluate_all(db, settings, client=AnthropicClient(api_key=None, model="test-model", client=sdk))

        assert sdk.messages.call_count == 0
        assert stats.cache_hits == 1

    def test_refresh_stale_re_screens_a_verdict_from_an_older_rubric(self, db: Database, settings):
        job = self._evaluate_once(db, settings)
        self._age(db, job)
        rejected = dict(VALID_EVAL_INPUT, verdict="reject", worth_applying=False, primary_rejection_reason="Requires Rust.")
        sdk = FakeSdkClient([tool_response(rejected)])

        stats = evaluate_all(
            db, settings, client=AnthropicClient(api_key=None, model="test-model", client=sdk), refresh_stale=True
        )

        assert sdk.messages.call_count == 1
        assert stats.sent_to_llm == 1
        refreshed = db.get_evaluation_for_job(job.id)
        assert refreshed.verdict == Verdict.REJECT
        assert refreshed.rubric_version == evaluation_rubric_version(profile_text(settings))

    def test_refresh_stale_leaves_current_verdicts_alone(self, db: Database, settings):
        self._evaluate_once(db, settings)
        sdk = FakeSdkClient([])

        stats = evaluate_all(
            db, settings, client=AnthropicClient(api_key=None, model="test-model", client=sdk), refresh_stale=True
        )

        assert sdk.messages.call_count == 0
        assert stats.cache_hits == 1

    def test_refresh_stale_dry_run_counts_stale_verdicts_without_touching_them(self, db: Database, settings):
        job = self._evaluate_once(db, settings)
        self._age(db, job)

        stats = evaluate_all(db, settings, refresh_stale=True)

        assert stats.unverified == 1
        assert stats.cache_hits == 0
        kept = db.get_evaluation_for_job(job.id)
        assert (kept.verdict, kept.rubric_version) == (Verdict.STRONG_MATCH, "older-rubric")
