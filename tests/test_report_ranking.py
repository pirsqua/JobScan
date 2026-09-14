"""Tests for the scope-calibration ranking/grouping logic in jobscan.reports.data, including
regression fixtures for the two real postings the ranking rework was built around.

TestRankingAndGrouping builds Evaluation objects directly (no LLM involved) to unit-test section
placement and within-section sort order precisely. TestRegressionFixtures runs the two real
postings (captured live from Doximity and RevenueCat) through the full evaluate_all() pipeline
with a mocked LLM response matching the scope-calibration spec's documented expected result, to
prove the whole pipeline — not just the sort function — places them correctly.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from jobscan.db import Database
from jobscan.evaluate import evaluate_all
from jobscan.llm.client import AnthropicClient
from jobscan.llm.schemas import JOB_EVALUATION_TOOL_NAME
from jobscan.models import (
    AtsType,
    ClassificationSource,
    Company,
    CompanyClassification,
    Evaluation,
    EvidenceClassification,
    RawPosting,
    RequirementEvidence,
    RequirementImportance,
    ScopeFit,
    SpecialistTenureAssessment,
    SpecialistTenureClassification,
    Verdict,
)
from jobscan.normalize import normalize_posting
from jobscan.reports.data import assemble_report_data
from jobscan.reports.markdown_report import render_markdown

NOW = datetime.now(timezone.utc)


def seed_company(db: Database, name: str = "Acme", classification=CompanyClassification.PRODUCT) -> Company:
    company = Company(
        name=name, domain=f"{name.lower().replace(' ', '')}.example.com", careers_url=None,
        ats_type=AtsType.GREENHOUSE, board_id=name.lower().replace(" ", ""),
        classification=classification, classification_source=ClassificationSource.SEED,
    )
    company.id = db.upsert_company(company)
    return company


def seed_job(
    db: Database, settings, company: Company, source_job_id: str, title: str, description: str,
    salary_min: int | None = None, salary_max: int | None = None,
):
    salary_line = f"<p>Base salary range: ${salary_min:,} - ${salary_max:,} per year.</p>" if salary_min else ""
    raw = RawPosting(
        source_job_id=source_job_id, title=title, location_raw="Remote - US",
        employment_type_raw="Full-time",
        description_html=f"<p>{description}</p>{salary_line}", description_text=None,
        posting_url=f"https://acme.example.com/jobs/{source_job_id}", apply_url=None,
    )
    job = normalize_posting(raw, company, settings)
    job_id, _, _ = db.upsert_job(job)
    return db.get_job(job_id)


def make_evaluation(
    job_id: int, verdict: Verdict, scope_fit: ScopeFit, evidence_coverage_percent: int = 50,
    central_directly_demonstrated: int = 0, required_gaps: list[str] | None = None,
    growth_dimensions: list[str] | None = None, hidden_staff_signals: list[str] | None = None,
) -> Evaluation:
    requirement_evidence = [
        RequirementEvidence(
            requirement=f"central requirement {i}", importance=RequirementImportance.CENTRAL,
            evidence_classification=EvidenceClassification.DIRECTLY_DEMONSTRATED,
            candidate_evidence="x", posting_evidence="y",
        )
        for i in range(central_directly_demonstrated)
    ]
    return Evaluation(
        job_id=job_id, description_hash=f"hash-{job_id}", verdict=verdict, confidence=0.7,
        scope_fit=scope_fit, evidence_coverage_percent=evidence_coverage_percent,
        specialist_tenure_assessment=SpecialistTenureAssessment(
            classification=SpecialistTenureClassification.NOT_APPLICABLE, specialty="", explanation="",
        ),
        requirement_evidence=requirement_evidence, growth_dimensions=growth_dimensions or [],
        hidden_staff_signals=hidden_staff_signals or [], compensation_assessment="ok",
        remote_employment_verification="ok", required_matches=[], required_gaps=required_gaps or [],
        preferred_only_gaps=[], minor_caveats=[], evidence=[], credibility_assessment="ok",
        why_this_is_or_is_not_gettable="ok", is_product_company=True, primary_rejection_reason=None,
        model_name="test-model", created_at=NOW,
    )


class TestRankingAndGrouping:
    def test_strong_match_at_level_is_best_bet(self, db: Database, settings):
        company = seed_company(db)
        job = seed_job(db, settings, company, "1", "Senior Backend Engineer", "x", 190000, 230000)
        db.save_evaluation(make_evaluation(job.id, Verdict.STRONG_MATCH, ScopeFit.AT_LEVEL))

        data = assemble_report_data(db, settings)

        assert len(data.best_bets) == 1
        assert len(data.growth_bets) == 0
        assert len(data.attractive_stretches) == 0

    def test_plausible_match_at_level_is_best_bet(self, db: Database, settings):
        company = seed_company(db)
        job = seed_job(db, settings, company, "1", "Senior Backend Engineer", "x", 190000, 230000)
        db.save_evaluation(make_evaluation(job.id, Verdict.PLAUSIBLE_MATCH, ScopeFit.AT_LEVEL))

        data = assemble_report_data(db, settings)

        assert len(data.best_bets) == 1

    def test_plausible_match_one_step_up_is_growth_bet(self, db: Database, settings):
        company = seed_company(db)
        job = seed_job(db, settings, company, "1", "Senior Backend Engineer", "x", 190000, 230000)
        db.save_evaluation(make_evaluation(job.id, Verdict.PLAUSIBLE_MATCH, ScopeFit.ONE_STEP_UP))

        data = assemble_report_data(db, settings)

        assert len(data.growth_bets) == 1
        assert len(data.best_bets) == 0

    def test_strong_match_one_step_up_is_not_a_best_bet(self, db: Database, settings):
        # strong_match is defined as requiring at_level; if the model ever produces this
        # combination anyway, it must not be silently treated as a confident Best Bet.
        company = seed_company(db)
        job = seed_job(db, settings, company, "1", "Senior Backend Engineer", "x", 190000, 230000)
        db.save_evaluation(make_evaluation(job.id, Verdict.STRONG_MATCH, ScopeFit.ONE_STEP_UP))

        data = assemble_report_data(db, settings)

        assert len(data.best_bets) == 0

    def test_borderline_one_step_up_is_attractive_stretch(self, db: Database, settings):
        company = seed_company(db)
        job = seed_job(db, settings, company, "1", "Senior Backend Engineer", "x", 190000, 230000)
        db.save_evaluation(make_evaluation(job.id, Verdict.BORDERLINE, ScopeFit.ONE_STEP_UP))

        data = assemble_report_data(db, settings)

        assert len(data.attractive_stretches) == 1

    def test_borderline_two_plus_steps_up_is_attractive_stretch(self, db: Database, settings):
        company = seed_company(db)
        job = seed_job(db, settings, company, "1", "Senior Backend Engineer", "x", 190000, 230000)
        db.save_evaluation(make_evaluation(job.id, Verdict.BORDERLINE, ScopeFit.TWO_PLUS_STEPS_UP))

        data = assemble_report_data(db, settings)

        assert len(data.attractive_stretches) == 1

    def test_reject_is_not_in_any_positive_section(self, db: Database, settings):
        company = seed_company(db)
        job = seed_job(db, settings, company, "1", "Senior Backend Engineer", "x", 190000, 230000)
        db.save_evaluation(make_evaluation(job.id, Verdict.REJECT, ScopeFit.TWO_PLUS_STEPS_UP))

        data = assemble_report_data(db, settings)

        assert len(data.best_bets) == 0
        assert len(data.growth_bets) == 0
        assert len(data.attractive_stretches) == 0

    def test_below_level_strong_match_is_excluded_from_best_bets(self, db: Database, settings):
        # Underusing experience isn't what "Best Bet" means, even with a strong verdict.
        company = seed_company(db)
        job = seed_job(db, settings, company, "1", "Backend Engineer II", "x", 190000, 230000)
        db.save_evaluation(make_evaluation(job.id, Verdict.STRONG_MATCH, ScopeFit.BELOW_LEVEL))

        data = assemble_report_data(db, settings)

        assert len(data.best_bets) == 0

    def test_sorts_by_evidence_coverage_percent_within_section(self, db: Database, settings):
        company = seed_company(db)
        low = seed_job(db, settings, company, "1", "Role A", "x", 190000, 230000)
        high = seed_job(db, settings, company, "2", "Role B", "x", 190000, 230000)
        db.save_evaluation(make_evaluation(low.id, Verdict.STRONG_MATCH, ScopeFit.AT_LEVEL, evidence_coverage_percent=60))
        db.save_evaluation(make_evaluation(high.id, Verdict.STRONG_MATCH, ScopeFit.AT_LEVEL, evidence_coverage_percent=95))

        data = assemble_report_data(db, settings)

        assert [r.job.title for r in data.best_bets] == ["Role B", "Role A"]

    def test_compensation_does_not_override_evidence_coverage(self, db: Database, settings):
        # The core anti-goal from the spec: a much higher salary must not out-rank a role the
        # candidate is more credibly positioned to win.
        company = seed_company(db)
        high_pay_weak_fit = seed_job(db, settings, company, "1", "Elite Co Role", "x", 260000, 300000)
        modest_pay_strong_fit = seed_job(db, settings, company, "2", "Solid Role", "x", 175000, 210000)
        db.save_evaluation(
            make_evaluation(high_pay_weak_fit.id, Verdict.STRONG_MATCH, ScopeFit.AT_LEVEL, evidence_coverage_percent=65)
        )
        db.save_evaluation(
            make_evaluation(modest_pay_strong_fit.id, Verdict.STRONG_MATCH, ScopeFit.AT_LEVEL, evidence_coverage_percent=92)
        )

        data = assemble_report_data(db, settings)

        assert [r.job.title for r in data.best_bets] == ["Solid Role", "Elite Co Role"]

    def test_compensation_breaks_ties_between_equally_credible_roles(self, db: Database, settings):
        company = seed_company(db)
        lower_pay = seed_job(db, settings, company, "1", "Role Low Pay", "x", 175000, 175000)
        higher_pay = seed_job(db, settings, company, "2", "Role High Pay", "x", 200000, 200000)
        db.save_evaluation(make_evaluation(lower_pay.id, Verdict.STRONG_MATCH, ScopeFit.AT_LEVEL, evidence_coverage_percent=85))
        db.save_evaluation(make_evaluation(higher_pay.id, Verdict.STRONG_MATCH, ScopeFit.AT_LEVEL, evidence_coverage_percent=85))

        data = assemble_report_data(db, settings)

        assert [r.job.title for r in data.best_bets] == ["Role High Pay", "Role Low Pay"]

    def test_sorts_by_required_gaps_count(self, db: Database, settings):
        company = seed_company(db)
        many_gaps = seed_job(db, settings, company, "1", "Role Gaps", "x", 190000, 230000)
        few_gaps = seed_job(db, settings, company, "2", "Role Clean", "x", 190000, 230000)
        db.save_evaluation(
            make_evaluation(many_gaps.id, Verdict.BORDERLINE, ScopeFit.ONE_STEP_UP, evidence_coverage_percent=70, required_gaps=["a", "b", "c"])
        )
        db.save_evaluation(
            make_evaluation(few_gaps.id, Verdict.BORDERLINE, ScopeFit.ONE_STEP_UP, evidence_coverage_percent=70, required_gaps=[])
        )

        data = assemble_report_data(db, settings)

        assert [r.job.title for r in data.attractive_stretches] == ["Role Clean", "Role Gaps"]

    def test_growth_bets_never_appear_in_best_bets_regardless_of_salary(self, db: Database, settings):
        company = seed_company(db)
        best = seed_job(db, settings, company, "1", "At Level Role", "x", 175000, 175000)
        growth = seed_job(db, settings, company, "2", "One Step Up Role", "x", 260000, 300000)
        db.save_evaluation(make_evaluation(best.id, Verdict.STRONG_MATCH, ScopeFit.AT_LEVEL, evidence_coverage_percent=80))
        db.save_evaluation(make_evaluation(growth.id, Verdict.PLAUSIBLE_MATCH, ScopeFit.ONE_STEP_UP, evidence_coverage_percent=95))

        data = assemble_report_data(db, settings)

        assert [r.job.title for r in data.best_bets] == ["At Level Role"]
        assert [r.job.title for r in data.growth_bets] == ["One Step Up Role"]


DOXIMITY_DESCRIPTION = (
    "About Doximity: Doximity is the leading clinical AI company with the largest network of "
    "U.S. clinicians. About the Role: Foremost a software engineer. You exemplify high code "
    "quality and guide others, and have at least 5 years of industry experience as a data "
    "engineer or similar role. You are an expert in Python and possess fluency in SQL and have "
    "developed maintainable data pipelines with each. Hands-on with architecting solutions. You "
    "have designed several data pipelines from start to finish. Choosing appropriate "
    "technologies, building, testing and rollout. Experience with asynchronous systems. You are "
    "comfortable identifying and applying APIs, streaming, multithreading, and other "
    "asynchronous approaches to the right problems. ML/AI-aware. You have working knowledge of "
    "how ML/AI applications are deployed and monitored. Here's How You Will Make an Impact: "
    "Establish data architecture processes and practices that can be scheduled, automated, "
    "replicated and serve as standards for other teams to leverage. Location: This role is "
    "remote in the United States, Canada, or some countries in South America. Compensation: The "
    "anticipated total compensation for this role is $165,000 - $221,000, depending on factors "
    "such as experience, skills, location, and internal equity. Our total compensation package "
    "may include base salary, equity, and annual bonus."
)

DOXIMITY_EXPECTED_RESPONSE = {
    "verdict": "borderline",
    "confidence": 0.6,
    "scope_fit": "two_plus_steps_up",
    "evidence_coverage_percent": 55,
    "specialist_tenure_assessment": {
        "classification": "adjacent",
        "specialty": "data engineering",
        "explanation": "Candidate has one demonstrated end-to-end Databricks pipeline but not "
        "5+ years specifically as a dedicated data engineer designing several pipelines from scratch.",
    },
    "requirement_evidence": [
        {
            "requirement": "5+ years as a data engineer or similar",
            "importance": "central",
            "evidence_classification": "credibly_transferable",
            "candidate_evidence": "~15 years general software engineering including one Databricks pipeline.",
            "posting_evidence": "have at least 5 years of industry experience as a data engineer or similar role",
        },
        {
            "requirement": "Designed several data pipelines from start to finish",
            "importance": "central",
            "evidence_classification": "weakly_inferred",
            "candidate_evidence": "One clearly demonstrated end-to-end pipeline, not several.",
            "posting_evidence": "You have designed several data pipelines from start to finish",
        },
    ],
    "growth_dimensions": [
        "Specialized data-engineering tenure (5+ years dedicated)",
        "Repeated pipeline design across several greenfield efforts",
        "Establishing org-wide data architecture standards",
    ],
    "hidden_staff_signals": [
        "Establish data architecture processes and standards for other teams to leverage",
    ],
    "is_product_company": True,
    "compensation_assessment": "Published as total compensation ($165,000-$221,000), not a "
    "clear base salary figure. Treat as compensation-unverified under the strict base-salary "
    "requirement despite the midpoint nominally clearing $170,000.",
    "remote_employment_verification": "Remote in the United States (also Canada/South America), "
    "no state-specific exclusion for Washington noted.",
    "required_matches": ["Python", "SQL", "automated testing", "maintainable pipeline experience"],
    "required_gaps": [
        "5+ years specifically as a dedicated data engineer",
        "Several independently designed data pipelines (only one demonstrated)",
        "Establishing reusable data-architecture standards across teams",
    ],
    "preferred_only_gaps": ["ML/AI deployment awareness"],
    "minor_caveats": ["Compensation published as total comp, not base"],
    "evidence": ["\"have at least 5 years of industry experience as a data engineer\"", "\"designed several data pipelines from start to finish\""],
    "credibility_assessment": "Attractive technical overlap but the role expects specialized, "
    "repeated data-engineering tenure and org-wide standard-setting the candidate hasn't demonstrated.",
    "why_this_is_or_is_not_gettable": "A genuine stretch: strong Python/SQL/pipeline overlap "
    "makes an application reasonable, but the specialized-tenure and repeated-pipeline "
    "expectations mean this is not a confident near-term win.",
    "primary_rejection_reason": "Requires specialized, repeated data-engineering tenure "
    "(5+ years dedicated, several pipelines from scratch) beyond the one demonstrated pipeline.",
}

REVENUECAT_DESCRIPTION = (
    "RevenueCat gives app businesses the infrastructure and tools to build, run, and improve "
    "their monetization. We process $16B+ in annual purchase volume. We're a remote-first team "
    "of 150+ people across 25+ countries. The systems you'll build here manage billions of "
    "dollars and touch hundreds of millions of end users. THE ROLE: design, build, ship, and own "
    "end-to-end product features used by thousands of developers and hundreds of millions of "
    "end-users. ABOUT YOU: You have 8+ years of experience as a backend engineer designing "
    "complex, fast-growing systems that didn't exist before you. You can write advanced SQL "
    "statements. You're a builder, not an implementer: scoping, prioritizing, and driving "
    "rollout in partnership with our Product team. WITHIN THE FIRST 12 MONTHS: Have your own "
    "initiatives for improving our products. Push the org to improve reliability, scalability, "
    "and performance. WHAT WE OFFER: Competitive equity in a fast-growing Series C startup. "
    "Compensation: $230K, Offers Equity."
)

REVENUECAT_EXPECTED_RESPONSE = {
    "verdict": "borderline",
    "confidence": 0.55,
    "scope_fit": "two_plus_steps_up",
    "evidence_coverage_percent": 50,
    "specialist_tenure_assessment": {
        "classification": "not_applicable",
        "specialty": "",
        "explanation": "Not a specialty-tenure question — this is about general backend scale/ownership depth.",
    },
    "requirement_evidence": [
        {
            "requirement": "8+ years building complex, fast-growing systems from scratch",
            "importance": "central",
            "evidence_classification": "weakly_inferred",
            "candidate_evidence": "~15 years general backend experience, not specifically repeated greenfield systems at this scale.",
            "posting_evidence": "8+ years of experience as a backend engineer designing complex, fast-growing systems that didn't exist before you",
        },
    ],
    "growth_dimensions": [
        "Systems operating at billions-of-dollars / hundreds-of-millions-of-users scale",
        "Repeated zero-to-one ownership of systems that didn't exist before",
        "Driving product rollout and org-wide reliability push, not just implementation",
    ],
    "hidden_staff_signals": [
        "Small, highly selective remote company with a compressed seniority ladder",
        "$230,000 salary paired with extremely broad ownership (problem scoping through long-term maintenance)",
        "Push the org to improve reliability, scalability, and performance",
        "Systems manage billions of dollars and touch hundreds of millions of end users",
    ],
    "is_product_company": True,
    "compensation_assessment": "Published as $230K flat, well above the $170,000 minimum — "
    "compensation is not the blocker here, but per the compensation rule this does not "
    "compensate for the scope mismatch.",
    "remote_employment_verification": "Remote-first team, described as 'Americas' — plausibly open to Washington.",
    "required_matches": ["SQL", "backend ownership", "production feature delivery", "mentorship"],
    "required_gaps": [
        "8+ years designing complex systems from scratch at this scale",
        "Track record operating extremely large distributed systems (billions of dollars, hundreds of millions of users)",
        "Repeated zero-to-one ownership",
    ],
    "preferred_only_gaps": ["SDK-building experience", "mobile in-app-purchase domain knowledge"],
    "minor_caveats": [],
    "evidence": ["\"systems you'll build here manage billions of dollars and touch hundreds of millions of end users\""],
    "credibility_assessment": "Strong general backend/SQL overlap, but the combination of scale, "
    "repeated zero-to-one ownership, and hidden staff-level signals make this a stretch, not an at-level match.",
    "why_this_is_or_is_not_gettable": "Aspirational: excellent pay and genuine backend overlap, "
    "but the compressed ladder and billions-of-dollars scale mean this is a two-plus-step stretch, not a credible near-term win.",
    "primary_rejection_reason": "Combines extreme production scale, repeated zero-to-one system "
    "ownership, and hidden staff-level scope at a small, highly selective company — two or more "
    "unproven dimensions beyond demonstrated experience.",
}

SYNTHETIC_AT_LEVEL_DESCRIPTION = (
    "We are a mid-size, established fintech product company with dedicated Staff and Principal "
    "Engineer tracks above Senior. We're hiring a Senior Backend Engineer to join our payments "
    "platform team. Requirements: 5+ years of professional software engineering experience. "
    "Strong experience with Python or C# and SQL. Experience building and maintaining REST APIs "
    "and integrating with third-party services. Experience with a major cloud provider (Azure, "
    "AWS, or GCP). You will own and improve existing services within an established team, "
    "participate in architecture discussions alongside our staff engineers, write automated "
    "tests, deploy through CI/CD, and take part in production on-call and troubleshooting. "
    "Mentoring junior engineers and participating in code review is expected. The base salary "
    "range for this role is $175,000 - $210,000 per year."
)

SYNTHETIC_AT_LEVEL_EXPECTED_RESPONSE = {
    "verdict": "strong_match",
    "confidence": 0.85,
    "scope_fit": "at_level",
    "evidence_coverage_percent": 88,
    "specialist_tenure_assessment": {"classification": "not_applicable", "specialty": "", "explanation": "No specialty tenure requirement beyond general backend experience."},
    "requirement_evidence": [
        {
            "requirement": "5+ years professional software engineering",
            "importance": "central",
            "evidence_classification": "directly_demonstrated",
            "candidate_evidence": "~15 years of general software-engineering experience.",
            "posting_evidence": "5+ years of professional software engineering experience",
        },
        {
            "requirement": "Python or C# and SQL",
            "importance": "central",
            "evidence_classification": "directly_demonstrated",
            "candidate_evidence": "Strong C#, Python and SQL Server experience.",
            "posting_evidence": "Strong experience with Python or C# and SQL",
        },
        {
            "requirement": "REST APIs and third-party integrations",
            "importance": "central",
            "evidence_classification": "directly_demonstrated",
            "candidate_evidence": "REST API design and third-party integration experience.",
            "posting_evidence": "Experience building and maintaining REST APIs and integrating with third-party services",
        },
        {
            "requirement": "Major cloud provider",
            "importance": "central",
            "evidence_classification": "directly_demonstrated",
            "candidate_evidence": "Strong Azure experience.",
            "posting_evidence": "Experience with a major cloud provider (Azure, AWS, or GCP)",
        },
    ],
    "growth_dimensions": [],
    "hidden_staff_signals": [],
    "is_product_company": True,
    "compensation_assessment": "Published base salary range $175,000-$210,000, midpoint "
    "$192,500, comfortably clears the $170,000 minimum.",
    "remote_employment_verification": "No location restriction noted; treated as open to Washington-based remote.",
    "required_matches": ["Python/C#", "SQL", "REST APIs", "Azure", "testing/CI/CD", "production support", "mentoring"],
    "required_gaps": [],
    "preferred_only_gaps": [],
    "minor_caveats": [],
    "evidence": ["\"own and improve existing services within an established team\"", "\"participate in architecture discussions alongside our staff engineers\""],
    "credibility_assessment": "Strong, direct match across every central requirement at ordinary senior scope.",
    "why_this_is_or_is_not_gettable": "Highly gettable: every central requirement is directly "
    "demonstrated, the company has an established Staff/Principal track above Senior (making the "
    "title credible), and the role is collaborative rather than sole-ownership scope.",
    "primary_rejection_reason": None,
}


def _tool_response(input_dict: dict, input_tokens: int = 500, output_tokens: int = 300):
    block = SimpleNamespace(type="tool_use", name=JOB_EVALUATION_TOOL_NAME, input=input_dict)
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(content=[block], usage=usage)


class _FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)

    def create(self, **kwargs):
        return self._responses.pop(0)


class _FakeSdkClient:
    def __init__(self, responses):
        self.messages = _FakeMessages(responses)


class TestRegressionFixtures:
    def test_doximity_lands_in_attractive_stretches_not_best_or_growth_bets(self, db: Database, settings):
        company = seed_company(db, "Doximity")
        seed_job(db, settings, company, "8155070", "Senior Software Engineer, Data", DOXIMITY_DESCRIPTION, 165000, 221000)
        client = AnthropicClient(api_key=None, model="test-model", client=_FakeSdkClient([_tool_response(DOXIMITY_EXPECTED_RESPONSE)]))

        evaluate_all(db, settings, client=client)
        data = assemble_report_data(db, settings)

        assert len(data.best_bets) == 0
        assert len(data.growth_bets) == 0
        assert len(data.attractive_stretches) == 1
        row = data.attractive_stretches[0]
        assert row.evaluation.scope_fit == ScopeFit.TWO_PLUS_STEPS_UP
        assert row.evaluation.verdict == Verdict.BORDERLINE
        assert row.evaluation.specialist_tenure_assessment.classification == SpecialistTenureClassification.ADJACENT

    def test_revenuecat_lands_in_attractive_stretches_with_hidden_staff_signals(self, db: Database, settings):
        company = seed_company(db, "RevenueCat")
        seed_job(
            db, settings, company, "c6d43e21", "Senior Backend Engineer", REVENUECAT_DESCRIPTION,
            230000, 230000,
        )
        client = AnthropicClient(api_key=None, model="test-model", client=_FakeSdkClient([_tool_response(REVENUECAT_EXPECTED_RESPONSE)]))

        evaluate_all(db, settings, client=client)
        data = assemble_report_data(db, settings)

        assert len(data.best_bets) == 0
        assert len(data.attractive_stretches) == 1
        row = data.attractive_stretches[0]
        assert row.evaluation.scope_fit == ScopeFit.TWO_PLUS_STEPS_UP
        assert len(row.evaluation.hidden_staff_signals) >= 2
        # The $230K salary must not have promoted this into Best Bets.
        assert row not in data.best_bets

    def test_synthetic_at_level_fixture_lands_in_best_bets(self, db: Database, settings):
        company = seed_company(db, "GoodFit Fintech")
        seed_job(db, settings, company, "1", "Senior Backend Engineer", SYNTHETIC_AT_LEVEL_DESCRIPTION, 175000, 210000)
        client = AnthropicClient(api_key=None, model="test-model", client=_FakeSdkClient([_tool_response(SYNTHETIC_AT_LEVEL_EXPECTED_RESPONSE)]))

        evaluate_all(db, settings, client=client)
        data = assemble_report_data(db, settings)

        assert len(data.best_bets) == 1
        assert data.best_bets[0].evaluation.scope_fit == ScopeFit.AT_LEVEL
        assert data.best_bets[0].evaluation.verdict == Verdict.STRONG_MATCH

    def test_synthetic_fixture_ranks_in_best_bets_above_both_higher_paying_stretches(self, db: Database, settings):
        """End-to-end proof of the spec's central anti-goal: RevenueCat ($230K) and Doximity
        must not outrank a genuinely at-level, lower-paying role — because they're not even in
        the same report section."""
        doximity = seed_company(db, "Doximity")
        seed_job(db, settings, doximity, "8155070", "Senior Software Engineer, Data", DOXIMITY_DESCRIPTION, 165000, 221000)
        revenuecat = seed_company(db, "RevenueCat")
        seed_job(db, settings, revenuecat, "c6d43e21", "Senior Backend Engineer", REVENUECAT_DESCRIPTION, 230000, 230000)
        goodfit = seed_company(db, "GoodFit Fintech")
        seed_job(db, settings, goodfit, "1", "Senior Backend Engineer", SYNTHETIC_AT_LEVEL_DESCRIPTION, 175000, 210000)

        client = AnthropicClient(
            api_key=None, model="test-model",
            client=_FakeSdkClient([
                _tool_response(DOXIMITY_EXPECTED_RESPONSE),
                _tool_response(REVENUECAT_EXPECTED_RESPONSE),
                _tool_response(SYNTHETIC_AT_LEVEL_EXPECTED_RESPONSE),
            ]),
        )
        evaluate_all(db, settings, client=client)
        data = assemble_report_data(db, settings)

        assert [r.company.name for r in data.best_bets] == ["GoodFit Fintech"]
        assert {r.company.name for r in data.attractive_stretches} == {"Doximity", "RevenueCat"}

        markdown = render_markdown(data)
        best_bets_index = markdown.index("## Best Bets")
        stretches_index = markdown.index("## Attractive Stretches")
        goodfit_index = markdown.index("GoodFit Fintech")
        revenuecat_index = markdown.index("RevenueCat")
        assert best_bets_index < stretches_index
        assert best_bets_index < goodfit_index < stretches_index
        assert stretches_index < revenuecat_index
