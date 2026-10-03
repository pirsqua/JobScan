from __future__ import annotations

from datetime import datetime, timezone

from jobscan.config import Settings
from jobscan.db import Database
from jobscan.models import (
    AtsType,
    ClassificationSource,
    Company,
    CompanyClassification,
    Evaluation,
    JobStatus,
    ManualOverride,
    RawPosting,
    RemoteScope,
    SalarySource,
    ScopeFit,
    SpecialistTenureAssessment,
    SpecialistTenureClassification,
    Verdict,
    WorkplaceType,
)
from jobscan.normalize import compute_description_hash, normalize_posting


def make_raw(source_job_id="1", title="Senior Backend Engineer", description="We build our own SaaS product. $180,000 - $220,000/year.", **overrides):
    defaults = dict(
        source_job_id=source_job_id,
        title=title,
        location_raw="Remote - US",
        employment_type_raw="Full-time",
        description_html=f"<p>{description}</p>",
        description_text=None,
        posting_url="https://acme.example.com/jobs/1",
        apply_url="https://acme.example.com/jobs/1/apply",
    )
    defaults.update(overrides)
    return RawPosting(**defaults)


class TestDescriptionHash:
    def test_hash_stable_for_identical_text(self):
        assert compute_description_hash("Hello  world") == compute_description_hash("hello world")

    def test_hash_changes_for_different_text(self):
        assert compute_description_hash("Hello world") != compute_description_hash("Goodbye world")


class TestUpsertCompany:
    def test_insert_then_update_by_ats_and_board(self, db: Database, sample_company: Company):
        first_id = db.upsert_company(sample_company)
        updated = Company(**{**sample_company.__dict__, "name": "Acme Corporation"})
        second_id = db.upsert_company(updated)
        assert first_id == second_id
        assert db.get_company(first_id).name == "Acme Corporation"

    def test_reimport_applies_a_real_classification_to_an_existing_unknown_company(
        self, db: Database, sample_company: Company
    ):
        # A registry re-curated with a known classification should actually take effect for a
        # company that already exists in the database as "unknown" — not be silently dropped.
        company_id = db.upsert_company(sample_company)
        assert db.get_company(company_id).classification == CompanyClassification.UNKNOWN

        curated = Company(
            **{
                **sample_company.__dict__,
                "classification": CompanyClassification.PRODUCT,
                "classification_source": ClassificationSource.SEED,
            }
        )
        db.upsert_company(curated)

        updated = db.get_company(company_id)
        assert updated.classification == CompanyClassification.PRODUCT
        assert updated.classification_source == ClassificationSource.SEED

    def test_reimport_does_not_overwrite_a_manual_classification(self, db: Database, sample_company: Company):
        company_id = db.upsert_company(sample_company)
        db.set_company_classification(company_id, CompanyClassification.CONSULTING, ClassificationSource.MANUAL, None, "confirmed manually")

        reimported = Company(
            **{
                **sample_company.__dict__,
                "classification": CompanyClassification.PRODUCT,
                "classification_source": ClassificationSource.SEED,
            }
        )
        db.upsert_company(reimported)

        updated = db.get_company(company_id)
        assert updated.classification == CompanyClassification.CONSULTING
        assert updated.classification_source == ClassificationSource.MANUAL

    def test_reimport_with_unknown_classification_does_not_clear_existing_one(
        self, db: Database, sample_company: Company
    ):
        company_id = db.upsert_company(sample_company)
        db.set_company_classification(company_id, CompanyClassification.PRODUCT, ClassificationSource.LLM, 0.9, "evidence")

        reimported = Company(**{**sample_company.__dict__, "classification": CompanyClassification.UNKNOWN})
        db.upsert_company(reimported)

        updated = db.get_company(company_id)
        assert updated.classification == CompanyClassification.PRODUCT
        assert updated.classification_source == ClassificationSource.LLM


class TestUpsertJob:
    def test_workplace_type_round_trips_and_reaches_the_llm_prompt(self, db: Database, settings: Settings, sample_company: Company):
        # A remote role that lists a city stays UNKNOWN (ambiguous region) and goes to the LLM,
        # which must see the ATS's "remote" field — otherwise "affirmative remote evidence" is
        # invisible to it and a genuinely remote role whose text never says so gets rejected.
        from jobscan.llm.prompts import build_job_eval_user_message, build_triage_user_message

        sample_company.id = db.upsert_company(sample_company)
        raw = make_raw(location_raw="Seattle", workplace_type=WorkplaceType.REMOTE)
        job_id, _, _ = db.upsert_job(normalize_posting(raw, sample_company, settings))
        stored = db.get_job(job_id)
        assert stored.workplace_type == WorkplaceType.REMOTE
        assert stored.remote_scope == RemoteScope.UNKNOWN
        for message in (build_job_eval_user_message(sample_company, stored), build_triage_user_message(sample_company, stored)):
            assert "Workplace type (the ATS's own structured field): remote" in message

    def test_new_job_is_new(self, db: Database, settings: Settings, sample_company: Company):
        sample_company.id = db.upsert_company(sample_company)
        job = normalize_posting(make_raw(), sample_company, settings)
        job_id, is_new, changed = db.upsert_job(job)
        assert is_new
        assert not changed
        assert db.get_job(job_id).status == JobStatus.ACTIVE

    def test_reseeing_unchanged_job_is_not_new_or_changed(self, db: Database, settings: Settings, sample_company: Company):
        sample_company.id = db.upsert_company(sample_company)
        raw = make_raw()
        job1 = normalize_posting(raw, sample_company, settings)
        db.upsert_job(job1)
        job2 = normalize_posting(raw, sample_company, settings)
        _, is_new, changed = db.upsert_job(job2)
        assert not is_new
        assert not changed

    def test_changed_description_is_flagged_and_snapshotted(self, db: Database, settings: Settings, sample_company: Company):
        sample_company.id = db.upsert_company(sample_company)
        job1 = normalize_posting(make_raw(description="Original description. $180,000-$220,000/year."), sample_company, settings)
        job_id, _, _ = db.upsert_job(job1)

        job2 = normalize_posting(make_raw(description="Updated description with new requirements. $180,000-$220,000/year."), sample_company, settings)
        _, is_new, changed = db.upsert_job(job2)
        assert not is_new
        assert changed

        cur = db.conn.execute("SELECT COUNT(*) FROM job_snapshots WHERE job_id=?", (job_id,))
        (count,) = cur.fetchone()
        assert count == 1

    def test_duplicate_source_job_id_upserts_not_duplicates(self, db: Database, settings: Settings, sample_company: Company):
        sample_company.id = db.upsert_company(sample_company)
        raw = make_raw()
        db.upsert_job(normalize_posting(raw, sample_company, settings))
        db.upsert_job(normalize_posting(raw, sample_company, settings))
        assert db.count_jobs() == 1

    def test_closed_job_reopened_on_reappearance(self, db: Database, settings: Settings, sample_company: Company):
        sample_company.id = db.upsert_company(sample_company)
        raw = make_raw()
        job_id, _, _ = db.upsert_job(normalize_posting(raw, sample_company, settings))
        db.close_missing_jobs(sample_company.id, seen_source_job_ids=set())
        assert db.get_job(job_id).status == JobStatus.CLOSED

        db.upsert_job(normalize_posting(raw, sample_company, settings))
        assert db.get_job(job_id).status == JobStatus.ACTIVE
        assert db.get_job(job_id).closed_at is None


class TestCloseMissingJobs:
    def test_jobs_not_seen_in_crawl_are_closed(self, db: Database, settings: Settings, sample_company: Company):
        sample_company.id = db.upsert_company(sample_company)
        raw1 = make_raw(source_job_id="1")
        raw2 = make_raw(source_job_id="2")
        db.upsert_job(normalize_posting(raw1, sample_company, settings))
        job2_id, _, _ = db.upsert_job(normalize_posting(raw2, sample_company, settings))

        closed = db.close_missing_jobs(sample_company.id, seen_source_job_ids={"1"})
        assert closed == 1
        assert db.get_job(job2_id).status == JobStatus.CLOSED
        assert len(db.get_active_jobs(sample_company.id)) == 1

    def test_active_jobs_still_present_remain_active(self, db: Database, settings: Settings, sample_company: Company):
        sample_company.id = db.upsert_company(sample_company)
        raw = make_raw(source_job_id="1")
        job_id, _, _ = db.upsert_job(normalize_posting(raw, sample_company, settings))
        db.close_missing_jobs(sample_company.id, seen_source_job_ids={"1"})
        assert db.get_job(job_id).status == JobStatus.ACTIVE


class TestManualOverrides:
    def test_set_and_get_job_override(self, db: Database):
        db.set_manual_override(
            ManualOverride(
                entity_type="job", entity_id=42, field="verdict", value="strong_match",
                reason="I know this team", created_at=datetime.now(timezone.utc),
            )
        )
        overrides = db.get_manual_overrides("job", 42)
        assert overrides["verdict"] == "strong_match"

    def test_override_upsert_replaces_value(self, db: Database):
        now = datetime.now(timezone.utc)
        db.set_manual_override(ManualOverride("job", 1, "verdict", "reject", None, now))
        db.set_manual_override(ManualOverride("job", 1, "verdict", "strong_match", "changed my mind", now))
        overrides = db.get_manual_overrides("job", 1)
        assert overrides["verdict"] == "strong_match"

    def test_company_classification_override(self, db: Database):
        db.set_manual_override(
            ManualOverride("company", 7, "classification", "product", "verified myself", datetime.now(timezone.utc))
        )
        assert db.get_manual_overrides("company", 7)["classification"] == "product"


class TestCompanyClassification:
    def test_set_and_read_classification(self, db: Database, sample_company: Company):
        company_id = db.upsert_company(sample_company)
        db.set_company_classification(company_id, CompanyClassification.PRODUCT, ClassificationSource.LLM, 0.9, "builds its own SaaS")
        updated = db.get_company(company_id)
        assert updated.classification == CompanyClassification.PRODUCT
        assert updated.classification_confidence == 0.9


def make_evaluation(job_id: int, worth_applying: bool = True) -> Evaluation:
    return Evaluation(
        job_id=job_id, description_hash="hash-1", verdict=Verdict.BORDERLINE,
        confidence=0.6, scope_fit=ScopeFit.TWO_PLUS_STEPS_UP, evidence_coverage_percent=45,
        specialist_tenure_assessment=SpecialistTenureAssessment(
            classification=SpecialistTenureClassification.NOT_APPLICABLE, specialty="", explanation="",
        ),
        requirement_evidence=[], growth_dimensions=[], hidden_staff_signals=[],
        compensation_assessment="ok", remote_employment_verification="ok",
        required_matches=[], required_gaps=[], preferred_only_gaps=[], minor_caveats=[],
        evidence=[], credibility_assessment="ok", why_this_is_or_is_not_gettable="ok",
        is_product_company=True, primary_rejection_reason="not gettable",
        worth_applying=worth_applying, model_name="test-model",
        created_at=datetime.now(timezone.utc),
    )


class TestEvaluationPersistence:
    def test_worth_applying_round_trips(self, db: Database, settings: Settings, sample_company: Company):
        sample_company.id = db.upsert_company(sample_company)
        job_id, _, _ = db.upsert_job(normalize_posting(make_raw(), sample_company, settings))

        db.save_evaluation(make_evaluation(job_id, worth_applying=False))

        loaded = db.get_evaluation_for_job(job_id)
        assert loaded.worth_applying is False

    def test_legacy_row_missing_worth_applying_column_defaults_true(self, db: Database, settings: Settings, sample_company: Company):
        # Simulates an evaluation cached before this column existed: a raw INSERT that never
        # touches worth_applying leaves it NULL, same as every pre-existing row after the
        # ADD COLUMN migration ran. Must not be silently treated as "not worth applying".
        sample_company.id = db.upsert_company(sample_company)
        job_id, _, _ = db.upsert_job(normalize_posting(make_raw(), sample_company, settings))
        db.save_evaluation(make_evaluation(job_id, worth_applying=False))
        db.conn.execute("UPDATE evaluations SET worth_applying = NULL WHERE job_id = ?", (job_id,))

        loaded = db.get_evaluation_for_job(job_id)
        assert loaded.worth_applying is True
