"""SQLite persistence layer.

Plain ``sqlite3`` + explicit SQL rather than an ORM, so the schema and the query behavior
(dedup keys, what counts as "changed", how history is retained) stay visible and easy to reason
about. Enums are stored as their ``.value`` strings; datetimes as ISO-8601 strings (UTC);
list/dict fields as JSON text.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from jobscan.models import (
    AtsType,
    ClassificationSource,
    Company,
    CompanyClassification,
    CrawlRunStats,
    EmploymentType,
    EvaluateStats,
    Evaluation,
    EvidenceClassification,
    FilterLogEntry,
    JobPosting,
    JobStatus,
    ManualOverride,
    RemoteScope,
    RequirementEvidence,
    RequirementImportance,
    RequirementStrength,
    SalarySource,
    ScopeFit,
    SpecialistTenureAssessment,
    SpecialistTenureClassification,
    Verdict,
    WorkplaceType,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS companies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    domain TEXT,
    careers_url TEXT,
    ats_type TEXT NOT NULL,
    board_id TEXT NOT NULL,
    classification TEXT NOT NULL DEFAULT 'unknown',
    classification_source TEXT,
    classification_confidence REAL,
    classification_evidence TEXT,
    active INTEGER NOT NULL DEFAULT 1,
    last_crawled_at TEXT,
    discovery_source TEXT,
    notes TEXT,
    applied INTEGER NOT NULL DEFAULT 0,
    UNIQUE (ats_type, board_id)
);

CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER NOT NULL REFERENCES companies(id),
    source TEXT NOT NULL,
    source_job_id TEXT NOT NULL,
    title TEXT NOT NULL,
    location_raw TEXT,
    remote_scope TEXT NOT NULL,
    employment_type_raw TEXT,
    employment_type TEXT NOT NULL,
    description_text TEXT NOT NULL,
    description_html TEXT,
    description_hash TEXT NOT NULL,
    salary_min REAL,
    salary_max REAL,
    salary_currency TEXT,
    salary_period TEXT,
    salary_source TEXT NOT NULL,
    posting_url TEXT,
    apply_url TEXT,
    department TEXT,
    published_at TEXT,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    closed_at TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    workplace_type TEXT,
    UNIQUE (source, source_job_id)
);

CREATE TABLE IF NOT EXISTS job_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL REFERENCES jobs(id),
    description_hash TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    title TEXT,
    location_raw TEXT,
    salary_min REAL,
    salary_max REAL,
    description_text TEXT
);

CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL REFERENCES jobs(id),
    description_hash TEXT NOT NULL,
    verdict TEXT NOT NULL,
    confidence REAL,
    scope_fit TEXT NOT NULL,
    evidence_coverage_percent INTEGER,
    specialist_tenure_json TEXT,
    requirement_evidence_json TEXT,
    growth_dimensions TEXT,
    hidden_staff_signals TEXT,
    compensation_assessment TEXT,
    remote_employment_verification TEXT,
    required_matches TEXT,
    required_gaps TEXT,
    preferred_only_gaps TEXT,
    minor_caveats TEXT,
    evidence TEXT,
    credibility_assessment TEXT,
    why_this_is_or_is_not_gettable TEXT,
    is_product_company INTEGER,
    primary_rejection_reason TEXT,
    worth_applying INTEGER,
    model_name TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    created_at TEXT NOT NULL,
    UNIQUE (job_id, description_hash)
);

CREATE TABLE IF NOT EXISTS manual_overrides (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,
    entity_id INTEGER NOT NULL,
    field TEXT NOT NULL,
    value TEXT NOT NULL,
    reason TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (entity_type, entity_id, field)
);

CREATE TABLE IF NOT EXISTS filter_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL REFERENCES jobs(id),
    stage TEXT NOT NULL,
    reason TEXT NOT NULL,
    detail TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS crawl_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    companies_attempted INTEGER,
    companies_succeeded INTEGER,
    boards_failed TEXT,
    postings_fetched INTEGER,
    postings_out_of_family INTEGER DEFAULT 0,
    new_postings INTEGER,
    changed_postings INTEGER,
    closed_postings INTEGER
);

CREATE TABLE IF NOT EXISTS evaluation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    jobs_considered INTEGER,
    factual_rejected INTEGER,
    companies_classified INTEGER,
    sent_to_llm INTEGER,
    cache_hits INTEGER,
    manual_overrides_applied INTEGER,
    verdict_counts TEXT,
    unverified INTEGER,
    llm_errors INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cache_creation_input_tokens INTEGER,
    cache_read_input_tokens INTEGER,
    triaged INTEGER,
    triage_skipped INTEGER,
    triage_input_tokens INTEGER,
    triage_output_tokens INTEGER,
    triage_cache_creation_input_tokens INTEGER,
    triage_cache_read_input_tokens INTEGER,
    triage_model TEXT
);
"""


def _dt(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _parse_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Database:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.migrate()

    def migrate(self) -> None:
        self.conn.executescript(SCHEMA)
        # CREATE TABLE IF NOT EXISTS above is a no-op against an already-existing table, so a
        # brand-new column needs an explicit, idempotent ADD COLUMN for existing databases.
        self._add_column_if_missing("evaluations", "worth_applying", "INTEGER")
        self._add_column_if_missing("evaluations", "cache_creation_input_tokens", "INTEGER")
        self._add_column_if_missing("evaluations", "cache_read_input_tokens", "INTEGER")
        self._add_column_if_missing("evaluation_runs", "cache_creation_input_tokens", "INTEGER")
        self._add_column_if_missing("evaluation_runs", "cache_read_input_tokens", "INTEGER")
        self._add_column_if_missing("evaluation_runs", "triaged", "INTEGER")
        self._add_column_if_missing("evaluation_runs", "triage_skipped", "INTEGER")
        self._add_column_if_missing("evaluation_runs", "triage_input_tokens", "INTEGER")
        self._add_column_if_missing("evaluation_runs", "triage_output_tokens", "INTEGER")
        self._add_column_if_missing("evaluation_runs", "triage_cache_creation_input_tokens", "INTEGER")
        self._add_column_if_missing("evaluation_runs", "triage_cache_read_input_tokens", "INTEGER")
        self._add_column_if_missing("evaluation_runs", "triage_model", "TEXT")
        self._add_column_if_missing("companies", "applied", "INTEGER")
        self._add_column_if_missing("jobs", "workplace_type", "TEXT")
        self._add_column_if_missing("evaluations", "rubric_version", "TEXT")

    def _add_column_if_missing(self, table: str, column: str, sql_type: str) -> None:
        existing = {row["name"] for row in self.conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @contextmanager
    def transaction(self):
        self.conn.execute("BEGIN")
        try:
            yield
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    # ------------------------------------------------------------------
    # Companies
    # ------------------------------------------------------------------

    def upsert_company(self, company: Company) -> int:
        cur = self.conn.execute(
            "SELECT id, classification_source FROM companies WHERE ats_type = ? AND board_id = ?",
            (company.ats_type.value, company.board_id),
        )
        row = cur.fetchone()
        if row:
            company_id = row["id"]
            existing_source = row["classification_source"]
            # A registry re-import can carry a real (non-"unknown") classification for a company
            # that already exists — apply it, so re-curating a registry actually takes effect,
            # unless a manual override already asserted one (that always wins).
            should_update_classification = (
                company.classification != CompanyClassification.UNKNOWN
                and existing_source != ClassificationSource.MANUAL.value
            )
            if should_update_classification:
                self.conn.execute(
                    """UPDATE companies SET name=?, domain=?, careers_url=?, active=?, applied=?,
                       discovery_source=COALESCE(?, discovery_source), notes=COALESCE(?, notes),
                       classification=?, classification_source=?,
                       classification_confidence=COALESCE(?, classification_confidence),
                       classification_evidence=COALESCE(?, classification_evidence)
                       WHERE id=?""",
                    (
                        company.name, company.domain, company.careers_url, int(company.active),
                        int(company.applied),
                        company.discovery_source, company.notes,
                        company.classification.value,
                        (company.classification_source or ClassificationSource.SEED).value,
                        company.classification_confidence, company.classification_evidence,
                        company_id,
                    ),
                )
            else:
                self.conn.execute(
                    """UPDATE companies SET name=?, domain=?, careers_url=?, active=?, applied=?,
                       discovery_source=COALESCE(?, discovery_source), notes=COALESCE(?, notes)
                       WHERE id=?""",
                    (
                        company.name,
                        company.domain,
                        company.careers_url,
                        int(company.active),
                        int(company.applied),
                        company.discovery_source,
                        company.notes,
                        company_id,
                    ),
                )
            return company_id

        cur = self.conn.execute(
            """INSERT INTO companies
               (name, domain, careers_url, ats_type, board_id, classification,
                classification_source, classification_confidence, classification_evidence,
                active, last_crawled_at, discovery_source, notes, applied)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                company.name,
                company.domain,
                company.careers_url,
                company.ats_type.value,
                company.board_id,
                company.classification.value,
                company.classification_source.value if company.classification_source else None,
                company.classification_confidence,
                company.classification_evidence,
                int(company.active),
                _dt(company.last_crawled_at),
                company.discovery_source,
                company.notes,
                int(company.applied),
            ),
        )
        return cur.lastrowid

    def _row_to_company(self, row: sqlite3.Row) -> Company:
        return Company(
            id=row["id"],
            name=row["name"],
            domain=row["domain"],
            careers_url=row["careers_url"],
            ats_type=AtsType(row["ats_type"]),
            board_id=row["board_id"],
            classification=CompanyClassification(row["classification"]),
            classification_source=ClassificationSource(row["classification_source"])
            if row["classification_source"]
            else None,
            classification_confidence=row["classification_confidence"],
            classification_evidence=row["classification_evidence"],
            active=bool(row["active"]),
            last_crawled_at=_parse_dt(row["last_crawled_at"]),
            discovery_source=row["discovery_source"],
            notes=row["notes"],
            applied=bool(row["applied"]) if row["applied"] is not None else False,
        )

    def get_company(self, company_id: int) -> Company | None:
        row = self.conn.execute("SELECT * FROM companies WHERE id=?", (company_id,)).fetchone()
        return self._row_to_company(row) if row else None

    def list_companies(self, active_only: bool = True) -> list[Company]:
        sql = "SELECT * FROM companies"
        if active_only:
            sql += " WHERE active=1"
        sql += " ORDER BY name"
        return [self._row_to_company(r) for r in self.conn.execute(sql)]

    def deactivate_companies_except(self, keys: set[tuple[AtsType, str]]) -> int:
        """Sets active=0 for every currently-active company whose (ats_type, board_id) is not in
        ``keys``. Returns the number deactivated. History (jobs, evaluations) is untouched."""
        deactivated = 0
        for company in self.list_companies(active_only=True):
            if (company.ats_type, company.board_id) not in keys:
                self.conn.execute("UPDATE companies SET active=0 WHERE id=?", (company.id,))
                deactivated += 1
        return deactivated

    def delete_companies_except(self, keys: set[tuple[AtsType, str]]) -> int:
        """Permanently deletes every company whose (ats_type, board_id) is not in ``keys``,
        along with every job, evaluation, snapshot and filter-log entry tied to it. Irreversible
        — prefer deactivate_companies_except unless the data is genuinely meant to be discarded.
        """
        removed = 0
        for company in self.list_companies(active_only=False):
            if (company.ats_type, company.board_id) in keys:
                continue
            job_ids = [
                r["id"] for r in self.conn.execute("SELECT id FROM jobs WHERE company_id=?", (company.id,))
            ]
            for job_id in job_ids:
                self.conn.execute("DELETE FROM filter_log WHERE job_id=?", (job_id,))
                self.conn.execute("DELETE FROM evaluations WHERE job_id=?", (job_id,))
                self.conn.execute("DELETE FROM job_snapshots WHERE job_id=?", (job_id,))
            self.conn.execute("DELETE FROM jobs WHERE company_id=?", (company.id,))
            self.conn.execute("DELETE FROM companies WHERE id=?", (company.id,))
            removed += 1
        return removed

    def set_company_classification(
        self,
        company_id: int,
        classification: CompanyClassification,
        source: ClassificationSource,
        confidence: float | None,
        evidence: str | None,
    ) -> None:
        self.conn.execute(
            """UPDATE companies SET classification=?, classification_source=?,
               classification_confidence=?, classification_evidence=? WHERE id=?""",
            (classification.value, source.value, confidence, evidence, company_id),
        )

    def touch_company_crawled(self, company_id: int, when: datetime | None = None) -> None:
        self.conn.execute(
            "UPDATE companies SET last_crawled_at=? WHERE id=?",
            (_dt(when or _now()), company_id),
        )

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    def get_job_by_source(self, source: AtsType, source_job_id: str) -> JobPosting | None:
        row = self.conn.execute(
            "SELECT * FROM jobs WHERE source=? AND source_job_id=?",
            (source.value, source_job_id),
        ).fetchone()
        return self._row_to_job(row) if row else None

    def get_job(self, job_id: int) -> JobPosting | None:
        row = self.conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    def _row_to_job(self, row: sqlite3.Row) -> JobPosting:
        return JobPosting(
            id=row["id"],
            company_id=row["company_id"],
            source=AtsType(row["source"]),
            source_job_id=row["source_job_id"],
            title=row["title"],
            location_raw=row["location_raw"],
            remote_scope=RemoteScope(row["remote_scope"]),
            employment_type_raw=row["employment_type_raw"],
            employment_type=EmploymentType(row["employment_type"]),
            description_text=row["description_text"],
            description_html=row["description_html"],
            description_hash=row["description_hash"],
            salary_min=row["salary_min"],
            salary_max=row["salary_max"],
            salary_currency=row["salary_currency"],
            salary_period=row["salary_period"],
            salary_source=SalarySource(row["salary_source"]),
            posting_url=row["posting_url"],
            apply_url=row["apply_url"],
            department=row["department"],
            published_at=_parse_dt(row["published_at"]),
            first_seen_at=_parse_dt(row["first_seen_at"]),
            last_seen_at=_parse_dt(row["last_seen_at"]),
            closed_at=_parse_dt(row["closed_at"]),
            status=JobStatus(row["status"]),
            workplace_type=WorkplaceType(row["workplace_type"]) if row["workplace_type"] else None,
        )

    def upsert_job(self, job: JobPosting) -> tuple[int, bool, bool]:
        """Insert or update a job. Returns (job_id, is_new, is_changed).

        A change is detected via description_hash; on change, the *previous* row contents are
        archived into job_snapshots before the row is overwritten, and a re-opened (previously
        closed) posting is reactivated.
        """
        existing = self.get_job_by_source(job.source, job.source_job_id)
        now = job.last_seen_at

        if existing is None:
            cur = self.conn.execute(
                """INSERT INTO jobs
                   (company_id, source, source_job_id, title, location_raw, remote_scope,
                    employment_type_raw, employment_type, description_text, description_html,
                    description_hash, salary_min, salary_max, salary_currency, salary_period,
                    salary_source, posting_url, apply_url, department, published_at,
                    first_seen_at, last_seen_at, closed_at, status, workplace_type)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    job.company_id, job.source.value, job.source_job_id, job.title,
                    job.location_raw, job.remote_scope.value, job.employment_type_raw,
                    job.employment_type.value, job.description_text, job.description_html,
                    job.description_hash, job.salary_min, job.salary_max, job.salary_currency,
                    job.salary_period, job.salary_source.value, job.posting_url, job.apply_url,
                    job.department, _dt(job.published_at), _dt(job.first_seen_at),
                    _dt(job.last_seen_at), _dt(job.closed_at), job.status.value,
                    job.workplace_type.value if job.workplace_type else None,
                ),
            )
            return cur.lastrowid, True, False

        changed = existing.description_hash != job.description_hash
        if changed:
            self.conn.execute(
                """INSERT INTO job_snapshots
                   (job_id, description_hash, captured_at, title, location_raw, salary_min,
                    salary_max, description_text)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    existing.id, existing.description_hash, _dt(now), existing.title,
                    existing.location_raw, existing.salary_min, existing.salary_max,
                    existing.description_text,
                ),
            )

        self.conn.execute(
            """UPDATE jobs SET title=?, location_raw=?, remote_scope=?, employment_type_raw=?,
               employment_type=?, description_text=?, description_html=?, description_hash=?,
               salary_min=?, salary_max=?, salary_currency=?, salary_period=?, salary_source=?,
               posting_url=?, apply_url=?, department=?, published_at=?, last_seen_at=?,
               closed_at=NULL, status='active', workplace_type=?
               WHERE id=?""",
            (
                job.title, job.location_raw, job.remote_scope.value, job.employment_type_raw,
                job.employment_type.value, job.description_text, job.description_html,
                job.description_hash, job.salary_min, job.salary_max, job.salary_currency,
                job.salary_period, job.salary_source.value, job.posting_url, job.apply_url,
                job.department, _dt(job.published_at), _dt(now),
                job.workplace_type.value if job.workplace_type else None, existing.id,
            ),
        )
        return existing.id, False, changed

    def close_missing_jobs(self, company_id: int, seen_source_job_ids: set[str], when: datetime | None = None) -> int:
        when = when or _now()
        rows = self.conn.execute(
            "SELECT id, source_job_id FROM jobs WHERE company_id=? AND status='active'",
            (company_id,),
        ).fetchall()
        closed = 0
        for row in rows:
            if row["source_job_id"] not in seen_source_job_ids:
                self.conn.execute(
                    "UPDATE jobs SET status='closed', closed_at=? WHERE id=?",
                    (_dt(when), row["id"]),
                )
                closed += 1
        return closed

    def get_active_jobs(self, company_id: int | None = None) -> list[JobPosting]:
        if company_id is not None:
            rows = self.conn.execute(
                "SELECT * FROM jobs WHERE status='active' AND company_id=? ORDER BY id", (company_id,)
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM jobs WHERE status='active' ORDER BY id").fetchall()
        return [self._row_to_job(r) for r in rows]

    def count_jobs(self, status: JobStatus | None = None) -> int:
        if status:
            (n,) = self.conn.execute("SELECT COUNT(*) FROM jobs WHERE status=?", (status.value,)).fetchone()
        else:
            (n,) = self.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()
        return n

    # ------------------------------------------------------------------
    # Filter log (audit trail)
    # ------------------------------------------------------------------

    def record_filter_log(self, entry: FilterLogEntry) -> int:
        cur = self.conn.execute(
            "INSERT INTO filter_log (job_id, stage, reason, detail, created_at) VALUES (?,?,?,?,?)",
            (entry.job_id, entry.stage, entry.reason, entry.detail, _dt(entry.created_at)),
        )
        return cur.lastrowid

    def filter_log_for_job(self, job_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM filter_log WHERE job_id=? ORDER BY created_at DESC", (job_id,)
        ).fetchall()

    def all_filter_log_entries(self, since: datetime | None = None) -> list[sqlite3.Row]:
        if since:
            return self.conn.execute(
                "SELECT * FROM filter_log WHERE created_at >= ? ORDER BY created_at", (_dt(since),)
            ).fetchall()
        return self.conn.execute("SELECT * FROM filter_log ORDER BY created_at").fetchall()

    def has_filter_log(self, job_id: int, reason: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM filter_log WHERE job_id=? AND reason=? LIMIT 1", (job_id, reason)
        ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # Evaluations
    # ------------------------------------------------------------------

    def get_evaluation_by_hash(
        self, description_hash: str, rubric_versions: tuple[str, ...] | None = None
    ) -> Evaluation | None:
        """The latest evaluation of this description; with ``rubric_versions``, the latest one made
        under any of those rubrics (None if there isn't one)."""
        if rubric_versions is None:
            row = self.conn.execute(
                "SELECT * FROM evaluations WHERE description_hash=? ORDER BY id DESC LIMIT 1",
                (description_hash,),
            ).fetchone()
        else:
            placeholders = ",".join("?" * len(rubric_versions))
            row = self.conn.execute(
                f"SELECT * FROM evaluations WHERE description_hash=? AND rubric_version IN ({placeholders}) "
                "ORDER BY id DESC LIMIT 1",
                (description_hash, *rubric_versions),
            ).fetchone()
        return self._row_to_evaluation(row) if row else None

    def get_evaluation_for_job(self, job_id: int) -> Evaluation | None:
        row = self.conn.execute(
            "SELECT * FROM evaluations WHERE job_id=? ORDER BY id DESC LIMIT 1", (job_id,)
        ).fetchone()
        return self._row_to_evaluation(row) if row else None

    def _row_to_evaluation(self, row: sqlite3.Row) -> Evaluation:
        specialist_tenure_raw = json.loads(row["specialist_tenure_json"] or "{}")
        requirement_evidence_raw = json.loads(row["requirement_evidence_json"] or "[]")
        return Evaluation(
            id=row["id"],
            job_id=row["job_id"],
            description_hash=row["description_hash"],
            verdict=Verdict(row["verdict"]),
            confidence=row["confidence"],
            scope_fit=ScopeFit(row["scope_fit"]),
            evidence_coverage_percent=row["evidence_coverage_percent"],
            specialist_tenure_assessment=SpecialistTenureAssessment(
                classification=SpecialistTenureClassification(specialist_tenure_raw.get("classification", "not_applicable")),
                specialty=specialist_tenure_raw.get("specialty", ""),
                explanation=specialist_tenure_raw.get("explanation", ""),
            ),
            requirement_evidence=[
                RequirementEvidence(
                    requirement=item["requirement"],
                    importance=RequirementImportance(item["importance"]),
                    evidence_classification=EvidenceClassification(item["evidence_classification"]),
                    candidate_evidence=item["candidate_evidence"],
                    posting_evidence=item["posting_evidence"],
                    stated_as=RequirementStrength(item["stated_as"]) if item.get("stated_as") else None,
                )
                for item in requirement_evidence_raw
            ],
            growth_dimensions=json.loads(row["growth_dimensions"] or "[]"),
            hidden_staff_signals=json.loads(row["hidden_staff_signals"] or "[]"),
            compensation_assessment=row["compensation_assessment"],
            remote_employment_verification=row["remote_employment_verification"],
            required_matches=json.loads(row["required_matches"] or "[]"),
            required_gaps=json.loads(row["required_gaps"] or "[]"),
            preferred_only_gaps=json.loads(row["preferred_only_gaps"] or "[]"),
            minor_caveats=json.loads(row["minor_caveats"] or "[]"),
            evidence=json.loads(row["evidence"] or "[]"),
            credibility_assessment=row["credibility_assessment"],
            why_this_is_or_is_not_gettable=row["why_this_is_or_is_not_gettable"],
            is_product_company=bool(row["is_product_company"]),
            primary_rejection_reason=row["primary_rejection_reason"],
            # NULL for evaluations cached before this column existed — default True so they
            # aren't retroactively treated as "not worth applying" (see Evaluation.worth_applying).
            worth_applying=bool(row["worth_applying"]) if row["worth_applying"] is not None else True,
            model_name=row["model_name"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            cache_creation_input_tokens=row["cache_creation_input_tokens"],
            cache_read_input_tokens=row["cache_read_input_tokens"],
            created_at=_parse_dt(row["created_at"]),
            rubric_version=row["rubric_version"],
        )

    def save_evaluation(self, evaluation: Evaluation) -> int:
        specialist_tenure_json = json.dumps(
            {
                "classification": evaluation.specialist_tenure_assessment.classification.value,
                "specialty": evaluation.specialist_tenure_assessment.specialty,
                "explanation": evaluation.specialist_tenure_assessment.explanation,
            }
        )
        requirement_evidence_json = json.dumps(
            [
                {
                    "requirement": item.requirement,
                    "importance": item.importance.value,
                    "evidence_classification": item.evidence_classification.value,
                    "candidate_evidence": item.candidate_evidence,
                    "posting_evidence": item.posting_evidence,
                    "stated_as": item.stated_as.value if item.stated_as else None,
                }
                for item in evaluation.requirement_evidence
            ]
        )
        cur = self.conn.execute(
            """INSERT INTO evaluations
               (job_id, description_hash, verdict, confidence, scope_fit, evidence_coverage_percent,
                specialist_tenure_json, requirement_evidence_json, growth_dimensions,
                hidden_staff_signals, compensation_assessment, remote_employment_verification,
                required_matches, required_gaps, preferred_only_gaps, minor_caveats, evidence,
                credibility_assessment, why_this_is_or_is_not_gettable, is_product_company,
                primary_rejection_reason, worth_applying, model_name, input_tokens, output_tokens,
                cache_creation_input_tokens, cache_read_input_tokens, created_at, rubric_version)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(job_id, description_hash) DO UPDATE SET
                 verdict=excluded.verdict, confidence=excluded.confidence,
                 scope_fit=excluded.scope_fit,
                 evidence_coverage_percent=excluded.evidence_coverage_percent,
                 specialist_tenure_json=excluded.specialist_tenure_json,
                 requirement_evidence_json=excluded.requirement_evidence_json,
                 growth_dimensions=excluded.growth_dimensions,
                 hidden_staff_signals=excluded.hidden_staff_signals,
                 compensation_assessment=excluded.compensation_assessment,
                 remote_employment_verification=excluded.remote_employment_verification,
                 required_matches=excluded.required_matches, required_gaps=excluded.required_gaps,
                 preferred_only_gaps=excluded.preferred_only_gaps,
                 minor_caveats=excluded.minor_caveats,
                 evidence=excluded.evidence, credibility_assessment=excluded.credibility_assessment,
                 why_this_is_or_is_not_gettable=excluded.why_this_is_or_is_not_gettable,
                 is_product_company=excluded.is_product_company,
                 primary_rejection_reason=excluded.primary_rejection_reason,
                 worth_applying=excluded.worth_applying,
                 model_name=excluded.model_name, input_tokens=excluded.input_tokens,
                 output_tokens=excluded.output_tokens,
                 cache_creation_input_tokens=excluded.cache_creation_input_tokens,
                 cache_read_input_tokens=excluded.cache_read_input_tokens,
                 created_at=excluded.created_at, rubric_version=excluded.rubric_version
               """,
            (
                evaluation.job_id, evaluation.description_hash, evaluation.verdict.value,
                evaluation.confidence, evaluation.scope_fit.value, evaluation.evidence_coverage_percent,
                specialist_tenure_json, requirement_evidence_json,
                json.dumps(evaluation.growth_dimensions), json.dumps(evaluation.hidden_staff_signals),
                evaluation.compensation_assessment, evaluation.remote_employment_verification,
                json.dumps(evaluation.required_matches), json.dumps(evaluation.required_gaps),
                json.dumps(evaluation.preferred_only_gaps), json.dumps(evaluation.minor_caveats),
                json.dumps(evaluation.evidence), evaluation.credibility_assessment,
                evaluation.why_this_is_or_is_not_gettable, int(evaluation.is_product_company),
                evaluation.primary_rejection_reason, int(evaluation.worth_applying),
                evaluation.model_name, evaluation.input_tokens, evaluation.output_tokens,
                evaluation.cache_creation_input_tokens, evaluation.cache_read_input_tokens,
                _dt(evaluation.created_at), evaluation.rubric_version,
            ),
        )
        return cur.lastrowid or self.get_evaluation_for_job(evaluation.job_id).id

    # ------------------------------------------------------------------
    # Manual overrides
    # ------------------------------------------------------------------

    def set_manual_override(self, override: ManualOverride) -> int:
        cur = self.conn.execute(
            """INSERT INTO manual_overrides (entity_type, entity_id, field, value, reason, created_at)
               VALUES (?,?,?,?,?,?)
               ON CONFLICT(entity_type, entity_id, field) DO UPDATE SET
                 value=excluded.value, reason=excluded.reason, created_at=excluded.created_at""",
            (
                override.entity_type, override.entity_id, override.field, override.value,
                override.reason, _dt(override.created_at),
            ),
        )
        return cur.lastrowid

    def get_manual_overrides(self, entity_type: str, entity_id: int) -> dict[str, str]:
        rows = self.conn.execute(
            "SELECT field, value FROM manual_overrides WHERE entity_type=? AND entity_id=?",
            (entity_type, entity_id),
        ).fetchall()
        return {r["field"]: r["value"] for r in rows}

    # ------------------------------------------------------------------
    # Crawl runs
    # ------------------------------------------------------------------

    def start_crawl_run(self, stats: CrawlRunStats) -> int:
        cur = self.conn.execute(
            "INSERT INTO crawl_runs (started_at, companies_attempted, companies_succeeded, "
            "boards_failed, postings_fetched, postings_out_of_family, new_postings, "
            "changed_postings, closed_postings) VALUES (?,0,0,'[]',0,0,0,0,0)",
            (_dt(stats.started_at),),
        )
        return cur.lastrowid

    def finish_crawl_run(self, run_id: int, stats: CrawlRunStats) -> None:
        self.conn.execute(
            """UPDATE crawl_runs SET finished_at=?, companies_attempted=?, companies_succeeded=?,
               boards_failed=?, postings_fetched=?, postings_out_of_family=?, new_postings=?,
               changed_postings=?, closed_postings=? WHERE id=?""",
            (
                _dt(stats.finished_at or _now()), stats.companies_attempted,
                stats.companies_succeeded, json.dumps(stats.boards_failed),
                stats.postings_fetched, stats.postings_out_of_family, stats.new_postings,
                stats.changed_postings, stats.closed_postings, run_id,
            ),
        )

    def latest_crawl_run(self) -> CrawlRunStats | None:
        row = self.conn.execute("SELECT * FROM crawl_runs ORDER BY id DESC LIMIT 1").fetchone()
        if row is None:
            return None
        return CrawlRunStats(
            id=row["id"],
            started_at=_parse_dt(row["started_at"]),
            finished_at=_parse_dt(row["finished_at"]),
            companies_attempted=row["companies_attempted"],
            companies_succeeded=row["companies_succeeded"],
            boards_failed=json.loads(row["boards_failed"] or "[]"),
            postings_fetched=row["postings_fetched"],
            postings_out_of_family=row["postings_out_of_family"] or 0,
            new_postings=row["new_postings"],
            changed_postings=row["changed_postings"],
            closed_postings=row["closed_postings"],
        )

    # ------------------------------------------------------------------
    # Evaluation runs
    # ------------------------------------------------------------------

    def start_evaluation_run(self, stats: EvaluateStats) -> int:
        cur = self.conn.execute(
            "INSERT INTO evaluation_runs (started_at, jobs_considered, factual_rejected, "
            "companies_classified, sent_to_llm, cache_hits, manual_overrides_applied, "
            "verdict_counts, unverified, llm_errors, input_tokens, output_tokens, "
            "cache_creation_input_tokens, cache_read_input_tokens, triaged, triage_skipped, "
            "triage_input_tokens, triage_output_tokens, triage_cache_creation_input_tokens, "
            "triage_cache_read_input_tokens) "
            "VALUES (?,0,0,0,0,0,0,'{}',0,0,0,0,0,0,0,0,0,0,0,0)",
            (_dt(stats.started_at),),
        )
        return cur.lastrowid

    def finish_evaluation_run(self, run_id: int, stats: EvaluateStats) -> None:
        self.conn.execute(
            """UPDATE evaluation_runs SET finished_at=?, jobs_considered=?, factual_rejected=?,
               companies_classified=?, sent_to_llm=?, cache_hits=?, manual_overrides_applied=?,
               verdict_counts=?, unverified=?, llm_errors=?, input_tokens=?, output_tokens=?,
               cache_creation_input_tokens=?, cache_read_input_tokens=?, triaged=?,
               triage_skipped=?, triage_input_tokens=?, triage_output_tokens=?,
               triage_cache_creation_input_tokens=?, triage_cache_read_input_tokens=?,
               triage_model=?
               WHERE id=?""",
            (
                _dt(stats.finished_at or _now()), stats.jobs_considered, stats.factual_rejected,
                stats.companies_classified, stats.sent_to_llm, stats.cache_hits,
                stats.manual_overrides_applied, json.dumps(stats.verdict_counts),
                stats.unverified, stats.llm_errors, stats.input_tokens, stats.output_tokens,
                stats.cache_creation_input_tokens, stats.cache_read_input_tokens,
                stats.triaged, stats.triage_skipped, stats.triage_input_tokens,
                stats.triage_output_tokens, stats.triage_cache_creation_input_tokens,
                stats.triage_cache_read_input_tokens, stats.triage_model,
                run_id,
            ),
        )

    def latest_evaluation_run(self) -> EvaluateStats | None:
        row = self.conn.execute("SELECT * FROM evaluation_runs ORDER BY id DESC LIMIT 1").fetchone()
        if row is None:
            return None
        return EvaluateStats(
            id=row["id"],
            started_at=_parse_dt(row["started_at"]),
            finished_at=_parse_dt(row["finished_at"]),
            jobs_considered=row["jobs_considered"],
            factual_rejected=row["factual_rejected"],
            companies_classified=row["companies_classified"],
            sent_to_llm=row["sent_to_llm"],
            cache_hits=row["cache_hits"],
            manual_overrides_applied=row["manual_overrides_applied"],
            verdict_counts=json.loads(row["verdict_counts"] or "{}"),
            unverified=row["unverified"],
            llm_errors=row["llm_errors"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            cache_creation_input_tokens=row["cache_creation_input_tokens"] or 0,
            cache_read_input_tokens=row["cache_read_input_tokens"] or 0,
            triaged=row["triaged"] or 0,
            triage_skipped=row["triage_skipped"] or 0,
            triage_input_tokens=row["triage_input_tokens"] or 0,
            triage_output_tokens=row["triage_output_tokens"] or 0,
            triage_cache_creation_input_tokens=row["triage_cache_creation_input_tokens"] or 0,
            triage_cache_read_input_tokens=row["triage_cache_read_input_tokens"] or 0,
            triage_model=row["triage_model"],
        )
