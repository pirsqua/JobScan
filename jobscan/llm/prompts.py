"""Prompt construction for the two LLM calls, built from config/candidate_profile.yaml."""
from __future__ import annotations

from pathlib import Path

import yaml

from jobscan.models import Company, JobPosting


def load_profile(profile_path: Path) -> dict:
    return yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def render_profile_text(profile: dict) -> str:
    return f"""Candidate headline: {profile.get('headline', '').strip()}
Location: {profile.get('location', '')}

Strongest skills:
{_bullets(profile.get('strongest_skills', []))}

Representative accomplishments:
{_bullets(profile.get('accomplishments', []))}

Specialty areas outside current focus (no direct demonstrated evidence — do not credit a
posting's overlap with these as if it were already proven):
{_bullets(profile.get('specialty_areas_outside_current_focus', []))}

Do not inflate the evidence above:
{_bullets(profile.get('non_inflation_rules', []))}

Prioritized target roles:
{_bullets(profile.get('target_roles_prioritized', []))}

Favored hands-on work:
{_bullets(profile.get('favored_work', []))}

Stretch categories (one_step_up at best, and only when the specific posting reads as ordinary
senior scope rather than de facto staff/org-wide scope):
{_bullets(profile.get('stretch_categories', []))}

Do NOT target (roles to avoid recommending):
{_bullets(profile.get('do_not_target', []))}

Company-fit preferences:
{_bullets(profile.get('company_preferences', []))}

Qualification-language interpretation rules:
{_bullets(profile.get('qualification_interpretation_rules', []))}

Compensation rule: {profile.get('compensation_rule', '').strip()}
Minimum acceptable base salary: ${profile.get('min_base_salary', 170000):,}

Hard employment filters (already partially checked by code, verify from the text):
{_bullets(profile.get('hard_filters', []))}
"""


JOB_EVAL_SYSTEM = """You are a meticulous technical recruiter screening job postings for one specific \
candidate, whose evidence boundary is given in the candidate profile below. Evaluate ONLY the \
posting text given to you — do not assume facts not stated, and do not inflate the candidate's \
background beyond what their profile documents. Do not treat ~15 years of general \
software-engineering experience as 15 years in every specialty a posting happens to mention. Do \
not convert one demonstrated project into "several" similar ones. Do not infer repeated \
experience, massive production scale, or organization-wide influence without direct evidence.

The candidate is targeting properly scoped Senior Engineer roles — not mid-level roles, and not \
roles that use a "Senior" title to disguise Staff-level or elite-startup scope. Your central job \
is to catch that gap: a role can share every keyword with the candidate's background and still \
expect specialized tenure, a repeated track record, production scale, or organizational ownership \
the candidate has not demonstrated.

Work through these steps, in order, before calling the tool:

1. IDENTIFY REQUIRED QUALIFICATIONS. Separate explicitly required qualifications, strongly implied \
requirements, preferred-only qualifications, and general promotional language. Never treat a \
preferred qualification as required.

2. MAP EVIDENCE. For every important required or strongly implied qualification, classify the \
candidate's evidence as exactly one of: directly_demonstrated, credibly_transferable, \
weakly_inferred, or not_demonstrated. Cite both the posting's own language and the specific \
candidate evidence (or lack of it) in `requirement_evidence`, tagging each entry's `importance` as \
central or secondary. Technology adjacency alone is not direct evidence of equivalent scope — e.g. \
knowing Python and Databricks does not itself prove several years architecting multiple data \
platforms from scratch.

3. EVALUATE SPECIALIST TENURE. Decide whether the posting requires substantial tenure specifically \
in a specialty — data engineering, distributed systems, infrastructure, security, machine \
learning, frontend engineering, or similar — beyond general software-engineering experience. \
Classify `specialist_tenure_assessment` as meets, adjacent, insufficient, or not_applicable. \
General tenure may support "adjacent" but must never automatically satisfy a specialized-years \
requirement.

4. IDENTIFY GROWTH DIMENSIONS. A growth dimension is a central part of the role the candidate has \
not already demonstrated at roughly the required level — e.g. an unfamiliar primary cloud \
provider or language, substantially greater production scale, deep distributed-systems \
architecture, specialized data-engineering tenure, designing several greenfield systems from \
scratch, repeated zero-to-one ownership, organization-wide technical influence, owning strategy \
across multiple teams, production AI/ML responsibility, deep frontend ownership, or formal people \
management. List only material dimensions in `growth_dimensions`, not minor tool differences.

5. DETECT HIDDEN STAFF-LEVEL SCOPE. Look for clusters (never a single ambitious-sounding phrase \
alone) among: establishing architecture/standards across teams, setting organizational technical \
direction, designing several major systems from scratch, acting as the principal technical \
authority, building foundational infrastructure with little existing structure, repeated \
zero-to-one success, extraordinarily high scale or near-perfect reliability, influencing several \
teams or the whole engineering org, combining architecture + product strategy + implementation + \
long-term ownership, requiring exceptional autonomy at a small/highly-selective company, or an \
exceptional salary paired with extremely broad ownership. Weigh the combination of \
responsibilities, qualifications, company structure and compensation — not any one sentence. Put \
relevant signals in `hidden_staff_signals`.

6. CLASSIFY SCOPE (`scope_fit`), exactly one of:
   - at_level: the candidate has already performed substantially similar work at comparable scope.
   - one_step_up: the candidate demonstrates the central capabilities and the role introduces no \
more than one material growth dimension.
   - two_plus_steps_up: the role combines at least two material unproven dimensions, or expects a \
substantially deeper repeated track record than demonstrated.
   - below_level: the role would materially underuse the candidate's experience.
   A role is NOT one_step_up merely because its title says "Senior" or because several \
technologies match.

7. CALCULATE EVIDENCE COVERAGE (`evidence_coverage_percent`, 0-100). Weight central required \
qualifications more heavily than secondary ones. Count directly_demonstrated evidence fully, \
credibly_transferable partially, weakly_inferred minimally, and not_demonstrated as zero. This is \
a weighted judgment call, not a mechanical keyword-overlap count.

8. ASSIGN THE VERDICT:
   - strong_match: the role is at_level, at least 80% of important required qualifications are \
directly demonstrated or strongly supported, and there are no material required gaps.
   - plausible_match: the role is at_level OR genuinely one_step_up, evidence coverage is \
normally at least 70%, and there is no more than one material growth dimension.
   - borderline: technical overlap is attractive, but the role expects repeated experience, \
specialization, scale, or organizational influence not demonstrated — OR the role is otherwise \
attractive but two_plus_steps_up. A borderline role can be a genuine aspirational target, but must \
never be framed as an immediate strong recommendation.
   - reject: a hard employment requirement fails, or a central mandatory requirement makes an \
interview professionally implausible.

Also determine whether the employer is a product company or a consulting/client-delivery shop \
from this posting's own language, whether frontend/deep-AWS-or-GCP/AI-agent/distributed-systems \
ownership is central, and whether a $170,000+ base offer is credible from the published range \
(explicitly distinguish base salary from total compensation if the posting blends them).

Write `why_this_is_or_is_not_gettable` as a concise, specific verdict on interview/offer \
plausibility grounded in scope_fit and evidence_coverage_percent — not a restatement of the \
verdict label. When the posting is ambiguous, say so in evidence/caveats rather than guessing.

9. SET `worth_applying` (boolean) — a forced yes/no distillation of the gettability judgment \
above, independent of the verdict label. False specifically when central requirements depend on \
a technology, domain, or scale the candidate has never demonstrably touched — a new primary \
language or datastore the role centers on (e.g. Rust, Kafka, Kubernetes as core systems, not a \
minor mention), a security clearance, formal people management, production AI/ML ownership — such \
that a real interview loop would almost certainly expose the mismatch, even when \
evidence_coverage_percent looks moderate because of secondary-requirement overlap. True for \
genuine aspirational stretches where the gap is depth/scale/tenure in already-familiar territory \
(more years, larger scale, broader ownership of tools/domains the candidate already works in) \
rather than unfamiliar core technology. Always True for strong_match and plausible_match — this \
field exists to separate real stretches from roles that only look attractive on a keyword scan.

Always call the submit_job_evaluation tool exactly once with your full structured evaluation."""


def build_candidate_profile_block(profile_text: str) -> str:
    """Split out from the rest of the user message so it can be sent as a separate, cached
    content block — identical across every posting evaluated in a run, since it depends only on
    the candidate profile, never on the specific job."""
    return f"""# Candidate profile

{profile_text}"""


def build_job_eval_user_message(company: Company, job: JobPosting) -> str:
    salary_line = "not published"
    if job.salary_min is not None or job.salary_max is not None:
        lo = f"${job.salary_min:,.0f}" if job.salary_min is not None else "?"
        hi = f"${job.salary_max:,.0f}" if job.salary_max is not None else "?"
        period = job.salary_period or "year"
        salary_line = f"{lo} - {hi} per {period} (source: {job.salary_source.value})"

    return f"""# Company

Name: {company.name}
Domain: {company.domain or 'unknown'}
Known classification: {company.classification.value} (source: {company.classification_source.value if company.classification_source else 'none'})

# Job posting

Title: {job.title}
Location (raw): {job.location_raw or 'unknown'}
Employment type (raw): {job.employment_type_raw or 'unknown'}
Published salary: {salary_line}
Posting URL: {job.posting_url or 'unknown'}

Full description:
---
{job.description_text}
---

Work through the 8-step procedure from your instructions and call submit_job_evaluation with the \
complete structured result, including requirement_evidence for each important required or \
strongly implied qualification."""


TRIAGE_SYSTEM = """You do a fast, deliberately lenient first-pass screen: is this job posting even \
worth a full detailed evaluation against the candidate profile, or is it such a clear mismatch that \
a detailed review would just waste money confirming what's already obvious?

Mark skip_full_evaluation=true ONLY when you're confident a careful reviewer would also reject it \
outright — a hard disqualifying requirement stated as a genuine central expectation, not a passing \
mention: a completely different primary technology stack/language than the candidate's (e.g. the \
role centers on Rust/Go/Java with no .NET/Python/C# presence), formal people-management \
responsibility, a specialized domain or production scale far beyond general backend/data-pipeline \
work, a hard employment-type/location mismatch not already caught by factual filters, or an \
explicit title/seniority far above Senior (Staff/Principal/Director) with no dual-track ambiguity.

When you're genuinely unsure, or the posting has real overlap alongside some gaps, do NOT skip — \
that's exactly the judgment call the full evaluation exists for. Skipping a posting that deserved a \
full look silently destroys a real opportunity; sending a clear reject through for full evaluation \
only costs a few cents. Treat those two mistakes as very different in severity.

Always call the submit_triage tool exactly once with a one-sentence reason."""


def build_triage_user_message(company: Company, job: JobPosting) -> str:
    salary_line = "not published"
    if job.salary_min is not None or job.salary_max is not None:
        lo = f"${job.salary_min:,.0f}" if job.salary_min is not None else "?"
        hi = f"${job.salary_max:,.0f}" if job.salary_max is not None else "?"
        period = job.salary_period or "year"
        salary_line = f"{lo} - {hi} per {period} (source: {job.salary_source.value})"

    return f"""# Company

Name: {company.name}
Domain: {company.domain or 'unknown'}

# Job posting

Title: {job.title}
Location (raw): {job.location_raw or 'unknown'}
Employment type (raw): {job.employment_type_raw or 'unknown'}
Published salary: {salary_line}

Full description:
---
{job.description_text}
---

Call submit_triage with your skip_full_evaluation decision and a one-sentence reason."""


COMPANY_EVAL_SYSTEM = """You classify companies as either a PRODUCT company (builds and operates its \
own software product or platform, hiring engineers to build features for its own users) or a \
CONSULTING/STAFFING/OUTSOURCING company (sells engineering time/delivery to other companies' \
projects, e.g. a digital agency, dev shop, staffing firm, or IT services vendor). Base your answer \
only on the text given. Always call the submit_company_classification tool exactly once."""


def build_company_eval_user_message(company_name: str, domain: str | None, homepage_text: str, sample_job_text: str) -> str:
    return f"""Company name: {company_name}
Domain: {domain or 'unknown'}

Homepage/About text (may be truncated):
---
{homepage_text or '(not available)'}
---

Sample job posting text from this company (may be truncated):
---
{sample_job_text or '(not available)'}
---

Classify this company as product, consulting, or unknown (only if genuinely indeterminate)."""
