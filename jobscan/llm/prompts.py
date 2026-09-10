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

Important limitations (do not assume expertise beyond these):
{_bullets(profile.get('limitations', []))}

Prioritized target roles:
{_bullets(profile.get('target_roles_prioritized', []))}

Favored hands-on work:
{_bullets(profile.get('favored_work', []))}

Do NOT target (roles to avoid recommending):
{_bullets(profile.get('do_not_target', []))}

Qualification-language interpretation rules:
{_bullets(profile.get('qualification_interpretation_rules', []))}

Compensation rule: {profile.get('compensation_rule', '').strip()}
Minimum acceptable base salary: ${profile.get('min_base_salary', 170000):,}

Hard employment filters (already partially checked by code, verify from the text):
{_bullets(profile.get('hard_filters', []))}
"""


JOB_EVAL_SYSTEM = """You are a meticulous technical recruiter screening job postings for one specific \
candidate. You evaluate ONLY the posting text given to you — do not assume facts not stated. Be \
skeptical of vague or padded postings. Distinguish REQUIRED qualifications from PREFERRED/nice-to-have \
ones explicitly. When the posting is ambiguous, say so in evidence/caveats rather than guessing. \
Always call the submit_job_evaluation tool exactly once with your full structured evaluation."""


def build_job_eval_user_message(profile_text: str, company: Company, job: JobPosting) -> str:
    salary_line = "not published"
    if job.salary_min is not None or job.salary_max is not None:
        lo = f"${job.salary_min:,.0f}" if job.salary_min is not None else "?"
        hi = f"${job.salary_max:,.0f}" if job.salary_max is not None else "?"
        period = job.salary_period or "year"
        salary_line = f"{lo} - {hi} per {period} (source: {job.salary_source.value})"

    return f"""# Candidate profile

{profile_text}

# Company

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

Evaluate this posting for the candidate above. Determine whether the employer is a product \
company or consulting/client-delivery shop based on this posting's language, separate required \
from preferred qualifications, identify exact matches and material gaps against the candidate's \
background, note minor/preferred-only gaps, judge whether frontend/AWS/AI-agent/distributed-\
systems ownership is central to the role, assess whether a $170,000+ base offer is credible from \
the published range, and give a concise credibility assessment of whether applying makes sense. \
Cite evidence from the posting for your conclusions."""


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
