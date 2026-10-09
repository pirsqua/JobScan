"""Prompt construction for the two LLM calls, built from config/candidate_profile.yaml."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from jobscan.llm.schemas import JOB_EVALUATION_TOOL_SCHEMA, TRIAGE_TOOL_SCHEMA
from jobscan.models import Company, JobPosting


def load_profile(profile_path: Path) -> dict:
    return yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def render_profile_text(profile: dict) -> str:
    return f"""Candidate headline: {profile.get('headline', '').strip()}
Location: {profile.get('location', '')}

Education:
{_bullets(profile.get('education', []))}

Certifications:
{_bullets(profile.get('certifications', []))}

Work history:
{_bullets(profile.get('work_history', []))}

Strongest skills:
{_bullets(profile.get('strongest_skills', []))}

Representative accomplishments:
{_bullets(profile.get('accomplishments', []))}

Other skills listed on the resume (working familiarity — credit as transferable, not as
demonstrated production tenure, unless an accomplishment above shows depth):
{_bullets(profile.get('additional_skills', []))}

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

The candidate is targeting properly scoped Senior Engineer roles — not roles that use a "Senior" \
title to disguise Staff-level or elite-startup scope. A below-Senior title (Software Engineer II, \
Intermediate, ...) is acceptable when the profile's conditions for it are met: judge the work and \
the pay, not the title. One job is to catch the over-scoping gap: a role can share every keyword with the candidate's background and still expect \
specialized tenure, a repeated track record, production scale, or organizational ownership the \
candidate has not demonstrated.

The other job matters just as much: do not manufacture gaps. Good fits are rare, and a real fit \
wrongly rejected is the costlier mistake — the candidate never sees it — while a generous \
borderline costs one read. Hold the posting to what it actually asks for, not to a stricter \
version of it: a nice-to-have is not a requirement, hoping for someone curious about a topic is \
not requiring experience in it, and ordinary senior ownership is not staff scope. Where a \
reasonable reading is genuinely uncertain, say so in the caveats rather than resolving it against \
the candidate.

Work through these steps, in order, before calling the tool:

1. IDENTIFY REQUIRED QUALIFICATIONS. Classify every qualification by the posting's own wording \
(`stated_as`):
   - required: "must", "required", "minimum qualifications", "N+ years of X", or listed plainly \
under "Requirements" / "What you'll need" / "You have" with no softening qualifier;
   - strongly_implied: never stated, but plainly unavoidable to do the described work;
   - preferred: "helpful", "a plus", "bonus", "nice to have", "preferred", "ideally", "familiarity \
with", "exposure to", "X or a willingness to learn it", anything under "Nice-to-haves" / "Bonus";
   - trait_or_interest: personal qualities and interests — curiosity, high agency, eagerness or \
boldness to learn, "interested in", "passion for".
   Read qualifiers inside a sentence: in "Proficiency in Python, ideally including numpy", Python \
is required and numpy is preferred. When a requirement accepts alternatives ("Go or a comparable \
language such as Python", "Python or C#"), judge it against the alternative the candidate has — \
that the team mostly works in the other one is at most a growth note. "Using technologies like \
AWS, MySQL and Kubernetes" or "such as ..." names examples, not requirements — equivalent \
experience (Azure, SQL Server) meets it. Read the posting's framing as a whole: when it says it doesn't \
expect every item to be met, or that eagerness to learn is what's essential, the technologies it \
lists are preferred unless one is individually marked as required. Never treat a preferred \
qualification or a trait/interest as required.

2. MAP EVIDENCE. In `requirement_evidence`, list each required, strongly implied or notable \
preferred qualification: quote the posting's language with its qualifiers, set `stated_as`, then \
`importance` — central only for a required or strongly implied item the hiring bar actually rests \
on; preferred items are always secondary — then the candidate's evidence and exactly one of \
directly_demonstrated, credibly_transferable, weakly_inferred, or not_demonstrated. Leave traits \
and interests out (if you do list one, mark it trait_or_interest): a profile can't demonstrate \
curiosity, and its absence from one is not a gap. An unmet preferred item goes in \
`preferred_only_gaps`, never `required_gaps`. Technology adjacency alone is not direct evidence of \
equivalent scope — e.g. knowing Python and Databricks does not itself prove several years \
architecting multiple data platforms from scratch.

3. EVALUATE SPECIALIST TENURE. Decide whether the posting requires substantial tenure specifically \
in a specialty — data engineering, distributed systems, infrastructure, security, machine \
learning, frontend engineering, or similar — beyond general software-engineering experience. \
Classify `specialist_tenure_assessment` as meets, adjacent, insufficient, or not_applicable. \
General tenure may support "adjacent" but must never automatically satisfy a specialized-years \
requirement.

4. IDENTIFY GROWTH DIMENSIONS. A growth dimension is a central, required part of the role the \
candidate has not already demonstrated at roughly the required level — e.g. a required unfamiliar \
primary cloud provider or language, substantially greater production scale, deep \
distributed-systems architecture, specialized data-engineering tenure, designing several \
greenfield systems from scratch, repeated zero-to-one ownership, organization-wide technical \
influence, owning strategy across multiple teams, required production AI/ML responsibility, deep \
frontend ownership, or formal people management. List only material dimensions in \
`growth_dimensions`, not minor tool differences. These are NOT growth dimensions: a preferred or \
learnable qualification; a new business or product domain (logistics, data-science tooling, \
finance, identity, ...) unless the posting requires prior specialist experience in it; working \
with ambiguity, prototyping, or taking initiative; interest in a topic. A primary language the \
team works in that the posting calls helpful or learnable — especially alongside one of the \
candidate's own languages — is at most one growth dimension.

5. DETECT HIDDEN STAFF-LEVEL SCOPE. Look for clusters (never a single ambitious-sounding phrase \
alone) among: establishing architecture/standards across teams, setting organizational technical \
direction, designing several major systems from scratch, acting as the principal technical \
authority, building foundational infrastructure with little existing structure, repeated \
zero-to-one success, extraordinarily high scale or near-perfect reliability, influencing several \
teams or the whole engineering org, combining architecture + product strategy + implementation + \
long-term ownership across a whole product area, or an exceptional salary paired with extremely \
broad ownership. Weigh the combination of responsibilities, qualifications, company structure and \
compensation — not any one sentence. Ordinary senior expectations are NOT staff signals, alone or \
together: owning features end to end, planning and driving one's own features, owning the team's \
goals or delivery, tech-leading or leading engineers on the team through a project, high agency or \
self-direction, comfort with ambiguity, prototyping, shipping quickly, mentoring, cross-functional \
collaboration, aspirational "build things we haven't imagined" language, or a mission-driven or \
generalist culture. Staff scope is breadth of influence beyond one team, not initiative within it. \
For example, "owning and delivering quarterly goals for your team, leading engineers on your team \
through ambiguity, advocating for good practices beyond your team through writing and tech talks" \
is an ordinary senior/tech-lead expectation at a structured company — not a staff signal and not a \
growth dimension (technical leadership is a match for this candidate); "set technical direction \
across several teams" or "be the technical authority for the platform" is a staff signal. Put \
relevant signals in `hidden_staff_signals`.

6. CLASSIFY SCOPE (`scope_fit`), exactly one of:
   - at_level: the candidate has already performed substantially similar work at comparable scope.
   - one_step_up: the candidate demonstrates the central capabilities and the role introduces no \
more than one material growth dimension.
   - two_plus_steps_up: the role combines at least two material unproven dimensions, or expects a \
substantially deeper repeated track record than demonstrated.
   - below_level: the described work itself would materially underuse the candidate (junior \
tasks, close supervision, an explicitly early-career or new-grad role) — never because of the \
title alone, and never because the stated requirements are modest or easy for the candidate to \
meet: a low years-of-experience minimum, "previous work or internship experience", or standard \
expectations (code review, debugging, designing a multi-component system) describe the hiring bar, \
not junior work (observed: a Senior backend role paying $195,000-$255,000 called below_level for \
that alone, pushing it out of the Best Bets). Being below level is not a reason \
to reject: when the pay passes the compensation rule and the work is a backend fit, give the \
verdict the evidence supports, exactly as for an at_level role (observed: an "early-career, 1.5+ \
years" backend role paying $165,000-$225,000 at 95% coverage was rejected for its level alone).
   A role is NOT one_step_up merely because its title says "Senior" or because several \
technologies match.

7. CALCULATE EVIDENCE COVERAGE (`evidence_coverage_percent`, 0-100). Weight central required \
qualifications more heavily than secondary ones. Count directly_demonstrated evidence fully, \
credibly_transferable partially, weakly_inferred minimally, and not_demonstrated as zero. A met \
preferred item may add a little; an unmet one subtracts nothing. This is a weighted judgment call, \
not a mechanical keyword-overlap count.

8. ASSIGN THE VERDICT:
   - strong_match: the role is at_level or below_level, at least 80% of important required qualifications are \
directly demonstrated or strongly supported, and there are no material required gaps.
   - plausible_match: the role is at_level, below_level OR genuinely one_step_up, evidence coverage is \
normally at least 70%, and there is no more than one material growth dimension.
   - borderline: technical overlap is attractive, but the role expects repeated experience, \
specialization, scale, or organizational influence not demonstrated — OR the role is otherwise \
attractive but two_plus_steps_up. A borderline role can be a genuine aspirational target, but must \
never be framed as an immediate strong recommendation.
   - reject: a hard employment requirement fails, or a required (not preferred) central \
qualification makes an interview professionally implausible. Remote eligibility needs affirmative evidence — the location \
field, the text, or a structured workplace type of "remote". If the posting does not clearly offer \
remote work, or it excludes Washington (including by limiting remote work to states or regions that \
leave it out), the remote hard filter has failed: reject; never downgrade that to a borderline "risk", \
however good the technical fit. A remote offer that simply doesn't list eligible states is not a \
failure — e.g. a role listed at a city whose text offers working "100% remotely" passes.

Also determine whether the employer is a product company or a consulting/client-delivery shop \
from this posting's own language, whether frontend/deep-AWS-or-GCP/AI-agent/distributed-systems \
ownership is central, and whether a $170,000+ base offer is credible from the published range \
under the candidate's compensation rule (explicitly distinguish base salary from total \
compensation if the posting blends them). Compensation is a pass/fail gate: once a range passes \
it, a midpoint below $170,000 is not a gap and must not lower scope_fit, coverage, the verdict or \
worth_applying — note it as a minor caveat at most.

Write `why_this_is_or_is_not_gettable` as a concise, specific verdict on interview/offer \
plausibility grounded in scope_fit and evidence_coverage_percent — not a restatement of the \
verdict label. When the posting is ambiguous, say so in evidence/caveats rather than guessing.

9. SET `worth_applying` (boolean) — a forced yes/no distillation of the gettability judgment \
above, independent of the verdict label. False specifically when the posting REQUIRES (stated_as \
required) a technology, domain, or scale the candidate has never demonstrably touched — a new \
primary language or datastore the role demands proficiency in (e.g. "5+ years of Rust", "expert \
Kafka", Kubernetes as core systems, not a minor mention), a security clearance, formal people \
management, production AI/ML ownership — such that a real interview loop would almost certainly \
expose the mismatch, even when evidence_coverage_percent looks moderate because of \
secondary-requirement overlap. An unfamiliar language or domain the posting calls helpful, \
preferred or learnable, or lists as an alternative to one the candidate knows, is never on its own \
a reason for False. True for genuine aspirational stretches where the gap is depth/scale/tenure in \
already-familiar territory (more years, larger scale, broader ownership of tools/domains the \
candidate already works in) or a learnable stack the posting invites. Always True for \
strong_match and plausible_match — this field exists to separate real stretches from roles that \
only look attractive on a keyword scan.

The tool's fields follow these steps in order, with the verdict near the end: fill them in that \
order and let the verdict follow from the analysis you've written — never settle it first and \
justify it afterwards. Always call the submit_job_evaluation tool exactly once with your full \
structured evaluation."""


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
Workplace type (the ATS's own structured field): {job.workplace_type.value if job.workplace_type else 'not provided'}
Employment type (raw): {job.employment_type_raw or 'unknown'}
Published salary: {salary_line}
Posting URL: {job.posting_url or 'unknown'}

Full description:
---
{job.description_text}
---

Work through the 9-step procedure from your instructions and call submit_job_evaluation with the \
complete structured result, including requirement_evidence for each important required or \
strongly implied qualification."""


TRIAGE_SYSTEM = """You do a fast, deliberately lenient first-pass screen: is this job posting even \
worth a full detailed evaluation against the candidate profile, or is it such a clear mismatch that \
a detailed review would just waste money confirming what's already obvious?

You may skip a posting only for one of these disqualifiers, and only when the posting itself states \
it as a genuine, central requirement — not a passing mention:
- staff_or_higher_title: the job title is Staff, Senior Staff, Principal, Distinguished, Director or \
above (the candidate doesn't target Staff-level roles) — but not a dual-level title that also hires \
at Senior ("Senior/Staff", "Senior or Staff").
- people_management: formal people management — direct reports, hiring, performance reviews. \
Tech-leading, or "leading engineers on the team" through a project, is ordinary senior work, not \
this.
- not_remote: the posting requires office or hybrid presence, or limits remote work to places, \
time zones or states that exclude a Washington resident. A remote offer that doesn't list eligible \
states is not a mismatch.
- excluded_industry: the employer is in an industry the candidate's hard filters exclude.
- not_engineering: not a software-engineering job at all — sales, support, solutions, developer \
advocacy and the like under an engineering-sounding title.
- required_unfamiliar_language: the posting requires proficiency in a primary language or \
framework the candidate lacks ("5+ years of Go", "expert Rust", "deep React/TypeScript ownership") \
and accepts none of the candidate's languages instead.
- required_specialty: the posting requires prior specialist experience the candidate lacks — \
production ML/LLM/AI-agent systems, security, SRE/infrastructure/Kubernetes platforms, low-level \
networking, mobile, embedded, or a frontend-centered role.

For a skip, copy into disqualifier_quote the posting's exact words that establish it — one phrase \
or sentence, character for character from the title or text, no paraphrase or ellipsis — and set \
disqualifier to its category. Code only checks that the quote really is in the posting — whether \
a skill is truly required is entirely your judgment, so read its qualifiers ("ideally", "a plus", \
"or similar", "such as", "you do not need experience with ...") before skipping on it.

Never skip over: something the posting calls helpful, preferred, a plus, or learnable ("Go or \
TypeScript are helpful", "Rails, or a willingness to learn it"); a language requirement that \
accepts one of the candidate's languages ("C#, C++, or Java", "Go or a comparable language such as \
Python"); an interest the posting hopes for; an unfamiliar business domain; ordinary senior \
expectations (owning or leading the team's delivery, mentoring, ambiguity, autonomy); the role's \
level looking lower than the candidate's; salary or compensation (code has already checked it); or \
anything else not in the list above. When you're genuinely unsure, or the posting has real overlap \
alongside some gaps, do NOT skip — set disqualifier to none and leave the quote empty. Skipping a \
posting that deserved a full look silently destroys a real opportunity; sending a clear reject \
through for full evaluation only costs a few cents. Treat those two mistakes as very different in \
severity.

Always call the submit_triage tool exactly once."""


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
Workplace type (the ATS's own structured field): {job.workplace_type.value if job.workplace_type else 'not provided'}
Employment type (raw): {job.employment_type_raw or 'unknown'}
Published salary: {salary_line}

Full description:
---
{job.description_text}
---

Call submit_triage with your skip_full_evaluation decision and a one-sentence reason."""


def _fingerprint(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:12]


def evaluation_rubric_version(profile_text: str) -> str:
    """Short fingerprint of everything that decides a full evaluation's verdict — its system
    prompt, tool schema and the rendered candidate profile. Recorded on every evaluation so verdicts
    made under an older rubric can be found and refreshed (``evaluate --refresh-stale``)."""
    return _fingerprint(JOB_EVAL_SYSTEM, json.dumps(JOB_EVALUATION_TOOL_SCHEMA, sort_keys=True), profile_text)


def triage_rubric_version(profile_text: str) -> str:
    """The same for a triage screen-out. Kept separate so a fix to the cheap triage prompt only
    re-screens triage screen-outs, not every full evaluation (which never sees that prompt)."""
    return _fingerprint(TRIAGE_SYSTEM, json.dumps(TRIAGE_TOOL_SCHEMA, sort_keys=True), profile_text)


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
