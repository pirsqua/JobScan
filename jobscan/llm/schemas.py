"""Pydantic models for the structured LLM calls.

These double as (a) the JSON Schema handed to Claude as its structured-output format, so the
response is structurally guaranteed, and (b) the validator applied to whatever comes back, so a
malformed or hallucinated response is rejected rather than silently cached.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Verdict = Literal["strong_match", "plausible_match", "borderline", "reject"]
CompanyClass = Literal["product", "consulting", "unknown"]
ScopeFit = Literal["at_level", "one_step_up", "two_plus_steps_up", "below_level"]
RequirementImportance = Literal["central", "secondary"]
RequirementStrength = Literal["required", "strongly_implied", "preferred", "trait_or_interest"]
EvidenceClassification = Literal[
    "directly_demonstrated", "credibly_transferable", "weakly_inferred", "not_demonstrated"
]
SpecialistTenureClassification = Literal["meets", "adjacent", "insufficient", "not_applicable"]
WorkingHoursFit = Literal["compatible", "conflicts"]
WorkFit = Literal["in_lane", "partly_outside", "outside"]
FrontendFit = Literal["compatible", "too_heavy"]


class RequirementEvidenceItem(BaseModel):
    # Field order is generation order: quote the posting, classify how it frames the item, and only
    # then decide importance and evidence — so the wording is on the page before it's weighed.
    requirement: str
    posting_evidence: str = Field(
        description='The posting language this requirement is drawn from, quoted with its qualifiers ("ideally", "helpful", "or a willingness to learn", ...).'
    )
    stated_as: RequirementStrength = Field(
        description="How the posting's own wording frames it: required (must/required/minimum/N+ "
        'years/"you have" in a requirements list), strongly_implied (unstated but plainly '
        "unavoidable to do the job), preferred (helpful, a plus, bonus, nice to have, preferred, "
        'ideally, familiarity with, exposure to, "or willingness to learn", and technologies '
        'named as examples — "technologies like ...", "such as ..."), or trait_or_interest '
        "(curiosity, high agency, eagerness to learn, interest in a topic). A requirement that "
        "interest can satisfy is trait_or_interest even in a minimum-requirements list: "
        '"Experience or demonstrated interest in AI-native development" is met by interest — '
        "never treat it as required experience."
    )
    importance: RequirementImportance = Field(
        description="central only for required or strongly_implied items the hiring bar rests on; "
        "preferred and trait_or_interest items are always secondary."
    )
    candidate_evidence: str = Field(
        description="What in the candidate's background supports this classification, if anything."
    )
    evidence_classification: EvidenceClassification

    @model_validator(mode="after")
    def _only_required_items_are_central(self) -> RequirementEvidenceItem:
        """Enforced rather than trusted: a nice-to-have the model also tagged central would
        otherwise drag down coverage and scope exactly as if it were a hard requirement."""
        if self.stated_as in ("preferred", "trait_or_interest") and self.importance == "central":
            self.importance = "secondary"
        return self


class SpecialistTenureAssessment(BaseModel):
    classification: SpecialistTenureClassification
    specialty: str = Field(
        description="The specialty in question, e.g. 'data engineering'. Empty string if not_applicable."
    )
    explanation: str


class JobEvaluationResult(BaseModel):
    # Field order is generation order — the model writes the JSON in schema order (verified live)
    # with thinking off, so this schema IS the model's reasoning sequence. Evidence and analysis
    # come first and the verdict near the end. Observed live with the verdict first: the model committed to
    # "reject" for Pilot, then wrote in primary_rejection_reason that it had no reason to reject
    # ("[correction: no healthcare issue here] ... This role should not be a reject"); and
    # rubric rules were rationalized past (preferred items listed as growth dimensions) to fit a
    # verdict already given.
    #
    # What the job's work is comes before any requirement. Observed (Sonnet 5.5): GitHub's npm team
    # ("a full-stack software engineer" on "the open source tools that empower the JavaScript
    # ecosystem") came out a plausible match because its qualifications list accepted C#, and the
    # npm/Node.js work was filed as a preferred item and an interview risk.
    day_to_day_work: str = Field(
        description="Before any requirement: what this team's engineers actually spend their days "
        "building or running, and in which stack or domain — read from the responsibilities and "
        "the team's description, not the qualifications list alone, which is often generic (\"4+ "
        'years in C#, Python, JavaScript, ..."). One or two sentences, e.g. "Maintaining and '
        "extending npm, a JavaScript/Node.js open-source package manager, across its CLI and "
        'registry services."'
    )
    work_fit: WorkFit = Field(
        description="That day-to-day work against the candidate's demonstrated lane (strongest "
        "skills and favored work) and the specialty areas outside current focus. in_lane: the "
        "work is mostly in stacks and domains the candidate has shown, or that the profile treats "
        "as compatible; a minor frontend or tooling share is fine. partly_outside: a substantial "
        "share of the work, roughly a third or more, is in a stack or specialty the candidate "
        "lacks, such as a full-stack role whose UI-framework work is a real half of the job. "
        "outside: the work centers on a stack or specialty the candidate lacks — maintaining a "
        "JavaScript/Node.js tool, a Go proxy codebase, ML infrastructure, mobile apps — even when "
        "the qualifications list accepts the candidate's languages. Judge the work, not whether "
        "the written bar can be met."
    )
    frontend_fit: FrontendFit = Field(
        description="How much of that day-to-day work is building user interfaces (React, "
        "Angular, Vue, TypeScript UIs), against what the candidate's profile accepts — its "
        "do-not-target list and qualification rules. too_heavy: the profile rules out "
        "frontend-heavy work and building UI is a substantial, regular part of this job, roughly "
        "a third of it or more — e.g. a full-stack role shipping product features across a React "
        "frontend and the API behind it. compatible: the profile accepts that share — "
        'backend-centered work with no frontend or a minor share ("approximately 80% backend '
        'focused and 20% frontend focused", occasional UI changes, one line asking for end-to-end '
        "features), where a required React or Angular item is a requirement gap judged with the "
        "others, not a reason for too_heavy — or a profile that targets frontend work. Judge the "
        'work described, not the title: a "Full-Stack" title on backend-centered work is '
        'compatible, and a plain "Software Engineer" title on mostly UI work is too_heavy.'
    )
    requirement_evidence: list[RequirementEvidenceItem] = Field(default_factory=list)
    specialist_tenure_assessment: SpecialistTenureAssessment
    required_matches: list[str] = Field(default_factory=list)
    required_gaps: list[str] = Field(
        default_factory=list,
        description="Unmet required or strongly implied qualifications only — never a preferred item.",
    )
    preferred_only_gaps: list[str] = Field(default_factory=list)
    growth_dimensions: list[str] = Field(
        default_factory=list,
        description="Material central, REQUIRED parts of the role not already demonstrated at "
        "roughly the required level. Never a preferred/helpful/learnable item, a new business "
        "domain the posting doesn't require prior experience in, working with ambiguity, "
        "tech-leading or owning delivery within the team (technical leadership is a match), a "
        'technology named only as an example ("using technologies like AWS, MySQL and '
        'Kubernetes" — equivalent Azure/SQL Server experience meets it), or an interest. Omit '
        "minor tool differences.",
    )
    hidden_staff_signals: list[str] = Field(
        default_factory=list,
        description="Signals that this 'Senior' posting actually carries staff-level or "
        "elite-startup scope (organization-wide influence, repeated zero-to-one ownership, "
        "extreme scale/reliability, sole technical authority, ...). Breadth of influence beyond "
        "one team — never ordinary senior expectations such as owning or driving one's own "
        "features, owning the team's goals or delivery, tech-leading or leading engineers on the "
        "team through a project, sharing practices through writing or tech talks, autonomy, "
        "comfort with ambiguity, prototyping, mentoring or ambitious wording. Empty when there are "
        "none — never list reasons something is NOT a signal.",
    )
    is_product_company: bool = Field(
        description="True if this specific posting is for building/operating the company's own "
        "product or platform, false if it is client-delivery/billable consulting work."
    )
    remote_employment_verification: str = Field(
        description="Assessment of whether the role is genuinely U.S. remote and open to an "
        "employee living in the candidate's state."
    )
    working_hours_quote: str = Field(
        description="The posting's own words about working hours, time-zone overlap or the time "
        'zone employees must live in, copied from the text — e.g. "ideally with the ability to '
        'work within Eastern Time hours". Empty string when the posting says nothing about hours.'
    )
    working_hours_fit: WorkingHoursFit = Field(
        description="Those words against the working-hours filter in the candidate's hard "
        "filters, applied exactly as that filter words it. conflicts: the stated hours, overlap or "
        "time zone fail that filter — as a requirement, or as an expectation or preference "
        '("ideally with the ability to work within ... hours", "core hours 9-5 ... '
        'preferred") when the filter says a stated preference fails it too. A conflict fails a '
        "hard filter, so the verdict is reject. compatible: nothing stated, flexible hours, or "
        "hours and overlap the filter allows."
    )
    compensation_assessment: str = Field(
        description="Whether the published range passes the candidate's compensation rule — a "
        "pass/fail gate, not a score."
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Short direct quotes or close paraphrases from the posting supporting the verdict.",
    )
    minor_caveats: list[str] = Field(default_factory=list)
    scope_fit: ScopeFit = Field(
        description="Whether the role's actual scope is at, one step above, two-plus steps "
        "above, or below demonstrated experience — independent of title or keyword overlap. "
        "below_level only when the work itself is junior (close supervision, early-career or "
        "new-grad scope) — never because the stated requirements are modest or easy to meet: a "
        '"Senior Software Engineer, Backend" asking for "3+ years" to own APIs and services is '
        "at_level for a candidate far past that bar, since the bar says who may apply, not what "
        "the work is. "
        "below_level is never by itself a reason to reject: with passing pay and backend work, "
        "the verdict follows the evidence as for an at_level role."
    )
    evidence_coverage_percent: int = Field(
        ge=0,
        le=100,
        description="0 to 100: weighted coverage of important required qualifications by "
        "demonstrated or transferable evidence — not a mechanical keyword-overlap count.",
    )
    credibility_assessment: str = Field(
        description="One or two sentences on whether applying is professionally credible given fit."
    )
    why_this_is_or_is_not_gettable: str = Field(
        description="Concise, specific explanation of interview/offer plausibility given scope_fit "
        "and evidence_coverage_percent — not a restatement of the verdict."
    )
    verdict: Verdict = Field(description="Follows from the analysis above, per the verdict rules.")
    confidence: float = Field(ge=0.0, le=1.0, description="0.0 to 1.0.")
    worth_applying: bool = Field(
        description="A forced yes/no distillation of why_this_is_or_is_not_gettable, independent "
        "of the verdict label — would a real interview loop plausibly survive contact with this "
        "posting's central requirements? False when the posting REQUIRES a technology, domain, or "
        "scale the candidate has never demonstrably touched (e.g. required Rust/Kafka/"
        "Kubernetes-as-core-systems, a security clearance, formal people management, production "
        "AI/ML ownership) such that interviewers would almost certainly expose the mismatch — even "
        "when evidence_coverage_percent looks moderate from secondary-requirement overlap. Never "
        "False merely for a language or domain the posting calls helpful, preferred or learnable. "
        "True for genuine aspirational stretches where the gap is depth/scale/tenure in "
        "already-familiar territory, and always True for strong_match/plausible_match.",
    )
    primary_rejection_reason: str | None = Field(
        default=None,
        description="REQUIRED (non-null) whenever verdict is 'reject' or 'borderline' — always "
        "set it, even when the reason is already implied by required_gaps or growth_dimensions. "
        "Leave null only for strong_match/plausible_match.",
    )

    @field_validator("working_hours_quote")
    @classmethod
    def _strip_quote_marks(cls, value: str) -> str:
        """Observed live: an empty quote sent as a literal '""', and real ones wrapped in quote
        marks — the report adds its own."""
        return value.strip().strip("\"'“”").strip()


JOB_EVALUATION_SCHEMA = JobEvaluationResult.model_json_schema()


TriageDisqualifier = Literal[
    "none",
    "staff_or_higher_title",
    "people_management",
    "not_remote",
    "excluded_industry",
    "not_engineering",
    "required_unfamiliar_language",
    "required_specialty",
]


class TriageResult(BaseModel):
    # Quote and category come before the decision, so the skip has to rest on something the
    # posting actually says, in a category the instructions allow; jobscan.llm.triage
    # .skip_is_substantiated checks both before any skip is honoured.
    disqualifier_quote: str = Field(
        description="When skipping: the posting's exact words that establish the disqualifier, "
        "copied character for character from the title or text — one phrase or sentence, no "
        "paraphrase or ellipsis. Empty string when not skipping. For a skill category the quote "
        "must be a stated requirement the candidate cannot meet — never a responsibility (\"You'll "
        'help build the authentication, authorization, and access-management primitives"), never '
        'a "familiarity with" or "exposure to" item, and never one with an alternative the '
        'candidate meets ("Go, Java, or another applicable backend language"; "Proficiency in Go '
        '(or a strong willingness to ramp quickly)"; "Exposure to, or strong interest in, '
        'building on top of large language models"). If that is the strongest quote there is, '
        "don't skip."
    )
    quote_is_hard_requirement: bool = Field(
        description="About the quote alone: is it a hard requirement this candidate cannot meet? "
        "False when it is a responsibility rather than a requirement; when it asks only for "
        "familiarity, exposure, interest or willingness to learn; when it offers an alternative "
        'the candidate meets ("or another applicable backend language", "or a strong '
        'willingness to ramp quickly", "or strong interest in"); when it is preferred, ideal '
        "or a plus; when it asks for full-stack, end-to-end ownership whose backend is in one of "
        "the candidate's languages (\"Own a feature end-to-end, from the React/TypeScript UI "
        'through the API it calls", with that API in a language the candidate uses); or when the '
        "candidate's profile "
        "shows they meet it (e.g. hands-on use of AI "
        "coding tools such as Claude Code). True for a quote that really rules the candidate out, "
        "and for the title and location categories. False when not skipping."
    )
    disqualifier: TriageDisqualifier = Field(
        description="The one category from the instructions that justifies skipping, or none."
    )
    skip_full_evaluation: bool = Field(
        description="True only when confident a careful reviewer would also reject this posting "
        "outright. False whenever there's real overlap alongside gaps, or any genuine uncertainty "
        "— the full evaluation exists for exactly that judgment call."
    )
    reason: str = Field(description="One short sentence.")


TRIAGE_SCHEMA = TriageResult.model_json_schema()


class CompanyClassificationResult(BaseModel):
    classification: CompanyClass
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str = Field(
        description="One or two sentences citing what indicates product vs. consulting."
    )


COMPANY_CLASSIFICATION_SCHEMA = CompanyClassificationResult.model_json_schema()
