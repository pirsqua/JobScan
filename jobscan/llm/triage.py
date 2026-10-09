"""Cheap first-pass screen: is a posting worth the full, expensive job_eval.py evaluation at all?

Wired into evaluate_all() when settings.triage_model is configured. Validated against 692
historical evaluations before enabling: 0 false negatives on the 3 known genuine matches, ~89% of
real rejects correctly flagged skippable — see the settings.yaml comment on triage_model. A false
negative here silently destroys a real opportunity, which is a far worse failure mode than the
wasted cost of a false positive reaching the full evaluation anyway — the prompt is written to be
lenient accordingly, and callers should treat a failed triage call as "don't skip", not as
evidence the posting is a safe reject.
"""
from __future__ import annotations

import re

from pydantic import ValidationError

from jobscan.llm.client import AnthropicClient, LlmCallError
from jobscan.llm.prompts import TRIAGE_SYSTEM, build_candidate_profile_block, build_triage_user_message
from jobscan.llm.schemas import TRIAGE_TOOL_NAME, TRIAGE_TOOL_SCHEMA, TriageResult
from jobscan.models import Company, JobPosting

# A skip is honoured only when the model names an allowed category and quotes the posting's own
# words for it — code checks the quote really is in the posting (observed live: a skip justified by
# salary "arithmetic" the posting never contained). Skills (required_unfamiliar_language,
# required_specialty) are then the triage model's call: parsing skill sentences with patterns
# proved too ambiguous ("Go or Python", "ideally", "you do not need experience with ..."). The
# remaining checks are near-objective: the title (Staff+, manager) and location wording. Every
# check can only turn a skip into a full evaluation — never the reverse.

_TITLE_ABOVE_SENIOR_RE = re.compile(r"\b(staff|principal|distinguished|director|head of|vice president|vp)\b")
_DUAL_LEVEL_TITLE_RE = re.compile(r"\bsenior\s*(/|or|and|,)\s*staff\b|\bstaff\s*(/|or|and|,)\s*senior\b")
# Observed live: "leading engineers on your team" (a Senior role) claimed as people management.
_MANAGER_TITLE_RE = re.compile(r"\b(manager|head of|director)\b")
_NOT_REMOTE_RE = re.compile(
    r"hybrid|office|on-?site|in[- ]person|relocat|commut|telecommut|days? (a|per) week|time ?zone|"
    r"\b(est|cst|mst|pst|et|ct|pt)\b|exclud|not (eligible|available)|must (live|reside|be (located|based))|"
    r"\b(located|based) in\b|\bresid\w*|open only to|only open to|eligib"
)
# A location quote that offers remote work, or binds only people near some office, isn't a
# disqualifier. Observed live (SeatGeek): "as many days a week in the office as you'd like or 100%
# remotely" was quoted as not_remote; (Discord) "Candidates in the San Francisco area are required
# to be in the office 2 days a week" binds only SF locals.
_REMOTE_OFFERED_RE = re.compile(
    r"100% remote|fully remote|remote[- ]first|remote[- ]friendly|or (work )?remotely|work from anywhere|"
    r"#li-remote|remote (option|eligible)|"
    r"(candidates|employees|those|people|you) (who are |who live |who reside )?(in|near|within|based in|located in) "
    r"(the |our )?[\w .,'-]{0,40}(area|metro|region|office)s? (are|will|must|should|is)|if you (live|are|reside)\b|local to"
)
# A quote that rules some places out disqualifies only when the candidate's state is one of them.
# Observed live (Samsara): "a remote position open to candidates residing in the US except the San
# Francisco Bay Metro Area, NYC Metro Area, and Washington, D.C. Metro Area" skipped as not_remote.
_EXCLUSION_RE = re.compile(r"except|exclud|not (eligible|available)|cannot|can't|unable to (hire|employ)")
# A time-zone quote that names the candidate's own zone isn't a disqualifier. Observed live (Scribe):
# "Remote based permanently in PST (Pacific Standard Time)" skipped as not_remote for a Seattle
# candidate.
_TIME_ZONE_RE = re.compile(r"time ?zone|\b(eastern|central|mountain|pacific)\b|\b[ecmp][sd]?t\b")
# A location or hours preference ("core hours 9-5 EST preferred", "ideally ...") is a negative the
# full evaluation weighs, not a disqualifier.
_PREFERENCE_RE = re.compile(r"\b(ideally|prefer(red|ably)?|a plus|nice to have|bonus|where possible|if possible)\b")
_TRANSLATE = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "‐": "-",
                            "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-",
                            " ": " ", " ": " ", "·": " ", "•": " "})


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(_TRANSLATE).lower()).strip()


def _quoted_from(quote: str, haystack: str) -> bool:
    parts = [part.strip(" .,:;!\"'()-") for part in re.split(r"\.\.\.|…", _normalize(quote))]
    parts = [part for part in parts if len(part) >= 4]
    return bool(parts) and all(part in haystack for part in parts)


def skip_is_substantiated(
    result: TriageResult,
    job: JobPosting,
    state_abbr: str = "WA",
    state_name: str = "Washington",
    time_zone: str = "Pacific",
) -> bool:
    """True only when the triage model's skip names an allowed category and quotes the posting's
    own words for it, and — for the title and location categories — the title or quote fits.
    ``state_abbr``/``state_name`` are the candidate's state, for quotes that exclude places, and
    ``time_zone`` the candidate's own time zone ("Pacific"), for quotes about time zones."""
    if not result.skip_full_evaluation or result.disqualifier == "none":
        return False
    title = _normalize(job.title)
    if result.disqualifier == "staff_or_higher_title":
        # Decided by the title alone — deterministic, whatever line the model chose to quote.
        return bool(_TITLE_ABOVE_SENIOR_RE.search(title)) and not _DUAL_LEVEL_TITLE_RE.search(title)
    posting = _normalize(" ".join(filter(None, [
        job.title, job.location_raw, job.workplace_type.value if job.workplace_type else None, job.description_text,
    ])))
    if not _quoted_from(result.disqualifier_quote, posting):
        return False
    if result.disqualifier == "people_management":
        return bool(_MANAGER_TITLE_RE.search(title))
    if result.disqualifier == "not_remote":
        quote = _normalize(result.disqualifier_quote)
        if _PREFERENCE_RE.search(quote):
            return False
        if _EXCLUSION_RE.search(quote):
            names_state = re.search(
                rf"\b({re.escape(state_abbr)}|{re.escape(state_name)})\b(?!,? ?d\.? ?c\b)", quote, re.IGNORECASE
            )
            if not names_state:
                return False
        if _TIME_ZONE_RE.search(quote):
            zone = time_zone.lower()
            if re.search(rf"\b({re.escape(zone)}|{re.escape(zone[0])}[sd]?t)\b", quote):
                return False
        return bool(_NOT_REMOTE_RE.search(quote)) and not _REMOTE_OFFERED_RE.search(quote)
    return True


def triage_job(
    client: AnthropicClient, profile_text: str, company: Company, job: JobPosting
) -> tuple[TriageResult, int, int, int, int]:
    """Returns (result, input_tokens, output_tokens, cache_creation_input_tokens,
    cache_read_input_tokens). Raises LlmCallError on any failure."""
    profile_block = build_candidate_profile_block(profile_text)
    user_message = build_triage_user_message(company, job)

    result = client.call_tool(
        system=TRIAGE_SYSTEM,
        user_message=user_message,
        tool_schema=TRIAGE_TOOL_SCHEMA,
        tool_name=TRIAGE_TOOL_NAME,
        # Observed live at 300: a full-sentence disqualifier_quote left no room for `reason`,
        # truncating the call. Still small — triage output is a quote, a label and one sentence.
        max_tokens=800,
        cacheable_prefix=profile_block,
    )

    try:
        parsed = TriageResult.model_validate(result.input)
    except ValidationError as exc:
        raise LlmCallError(f"triage response failed schema validation: {exc}") from exc
    return (
        parsed, result.input_tokens, result.output_tokens,
        result.cache_creation_input_tokens, result.cache_read_input_tokens,
    )
