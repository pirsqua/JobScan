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
from collections.abc import Sequence
from functools import lru_cache

from pydantic import ValidationError

from jobscan.job_family import names_software_role
from jobscan.llm.client import AnthropicClient, LlmCallError
from jobscan.llm.prompts import (
    TRIAGE_SYSTEM,
    build_candidate_profile_block,
    build_triage_user_message,
)
from jobscan.llm.schemas import TRIAGE_SCHEMA, TriageResult
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
# A Staff-titled posting whose text hires at Senior too. Observed live: Twilio's "Staff Software
# Engineer, Fraud" ("Hiring at either the Senior or Staff level") and GitLab's "Staff Backend
# Engineer" ("As a Senior or Staff Backend Engineer"). Only "or" and "/": "mentor Senior and Staff
# engineers" is common and says nothing about the level being hired.
_HIRES_AT_SENIOR_TOO_RE = re.compile(r"\bsenior\s*(/|or)\s*staff\b|\bstaff\s*(/|or)\s*senior\b")
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
# to be in the office 2 days a week" binds only SF locals; (Mesh) "Employees based in our San
# Francisco, New York, and Bangalore hubs are expected to work from the office" skipped a posting
# located "United States - Remote".
_REMOTE_OFFERED_RE = re.compile(
    r"100% remote|fully remote|remote[- ]first|remote[- ]friendly|or (work )?remotely|work from anywhere|"
    r"#li-remote|remote (option|eligible)|"
    r"(candidates|employees|those|people|you) (who are |who live |who reside )?(in|near|within|based in|located in) "
    r"(the |our |one of our )?[\w .,'-]{0,60}(area|metro|region|office|hub|location|campus)(e?s)? "
    r"(are|will|must|should|is)|if you (live|are|reside)\b|local to"
)
# A quote that rules some places out disqualifies only when the candidate's state is one of them.
# Observed live (Samsara): "a remote position open to candidates residing in the US except the San
# Francisco Bay Metro Area, NYC Metro Area, and Washington, D.C. Metro Area" skipped as not_remote.
_EXCLUSION_RE = re.compile(r"except|exclud|not (eligible|available)|cannot|can't|unable to (hire|employ)")
# A time-zone quote that names the candidate's own zone isn't a disqualifier. Observed live (Scribe):
# "Remote based permanently in PST (Pacific Standard Time)" skipped as not_remote for a Seattle
# candidate.
_TIME_ZONE_RE = re.compile(r"time ?zone|\b(eastern|central|mountain|pacific)\b|\b[ecmp][sd]?t\b")

# Any other location preference ("Preferred locations: New York or San Francisco") is a negative
# the full evaluation weighs, not a disqualifier.
_PREFERENCE_RE = re.compile(r"\b(ideally|prefer(red|ably)?|a plus|nice to have|bonus|where possible|if possible)\b")
_TRANSLATE = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "‐": "-",
                            "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-",
                            " ": " ", " ": " ", "·": " ", "•": " "})


@lru_cache
def _hours_in_zones_re(zones: tuple[str, ...]) -> re.Pattern[str] | None:
    """Hours in any of the time zones the candidate's profile rules out (``excluded_time_zones``,
    e.g. Eastern and Central), which disqualify required or merely preferred ("ideally with the
    ability to work within Eastern Time hours"). The zone's name beside "time"/"hours", its
    three-letter abbreviation (EST, CDT), or the bare two-letter one (ET, CT) only next to hours or
    time, so a "CA, CT, NJ" state list isn't read as Central Time."""
    if not zones:
        return None
    names = "|".join(re.escape(zone.lower()) for zone in zones)
    initials = "".join(sorted({zone[0].lower() for zone in zones}))
    return re.compile(
        rf"\b({names})\b[^.]{{0,30}}\b(time|hours)\b|\b[{initials}][sd]t\b|\b[{initials}]t\s+(time|hours|business)|"
        rf"\bhours\b[^.]{{0,15}}\b[{initials}]t\b"
    )


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
    excluded_time_zones: Sequence[str] = (),
) -> bool:
    """True only when the triage model's skip names an allowed category and quotes the posting's
    own words for it, and — for the title and location categories — the title or quote fits.
    ``state_abbr``/``state_name`` are the candidate's state, for quotes that exclude places, and
    ``time_zone`` the candidate's own time zone ("Pacific"), for quotes about time zones, and
    ``excluded_time_zones`` the zones whose working hours rule a posting out (none by default)."""
    if not result.skip_full_evaluation or result.disqualifier == "none":
        return False
    title = _normalize(job.title)
    if result.disqualifier == "staff_or_higher_title":
        # Decided by the title — deterministic, whatever line the model chose to quote — unless the
        # posting says it also hires at Senior.
        if not _TITLE_ABOVE_SENIOR_RE.search(title) or _DUAL_LEVEL_TITLE_RE.search(title):
            return False
        return not _HIRES_AT_SENIOR_TOO_RE.search(_normalize(job.description_text or ""))
    posting = _normalize(" ".join(filter(None, [
        job.title, job.location_raw, job.workplace_type.value if job.workplace_type else None, job.description_text,
    ])))
    if not _quoted_from(result.disqualifier_quote, posting):
        return False
    if result.disqualifier in ("required_unfamiliar_language", "required_specialty") and not result.quote_is_hard_requirement:
        # The model's own answer about its quote, asked narrowly before the decision. Observed: a
        # skip quoting "Exposure to, or strong interest in, building on top of large language
        # models" — the very example its instructions forbid — while the decision field said skip.
        return False
    if result.disqualifier == "people_management":
        return bool(_MANAGER_TITLE_RE.search(title))
    if result.disqualifier == "not_engineering":
        # Observed live: "Software Engineer, AI Enablement" skipped as not engineering.
        return not names_software_role(job.title)
    if result.disqualifier == "not_remote":
        quote = _normalize(result.disqualifier_quote)
        if _TIME_ZONE_RE.search(quote):
            zone = time_zone.lower()
            if re.search(rf"\b({re.escape(zone)}|{re.escape(zone[0])}[sd]?t)\b", quote):
                return False
            excluded = _hours_in_zones_re(tuple(excluded_time_zones))
            if excluded is not None and excluded.search(quote):
                return True
        if _PREFERENCE_RE.search(quote):
            return False
        if _EXCLUSION_RE.search(quote):
            names_state = re.search(
                rf"\b({re.escape(state_abbr)}|{re.escape(state_name)})\b(?!,? ?d\.? ?c\b)", quote, re.IGNORECASE
            )
            if not names_state:
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

    result = client.call_structured(
        system=TRIAGE_SYSTEM,
        user_message=user_message,
        schema=TRIAGE_SCHEMA,
        # Observed live at 300: a full-sentence disqualifier_quote left no room for `reason`,
        # truncating the call. Still small — triage output is a quote, a label and one sentence.
        max_tokens=800,
        cacheable_prefix=profile_block,
    )

    try:
        parsed = TriageResult.model_validate(result.data)
    except ValidationError as exc:
        raise LlmCallError(f"triage response failed schema validation: {exc}") from exc
    return (
        parsed, result.input_tokens, result.output_tokens,
        result.cache_creation_input_tokens, result.cache_read_input_tokens,
    )
