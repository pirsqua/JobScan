"""Free-text and structured-field parsing: salary ranges, remote/location scope, employment type.

These are deliberately narrow, well-tested heuristics — not an attempt to understand
qualification language. Anything the regexes can't confidently resolve is left as
``unknown``/``None`` so the caller can route it to the LLM instead of guessing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from jobscan.models import EmploymentType, RemoteScope, SalarySource, WorkplaceType

HOURS_PER_YEAR = 2080

_NUMBER = r"(\d{1,3}(?:,\d{3})+|\d{2,6})(?:\.\d+)?"
_K_SUFFIX = r"\s?[kK]\b"

# Optional unit annotation directly after a number, e.g. "$85/hr" or "$170,000/yr".
_UNIT_SUFFIX = r"(?:\s?/\s?(?:hr|hour|yr|year))?"

# "$170,000 - $210,000", "$170,000—$210,000 USD", "170,000 to 210,000", "$170K-$210K",
# "$85/hr - $100/hr", "USD $129,500.00 - USD $220,500.00 /Yr", "between $112,000 and $269,000".
# "and" only separates a range after "between" — "$120,000 and a $40,000 bonus" is not a range.
_RANGE_RE = re.compile(
    rf"(?P<between>\bbetween\s+)?"
    rf"(?:(?P<pre_currency>USD|CAD)\s?)?\$?\s?(?P<low>{_NUMBER})(?P<low_k>{_K_SUFFIX})?{_UNIT_SUFFIX}"
    rf"\s?(?:-|to|–|—|~|(?(between)and|(?!)))\s?"
    rf"(?:(?:USD|CAD)\s?)?\$?\s?(?P<high>{_NUMBER})(?P<high_k>{_K_SUFFIX})?{_UNIT_SUFFIX}"
    rf"(?:\s?(?P<currency>USD|CAD))?",
    re.IGNORECASE,
)

# "$190,000 per year", "$95/hr", "$95 per hour" (single value)
_SINGLE_RE = re.compile(
    rf"\$\s?(?P<value>{_NUMBER})(?P<k>{_K_SUFFIX})?",
    re.IGNORECASE,
)

_HOURLY_HINT_RE = re.compile(r"/\s?hr\b|/\s?hour\b|per\s+hour|hourly", re.IGNORECASE)
_ANNUAL_HINT_RE = re.compile(r"per\s+year|/\s?yr\b|annually|per\s+annum|/\s?year\b", re.IGNORECASE)

# Salary lines are usually near words like "salary", "compensation", "pay range" — used to
# prioritize candidate matches when several dollar amounts appear in a long description.
_SALARY_CONTEXT_RE = re.compile(
    r"(salary|compensation|pay)\s+(range|is|of|band)|base\s+(salary|pay)|"
    r"annual\s+(salary|base)",
    re.IGNORECASE,
)

MIN_PLAUSIBLE_ANNUAL = 20_000
MAX_PLAUSIBLE_ANNUAL = 600_000
MIN_PLAUSIBLE_HOURLY = 15
MAX_PLAUSIBLE_HOURLY = 300


def _to_number(raw: str, has_k_suffix: bool) -> float:
    value = float(raw.replace(",", ""))
    if has_k_suffix and value < 1000:
        value *= 1000
    return value


@dataclass
class ParsedSalary:
    salary_min: float | None
    salary_max: float | None
    salary_currency: str | None
    salary_period: str | None  # "year" | "hour"
    salary_source: SalarySource


_HOURLY_PROXIMITY_CHARS = 45


def _is_hourly_nearby(text: str, start: int, end: int) -> bool:
    """Whether an hourly-pay indicator appears close to this specific match — not just anywhere
    in the document. Job descriptions commonly include a generic "(or hourly wage range, if
    applicable)" disclaimer as pay-transparency boilerplate even for salaried roles; treating the
    whole text as hourly-paid because that phrase appears somewhere makes an unrelated
    $155,000-$400,000 range look like an implausible hourly rate and get discarded."""
    nearby = text[max(0, start - _HOURLY_PROXIMITY_CHARS) : end + _HOURLY_PROXIMITY_CHARS]
    return bool(_HOURLY_HINT_RE.search(nearby))


def parse_salary_from_text(text: str | None) -> ParsedSalary:
    """Best-effort extraction of a salary range from free-form job description text."""
    if not text:
        return ParsedSalary(None, None, None, None, SalarySource.NONE)

    candidates: list[tuple[int, float, float, str | None, str]] = []  # (score, min, max, currency, period)
    for match in _RANGE_RE.finditer(text):
        low = _to_number(match.group("low"), bool(match.group("low_k")))
        high = _to_number(match.group("high"), bool(match.group("high_k")))
        if low > high:
            low, high = high, low
        period = "hour" if _is_hourly_nearby(text, match.start(), match.end()) else "year"
        lo_bound, hi_bound = (
            (MIN_PLAUSIBLE_HOURLY, MAX_PLAUSIBLE_HOURLY)
            if period == "hour"
            else (MIN_PLAUSIBLE_ANNUAL, MAX_PLAUSIBLE_ANNUAL)
        )
        if not (lo_bound <= low <= hi_bound and lo_bound <= high <= hi_bound):
            continue
        window_start = max(0, match.start() - 60)
        context = text[window_start : match.start()]
        score = 1 if _SALARY_CONTEXT_RE.search(context) else 0
        candidates.append((score, low, high, match.group("currency") or match.group("pre_currency"), period))

    if candidates:
        candidates.sort(key=lambda c: c[0], reverse=True)
        _, low, high, currency, period = candidates[0]
        return ParsedSalary(low, high, (currency or "USD").upper(), period, SalarySource.DESCRIPTION)

    # Fall back to a single dollar figure (e.g. "starting at $190,000").
    single_candidates: list[tuple[int, float, str]] = []
    for match in _SINGLE_RE.finditer(text):
        value = _to_number(match.group("value"), bool(match.group("k")))
        period = "hour" if _is_hourly_nearby(text, match.start(), match.end()) else "year"
        lo_bound, hi_bound = (
            (MIN_PLAUSIBLE_HOURLY, MAX_PLAUSIBLE_HOURLY)
            if period == "hour"
            else (MIN_PLAUSIBLE_ANNUAL, MAX_PLAUSIBLE_ANNUAL)
        )
        if not (lo_bound <= value <= hi_bound):
            continue
        window_start = max(0, match.start() - 60)
        context = text[window_start : match.start()]
        score = 1 if _SALARY_CONTEXT_RE.search(context) else 0
        single_candidates.append((score, value, period))

    if single_candidates:
        single_candidates.sort(key=lambda c: c[0], reverse=True)
        _, value, period = single_candidates[0]
        return ParsedSalary(value, value, "USD", period, SalarySource.DESCRIPTION)

    return ParsedSalary(None, None, None, None, SalarySource.NONE)


def annualize(value: float | None, period: str | None) -> float | None:
    if value is None:
        return None
    if period == "hour":
        return value * HOURS_PER_YEAR
    return value


def resolve_salary(
    structured_min: float | None,
    structured_max: float | None,
    structured_currency: str | None,
    structured_period: str | None,
    description_text: str | None,
) -> ParsedSalary:
    """Prefer structured API fields (Ashby compensation, Greenhouse pay metadata); fall back to
    regex-parsing the description text, which is how most Greenhouse/Lever boards actually
    publish pay-transparency ranges."""
    if structured_min is not None or structured_max is not None:
        return ParsedSalary(
            structured_min,
            structured_max if structured_max is not None else structured_min,
            (structured_currency or "USD").upper(),
            structured_period or "year",
            SalarySource.STRUCTURED,
        )
    return parse_salary_from_text(description_text)


# ---------------------------------------------------------------------------
# Location / remote-scope parsing
# ---------------------------------------------------------------------------

_REMOTE_RE = re.compile(r"\bremote\b", re.IGNORECASE)
# "U.S." needs its own branch: a trailing \b after the final "." never matches before ")" or
# end-of-string, which made "Remote (U.S.)" fall through to UNKNOWN.
_US_RE = re.compile(r"\b(?:us|usa|united states|north america)\b|\bu\.s\.?(?!\w)", re.IGNORECASE)
# Any sign the posting offers remote work at all, anywhere in the location field or the text.
_REMOTE_MENTION_RE = re.compile(r"\bremote(ly)?\b|\bwork(ing)? from (home|anywhere)\b|\btelecommut", re.IGNORECASE)
_HYBRID_RE = re.compile(r"\bhybrid\b", re.IGNORECASE)
# Deliberately excludes generic "on-site" mentions inside benefits blurbs by requiring the word
# to appear where a location would (raw location field, or immediately near "only"/"required").
_ONSITE_LOCATION_RE = re.compile(r"\bon-?site\b", re.IGNORECASE)

_US_STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin",
    "WY": "Wyoming",
}


def _state_excluded(text: str, state_abbr: str, state_name: str) -> bool:
    """Detect explicit "remote, except CA/NY/WA" style exclusions naming the candidate's state."""
    exclusion_re = re.compile(
        rf"(except|excluding|not\s+(?:available|eligible)\s+(?:in|for)|cannot\s+(?:be\s+)?"
        rf"(?:based|located)\s+in)[^.]{{0,80}}\b({re.escape(state_abbr)}|{re.escape(state_name)})\b",
        re.IGNORECASE,
    )
    return bool(exclusion_re.search(text))


# Locations that are confidently outside the U.S. — a genuine geographic fact, not a judgment
# call, so a "remote" posting tied to one of these is safely excluded rather than deferred to
# the LLM. Deliberately does NOT include ambiguous U.S. hub-city names (e.g. "San Francisco",
# "New York, NY (HQ)") — those commonly appear on postings that are actually open nationwide
# with a listed HQ, so treating them as a hard rejection would be guessing, not fact-checking.
_NON_US_LOCATION_RE = re.compile(
    r"\b("
    r"canada|ontario|toronto|vancouver|montreal|quebec|"
    r"united kingdom|\buk\b|london|england|scotland|"
    r"india|mumbai|bangalore|bengaluru|hyderabad|pune|delhi|"
    r"germany|berlin|munich|"
    r"ireland|dublin|"
    r"poland|warsaw|krakow|"
    r"australia|sydney|melbourne|"
    r"mexico|"
    r"philippines|manila|"
    r"singapore|"
    r"brazil|sao paulo|"
    r"france|paris|"
    r"spain|madrid|barcelona|"
    r"netherlands|amsterdam|"
    r"japan|tokyo|"
    r"china|beijing|shanghai|"
    r"emea|apac|latam"
    r")\b",
    re.IGNORECASE,
)


def parse_workplace_type(label: str | None) -> WorkplaceType | None:
    """Map an ATS's own workplace label — "Remote", "Hybrid (Remote/Office)", "OnSite", "ON_SITE",
    "Office/Site only", "TELECOMMUTE", ... — onto WorkplaceType. Unrecognized/"unspecified" -> None."""
    norm = re.sub(r"[^a-z]", "", (label or "").lower())
    if not norm:
        return None
    if "hybrid" in norm or ("remote" in norm and ("office" in norm or "site" in norm)):
        return WorkplaceType.HYBRID
    if "remote" in norm or "telecommute" in norm:
        return WorkplaceType.REMOTE
    if "onsite" in norm or "office" in norm or "site" in norm:
        return WorkplaceType.ONSITE
    return None


def normalize_location(
    location_raw: str | None,
    description_text: str | None,
    workplace_type: WorkplaceType | None,
    candidate_state_abbr: str = "WA",
    candidate_state_name: str = "Washington",
) -> tuple[RemoteScope, str | None]:
    """Returns (scope, note). ``note`` explains ambiguous/restricted determinations.

    Precedence: the ATS's own structured workplace field, then explicit words in the location
    field, then — only when nothing anywhere offers remote work — onsite. A posting that never
    mentions remote work in its location or its text is not a remote posting, whatever city it
    names; one that does mention it but ties it to a city stays UNKNOWN for the LLM, since text
    like "work in the office as many days as you'd like or 100% remotely" is real remote
    eligibility that no location-string heuristic can see."""
    location = location_raw or ""
    text_blob = f"{location}\n{description_text or ''}"

    if _state_excluded(text_blob, candidate_state_abbr, candidate_state_name):
        return RemoteScope.REMOTE_US_RESTRICTED, f"{candidate_state_name} explicitly excluded from remote eligibility"

    if workplace_type == WorkplaceType.HYBRID or _HYBRID_RE.search(location):
        return RemoteScope.HYBRID, "source marks it hybrid"
    if workplace_type == WorkplaceType.ONSITE:
        return RemoteScope.ONSITE, "source marks it on-site"

    is_remote = workplace_type == WorkplaceType.REMOTE or bool(_REMOTE_RE.search(location))

    if is_remote:
        if _US_RE.search(location) or "remote" == location.strip().lower():
            return RemoteScope.REMOTE_US, None
        if _NON_US_LOCATION_RE.search(location):
            return (
                RemoteScope.REMOTE_US_RESTRICTED,
                f"remote but tied to a non-U.S. location '{location.strip()}'",
            )
        if re.search(r"[A-Za-z]", location):
            # Remote, tied to a named region that isn't clearly non-U.S. (e.g. a U.S. hub city
            # like "San Francisco" or "New York, NY (HQ)") — could be nationwide-with-a-hub or a
            # genuine region restriction. Not a fact Python can safely resolve; the LLM verifies
            # remote eligibility from the full posting text instead of a guess here.
            return (
                RemoteScope.UNKNOWN,
                f"remote but location field specifies '{location.strip()}' — ambiguous U.S. region, deferred to LLM",
            )
        return RemoteScope.REMOTE_US, None

    if _ONSITE_LOCATION_RE.search(location):
        return RemoteScope.ONSITE, "location field says on-site"

    if not _REMOTE_MENTION_RE.search(text_blob):
        return RemoteScope.ONSITE, "posting never mentions remote work in its location or text"

    return RemoteScope.UNKNOWN, "remote work mentioned in the text but not in the location field — deferred to LLM"


# ---------------------------------------------------------------------------
# Employment type parsing
# ---------------------------------------------------------------------------

_CONTRACT_RE = re.compile(r"\bcontract(?:or)?\b|\bcontract-to-hire\b|\bc2h\b|\btemporary\b|\btemp\b", re.IGNORECASE)
_PART_TIME_RE = re.compile(r"\bpart[\s-]?time\b", re.IGNORECASE)
_INTERN_RE = re.compile(r"\bintern(ship)?\b", re.IGNORECASE)
_FULL_TIME_RE = re.compile(r"\bfull[\s-]?time\b", re.IGNORECASE)


def normalize_employment_type(employment_type_raw: str | None) -> EmploymentType:
    """Employment type is read from the source's own structured field only — free-text
    descriptions are too easy to misread (e.g. "no contract work" would false-positive on a
    naive text search), so ambiguity here is left to the LLM rather than guessed at."""
    raw = employment_type_raw or ""
    if _CONTRACT_RE.search(raw):
        return EmploymentType.CONTRACT
    if _PART_TIME_RE.search(raw):
        return EmploymentType.PART_TIME
    if _INTERN_RE.search(raw):
        return EmploymentType.INTERN
    if _FULL_TIME_RE.search(raw):
        return EmploymentType.FULL_TIME
    return EmploymentType.UNKNOWN
