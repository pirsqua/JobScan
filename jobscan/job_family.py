"""Job-family gate: is this posting even a software/backend/data engineering role?

This is a structural, ingestion-time check — not a qualification judgment. It runs once per
posting, right after normalization and before anything is persisted, so a company's Sales,
Support, Legal, People, Design, Product, and warehouse/retail postings never enter the database
or get anywhere near an LLM call. It deliberately stays permissive on specialty and seniority
(DevOps, SRE, QA, frontend, Principal, EM titles all still pass) — sorting those against the
candidate's actual fit is the LLM's job, not a title regex's.

The rule is generic on purpose: any engineer/developer-type title passes unless it names a
different profession. An earlier allowlist of specific phrases ("data engineer", "platform
engineer", ...) silently dropped real backend roles whose titles nobody had thought of — Product
Engineer, Analytics Engineer, Managing Engineer (C#/ASP.NET), Software Architect. A posting this
gate drops is never seen again, while a non-software posting it lets through costs one cheap
triage call, so ambiguity resolves toward keeping the posting.
"""
from __future__ import annotations

import re

# Titles that name software development outright. These win over the profession list below, so a
# team name can't drop a real role ("Senior Full-Stack Software Engineer, Developer Success",
# "Software Engineer, Support Tools").
_SOFTWARE_ROLE_RE = re.compile(
    r"\b("
    r"software\s+(development\s+|dev\s+)?(engineer|developer)\w*|"
    r"back-?end\s+(software\s+)?(engineer|developer)\w*|"
    r"full[\s-]?stack\s+(software\s+)?(engineer|developer)\w*|"
    r"swe|sde|sdet|programmer\w*"
    r")\b",
    re.IGNORECASE,
)

# Professions that put "engineer" (or a department name) in the title but aren't software
# development. "engineer\w*" so "Sales Engineering Manager" is caught as well as "Sales Engineer".
# The physical-engineering disciplines allow one word in between ("Hardware Testing Engineer",
# "Electrical Infrastructure Engineer"); _SOFTWARE_ROLE_RE above keeps "Hardware Software
# Engineer". "network" stays adjacent-only so "Network Security Engineer" still passes as security.
_OTHER_PROFESSION_RE = re.compile(
    r"\b("
    r"(pre-?sales|sales|solutions?|customer(\s+success)?|support|field|escalation|"
    r"professional\s+services|consulting|technical\s+services|network|business\s+(development|value)|"
    r"(technical|product)\s+marketing)\s+engineer\w*|"
    r"(process|manufacturing|mechanical|electrical|electronics?|civil|chemical|structural|"
    r"hardware|industrial|validation|rf|radio\s+frequency|antenna|mechatronics|drilling|mine|mining|"
    r"equipment|silicon|fpga|rtl|asic|facilities|voip)(\s+\w+)?\s+engineer\w*|"
    r"(pre-?sales|sales|solutions?|professional\s+services(\s+technical)?)\s+architect\w*|"
    r"developer\s+(relations|advocate|advocacy|evangelis\w+)|developer\s+success\s+engineer\w*|"
    r"(product|ux|ui|visual|graphic|interaction)\s+designer|"
    r"(product|program|project)\s+manager|technical\s+writer|account\s+(executive|manager)|"
    r"technician|recruit\w*|sourcer"
    r")\b",
    re.IGNORECASE,
)

# Any engineer/developer-type role, however the specialty is phrased.
_ENGINEERING_ROLE_RE = re.compile(
    r"\b("
    r"engineer\w*|developer\w*|swe|sde|sdet|sre|devops|devsecops|"
    r"member\s+of\s+technical\s+staff|tech(nical)?\s+lead|software\s+development|"
    r"back-?end|full[\s-]?stack|"
    r"(software|data|application|cloud|platform|integration|api|backend|systems?|technical)"
    r"\s+architect\w*"
    r")\b",
    re.IGNORECASE,
)

# A programming language or stack in the title marks a development role even without "engineer"
# ("Platform (Lead) Consultant - .Net, Azure, API"). Custom boundaries instead of \b: a leading
# "." or a trailing "#"/"+" is not a word character, so \b would never match around ".NET",
# "C#" or "C++"; the lookarounds also keep "Java" from matching inside "JavaScript".
_LANGUAGE_RE = re.compile(
    r"(?<![\w.])(\.net|c#|c\+\+|python|java|golang|typescript|kotlin|scala|ruby|rails|node\.?js|rust)"
    r"(?![\w+#])",
    re.IGNORECASE,
)


def names_software_role(title: str) -> bool:
    """True if the title names software development outright ("Software Engineer, AI Enablement")."""
    return bool(_SOFTWARE_ROLE_RE.search(title))


def is_engineering_title(title: str) -> bool:
    """True if this title reads as a software/backend/data engineering role worth keeping in a
    software-engineering job search. Errs toward inclusion on genuinely ambiguous titles."""
    if _SOFTWARE_ROLE_RE.search(title):
        return True
    if _OTHER_PROFESSION_RE.search(title):
        return False
    return bool(_ENGINEERING_ROLE_RE.search(title) or _LANGUAGE_RE.search(title))
