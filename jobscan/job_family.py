"""Job-family gate: is this posting even a software/backend/data engineering role?

This is a structural, ingestion-time check — not a qualification judgment. It runs once per
posting, right after normalization and before anything is persisted, so a company's Sales,
Support, Legal, People, Design, Product, and warehouse/retail postings never enter the database
or get anywhere near an LLM call. It deliberately stays permissive on specialty and seniority
(DevOps, SRE, QA, frontend, Principal, EM titles all still pass) — sorting those against the
candidate's actual fit is the LLM's job, not a title regex's.
"""
from __future__ import annotations

import re

# Titles that are a software/backend/data/platform engineering role. Matched as substrings, so
# "Senior Software Engineer II" and "Software Engineer, Growth" both match "software engineer".
_ENGINEERING_TITLE_RE = re.compile(
    r"\b("
    r"software\s+(development\s+)?engineer|software\s+developer|"
    r"backend\s+engineer|back-end\s+engineer|"
    r"full[\s-]?stack\s+(software\s+)?engineer|"
    r"data\s+engineer|platform\s+engineer|applications?\s+engineer|"
    r"api\s+engineer|integration\s+engineer|"
    r"infrastructure\s+engineer|cloud\s+engineer|devops\s+engineer|"
    r"site\s+reliability\s+engineer|\bsre\b|"
    r"machine\s+learning\s+engineer|\bml\s+engineer|ai\s+engineer|"
    r"security\s+engineer|"
    r"principal\s+engineer|staff\s+engineer|lead\s+engineer|senior\s+engineer|"
    r"engineering\s+manager|director\s+of\s+engineering|vp\s+of\s+engineering|"
    r"\bswe\b|\bsde\b|\bdeveloper\b|\bprogrammer\b|"
    r"member\s+of\s+technical\s+staff|technical\s+lead|tech\s+lead|"
    r"qa\s+engineer|quality\s+assurance\s+engineer|test\s+engineer|automation\s+engineer|sdet|"
    r"frontend\s+engineer|front-end\s+engineer|mobile\s+engineer|"
    r"android\s+(engineer|developer)|ios\s+(engineer|developer)|"
    r"firmware\s+engineer|embedded\s+(engineer|software)|reliability\s+engineer|"
    r"systems?\s+engineer"
    r")\b",
    re.IGNORECASE,
)

# Matched separately from _ENGINEERING_TITLE_RE: a leading "." (as in ".NET Engineer") is not a
# word character, so it can never satisfy a preceding \b boundary — wrapping it in the same
# \b(...)\b group above would silently never match.
_DOTNET_RE = re.compile(r"\.net\s+(engineer|developer)", re.IGNORECASE)

# "Engineer" titles that are almost never software-development roles. Checked first so they
# aren't swept in by a broader match (none of these currently overlap _ENGINEERING_TITLE_RE, but
# this keeps that guarantee explicit rather than implicit).
_NON_SOFTWARE_ENGINEER_RE = re.compile(
    r"\b("
    r"sales\s+engineer|support\s+engineer|field\s+engineer|network\s+engineer|"
    r"solutions?\s+engineer|customer\s+engineer|"
    r"process\s+engineer|manufacturing\s+engineer|mechanical\s+engineer|"
    r"electrical\s+engineer|civil\s+engineer|chemical\s+engineer|"
    r"hardware\s+engineer|industrial\s+engineer|validation\s+engineer|"
    r"escalation\s+engineer"
    r")\b",
    re.IGNORECASE,
)


def is_engineering_title(title: str) -> bool:
    """True if this title reads as a software/backend/data engineering role worth keeping in a
    software-engineering job search. Errs toward inclusion on genuinely ambiguous titles."""
    if _NON_SOFTWARE_ENGINEER_RE.search(title):
        return False
    return bool(_ENGINEERING_TITLE_RE.search(title) or _DOTNET_RE.search(title))
