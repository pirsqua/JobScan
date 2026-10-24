"""The candidate's job applications: one entry per posting applied to, kept in a hand-editable
YAML file in the personal data folder (``Settings.applications_path``, under ``JOBSCAN_DATA_DIR``)
and matched to crawled postings by URL, so a report can mark each applied role wherever it now
stands — shortlisted, rejected, or closed since — and how the application itself is going. See
data/applications.example.yaml.

    applications:
      - company: Acme
        title: Senior Software Engineer
        url: https://acme.example.com/job-detail/?gh_jid=1234567
        applied_on: 2026-01-15
        status: rejected      # optional; one of STATUSES, "applied" when absent
        status_on: 2026-02-02 # optional; when the status last changed
    paused:
      - company: Initech
        until: 2026-03-01
        reason: accepts 3 applications per 60 days

A paused company's postings stay crawled and evaluated but are held out of the shortlist until the
``until`` date, when they return on their own (an employer limiting how often you can apply).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import yaml


# Where an application has got to, in the order applications usually move. "applied" means no
# reply yet; the report counts how many days it has been waiting.
STATUSES = ("applied", "screening", "interviewing", "offer", "rejected", "withdrawn")


@dataclass
class Application:
    company: str
    title: str
    url: str
    applied_on: dt.date
    status: str = "applied"
    status_on: dt.date | None = None


def normalize_url(url: str) -> str:
    """The same posting can be written with a trailing slash or different host case
    ("https://x/jobs/1/" vs "https://x/jobs/1"); the query string is kept because some boards
    identify the posting by it (``?gh_jid=...``), and a fragment is dropped."""
    parts = urlsplit(url.strip())
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), parts.query, ""))


def load_applications(path: Path) -> list[Application]:
    """No file means no applications recorded yet. Accepts a bare list or ``{applications: [...]}``."""
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    entries = raw.get("applications", []) if isinstance(raw, dict) else raw
    applications = []
    for entry in entries:
        status = str(entry.get("status") or "applied").strip().lower()
        if status not in STATUSES:
            # A typo would otherwise quietly count the application as still waiting for a reply.
            raise ValueError(f"{entry['company']} — {entry['title']}: unknown status {status!r} in {path}; "
                             f"use one of {', '.join(STATUSES)}")
        applications.append(Application(
            company=str(entry["company"]), title=str(entry["title"]), url=str(entry["url"]),
            applied_on=_date(entry["applied_on"]), status=status,
            status_on=_date(entry["status_on"]) if entry.get("status_on") else None,
        ))
    return applications


def _date(value: dt.date | str) -> dt.date:
    """YAML reads an unquoted 2026-01-15 as a date already, a quoted one as a string."""
    return dt.date.fromisoformat(value) if isinstance(value, str) else value


@dataclass
class Pause:
    company: str
    until: dt.date  # the first day the company's postings are shown again
    reason: str = ""

    def holds(self, company_name: str, today: dt.date) -> bool:
        return company_name.casefold() == self.company.casefold() and today < self.until


def load_pauses(path: Path) -> list[Pause]:
    """The ``paused`` list of the applications file; none without the file or the key."""
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    entries = raw.get("paused", []) if isinstance(raw, dict) else []
    pauses = []
    for entry in entries:
        pauses.append(Pause(company=str(entry["company"]), until=_date(entry["until"]),
                            reason=str(entry.get("reason") or "")))
    return pauses
