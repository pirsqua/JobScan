"""The candidate's job applications: one entry per posting applied to, kept in a hand-editable,
version-controlled YAML file (``Settings.applications_path``, default ``data/applications.yaml``)
and matched to crawled postings by URL, so a report can mark each applied role wherever it now
stands — shortlisted, rejected, or closed since.

    applications:
      - company: Posit
        title: Senior Software Engineer
        url: https://posit.co/job-detail/?gh_jid=7984391003
        applied_on: 2026-10-06
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import yaml


@dataclass
class Application:
    company: str
    title: str
    url: str
    applied_on: dt.date


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
        applied_on = entry["applied_on"]
        if isinstance(applied_on, str):
            applied_on = dt.date.fromisoformat(applied_on)
        applications.append(Application(
            company=str(entry["company"]), title=str(entry["title"]), url=str(entry["url"]), applied_on=applied_on,
        ))
    return applications
