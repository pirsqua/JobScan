"""Seattle-local (Pacific) time for human-facing output (report/audit filenames, report headers).

Data timestamps stored in the database (first_seen_at, published_at, evaluation created_at, ...)
stay in UTC — that's what makes stored history unambiguous across a DST transition. Only
display-facing timestamps generated at report/output time are shown in Seattle local time.

Needs the ``tzdata`` package on Windows, which doesn't ship an IANA timezone database — that's
still a better trade than hand-rolling DST transition math, which is one law change away from
being silently wrong.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

SEATTLE_TZ = ZoneInfo("America/Los_Angeles")


def now_seattle() -> datetime:
    return datetime.now(SEATTLE_TZ)


def filename_timestamp() -> str:
    """e.g. '20260914T143022PT' — sortable and unambiguous about which timezone it's in."""
    return now_seattle().strftime("%Y%m%dT%H%M%S") + "PT"
