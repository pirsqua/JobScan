"""Seattle-local (Pacific) time for human-facing output (report/audit filenames, report headers).

Data timestamps stored in the database (first_seen_at, published_at, evaluation created_at, ...)
stay in UTC — that's what makes stored history unambiguous across a DST transition. Only
display-facing timestamps generated at report/output time are shown in Seattle local time.

Computed from stdlib only (no ``zoneinfo``/``tzdata``): US daylight saving is fixed by federal
law (2nd Sunday in March to 1st Sunday in November), so the PST/PDT offset can be derived
directly rather than pulled from an IANA timezone database.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

_PST = timezone(timedelta(hours=-8), name="PST")
_PDT = timezone(timedelta(hours=-7), name="PDT")


def _nth_sunday_2am_utc(year: int, month: int, n: int, standard_utc_offset_hours: int) -> datetime:
    """The nth Sunday of `month`/`year` at 2:00 AM local time, expressed in UTC."""
    first_of_month = datetime(year, month, 1)
    days_to_first_sunday = (6 - first_of_month.weekday()) % 7  # Monday=0 ... Sunday=6
    sunday = first_of_month + timedelta(days=days_to_first_sunday, weeks=n - 1)
    return (sunday.replace(hour=2) - timedelta(hours=standard_utc_offset_hours)).replace(tzinfo=timezone.utc)


def _is_us_dst(utc_dt: datetime) -> bool:
    dst_start = _nth_sunday_2am_utc(utc_dt.year, 3, 2, standard_utc_offset_hours=-8)  # 2 AM PST
    dst_end = _nth_sunday_2am_utc(utc_dt.year, 11, 1, standard_utc_offset_hours=-7)  # 2 AM PDT
    return dst_start <= utc_dt < dst_end


def now_seattle() -> datetime:
    utc_now = datetime.now(timezone.utc)
    return utc_now.astimezone(_PDT if _is_us_dst(utc_now) else _PST)


def filename_timestamp() -> str:
    """e.g. '20260914T143022PT' — sortable and unambiguous about which timezone it's in."""
    return now_seattle().strftime("%Y%m%dT%H%M%S") + "PT"
