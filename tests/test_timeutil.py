from __future__ import annotations

from datetime import datetime, timezone

from jobscan.timeutil import _is_us_dst, filename_timestamp, now_seattle


class TestUsDstBoundaries:
    def test_summer_date_is_dst(self):
        # September is well within DST.
        assert _is_us_dst(datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc))

    def test_winter_date_is_not_dst(self):
        # January is well outside DST.
        assert not _is_us_dst(datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc))

    def test_dst_starts_second_sunday_of_march_2026(self):
        # 2026's second Sunday of March is the 8th; DST begins 2:00 AM PST (10:00 UTC).
        just_before = datetime(2026, 3, 8, 9, 59, tzinfo=timezone.utc)
        just_after = datetime(2026, 3, 8, 10, 1, tzinfo=timezone.utc)
        assert not _is_us_dst(just_before)
        assert _is_us_dst(just_after)

    def test_dst_ends_first_sunday_of_november_2026(self):
        # 2026's first Sunday of November is the 1st; DST ends 2:00 AM PDT (09:00 UTC).
        just_before = datetime(2026, 11, 1, 8, 59, tzinfo=timezone.utc)
        just_after = datetime(2026, 11, 1, 9, 1, tzinfo=timezone.utc)
        assert _is_us_dst(just_before)
        assert not _is_us_dst(just_after)


class TestNowSeattle:
    def test_offset_is_pacific_daylight_or_standard(self):
        offset = now_seattle().utcoffset()
        assert offset.total_seconds() / 3600 in (-7, -8)

    def test_matches_utc_now_converted(self):
        seattle = now_seattle()
        utc = datetime.now(timezone.utc)
        assert abs((seattle.astimezone(timezone.utc) - utc).total_seconds()) < 5


class TestFilenameTimestamp:
    def test_format(self):
        stamp = filename_timestamp()
        assert stamp.endswith("PT")
        assert len(stamp) == len("20260914T143022") + len("PT")
        # Sortable: no separators that would break lexicographic == chronological ordering.
        assert stamp[:8].isdigit()
