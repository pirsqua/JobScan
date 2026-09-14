from __future__ import annotations

from datetime import datetime, timezone

from jobscan.timeutil import filename_timestamp, now_seattle


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
