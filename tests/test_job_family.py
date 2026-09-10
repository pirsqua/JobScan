from __future__ import annotations

import pytest

from jobscan.job_family import is_engineering_title


class TestPositiveMatches:
    @pytest.mark.parametrize(
        "title",
        [
            "Senior Software Engineer",
            "Software Engineer II",
            "Software Engineer, Growth",
            "Software Development Engineer",
            "Senior Backend Engineer",
            "Staff Backend Engineer, Payments",
            "Full-Stack Engineer",
            "Full Stack Software Engineer",
            "Senior Data Engineer",
            "Data Platform Engineer",
            "Senior .NET Engineer",
            "API Engineer",
            "Systems Integration Engineer, Build Systems",
            "Machine Learning Engineer, Integrity",
            "Site Reliability Engineer",
            "DevOps Engineer",
            "QA Engineer",
            "Principal Engineer",
            "Staff Engineer",
            "Engineering Manager",
            "Backend Developer",
            "Python Developer",
        ],
    )
    def test_engineering_titles_pass(self, title: str):
        assert is_engineering_title(title)


class TestNegativeMatches:
    @pytest.mark.parametrize(
        "title",
        [
            "Barista",
            "Forklift Driver",
            "Senior Account Executive",
            "Product Manager",
            "Product Designer",
            "Technical Program Manager",
            "Technical Writer",
            "Registered Nurse",
            "Executive Assistant",
            "Total Rewards Manager",
            "Community Manager",
        ],
    )
    def test_non_engineering_titles_fail(self, title: str):
        assert not is_engineering_title(title)


class TestEngineerTitledButNotSoftware:
    @pytest.mark.parametrize(
        "title",
        [
            "Sales Engineer",
            "Support Engineer",
            "Field Engineer",
            "Network Engineer",
            "Solutions Engineer",
            "Mechanical Engineer",
            "Civil Engineer",
        ],
    )
    def test_engineer_titled_non_software_roles_fail(self, title: str):
        # These contain "Engineer" but are a different profession — a naive keyword match would
        # wrongly include them.
        assert not is_engineering_title(title)
