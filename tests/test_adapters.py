from __future__ import annotations

import httpx
import pytest
import respx

from jobscan.adapters.ashby import AshbyAdapter
from jobscan.adapters.base import AdapterError
from jobscan.adapters.greenhouse import GreenhouseAdapter
from jobscan.adapters.lever import LeverAdapter
from jobscan.adapters.workday import WorkdayAdapter
from jobscan.models import SalarySource


@pytest.fixture()
def http_client():
    with httpx.Client() as client:
        yield client


class TestGreenhouseAdapter:
    @respx.mock
    def test_parses_jobs_and_salary_metadata_hint(self, http_client):
        respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
            return_value=httpx.Response(
                200,
                json={
                    "jobs": [
                        {
                            "id": 1001,
                            "title": "Senior Backend Engineer",
                            "location": {"name": "Remote - US"},
                            "absolute_url": "https://acme.example.com/jobs/1001",
                            "content": "<p>Join our backend team.</p>",
                            "metadata": [{"name": "Salary Range", "value": "$170,000 - $210,000"}],
                            "departments": [{"name": "Engineering"}],
                            "updated_at": "2026-08-01T00:00:00Z",
                        }
                    ],
                    "meta": {"total": 1},
                },
            )
        )
        adapter = GreenhouseAdapter(http_client)
        postings = list(adapter.fetch_postings("acme"))
        assert len(postings) == 1
        posting = postings[0]
        assert posting.source_job_id == "1001"
        assert posting.title == "Senior Backend Engineer"
        assert "170,000" in posting.description_html

    @respx.mock
    def test_missing_jobs_key_raises_adapter_error(self, http_client):
        respx.get("https://boards-api.greenhouse.io/v1/boards/broken/jobs").mock(
            return_value=httpx.Response(200, json={"unexpected": "shape"})
        )
        adapter = GreenhouseAdapter(http_client)
        with pytest.raises(AdapterError):
            list(adapter.fetch_postings("broken"))

    @respx.mock
    def test_http_error_raises_adapter_error(self, http_client):
        respx.get("https://boards-api.greenhouse.io/v1/boards/gone/jobs").mock(
            return_value=httpx.Response(404)
        )
        adapter = GreenhouseAdapter(http_client)
        with pytest.raises(AdapterError):
            list(adapter.fetch_postings("gone"))

    @respx.mock
    def test_null_metadata_and_departments_do_not_fail_validation(self, http_client):
        # Observed live: some real Greenhouse boards send an explicit JSON null (not an omitted
        # key) for these fields when there's nothing to report, rather than an empty list.
        respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
            return_value=httpx.Response(
                200,
                json={
                    "jobs": [
                        {
                            "id": 42,
                            "title": "Senior Backend Engineer",
                            "location": {"name": "Remote - US"},
                            "absolute_url": "https://acme.example.com/jobs/42",
                            "content": "<p>Join us.</p>",
                            "metadata": None,
                            "departments": None,
                        }
                    ]
                },
            )
        )
        adapter = GreenhouseAdapter(http_client)
        postings = list(adapter.fetch_postings("acme"))
        assert len(postings) == 1
        assert postings[0].source_job_id == "42"

    @respx.mock
    def test_list_valued_metadata_value_does_not_fail_validation(self, http_client):
        # Observed live: Greenhouse custom fields (e.g. a multi-select "Office Locations" field)
        # can have a list value, not just a scalar string/number.
        respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
            return_value=httpx.Response(
                200,
                json={
                    "jobs": [
                        {
                            "id": 43,
                            "title": "Senior Backend Engineer",
                            "location": {"name": "Remote - US"},
                            "absolute_url": "https://acme.example.com/jobs/43",
                            "content": "<p>Join us.</p>",
                            "metadata": [{"name": "Office Locations", "value": ["Seattle", "Remote"]}],
                            "departments": [],
                        }
                    ]
                },
            )
        )
        adapter = GreenhouseAdapter(http_client)
        postings = list(adapter.fetch_postings("acme"))
        assert len(postings) == 1


class TestAshbyAdapter:
    @respx.mock
    def test_parses_jobs_with_compensation(self, http_client):
        respx.get("https://api.ashbyhq.com/posting-api/job-board/acme").mock(
            return_value=httpx.Response(
                200,
                json={
                    "jobs": [
                        {
                            "id": "abc-123",
                            "title": "Staff Software Engineer",
                            "location": "Remote - US",
                            "isRemote": True,
                            "employmentType": "FullTime",
                            "descriptionPlain": "Build our platform.",
                            "jobUrl": "https://jobs.ashbyhq.com/acme/abc-123",
                            "applyUrl": "https://jobs.ashbyhq.com/acme/abc-123/apply",
                            "department": "Engineering",
                            "publishedAt": "2026-08-01T00:00:00Z",
                            "compensation": {"compensationTierSummary": "$180K – $230K"},
                        }
                    ]
                },
            )
        )
        adapter = AshbyAdapter(http_client)
        postings = list(adapter.fetch_postings("acme"))
        assert len(postings) == 1
        posting = postings[0]
        assert posting.remote_flag is True
        assert "180K" in posting.description_text

    @respx.mock
    def test_missing_jobs_key_raises(self, http_client):
        respx.get("https://api.ashbyhq.com/posting-api/job-board/broken").mock(
            return_value=httpx.Response(200, json={})
        )
        adapter = AshbyAdapter(http_client)
        with pytest.raises(AdapterError):
            list(adapter.fetch_postings("broken"))


class TestLeverAdapter:
    @respx.mock
    def test_parses_jobs_with_structured_salary(self, http_client):
        respx.get("https://api.lever.co/v0/postings/acme").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {
                        "id": "lev-1",
                        "text": "Backend Engineer",
                        "categories": {"location": "Remote - US", "commitment": "Full-time", "team": "Platform"},
                        "workplaceType": "remote",
                        "description": "<p>About the role</p>",
                        "lists": [{"text": "Requirements", "content": "<ul><li>SQL</li></ul>"}],
                        "hostedUrl": "https://jobs.lever.co/acme/lev-1",
                        "salaryRange": {"min": 160000, "max": 200000, "currency": "USD", "interval": "per-year-salary"},
                        "createdAt": 1750000000000,
                    }
                ],
            )
        )
        adapter = LeverAdapter(http_client)
        postings = list(adapter.fetch_postings("acme"))
        assert len(postings) == 1
        posting = postings[0]
        assert posting.salary_min == 160000
        assert posting.salary_max == 200000
        assert posting.salary_source == SalarySource.STRUCTURED
        assert posting.remote_flag is True
        assert posting.published_at is not None

    @respx.mock
    def test_pagination_follows_skip_until_short_page(self, http_client, monkeypatch):
        import jobscan.adapters.lever as lever_module

        monkeypatch.setattr(lever_module, "PAGE_SIZE", 2)

        def make_job(job_id: str) -> dict:
            return {
                "id": job_id,
                "text": f"Engineer {job_id}",
                "categories": {"location": "Remote - US"},
                "hostedUrl": f"https://jobs.lever.co/acme/{job_id}",
            }

        route = respx.get("https://api.lever.co/v0/postings/acme")
        route.side_effect = [
            httpx.Response(200, json=[make_job("1"), make_job("2")]),
            httpx.Response(200, json=[make_job("3")]),
        ]

        adapter = LeverAdapter(http_client)
        postings = list(adapter.fetch_postings("acme"))
        assert [p.source_job_id for p in postings] == ["1", "2", "3"]
        assert route.call_count == 2

    @respx.mock
    def test_non_list_response_raises_adapter_error(self, http_client):
        respx.get("https://api.lever.co/v0/postings/broken").mock(
            return_value=httpx.Response(200, json={"error": "not found"})
        )
        adapter = LeverAdapter(http_client)
        with pytest.raises(AdapterError):
            list(adapter.fetch_postings("broken"))

    @respx.mock
    def test_network_error_raises_adapter_error(self, http_client):
        respx.get("https://api.lever.co/v0/postings/timeout").mock(
            side_effect=httpx.ConnectTimeout("timed out")
        )
        adapter = LeverAdapter(http_client)
        with pytest.raises(AdapterError):
            list(adapter.fetch_postings("timeout"))

    @respx.mock
    def test_null_categories_and_lists_do_not_fail_validation(self, http_client):
        # categories/lists are typed Optional defensively, matching the same "APIs send explicit
        # null instead of omitting the key" pattern found live on Greenhouse boards.
        respx.get("https://api.lever.co/v0/postings/acme").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {
                        "id": "lev-2",
                        "text": "Backend Engineer",
                        "categories": None,
                        "lists": None,
                        "hostedUrl": "https://jobs.lever.co/acme/lev-2",
                    }
                ],
            )
        )
        adapter = LeverAdapter(http_client)
        postings = list(adapter.fetch_postings("acme"))
        assert len(postings) == 1
        assert postings[0].source_job_id == "lev-2"
        assert postings[0].location_raw is None
        assert postings[0].employment_type_raw is None


class TestWorkdayAdapter:
    LIST_URL = "https://acme.wd1.myworkdayjobs.com/wday/cxs/acme/Acme/jobs"
    DETAIL_URL = "https://acme.wd1.myworkdayjobs.com/wday/cxs/acme/Acme/job/US-Remote/Senior-Backend-Engineer_JR1"

    @respx.mock
    def test_fetches_detail_only_for_engineering_titles(self, http_client):
        # The board lists both an engineering and a non-engineering role; only the engineering
        # one's title should trigger a (relatively expensive) per-posting detail request.
        respx.post(self.LIST_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "jobPostings": [
                        {
                            "title": "Senior Backend Engineer",
                            "externalPath": "/job/US-Remote/Senior-Backend-Engineer_JR1",
                            "locationsText": "US Remote",
                        },
                        {
                            "title": "Account Executive",
                            "externalPath": "/job/US-Remote/Account-Executive_JR2",
                            "locationsText": "US Remote",
                        },
                    ]
                },
            )
        )
        detail_route = respx.get(self.DETAIL_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "jobPostingInfo": {
                        "title": "Senior Backend Engineer",
                        "jobDescription": "<p>Build our platform.</p>",
                        "location": "US Remote",
                        "timeType": "Full time",
                        "jobReqId": "JR1",
                        "externalUrl": "https://acme.wd1.myworkdayjobs.com/Acme/job/1",
                    }
                },
            )
        )

        adapter = WorkdayAdapter(http_client)
        postings = list(adapter.fetch_postings("acme/wd1/Acme"))

        assert detail_route.call_count == 1
        assert len(postings) == 1
        posting = postings[0]
        assert posting.source_job_id == "JR1"
        assert posting.title == "Senior Backend Engineer"
        assert posting.employment_type_raw == "Full time"
        assert "Build our platform" in posting.description_html

    @respx.mock
    def test_pagination_follows_offset_until_short_page(self, http_client, monkeypatch):
        import jobscan.adapters.workday as workday_module

        monkeypatch.setattr(workday_module, "PAGE_SIZE", 2)

        def make_brief(n: int) -> dict:
            return {
                "title": f"Software Engineer {n}",
                "externalPath": f"/job/US-Remote/Software-Engineer-{n}_JR{n}",
                "locationsText": "US Remote",
            }

        list_route = respx.post(self.LIST_URL)
        list_route.side_effect = [
            httpx.Response(200, json={"jobPostings": [make_brief(1), make_brief(2)]}),
            httpx.Response(200, json={"jobPostings": [make_brief(3)]}),
        ]
        respx.get(url__regex=r".*/job/US-Remote/Software-Engineer-\d_JR\d").mock(
            return_value=httpx.Response(
                200,
                json={
                    "jobPostingInfo": {
                        "title": "Software Engineer",
                        "jobDescription": "<p>Role.</p>",
                        "timeType": "Full time",
                        "jobReqId": "JRX",
                    }
                },
            )
        )

        adapter = WorkdayAdapter(http_client)
        postings = list(adapter.fetch_postings("acme/wd1/Acme"))

        assert list_route.call_count == 2
        assert len(postings) == 3

    @respx.mock
    def test_pagination_stops_at_reported_total_even_if_pages_never_go_short(self, http_client, monkeypatch):
        # Observed live (Motorola Solutions): past the real end of results, this API doesn't
        # return a short/empty page to signal "done" — it wraps around and re-serves page 1
        # forever, with `total` still reported normally. Without using `total` to stop, this
        # loops until MAX_PAGES (or forever, before that backstop existed).
        import jobscan.adapters.workday as workday_module

        monkeypatch.setattr(workday_module, "PAGE_SIZE", 2)

        def make_brief(n: int) -> dict:
            return {"title": f"Software Engineer {n}", "externalPath": f"/job/US-Remote/Software-Engineer-{n}_JR{n}"}

        # total=4 real postings, but every page — including ones past the real end — comes back
        # full length (2), simulating the wrap-around.
        list_route = respx.post(self.LIST_URL).mock(
            return_value=httpx.Response(
                200, json={"total": 4, "jobPostings": [make_brief(1), make_brief(2)]}
            )
        )
        respx.get(url__regex=r".*/job/US-Remote/Software-Engineer-\d_JR\d").mock(
            return_value=httpx.Response(
                200,
                json={"jobPostingInfo": {"title": "Software Engineer", "jobDescription": "<p>Role.</p>", "jobReqId": "JRX"}},
            )
        )

        adapter = WorkdayAdapter(http_client)
        postings = list(adapter.fetch_postings("acme/wd1/Acme"))

        # Stops after offset reaches total=4 (2 pages of 2), not MAX_PAGES.
        assert list_route.call_count == 2

    @respx.mock
    def test_pagination_ignores_a_flaky_zero_total_on_an_otherwise_normal_page(self, http_client, monkeypatch):
        import jobscan.adapters.workday as workday_module

        monkeypatch.setattr(workday_module, "PAGE_SIZE", 2)

        def make_brief(n: int) -> dict:
            return {"title": f"Software Engineer {n}", "externalPath": f"/job/US-Remote/Software-Engineer-{n}_JR{n}"}

        list_route = respx.post(self.LIST_URL)
        list_route.side_effect = [
            # First page inconsistently reports total=0 despite having real postings.
            httpx.Response(200, json={"total": 0, "jobPostings": [make_brief(1), make_brief(2)]}),
            httpx.Response(200, json={"total": 3, "jobPostings": [make_brief(3)]}),
        ]
        respx.get(url__regex=r".*/job/US-Remote/Software-Engineer-\d_JR\d").mock(
            return_value=httpx.Response(
                200,
                json={"jobPostingInfo": {"title": "Software Engineer", "jobDescription": "<p>Role.</p>", "jobReqId": "JRX"}},
            )
        )

        adapter = WorkdayAdapter(http_client)
        postings = list(adapter.fetch_postings("acme/wd1/Acme"))

        assert list_route.call_count == 2
        assert len(postings) == 3

    @respx.mock
    def test_pagination_stops_at_max_pages_if_total_is_never_reached(self, http_client, monkeypatch):
        # Backstop for the case where `total` is missing/wrong for every page — must not hang.
        import jobscan.adapters.workday as workday_module

        monkeypatch.setattr(workday_module, "PAGE_SIZE", 2)
        monkeypatch.setattr(workday_module, "MAX_PAGES", 3)

        def make_brief(n: int) -> dict:
            return {"title": f"Software Engineer {n}", "externalPath": f"/job/US-Remote/Software-Engineer-{n}_JR{n}"}

        list_route = respx.post(self.LIST_URL).mock(
            return_value=httpx.Response(200, json={"jobPostings": [make_brief(1), make_brief(2)]})
        )
        respx.get(url__regex=r".*/job/US-Remote/Software-Engineer-\d_JR\d").mock(
            return_value=httpx.Response(
                200,
                json={"jobPostingInfo": {"title": "Software Engineer", "jobDescription": "<p>Role.</p>", "jobReqId": "JRX"}},
            )
        )

        adapter = WorkdayAdapter(http_client)
        postings = list(adapter.fetch_postings("acme/wd1/Acme"))

        assert list_route.call_count == 3

    @respx.mock
    def test_detail_fetch_failure_is_skipped_not_fatal(self, http_client):
        # A single bad requisition (removed mid-crawl, malformed detail page, ...) shouldn't drop
        # the rest of an otherwise-healthy board.
        respx.post(self.LIST_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "jobPostings": [
                        {
                            "title": "Senior Backend Engineer",
                            "externalPath": "/job/US-Remote/Senior-Backend-Engineer_JR1",
                        }
                    ]
                },
            )
        )
        respx.get(self.DETAIL_URL).mock(return_value=httpx.Response(404))

        adapter = WorkdayAdapter(http_client)
        postings = list(adapter.fetch_postings("acme/wd1/Acme"))
        assert postings == []

    @respx.mock
    def test_list_http_error_raises_adapter_error(self, http_client):
        respx.post(self.LIST_URL).mock(return_value=httpx.Response(500))
        adapter = WorkdayAdapter(http_client)
        with pytest.raises(AdapterError):
            list(adapter.fetch_postings("acme/wd1/Acme"))

    def test_malformed_board_id_raises_adapter_error(self, http_client):
        with pytest.raises(AdapterError):
            list(WorkdayAdapter(http_client).fetch_postings("acme"))

    def test_board_id_with_blank_segment_raises_adapter_error(self, http_client):
        with pytest.raises(AdapterError):
            list(WorkdayAdapter(http_client).fetch_postings("acme//Acme"))
