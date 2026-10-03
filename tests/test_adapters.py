from __future__ import annotations

import json

import httpx
import pytest
import respx

from jobscan.adapters.ashby import AshbyAdapter
from jobscan.adapters.avature import AvatureAdapter
from jobscan.adapters.base import AdapterError
from jobscan.adapters.esri import EsriAdapter
from jobscan.adapters.greenhouse import GreenhouseAdapter
from jobscan.adapters.icims import ICIMSAdapter
from jobscan.adapters.jazzhr import JazzHRAdapter
from jobscan.adapters.jobvite import JobviteAdapter
from jobscan.adapters.lever import LeverAdapter
from jobscan.adapters.rippling import RipplingAdapter
from jobscan.adapters.smartrecruiters import SmartRecruitersAdapter
from jobscan.adapters.workday import WorkdayAdapter
from jobscan.models import SalarySource, WorkplaceType


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
                            "workplaceType": "Remote",
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
        assert posting.workplace_type == WorkplaceType.REMOTE
        assert "180K" in posting.description_text

    @respx.mock
    def test_missing_jobs_key_raises(self, http_client):
        respx.get("https://api.ashbyhq.com/posting-api/job-board/broken").mock(
            return_value=httpx.Response(200, json={})
        )
        adapter = AshbyAdapter(http_client)
        with pytest.raises(AdapterError):
            list(adapter.fetch_postings("broken"))

    @respx.mock
    def test_workplace_type_wins_over_misleading_is_remote(self, http_client):
        # Observed live (Plaid): isRemote=True on Hybrid postings located at "San Francisco HQ".
        # Trusting isRemote let them through as ambiguous-remote; workplaceType says Hybrid.
        def job(job_id, **fields):
            return {"id": job_id, "title": "Senior Software Engineer", "location": "San Francisco HQ", **fields}

        respx.get("https://api.ashbyhq.com/posting-api/job-board/acme").mock(
            return_value=httpx.Response(200, json={"jobs": [
                job("hybrid", isRemote=True, workplaceType="Hybrid"),
                job("no-type-remote", isRemote=True),
                job("no-type-onsite", isRemote=False),
            ]})
        )
        postings = {p.source_job_id: p for p in AshbyAdapter(http_client).fetch_postings("acme")}
        assert postings["hybrid"].workplace_type == WorkplaceType.HYBRID
        assert postings["no-type-remote"].workplace_type is None  # isRemote=True alone proves nothing
        assert postings["no-type-onsite"].workplace_type == WorkplaceType.ONSITE


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
        assert posting.workplace_type == WorkplaceType.REMOTE
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

    @respx.mock
    def test_board_id_query_string_becomes_applied_facets(self, http_client):
        # A tenant's own search facets ride along in board_id and must reach the server as the
        # list request's appliedFacets (repeated names collected into one list), while the
        # list/detail URLs are built from only the tenant/cluster/site part.
        bodies = []

        def list_handler(request: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={"total": 1, "jobPostings": [{"title": "Senior Backend Engineer",
                                                   "externalPath": "/job/US-Remote/Senior-Backend-Engineer_JR1"}]},
            )

        respx.post(self.LIST_URL).mock(side_effect=list_handler)
        detail_route = respx.get(self.DETAIL_URL).mock(
            return_value=httpx.Response(200, json={"jobPostingInfo": {"title": "Senior Backend Engineer", "jobReqId": "JR1"}})
        )

        adapter = WorkdayAdapter(http_client)
        postings = list(adapter.fetch_postings("acme/wd1/Acme?locationCountry=US1&locationCountry=CA2&jobFamilyGroup=ENG"))

        assert bodies[0]["appliedFacets"] == {"locationCountry": ["US1", "CA2"], "jobFamilyGroup": ["ENG"]}
        assert detail_route.call_count == 1
        assert postings[0].source_job_id == "JR1"


class TestJobviteAdapter:
    LIST_URL = "https://jobs.jobvite.com/acme"
    DETAIL_URL = "https://jobs.jobvite.com/acme/job/abc123"

    LIST_HTML = """
    <html><body>
        <h3 class="h2">Engineering</h3>
        <table class="jv-job-list">
            <tbody>
                <tr>
                    <td class="jv-job-list-name"><a href="/acme/job/abc123">Senior Backend Engineer</a></td>
                    <td class="jv-job-list-location">United States</td>
                </tr>
            </tbody>
        </table>
        <h3 class="h2">Sales</h3>
        <table class="jv-job-list">
            <tbody>
                <tr>
                    <td class="jv-job-list-name"><a href="/acme/job/xyz789">Account Executive</a></td>
                    <td class="jv-job-list-location">Remote</td>
                </tr>
            </tbody>
        </table>
    </body></html>
    """

    @staticmethod
    def detail_html(json_ld: str, extra_blocks: str = "") -> str:
        return f"""
        <html><head>
        {extra_blocks}
        <script type="application/ld+json">{json_ld}</script>
        </head><body>Detail page</body></html>
        """

    @respx.mock
    def test_fetches_detail_only_for_engineering_titles(self, http_client):
        respx.get(self.LIST_URL).mock(return_value=httpx.Response(200, text=self.LIST_HTML))
        detail_route = respx.get(self.DETAIL_URL).mock(
            return_value=httpx.Response(
                200,
                text=self.detail_html(
                    """{
                        "@type": "JobPosting",
                        "title": "Senior Backend Engineer",
                        "description": "<p>Build our platform.</p>",
                        "datePosted": "2026-08-01",
                        "employmentType": "Full-Time"
                    }"""
                ),
            )
        )

        adapter = JobviteAdapter(http_client)
        postings = list(adapter.fetch_postings("acme"))

        assert detail_route.call_count == 1
        assert len(postings) == 1
        posting = postings[0]
        assert posting.source_job_id == "abc123"
        assert posting.title == "Senior Backend Engineer"
        assert posting.location_raw == "United States"
        assert posting.employment_type_raw == "Full-Time"
        assert "Build our platform" in posting.description_html
        assert posting.salary_source == SalarySource.NONE

    @respx.mock
    def test_extracts_structured_salary_when_published(self, http_client):
        respx.get(self.LIST_URL).mock(return_value=httpx.Response(200, text=self.LIST_HTML))
        respx.get(self.DETAIL_URL).mock(
            return_value=httpx.Response(
                200,
                text=self.detail_html(
                    """{
                        "@type": "JobPosting",
                        "title": "Senior Backend Engineer",
                        "description": "<p>Build our platform.</p>",
                        "baseSalary": {
                            "@type": "MonetaryAmount",
                            "currency": "USD",
                            "value": {"@type": "QuantitativeValue", "minValue": "170000", "maxValue": "210000", "unitText": "YEAR"}
                        }
                    }"""
                ),
            )
        )

        adapter = JobviteAdapter(http_client)
        postings = list(adapter.fetch_postings("acme"))

        assert len(postings) == 1
        posting = postings[0]
        assert posting.salary_min == 170000
        assert posting.salary_max == 210000
        assert posting.salary_currency == "USD"
        assert posting.salary_period == "year"
        assert posting.salary_source == SalarySource.STRUCTURED

    @respx.mock
    def test_empty_salary_strings_are_not_treated_as_structured(self, http_client):
        # Observed live: unpublished salary still sends the baseSalary object, just with every
        # value as an empty string rather than the field being absent.
        respx.get(self.LIST_URL).mock(return_value=httpx.Response(200, text=self.LIST_HTML))
        respx.get(self.DETAIL_URL).mock(
            return_value=httpx.Response(
                200,
                text=self.detail_html(
                    """{
                        "@type": "JobPosting",
                        "title": "Senior Backend Engineer",
                        "description": "<p>Build our platform.</p>",
                        "baseSalary": {
                            "@type": "MonetaryAmount",
                            "currency": "",
                            "value": {"@type": "QuantitativeValue", "minValue": "", "maxValue": "", "unitText": ""}
                        }
                    }"""
                ),
            )
        )

        adapter = JobviteAdapter(http_client)
        postings = list(adapter.fetch_postings("acme"))

        assert postings[0].salary_source == SalarySource.NONE
        assert postings[0].salary_min is None

    @respx.mock
    def test_ignores_non_jobposting_json_ld_blocks(self, http_client):
        # Detail pages can carry other ld+json blocks (breadcrumbs, organization, ...) — only the
        # one typed JobPosting should be used.
        respx.get(self.LIST_URL).mock(return_value=httpx.Response(200, text=self.LIST_HTML))
        respx.get(self.DETAIL_URL).mock(
            return_value=httpx.Response(
                200,
                text=self.detail_html(
                    json_ld="""{"@type": "JobPosting", "title": "Senior Backend Engineer", "description": "<p>Real.</p>"}""",
                    extra_blocks='<script type="application/ld+json">{"@type": "Organization", "name": "Acme"}</script>',
                ),
            )
        )

        adapter = JobviteAdapter(http_client)
        postings = list(adapter.fetch_postings("acme"))

        assert len(postings) == 1
        assert "Real." in postings[0].description_html

    @respx.mock
    def test_detail_missing_json_ld_is_skipped_not_fatal(self, http_client):
        respx.get(self.LIST_URL).mock(return_value=httpx.Response(200, text=self.LIST_HTML))
        respx.get(self.DETAIL_URL).mock(return_value=httpx.Response(200, text="<html><body>No data here.</body></html>"))

        adapter = JobviteAdapter(http_client)
        postings = list(adapter.fetch_postings("acme"))
        assert postings == []

    @respx.mock
    def test_list_http_error_raises_adapter_error(self, http_client):
        respx.get(self.LIST_URL).mock(return_value=httpx.Response(500))
        adapter = JobviteAdapter(http_client)
        with pytest.raises(AdapterError):
            list(adapter.fetch_postings("acme"))

    @respx.mock
    def test_multi_node_location_whitespace_is_collapsed(self, http_client):
        # Observed live: a location cell split across text nodes (e.g. "Stockholm," and "Sweden"
        # on separate lines in the source HTML) leaves embedded newlines/indentation that
        # get_text(strip=True) alone doesn't collapse.
        list_html = """
        <html><body>
            <table class="jv-job-list"><tbody><tr>
                <td class="jv-job-list-name"><a href="/acme/job/abc123">Senior Backend Engineer</a></td>
                <td class="jv-job-list-location">
                    Stockholm,
                    Sweden
                </td>
            </tr></tbody></table>
        </body></html>
        """
        respx.get(self.LIST_URL).mock(return_value=httpx.Response(200, text=list_html))
        respx.get(self.DETAIL_URL).mock(
            return_value=httpx.Response(
                200,
                text=self.detail_html(
                    """{"@type": "JobPosting", "title": "Senior Backend Engineer", "description": "<p>Role.</p>"}"""
                ),
            )
        )

        adapter = JobviteAdapter(http_client)
        postings = list(adapter.fetch_postings("acme"))
        assert postings[0].location_raw == "Stockholm, Sweden"


class TestJazzHRAdapter:
    LIST_URL = "https://acme.applytojob.com/apply/jobs/"
    DETAIL_URL = "https://acme.applytojob.com/apply/jobs/details/abc123"

    LIST_HTML = """
    <html><body>
        <table id="jobs_table">
            <tbody>
                <tr class="resumator_department_heading"><td colspan="3">Engineering</td></tr>
                <tr id="row_job_1">
                    <td><a class="job_title_link" href="/apply/jobs/details/abc123?&amp;">Senior Backend Engineer</a></td>
                    <td>Remote - US</td>
                </tr>
                <tr class="resumator_department_heading"><td colspan="3">Sales</td></tr>
                <tr id="row_job_2">
                    <td><a class="job_title_link" href="/apply/jobs/details/xyz789?&amp;">Account Executive</a></td>
                    <td>Chicago, IL</td>
                </tr>
            </tbody>
        </table>
    </body></html>
    """

    @staticmethod
    def detail_html(json_ld: str) -> str:
        return f"""
        <html><head>
        <script type="application/ld+json">{{"@type": "Organization", "name": "Acme"}}</script>
        <script type="application/ld+json">{json_ld}</script>
        </head><body>Detail page</body></html>
        """

    @respx.mock
    def test_fetches_detail_only_for_engineering_titles(self, http_client):
        respx.get(self.LIST_URL).mock(return_value=httpx.Response(200, text=self.LIST_HTML))
        detail_route = respx.get(self.DETAIL_URL).mock(
            return_value=httpx.Response(
                200,
                text=self.detail_html(
                    """{
                        "@type": "JobPosting",
                        "title": "Senior Backend Engineer",
                        "description": "<p>Build our platform.</p>",
                        "datePosted": "2026-08-01",
                        "employmentType": "FULL_TIME",
                        "baseSalary": {
                            "@type": "MonetaryAmount",
                            "currency": "USD",
                            "value": {"@type": "QuantitativeValue", "minValue": 180000, "maxValue": 220000, "unitText": "YEAR"}
                        }
                    }"""
                ),
            )
        )

        adapter = JazzHRAdapter(http_client)
        postings = list(adapter.fetch_postings("acme"))

        assert detail_route.call_count == 1
        assert len(postings) == 1
        posting = postings[0]
        assert posting.source_job_id == "abc123"
        assert posting.title == "Senior Backend Engineer"
        assert posting.location_raw == "Remote - US"
        assert posting.employment_type_raw == "FULL_TIME"
        assert "Build our platform" in posting.description_html
        assert posting.salary_min == 180000
        assert posting.salary_max == 220000
        assert posting.salary_currency == "USD"
        assert posting.salary_period == "year"
        assert posting.salary_source == SalarySource.STRUCTURED

    @respx.mock
    def test_job_id_strips_trailing_query_string(self, http_client):
        # Observed live (iManage): JazzHR list-page hrefs carry a trailing "?&" with no real
        # query params — the job id must come from the path, not a naive last-segment split.
        respx.get(self.LIST_URL).mock(return_value=httpx.Response(200, text=self.LIST_HTML))
        respx.get(self.DETAIL_URL).mock(
            return_value=httpx.Response(
                200,
                text=self.detail_html(
                    """{"@type": "JobPosting", "title": "Senior Backend Engineer", "description": "<p>Role.</p>"}"""
                ),
            )
        )
        adapter = JazzHRAdapter(http_client)
        postings = list(adapter.fetch_postings("acme"))
        assert postings[0].source_job_id == "abc123"

    @respx.mock
    def test_detail_missing_json_ld_falls_back_to_the_page_html(self, http_client):
        # Observed live (iManage): some postings' detail pages only carry the Organization
        # ld+json block, with no JobPosting block at all — the rendered page still has the full
        # description, so the posting must not be silently dropped.
        respx.get(self.LIST_URL).mock(return_value=httpx.Response(200, text=self.LIST_HTML))
        respx.get(self.DETAIL_URL).mock(
            return_value=httpx.Response(
                200,
                text='<html><head><script type="application/ld+json">{"@type": "Organization"}</script></head>'
                '<body><h1 class="job_title">Senior Backend Engineer</h1>'
                '<div class="job_description"><p>Build our platform. $180,000 - $220,000.</p></div></body></html>',
            )
        )
        adapter = JazzHRAdapter(http_client)
        postings = list(adapter.fetch_postings("acme"))
        assert len(postings) == 1
        assert postings[0].source_job_id == "abc123"
        assert postings[0].title == "Senior Backend Engineer"
        assert postings[0].location_raw == "Remote - US"
        assert "Build our platform" in postings[0].description_html

    @respx.mock
    def test_detail_with_neither_json_ld_nor_description_is_skipped_not_fatal(self, http_client):
        respx.get(self.LIST_URL).mock(return_value=httpx.Response(200, text=self.LIST_HTML))
        respx.get(self.DETAIL_URL).mock(
            return_value=httpx.Response(
                200, text='<html><head><script type="application/ld+json">{"@type": "Organization"}</script></head></html>'
            )
        )
        adapter = JazzHRAdapter(http_client)
        assert list(adapter.fetch_postings("acme")) == []

    @respx.mock
    def test_list_http_error_raises_adapter_error(self, http_client):
        respx.get(self.LIST_URL).mock(return_value=httpx.Response(500))
        adapter = JazzHRAdapter(http_client)
        with pytest.raises(AdapterError):
            list(adapter.fetch_postings("acme"))

    @respx.mock
    def test_no_jobs_table_returns_empty(self, http_client):
        respx.get(self.LIST_URL).mock(return_value=httpx.Response(200, text="<html><body>No board here.</body></html>"))
        adapter = JazzHRAdapter(http_client)
        assert list(adapter.fetch_postings("acme")) == []


class TestICIMSAdapter:
    SEARCH_URL = "https://uscareers-acme.icims.com/jobs/search"
    DETAIL_URL = "https://uscareers-acme.icims.com/jobs/101/senior-backend-engineer/job"

    @staticmethod
    def row(job_id: str, title: str, location: str) -> str:
        slug = title.lower().replace(" ", "-")
        return f"""
        <div class="row">
            <div class="col-xs-6 header left">
                <span class="sr-only field-label">Location</span>
                <span>{location}</span>
            </div>
            <div class="col-xs-12 title">
                <a class="iCIMS_Anchor" href="https://uscareers-acme.icims.com/jobs/{job_id}/{slug}/job?in_iframe=1"
                   title="{job_id} - {title}"><span class="sr-only field-label">Title</span><h3>{title}</h3></a>
            </div>
        </div>
        """

    @classmethod
    def list_page(cls, rows: list[str], page: int, of: int) -> str:
        return f"""<html><body><div class="container-fluid iCIMS_JobsTable">{''.join(rows)}</div>
        <div class="iCIMS_Paging">Page {page} of {of}</div></body></html>"""

    @staticmethod
    def detail_page(title: str, extra: str = "") -> str:
        return f"""<html><head><script type="application/ld+json">{{
            "@type": "JobPosting", "title": "{title}",
            "description": "<p>Build our platform.</p>", "datePosted": "2026-09-01T04:00:00.000Z",
            "employmentType": "FULL_TIME", "jobLocationType": "TELECOMMUTE"{extra}
        }}</script></head><body></body></html>"""

    @respx.mock
    def test_fetches_detail_only_for_engineering_titles(self, http_client):
        respx.get(self.SEARCH_URL).mock(
            return_value=httpx.Response(
                200,
                text=self.list_page(
                    [
                        self.row("101", "Senior Backend Engineer", "US-Remote-Remote | US-CA-San Francisco"),
                        self.row("102", "Account Executive", "US-IL-Chicago"),
                    ],
                    1, 1,
                ),
            )
        )
        salary = ', "baseSalary": {"currency": "USD", "value": {"minValue": 180000, "maxValue": 220000, "unitText": "YEAR"}}'
        detail_route = respx.get(self.DETAIL_URL).mock(
            return_value=httpx.Response(200, text=self.detail_page("Senior Backend Engineer", salary))
        )

        postings = list(ICIMSAdapter(http_client).fetch_postings("uscareers-acme"))

        assert detail_route.call_count == 1
        assert len(postings) == 1
        posting = postings[0]
        assert posting.source_job_id == "101"
        assert posting.title == "Senior Backend Engineer"
        assert posting.location_raw == "US-Remote-Remote | US-CA-San Francisco"
        assert posting.posting_url == self.DETAIL_URL
        assert posting.employment_type_raw == "FULL_TIME"
        assert posting.workplace_type == WorkplaceType.REMOTE
        assert "Build our platform" in posting.description_html
        assert (posting.salary_min, posting.salary_max, posting.salary_source) == (180000, 220000, SalarySource.STRUCTURED)

    @respx.mock
    def test_pages_through_until_the_reported_page_count(self, http_client):
        pages = {
            "0": self.list_page([self.row("101", "Senior Backend Engineer", "US-Remote-Remote")], 1, 2),
            "1": self.list_page([self.row("202", "Senior Data Engineer", "US-TX-Remote")], 2, 2),
        }
        requested = []

        def list_handler(request: httpx.Request) -> httpx.Response:
            assert request.url.params.get("in_iframe") == "1"
            requested.append(request.url.params.get("pr"))
            return httpx.Response(200, text=pages[request.url.params.get("pr")])

        respx.get(self.SEARCH_URL).mock(side_effect=list_handler)
        respx.get(self.DETAIL_URL).mock(return_value=httpx.Response(200, text=self.detail_page("Senior Backend Engineer")))
        respx.get("https://uscareers-acme.icims.com/jobs/202/senior-data-engineer/job").mock(
            return_value=httpx.Response(200, text=self.detail_page("Senior Data Engineer"))
        )

        postings = list(ICIMSAdapter(http_client).fetch_postings("uscareers-acme"))

        assert requested == ["0", "1"]
        assert [p.source_job_id for p in postings] == ["101", "202"]

    @respx.mock
    def test_detail_without_json_ld_is_skipped_not_fatal(self, http_client):
        respx.get(self.SEARCH_URL).mock(
            return_value=httpx.Response(
                200, text=self.list_page([self.row("101", "Senior Backend Engineer", "US-Remote-Remote")], 1, 1)
            )
        )
        respx.get(self.DETAIL_URL).mock(return_value=httpx.Response(200, text="<html><body>Gone.</body></html>"))
        assert list(ICIMSAdapter(http_client).fetch_postings("uscareers-acme")) == []

    @respx.mock
    def test_list_http_error_raises_adapter_error(self, http_client):
        respx.get(self.SEARCH_URL).mock(return_value=httpx.Response(500))
        with pytest.raises(AdapterError):
            list(ICIMSAdapter(http_client).fetch_postings("uscareers-acme"))

    @respx.mock
    def test_portal_with_no_listings_returns_empty(self, http_client):
        respx.get(self.SEARCH_URL).mock(return_value=httpx.Response(200, text="<html><body>Search jobs</body></html>"))
        assert list(ICIMSAdapter(http_client).fetch_postings("uscareers-acme")) == []


class FakeRenderer:
    """Stands in for the real Playwright-backed renderer in tests — see EsriAdapter.__init__."""

    def __init__(self, text: str = "Full rendered description.", raise_for: set[str] | None = None):
        self.text = text
        self.raise_for = raise_for or set()
        self.rendered_urls: list[str] = []
        self.closed = False

    def render(self, url: str) -> str:
        self.rendered_urls.append(url)
        if url in self.raise_for:
            raise RuntimeError("simulated render failure")
        return self.text

    def close(self) -> None:
        self.closed = True


def esri_hit(title: str, job_id: str, job_title: str | None = None, locations: str | None = None) -> dict:
    meta = {}
    if job_title is not None:
        meta["JobTitle"] = job_title
    if locations is not None:
        meta["locations"] = locations
    return {"doc": {"displayurl": f"https://www.esri.com/careers/{job_id}", "title": title, "metaFields": meta}}


class TestEsriAdapter:
    SEARCH_URL = "https://esearchapi.esri.com/search"

    @respx.mock
    def test_fetches_detail_only_for_engineering_titles(self, http_client):
        respx.post(self.SEARCH_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "search": {
                        "count": 2,
                        "hits": [
                            esri_hit(
                                "Software Development Engineer II Job | Esri Career Opportunity",
                                "1001", job_title="Software Development Engineer II", locations="Redlands-CA",
                            ),
                            esri_hit("Account Executive Job | Esri Career Opportunity", "1002", job_title="Account Executive"),
                        ],
                    }
                },
            )
        )
        renderer = FakeRenderer(text="Full job description text.")
        adapter = EsriAdapter(http_client, renderer=renderer)
        postings = list(adapter.fetch_postings("esri"))

        assert len(renderer.rendered_urls) == 1
        assert len(postings) == 1
        posting = postings[0]
        assert posting.source_job_id == "1001"
        assert posting.title == "Software Development Engineer II"
        assert posting.location_raw == "Redlands, CA"
        assert posting.description_text == "Full job description text."
        # An injected renderer is caller-owned — the adapter doesn't close it.
        assert renderer.closed is False

    @respx.mock
    def test_falls_back_to_stripped_title_when_job_title_metafield_missing(self, http_client):
        respx.post(self.SEARCH_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "search": {
                        "count": 1,
                        "hits": [esri_hit("Senior Software Engineer Job | Esri Career Opportunity", "2001")],
                    }
                },
            )
        )
        adapter = EsriAdapter(http_client, renderer=FakeRenderer())
        postings = list(adapter.fetch_postings("esri"))
        assert postings[0].title == "Senior Software Engineer"

    @respx.mock
    def test_list_valued_metafield_does_not_crash(self, http_client):
        # Observed live elsewhere (Greenhouse): a metadata field can come back as a list instead
        # of a bare string.
        hit = esri_hit("Software Engineer Job | Esri Career Opportunity", "3001", job_title="Software Engineer")
        hit["doc"]["metaFields"]["locations"] = ["Redlands-CA"]
        respx.post(self.SEARCH_URL).mock(return_value=httpx.Response(200, json={"search": {"count": 1, "hits": [hit]}}))
        adapter = EsriAdapter(http_client, renderer=FakeRenderer())
        postings = list(adapter.fetch_postings("esri"))
        assert postings[0].location_raw == "Redlands, CA"

    @respx.mock
    def test_pagination_stops_at_reported_count(self, http_client, monkeypatch):
        import jobscan.adapters.esri as esri_module

        monkeypatch.setattr(esri_module, "PAGE_SIZE", 1)

        route = respx.post(self.SEARCH_URL)
        route.side_effect = [
            httpx.Response(200, json={"search": {"count": 2, "hits": [esri_hit("Software Engineer I Job | Esri Career Opportunity", "1", job_title="Software Engineer I")]}}),
            httpx.Response(200, json={"search": {"count": 2, "hits": [esri_hit("Software Engineer II Job | Esri Career Opportunity", "2", job_title="Software Engineer II")]}}),
        ]
        adapter = EsriAdapter(http_client, renderer=FakeRenderer())
        postings = list(adapter.fetch_postings("esri"))

        assert route.call_count == 2
        assert len(postings) == 2

    @respx.mock
    def test_render_failure_is_skipped_not_fatal(self, http_client):
        respx.post(self.SEARCH_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "search": {
                        "count": 1,
                        "hits": [esri_hit("Software Engineer Job | Esri Career Opportunity", "9001", job_title="Software Engineer")],
                    }
                },
            )
        )
        renderer = FakeRenderer(raise_for={"https://www.esri.com/careers/9001"})
        adapter = EsriAdapter(http_client, renderer=renderer)
        postings = list(adapter.fetch_postings("esri"))
        assert postings == []

    @respx.mock
    def test_search_http_error_raises_adapter_error(self, http_client):
        respx.post(self.SEARCH_URL).mock(return_value=httpx.Response(500))
        adapter = EsriAdapter(http_client, renderer=FakeRenderer())
        with pytest.raises(AdapterError):
            list(adapter.fetch_postings("esri"))

    @respx.mock
    def test_owned_renderer_is_closed_after_fetch(self, http_client, monkeypatch):
        import jobscan.adapters.esri as esri_module

        respx.post(self.SEARCH_URL).mock(return_value=httpx.Response(200, json={"search": {"count": 0, "hits": []}}))

        created = {}

        class FakeOwnedRenderer(FakeRenderer):
            def __init__(self):
                super().__init__()
                created["instance"] = self

        monkeypatch.setattr(esri_module, "_PlaywrightRenderer", FakeOwnedRenderer)
        adapter = EsriAdapter(http_client)  # no renderer injected -> adapter creates + owns one
        list(adapter.fetch_postings("esri"))
        assert created["instance"].closed is True


class TestSmartRecruitersAdapter:
    LIST_URL = "https://api.smartrecruiters.com/v1/companies/acme/postings"
    DETAIL_URL = "https://api.smartrecruiters.com/v1/companies/acme/postings/sr-1"

    @respx.mock
    def test_fetches_detail_only_for_engineering_titles_with_compensation(self, http_client):
        respx.get(self.LIST_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "totalFound": 2,
                    "content": [
                        {"id": "sr-1", "name": "Senior Backend Engineer"},
                        {"id": "sr-2", "name": "Account Executive"},
                    ],
                },
            )
        )
        detail_route = respx.get(self.DETAIL_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "sr-1",
                    "name": "Senior Backend Engineer",
                    "location": {"fullLocation": "Remote, US", "remote": True},
                    "department": {"label": "Engineering"},
                    "typeOfEmployment": {"label": "Full-time"},
                    "releasedDate": "2026-08-01T00:00:00.000Z",
                    "postingUrl": "https://jobs.smartrecruiters.com/acme/sr-1",
                    "applyUrl": "https://jobs.smartrecruiters.com/acme/sr-1/apply",
                    "jobAd": {
                        "sections": {
                            "jobDescription": {"title": "Job Description", "text": "<p>Build our platform.</p>"},
                            "qualifications": {"title": "Qualifications", "text": "<p>5+ years backend.</p>"},
                        }
                    },
                    "compensation": {"min": 160000, "max": 200000, "currency": "USD", "period": "YEARLY"},
                },
            )
        )

        adapter = SmartRecruitersAdapter(http_client)
        postings = list(adapter.fetch_postings("acme"))

        assert detail_route.call_count == 1
        assert len(postings) == 1
        posting = postings[0]
        assert posting.source_job_id == "sr-1"
        assert posting.title == "Senior Backend Engineer"
        assert posting.location_raw == "Remote, US"
        assert posting.workplace_type == WorkplaceType.REMOTE
        assert posting.department == "Engineering"
        assert posting.employment_type_raw == "Full-time"
        assert "Build our platform" in posting.description_html
        assert "5+ years backend" in posting.description_html
        assert posting.posting_url == "https://jobs.smartrecruiters.com/acme/sr-1"
        assert posting.apply_url == "https://jobs.smartrecruiters.com/acme/sr-1/apply"
        assert posting.salary_min == 160000
        assert posting.salary_max == 200000
        assert posting.salary_currency == "USD"
        assert posting.salary_period == "year"
        assert posting.salary_source == SalarySource.STRUCTURED

    @respx.mock
    def test_hourly_compensation_maps_to_hour_period(self, http_client):
        respx.get(self.LIST_URL).mock(
            return_value=httpx.Response(
                200, json={"totalFound": 1, "content": [{"id": "sr-1", "name": "Backend Engineer"}]}
            )
        )
        respx.get(self.DETAIL_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "sr-1",
                    "name": "Backend Engineer (Hybrid)",
                    "compensation": {"min": 45.0, "max": 60.0, "currency": "USD", "period": "HOURLY"},
                },
            )
        )
        adapter = SmartRecruitersAdapter(http_client)
        postings = list(adapter.fetch_postings("acme"))
        assert postings[0].salary_period == "hour"

    @respx.mock
    def test_missing_compensation_leaves_salary_source_none(self, http_client):
        respx.get(self.LIST_URL).mock(
            return_value=httpx.Response(
                200, json={"totalFound": 1, "content": [{"id": "sr-1", "name": "Backend Engineer"}]}
            )
        )
        respx.get(self.DETAIL_URL).mock(
            return_value=httpx.Response(200, json={"id": "sr-1", "name": "Backend Engineer"})
        )
        adapter = SmartRecruitersAdapter(http_client)
        postings = list(adapter.fetch_postings("acme"))
        assert postings[0].salary_source == SalarySource.NONE
        assert postings[0].salary_min is None

    @respx.mock
    def test_board_id_query_string_is_forwarded_as_list_filters(self, http_client):
        seen = []

        def list_handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.url.params.get("country"))
            return httpx.Response(200, json={"totalFound": 1, "content": [{"id": "sr-1", "name": "Backend Engineer"}]})

        respx.get(self.LIST_URL).mock(side_effect=list_handler)
        detail_route = respx.get(self.DETAIL_URL).mock(
            return_value=httpx.Response(200, json={"id": "sr-1", "name": "Backend Engineer"})
        )

        adapter = SmartRecruitersAdapter(http_client)
        postings = list(adapter.fetch_postings("acme?country=us"))

        assert seen == ["us"]
        assert detail_route.call_count == 1
        assert postings[0].source_job_id == "sr-1"

    @respx.mock
    def test_pagination_follows_offset_until_short_page(self, http_client, monkeypatch):
        import jobscan.adapters.smartrecruiters as smartrecruiters_module

        monkeypatch.setattr(smartrecruiters_module, "PAGE_SIZE", 2)

        def make_brief(n: int) -> dict:
            return {"id": f"sr-{n}", "name": f"Software Engineer {n}"}

        list_route = respx.get(self.LIST_URL)
        list_route.side_effect = [
            httpx.Response(200, json={"totalFound": 3, "content": [make_brief(1), make_brief(2)]}),
            httpx.Response(200, json={"totalFound": 3, "content": [make_brief(3)]}),
        ]
        respx.get(url__regex=r".*/postings/sr-\d$").mock(
            return_value=httpx.Response(200, json={"id": "sr-x", "name": "Software Engineer"})
        )

        adapter = SmartRecruitersAdapter(http_client)
        postings = list(adapter.fetch_postings("acme"))

        assert list_route.call_count == 2
        assert len(postings) == 3

    @respx.mock
    def test_detail_fetch_failure_is_skipped_not_fatal(self, http_client):
        respx.get(self.LIST_URL).mock(
            return_value=httpx.Response(
                200, json={"totalFound": 1, "content": [{"id": "sr-1", "name": "Senior Backend Engineer"}]}
            )
        )
        respx.get(self.DETAIL_URL).mock(return_value=httpx.Response(404))

        adapter = SmartRecruitersAdapter(http_client)
        postings = list(adapter.fetch_postings("acme"))
        assert postings == []

    @respx.mock
    def test_unexpected_list_shape_raises_adapter_error(self, http_client):
        respx.get(self.LIST_URL).mock(return_value=httpx.Response(200, json={"content": "not-a-list"}))
        adapter = SmartRecruitersAdapter(http_client)
        with pytest.raises(AdapterError):
            list(adapter.fetch_postings("acme"))

    @respx.mock
    def test_list_http_error_raises_adapter_error(self, http_client):
        respx.get(self.LIST_URL).mock(return_value=httpx.Response(500))
        adapter = SmartRecruitersAdapter(http_client)
        with pytest.raises(AdapterError):
            list(adapter.fetch_postings("acme"))

    @respx.mock
    def test_network_error_raises_adapter_error(self, http_client):
        respx.get(self.LIST_URL).mock(side_effect=httpx.ConnectTimeout("timed out"))
        adapter = SmartRecruitersAdapter(http_client)
        with pytest.raises(AdapterError):
            list(adapter.fetch_postings("acme"))


def avature_detail_html(job_id: str, job_type: str = "Full-time", field_of_work: str = "Software") -> str:
    return f"""
    <article class="article article--details">
      <div class="article__content__view">
        <div class="article__content__view__field">
          <div class="article__content__view__field__label">Job ID</div>
          <div class="article__content__view__field__value">{job_id}</div>
        </div>
        <div class="article__content__view__field">
          <div class="article__content__view__field__label">Job type</div>
          <div class="article__content__view__field__value">{job_type}</div>
        </div>
        <div class="article__content__view__field">
          <div class="article__content__view__field__label">Field of work</div>
          <div class="article__content__view__field__value">{field_of_work}</div>
        </div>
        <div class="article__content__view__field">
          <div class="article__content__view__field__label">Posted since</div>
          <div class="article__content__view__field__value">01-Oct-2026</div>
        </div>
        <div class="article__content__view__field">
          <div class="article__content__view__field__label">Location(s)</div>
          <div class="article__content__view__field__value">
            <ul class="list--locations"><li>Seattle, WA, USA</li><li>Remote, USA</li></ul>
          </div>
        </div>
      </div>
    </article>
    <article class="article article--details">
      <div class="article__content__view"><p>Build great backend systems.</p></div>
    </article>
    """


class TestAvatureAdapter:
    BOARD_ID = "jobs.example.com/en_US/externaljobs"
    LIST_URL = "https://jobs.example.com/en_US/externaljobs/SearchJobs/"
    DETAIL_URL = "https://jobs.example.com/en_US/externaljobs/JobDetail/1001"

    @respx.mock
    def test_fetches_detail_only_for_engineering_titles(self, http_client):
        respx.get(self.LIST_URL).mock(
            return_value=httpx.Response(
                200,
                text="""
                <article class="article article--result">
                  <div class="article__header__text__title"><a href="https://jobs.example.com/en_US/externaljobs/JobDetail/1001">Senior Backend Engineer</a></div>
                  <span class="list-item-location">Remote - US</span>
                </article>
                <article class="article article--result">
                  <div class="article__header__text__title"><a href="https://jobs.example.com/en_US/externaljobs/JobDetail/1002">Account Executive</a></div>
                  <span class="list-item-location">New York</span>
                </article>
                """,
            )
        )
        detail_route = respx.get(self.DETAIL_URL).mock(
            return_value=httpx.Response(200, text=avature_detail_html("1001"))
        )

        adapter = AvatureAdapter(http_client)
        postings = list(adapter.fetch_postings(self.BOARD_ID))

        assert detail_route.call_count == 1
        assert len(postings) == 1
        posting = postings[0]
        assert posting.source_job_id == "1001"
        assert posting.title == "Senior Backend Engineer"
        assert posting.location_raw == "Seattle, WA, USA; Remote, USA"
        assert posting.employment_type_raw == "Full-time"
        assert posting.department == "Software"
        assert posting.published_at == "2026-10-01"
        assert "Build great backend systems" in posting.description_html
        assert posting.apply_url == "https://jobs.example.com/en_US/externaljobs/ApplicationMethods?folderId=1001"
        assert posting.salary_source == SalarySource.NONE

    @respx.mock
    def test_pagination_stops_within_first_concurrent_batch(self, http_client, monkeypatch):
        # List pages are fetched CONCURRENCY-at-a-time (this tenant's hard-coded page size of 6
        # means a real board needs hundreds of sequential round trips otherwise — see the module
        # docstring). The mock is offset-aware (keyed by the real `folderOffset` query param)
        # rather than a fixed-order list, since a batch fires several requests at once and their
        # completion order isn't guaranteed. Real items exist only at offsets 0 and 2 (3 items
        # total, PAGE_SIZE=2) — every other offset in the first batch comes back empty, exactly
        # like the real board past its real end — so the whole board resolves within one batch.
        import jobscan.adapters.avature as avature_module

        monkeypatch.setattr(avature_module, "PAGE_SIZE", 2)

        def make_article(job_id: str) -> str:
            return f"""
            <article class="article article--result">
              <div class="article__header__text__title"><a href="https://jobs.example.com/en_US/externaljobs/JobDetail/{job_id}">Software Engineer {job_id}</a></div>
              <span class="list-item-location">Remote</span>
            </article>
            """

        pages_by_offset = {0: make_article("1") + make_article("2"), 2: make_article("3")}

        def list_handler(request: httpx.Request) -> httpx.Response:
            offset = int(request.url.params.get("folderOffset", "0"))
            return httpx.Response(200, text=pages_by_offset.get(offset, ""))

        list_route = respx.get(self.LIST_URL).mock(side_effect=list_handler)
        respx.get(url__regex=r".*/JobDetail/\d$").mock(return_value=httpx.Response(200, text=avature_detail_html("x")))

        adapter = AvatureAdapter(http_client)
        postings = list(adapter.fetch_postings(self.BOARD_ID))

        assert list_route.call_count == avature_module.CONCURRENCY
        assert len(postings) == 3

    @respx.mock
    def test_pagination_advances_past_a_full_batch(self, http_client, monkeypatch):
        # The real end lies beyond the first batch's span (CONCURRENCY * PAGE_SIZE), so a second
        # round of concurrent requests must be issued starting at the next offset.
        import jobscan.adapters.avature as avature_module

        monkeypatch.setattr(avature_module, "PAGE_SIZE", 2)
        monkeypatch.setattr(avature_module, "CONCURRENCY", 2)

        def make_article(job_id: str) -> str:
            return f"""
            <article class="article article--result">
              <div class="article__header__text__title"><a href="https://jobs.example.com/en_US/externaljobs/JobDetail/{job_id}">Software Engineer {job_id}</a></div>
              <span class="list-item-location">Remote</span>
            </article>
            """

        # Batch 1 covers offsets 0, 2 (both full pages); batch 2 covers offsets 4, 6 (the real
        # end — a short page — is at offset 6).
        pages_by_offset = {
            0: make_article("1") + make_article("2"),
            2: make_article("3") + make_article("4"),
            4: make_article("5") + make_article("6"),
            6: make_article("7"),
        }

        def list_handler(request: httpx.Request) -> httpx.Response:
            offset = int(request.url.params.get("folderOffset", "0"))
            return httpx.Response(200, text=pages_by_offset.get(offset, ""))

        list_route = respx.get(self.LIST_URL).mock(side_effect=list_handler)
        respx.get(url__regex=r".*/JobDetail/\d$").mock(return_value=httpx.Response(200, text=avature_detail_html("x")))

        adapter = AvatureAdapter(http_client)
        postings = list(adapter.fetch_postings(self.BOARD_ID))

        assert list_route.call_count == 4  # two full batches of CONCURRENCY=2
        assert len(postings) == 7

    @respx.mock
    def test_detail_fetch_failure_is_skipped_not_fatal(self, http_client):
        respx.get(self.LIST_URL).mock(
            return_value=httpx.Response(
                200,
                text="""
                <article class="article article--result">
                  <div class="article__header__text__title"><a href="https://jobs.example.com/en_US/externaljobs/JobDetail/1001">Senior Backend Engineer</a></div>
                  <span class="list-item-location">Remote - US</span>
                </article>
                """,
            )
        )
        respx.get(self.DETAIL_URL).mock(return_value=httpx.Response(404))

        adapter = AvatureAdapter(http_client)
        postings = list(adapter.fetch_postings(self.BOARD_ID))
        assert postings == []

    @respx.mock
    def test_board_id_query_string_is_forwarded_as_search_filters(self, http_client):
        # A tenant's own search-form filters (opaque field/option ids) ride along in board_id and
        # must reach the server on every list request, repeated keys included — while every URL
        # the adapter builds itself uses only the portal part before the "?".
        seen = []

        def list_handler(request: httpx.Request) -> httpx.Response:
            seen.append((request.url.params.get_list("42386[]"), request.url.params.get_list("42389[]")))
            return httpx.Response(
                200,
                text="""
                <article class="article article--result">
                  <div class="article__header__text__title"><a href="https://jobs.example.com/en_US/externaljobs/JobDetail/1001">Senior Backend Engineer</a></div>
                  <span class="list-item-location">Remote - US</span>
                </article>
                """,
            )

        respx.get(self.LIST_URL).mock(side_effect=list_handler)
        respx.get(self.DETAIL_URL).mock(return_value=httpx.Response(200, text=avature_detail_html("1001")))

        adapter = AvatureAdapter(http_client)
        postings = list(adapter.fetch_postings(f"{self.BOARD_ID}?42386[]=812209&42389[]=102127&42389[]=102122"))

        assert seen and all(s == (["812209"], ["102127", "102122"]) for s in seen)
        assert postings[0].apply_url == "https://jobs.example.com/en_US/externaljobs/ApplicationMethods?folderId=1001"

    @respx.mock
    def test_list_http_error_raises_adapter_error(self, http_client):
        respx.get(self.LIST_URL).mock(return_value=httpx.Response(500))
        adapter = AvatureAdapter(http_client)
        with pytest.raises(AdapterError):
            list(adapter.fetch_postings(self.BOARD_ID))

    @respx.mock
    def test_network_error_raises_adapter_error(self, http_client):
        respx.get(self.LIST_URL).mock(side_effect=httpx.ConnectTimeout("timed out"))
        adapter = AvatureAdapter(http_client)
        with pytest.raises(AdapterError):
            list(adapter.fetch_postings(self.BOARD_ID))


def rippling_next_data_html(build_id: str) -> str:
    import json as _json

    return f'<html><body><script id="__NEXT_DATA__" type="application/json">{_json.dumps({"buildId": build_id})}</script></body></html>'


def rippling_jobs_page(items: list[dict], page: int = 0, total_pages: int = 1) -> dict:
    return {
        "pageProps": {
            "dehydratedState": {
                "queries": [
                    {
                        "queryKey": ["board", "acme", "job-posts", False, {"page": page}],
                        "state": {"data": {"items": items, "page": page, "totalPages": total_pages}},
                    }
                ]
            }
        }
    }


def rippling_job_detail(job_post: dict) -> dict:
    return {"pageProps": {"apiData": {"jobPost": job_post}}}


class TestRipplingAdapter:
    BOARD_URL = "https://ats.rippling.com/acme/jobs"
    DATA_URL = "https://ats.rippling.com/_next/data/BUILD123/en-GB/acme/jobs.json"
    DETAIL_URL = "https://ats.rippling.com/_next/data/BUILD123/en-GB/acme/jobs/job-1.json"

    @respx.mock
    def test_fetches_detail_only_for_engineering_titles(self, http_client):
        respx.get(self.BOARD_URL).mock(return_value=httpx.Response(200, text=rippling_next_data_html("BUILD123")))
        respx.get(self.DATA_URL).mock(
            return_value=httpx.Response(
                200,
                json=rippling_jobs_page(
                    [{"id": "job-1", "name": "Senior Backend Engineer"}, {"id": "job-2", "name": "Account Executive"}]
                ),
            )
        )
        detail_route = respx.get(self.DETAIL_URL).mock(
            return_value=httpx.Response(
                200,
                json=rippling_job_detail(
                    {
                        "uuid": "job-1",
                        "name": "Senior Backend Engineer",
                        "description": {"company": "<p>About us.</p>", "role": "<p>Build things.</p>"},
                        "workLocations": ["Remote - US"],
                        "department": {"name": "Engineering"},
                        "employmentType": {"id": "Salaried, full-time", "label": "SALARIED_FT"},
                        "createdOn": "2026-09-01T00:00:00-07:00",
                        "url": "https://ats.rippling.com/acme/jobs/job-1",
                    }
                ),
            )
        )

        adapter = RipplingAdapter(http_client)
        postings = list(adapter.fetch_postings("acme"))

        assert detail_route.call_count == 1
        assert len(postings) == 1
        posting = postings[0]
        assert posting.source_job_id == "job-1"
        assert posting.title == "Senior Backend Engineer"
        assert posting.location_raw == "Remote - US"
        assert posting.employment_type_raw == "Salaried, full-time"
        assert posting.department == "Engineering"
        assert "About us" in posting.description_html
        assert "Build things" in posting.description_html
        assert posting.posting_url == "https://ats.rippling.com/acme/jobs/job-1"
        assert posting.published_at == "2026-09-01T00:00:00-07:00"
        assert posting.salary_source == SalarySource.NONE

    @respx.mock
    def test_pagination_follows_page_until_total_pages(self, http_client):
        respx.get(self.BOARD_URL).mock(return_value=httpx.Response(200, text=rippling_next_data_html("BUILD123")))
        data_route = respx.get(self.DATA_URL)
        data_route.side_effect = [
            httpx.Response(
                200, json=rippling_jobs_page([{"id": "job-1", "name": "Account Executive"}], page=0, total_pages=2)
            ),
            httpx.Response(
                200, json=rippling_jobs_page([{"id": "job-2", "name": "Account Manager"}], page=1, total_pages=2)
            ),
        ]

        adapter = RipplingAdapter(http_client)
        list(adapter.fetch_postings("acme"))

        assert data_route.call_count == 2

    @respx.mock
    def test_missing_build_id_raises_adapter_error(self, http_client):
        respx.get(self.BOARD_URL).mock(return_value=httpx.Response(200, text="<html><body>no next data here</body></html>"))
        adapter = RipplingAdapter(http_client)
        with pytest.raises(AdapterError):
            list(adapter.fetch_postings("acme"))

    @respx.mock
    def test_missing_job_posts_query_raises_adapter_error(self, http_client):
        respx.get(self.BOARD_URL).mock(return_value=httpx.Response(200, text=rippling_next_data_html("BUILD123")))
        respx.get(self.DATA_URL).mock(return_value=httpx.Response(200, json={"pageProps": {"dehydratedState": {"queries": []}}}))
        adapter = RipplingAdapter(http_client)
        with pytest.raises(AdapterError):
            list(adapter.fetch_postings("acme"))

    @respx.mock
    def test_detail_fetch_failure_is_skipped_not_fatal(self, http_client):
        respx.get(self.BOARD_URL).mock(return_value=httpx.Response(200, text=rippling_next_data_html("BUILD123")))
        respx.get(self.DATA_URL).mock(
            return_value=httpx.Response(200, json=rippling_jobs_page([{"id": "job-1", "name": "Backend Engineer"}]))
        )
        respx.get(self.DETAIL_URL).mock(return_value=httpx.Response(404))

        adapter = RipplingAdapter(http_client)
        postings = list(adapter.fetch_postings("acme"))
        assert postings == []

    @respx.mock
    def test_board_page_http_error_raises_adapter_error(self, http_client):
        respx.get(self.BOARD_URL).mock(return_value=httpx.Response(500))
        adapter = RipplingAdapter(http_client)
        with pytest.raises(AdapterError):
            list(adapter.fetch_postings("acme"))

    @respx.mock
    def test_data_endpoint_network_error_raises_adapter_error(self, http_client):
        respx.get(self.BOARD_URL).mock(return_value=httpx.Response(200, text=rippling_next_data_html("BUILD123")))
        respx.get(self.DATA_URL).mock(side_effect=httpx.ConnectTimeout("timed out"))
        adapter = RipplingAdapter(http_client)
        with pytest.raises(AdapterError):
            list(adapter.fetch_postings("acme"))
