from __future__ import annotations

import httpx
import pytest
import respx

from jobscan.adapters.ashby import AshbyAdapter
from jobscan.adapters.base import AdapterError
from jobscan.adapters.esri import EsriAdapter
from jobscan.adapters.greenhouse import GreenhouseAdapter
from jobscan.adapters.jobvite import JobviteAdapter
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
