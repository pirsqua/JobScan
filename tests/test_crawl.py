from __future__ import annotations

import threading
from dataclasses import replace

import httpx
import respx

from jobscan.crawl import crawl_all
from jobscan.db import Database
from jobscan.models import AtsType, Company


def make_company(name, board_id, ats_type=AtsType.GREENHOUSE) -> Company:
    return Company(name=name, domain=f"{board_id}.example.com", careers_url=None, ats_type=ats_type, board_id=board_id)


class TestCrawlOrchestration:
    @respx.mock
    def test_partial_failure_does_not_abort_other_boards(self, db: Database, settings):
        good = make_company("Good Co", "goodco")
        bad = make_company("Bad Co", "badco")
        db.upsert_company(good)
        db.upsert_company(bad)

        respx.get("https://boards-api.greenhouse.io/v1/boards/goodco/jobs").mock(
            return_value=httpx.Response(
                200,
                json={
                    "jobs": [
                        {
                            "id": 1,
                            "title": "Senior Backend Engineer",
                            "location": {"name": "Remote - US"},
                            "absolute_url": "https://goodco.example.com/jobs/1",
                            "content": "<p>$180,000 - $220,000</p>",
                            "metadata": [],
                            "departments": [],
                        }
                    ]
                },
            )
        )
        respx.get("https://boards-api.greenhouse.io/v1/boards/badco/jobs").mock(
            return_value=httpx.Response(500)
        )

        companies = db.list_companies(active_only=True)
        stats = crawl_all(db, settings, companies=companies)

        assert stats.companies_attempted == 2
        assert stats.companies_succeeded == 1
        assert len(stats.boards_failed) == 1
        assert "Bad Co" in stats.boards_failed[0]
        assert stats.postings_fetched == 1
        assert stats.new_postings == 1
        assert db.count_jobs() == 1

    @respx.mock
    def test_second_crawl_closes_disappeared_postings(self, db: Database, settings):
        company = make_company("Acme", "acme")
        db.upsert_company(company)

        route = respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs")
        route.side_effect = [
            httpx.Response(
                200,
                json={
                    "jobs": [
                        {"id": 1, "title": "Software Engineer One", "location": {"name": "Remote - US"},
                         "absolute_url": "https://acme.example.com/1", "content": "<p>$180,000 - $220,000</p>",
                         "metadata": [], "departments": []},
                        {"id": 2, "title": "Software Engineer Two", "location": {"name": "Remote - US"},
                         "absolute_url": "https://acme.example.com/2", "content": "<p>$180,000 - $220,000</p>",
                         "metadata": [], "departments": []},
                    ]
                },
            ),
            httpx.Response(
                200,
                json={
                    "jobs": [
                        {"id": 1, "title": "Software Engineer One", "location": {"name": "Remote - US"},
                         "absolute_url": "https://acme.example.com/1", "content": "<p>$180,000 - $220,000</p>",
                         "metadata": [], "departments": []},
                    ]
                },
            ),
        ]

        companies = db.list_companies(active_only=True)
        stats1 = crawl_all(db, settings, companies=companies)
        assert stats1.new_postings == 2
        assert stats1.closed_postings == 0

        stats2 = crawl_all(db, settings, companies=companies)
        assert stats2.closed_postings == 1
        assert len(db.get_active_jobs()) == 1

    @respx.mock
    def test_a_posting_whose_detail_page_failed_is_not_closed(self, db: Database, settings):
        # Observed live: five Siemens postings whose detail pages failed transiently were marked
        # closed. Only a posting missing from a complete fetch may be closed.
        db.upsert_company(make_company("FMX", "fmx", ats_type=AtsType.WORKABLE))
        list_url = "https://apply.workable.com/api/v3/accounts/fmx/jobs"
        detail_url = "https://apply.workable.com/api/v2/accounts/fmx/jobs/{}"
        both = {"results": [{"shortcode": "A", "title": "Software Developer"}, {"shortcode": "B", "title": "Software Developer"}]}
        detail = {"workplace": "remote", "type": "full", "title": "Software Developer",
                  "locations": [{"country": "United States"}], "description": "<p>$180,000 - $220,000</p>"}
        respx.post(list_url).mock(side_effect=[
            httpx.Response(200, json=both), httpx.Response(200, json=both),
            httpx.Response(200, json={"results": [{"shortcode": "A", "title": "Software Developer"}]}),
        ])
        respx.get(detail_url.format("A")).mock(return_value=httpx.Response(200, json={**detail, "shortcode": "A"}))
        respx.get(detail_url.format("B")).mock(side_effect=[
            httpx.Response(200, json={**detail, "shortcode": "B"}),
            httpx.Response(500), httpx.Response(500), httpx.Response(500),  # fails on every retry too
        ])
        companies = db.list_companies(active_only=True)

        assert crawl_all(db, settings, companies=companies).new_postings == 2
        assert crawl_all(db, settings, companies=companies).closed_postings == 0  # B's detail failed
        assert len(db.get_active_jobs()) == 2
        assert crawl_all(db, settings, companies=companies).closed_postings == 1  # B really gone
        assert [j.source_job_id for j in db.get_active_jobs()] == ["A"]

    @respx.mock
    def test_non_engineering_postings_are_never_persisted(self, db: Database, settings):
        company = make_company("Acme", "acme")
        db.upsert_company(company)

        respx.get("https://boards-api.greenhouse.io/v1/boards/acme/jobs").mock(
            return_value=httpx.Response(
                200,
                json={
                    "jobs": [
                        {"id": 1, "title": "Senior Backend Engineer", "location": {"name": "Remote - US"},
                         "absolute_url": "https://acme.example.com/1", "content": "<p>$180,000 - $220,000</p>",
                         "metadata": [], "departments": []},
                        {"id": 2, "title": "Senior Account Executive", "location": {"name": "Remote - US"},
                         "absolute_url": "https://acme.example.com/2", "content": "<p>$180,000 - $220,000</p>",
                         "metadata": [], "departments": []},
                        {"id": 3, "title": "Warehouse Associate", "location": {"name": "Seattle, WA"},
                         "absolute_url": "https://acme.example.com/3", "content": "<p>$50,000 - $60,000</p>",
                         "metadata": [], "departments": []},
                    ]
                },
            )
        )

        companies = db.list_companies(active_only=True)
        stats = crawl_all(db, settings, companies=companies)

        assert stats.postings_fetched == 3
        assert stats.postings_out_of_family == 2
        assert stats.new_postings == 1
        assert db.count_jobs() == 1
        assert db.get_active_jobs()[0].title == "Senior Backend Engineer"

        # A standalone `report` invocation recovers this run's stats from the database.
        recovered = db.latest_crawl_run()
        assert recovered is not None
        assert recovered.postings_fetched == stats.postings_fetched
        assert recovered.postings_out_of_family == stats.postings_out_of_family
        assert recovered.companies_succeeded == stats.companies_succeeded

    @respx.mock
    def test_company_with_unsupported_ats_is_reported_not_crashed(self, db: Database, settings):
        # A registry can record a company on an ATS with no adapter yet (Taleo, Phenom, a custom
        # site) — the run should report it as a failed board and continue, not raise.
        good = make_company("Good Co", "goodco")
        unsupported = make_company("Taleo Co", "taleoco", ats_type=AtsType.TALEO)
        db.upsert_company(good)
        db.upsert_company(unsupported)

        respx.get("https://boards-api.greenhouse.io/v1/boards/goodco/jobs").mock(
            return_value=httpx.Response(
                200,
                json={
                    "jobs": [
                        {"id": 1, "title": "Senior Backend Engineer", "location": {"name": "Remote - US"},
                         "absolute_url": "https://goodco.example.com/jobs/1", "content": "<p>$180,000 - $220,000</p>",
                         "metadata": [], "departments": []}
                    ]
                },
            )
        )

        companies = db.list_companies(active_only=True)
        stats = crawl_all(db, settings, companies=companies)

        assert stats.companies_attempted == 2
        assert stats.companies_succeeded == 1
        assert len(stats.boards_failed) == 1
        assert "Taleo Co" in stats.boards_failed[0]
        assert "no adapter registered" in stats.boards_failed[0]


def _greenhouse_board(job_id: int, title: str = "Senior Backend Engineer") -> dict:
    return {"jobs": [{"id": job_id, "title": title, "location": {"name": "Remote - US"},
                      "absolute_url": f"https://example.com/jobs/{job_id}", "content": "<p>$180,000 - $220,000</p>",
                      "metadata": [], "departments": []}]}


class TestParallelFetch:
    @respx.mock
    def test_boards_are_fetched_concurrently(self, db: Database, settings):
        # Each board's response waits for the other's request to arrive: fetched one at a time,
        # the barrier times out and both boards fail.
        barrier = threading.Barrier(2, timeout=5)

        def meet_then_respond(job_id):
            def respond(request):
                barrier.wait()
                return httpx.Response(200, json=_greenhouse_board(job_id))
            return respond

        for i, board in enumerate(["one", "two"], start=1):
            db.upsert_company(make_company(f"Co {board}", board))
            respx.get(f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs").mock(side_effect=meet_then_respond(i))

        stats = crawl_all(db, replace(settings, crawl_workers=2), companies=db.list_companies(active_only=True))

        assert stats.boards_failed == []
        assert stats.new_postings == 2

    @respx.mock
    def test_results_keep_registry_order_and_failures_stay_isolated(self, db: Database, settings):
        for board in ["alpha", "bravo", "charlie"]:
            db.upsert_company(make_company(f"Co {board}", board))
        respx.get("https://boards-api.greenhouse.io/v1/boards/alpha/jobs").mock(return_value=httpx.Response(200, json=_greenhouse_board(1)))
        respx.get("https://boards-api.greenhouse.io/v1/boards/bravo/jobs").mock(return_value=httpx.Response(500))
        respx.get("https://boards-api.greenhouse.io/v1/boards/charlie/jobs").mock(return_value=httpx.Response(200, json=_greenhouse_board(3)))

        companies = db.list_companies(active_only=True)
        stats = crawl_all(db, replace(settings, crawl_workers=3), companies=companies)

        assert stats.companies_attempted == 3
        assert stats.companies_succeeded == 2
        assert [failure.split(" (")[0] for failure in stats.boards_failed] == ["Co bravo"]
        assert db.count_jobs() == 2
