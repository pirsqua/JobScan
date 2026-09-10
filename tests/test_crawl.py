from __future__ import annotations

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
