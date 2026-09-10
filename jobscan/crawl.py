"""Crawl orchestration: fetch every active registry company's board, keep only postings that are
actually software-engineering roles, persist those, and mark ones that disappeared from a
successfully-crawled board as closed.

Each company's board typically spans every department (sales, support, legal, design, ...); the
job-family gate (jobscan.job_family) runs immediately after fetching so non-engineering postings
are never normalized, never stored, and never reach the LLM — this is a search for software
engineering jobs, not a generic store-everything-then-filter pipeline.

A failure on one board (network error, malformed API response) is recorded and skipped — it
never aborts the run for the remaining boards. The run's exact company/board/posting counts are
what the report surfaces, so "comprehensive" is never claimed beyond what was actually crawled.
"""
from __future__ import annotations

from datetime import datetime, timezone

import httpx

from jobscan.adapters import ADAPTERS
from jobscan.adapters.base import AdapterError
from jobscan.config import Settings
from jobscan.db import Database
from jobscan.job_family import is_engineering_title
from jobscan.logging_setup import get_logger, log_extra
from jobscan.models import Company, CrawlRunStats
from jobscan.normalize import normalize_posting

logger = get_logger("crawl")


def crawl_all(db: Database, settings: Settings, companies: list[Company] | None = None) -> CrawlRunStats:
    companies = companies if companies is not None else db.list_companies(active_only=True)
    stats = CrawlRunStats(started_at=datetime.now(timezone.utc))
    run_id = db.start_crawl_run(stats)

    headers = {"User-Agent": settings.http_user_agent}
    with httpx.Client(timeout=settings.http_timeout_seconds, headers=headers) as http_client:
        for company in companies:
            stats.companies_attempted += 1
            adapter_cls = ADAPTERS.get(company.ats_type)
            if adapter_cls is None:
                stats.boards_failed.append(f"{company.name} ({company.ats_type.value}/{company.board_id}): no adapter registered")
                continue

            adapter = adapter_cls(http_client)
            try:
                raw_postings = list(adapter.fetch_postings(company.board_id))
            except AdapterError as exc:
                logger.warning(
                    "board fetch failed",
                    extra=log_extra(company=company.name, ats_type=company.ats_type.value, board_id=company.board_id, error=str(exc)),
                )
                stats.boards_failed.append(f"{company.name} ({company.ats_type.value}/{company.board_id}): {exc}")
                continue
            except Exception as exc:  # noqa: BLE001 - adapters are third-party-facing, isolate the run from surprises
                logger.error(
                    "board fetch raised unexpected error",
                    extra=log_extra(company=company.name, board_id=company.board_id, error=str(exc)),
                )
                stats.boards_failed.append(f"{company.name} ({company.ats_type.value}/{company.board_id}): unexpected error: {exc}")
                continue

            now = datetime.now(timezone.utc)
            seen_ids: set[str] = set()
            for raw in raw_postings:
                stats.postings_fetched += 1
                if not is_engineering_title(raw.title):
                    stats.postings_out_of_family += 1
                    continue

                job = normalize_posting(raw, company, settings, now=now)
                job_id, is_new, changed = db.upsert_job(job)
                seen_ids.add(raw.source_job_id)
                if is_new:
                    stats.new_postings += 1
                elif changed:
                    stats.changed_postings += 1

            closed = db.close_missing_jobs(company.id, seen_ids, when=now)
            stats.closed_postings += closed
            db.touch_company_crawled(company.id, when=now)
            stats.companies_succeeded += 1

            logger.info(
                "board crawled",
                extra=log_extra(
                    company=company.name, ats_type=company.ats_type.value, board_id=company.board_id,
                    postings=len(raw_postings), closed=closed,
                ),
            )

    stats.finished_at = datetime.now(timezone.utc)
    db.finish_crawl_run(run_id, stats)
    return stats
