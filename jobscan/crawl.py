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

Boards are fetched ``settings.crawl_workers`` at a time — the crawl is network-bound, and one
board at a time took ~4.5 minutes for ~90 companies. Only the fetching runs in worker threads:
normalizing, persisting and closing happen on the calling thread, in registry order, because the
Database (one SQLite connection) isn't shared across threads.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

from jobscan.adapters import ADAPTERS
from jobscan.adapters.base import AdapterError
from jobscan.config import Settings
from jobscan.db import Database
from jobscan.job_family import is_engineering_title
from jobscan.logging_setup import get_logger, log_extra
from jobscan.models import Company, CrawlRunStats, RawPosting
from jobscan.normalize import normalize_posting

logger = get_logger("crawl")


@dataclass
class _BoardFetch:
    postings: list[RawPosting] | None = None
    incomplete: bool = False
    failure: str | None = None


def _fetch_board(company: Company, http_client: httpx.Client) -> _BoardFetch:
    """Runs in a worker thread: network only, no DB access."""
    adapter_cls = ADAPTERS.get(company.ats_type)
    if adapter_cls is None:
        return _BoardFetch(failure="no adapter registered")
    adapter = adapter_cls(http_client)
    try:
        postings = list(adapter.fetch_postings(company.board_id))
    except AdapterError as exc:
        logger.warning(
            "board fetch failed",
            extra=log_extra(company=company.name, ats_type=company.ats_type.value, board_id=company.board_id, error=str(exc)),
        )
        return _BoardFetch(failure=str(exc))
    except Exception as exc:  # noqa: BLE001 - adapters are third-party-facing, isolate the run from surprises
        logger.error(
            "board fetch raised unexpected error",
            extra=log_extra(company=company.name, board_id=company.board_id, error=str(exc)),
        )
        return _BoardFetch(failure=f"unexpected error: {exc}")
    return _BoardFetch(postings=postings, incomplete=adapter.incomplete)


def crawl_all(db: Database, settings: Settings, companies: list[Company] | None = None) -> CrawlRunStats:
    companies = companies if companies is not None else db.list_companies(active_only=True)
    stats = CrawlRunStats(started_at=datetime.now(timezone.utc))
    run_id = db.start_crawl_run(stats)

    headers = {"User-Agent": settings.http_user_agent}
    with httpx.Client(timeout=settings.http_timeout_seconds, headers=headers) as http_client, \
            ThreadPoolExecutor(max_workers=max(1, settings.crawl_workers)) as pool:
        fetches = pool.map(lambda company: _fetch_board(company, http_client), companies)
        for company, fetch in zip(companies, fetches):
            stats.companies_attempted += 1
            if fetch.failure is not None:
                stats.boards_failed.append(f"{company.name} ({company.ats_type.value}/{company.board_id}): {fetch.failure}")
                continue
            raw_postings = fetch.postings

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

            if fetch.incomplete:
                # Some listed postings couldn't be read this run; absence isn't evidence of closing.
                closed = 0
                logger.warning(
                    "board fetched incompletely — closing nothing on it this run",
                    extra=log_extra(company=company.name, ats_type=company.ats_type.value, board_id=company.board_id),
                )
            else:
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
