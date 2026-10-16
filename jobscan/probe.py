"""Dry-run one board before adding it to the registry: fetch it with the real adapter, apply the
same title gate, normalization and factual filters the pipeline uses, and report what would reach
evaluation. Nothing is persisted and no LLM is called — it answers "is this company worth adding?"
from its live postings instead of a guess."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import httpx

from jobscan.adapters import ADAPTERS
from jobscan.config import Settings
from jobscan.filters import apply_factual_filters
from jobscan.job_family import is_engineering_title
from jobscan.models import AtsType, Company, CompanyClassification, JobPosting
from jobscan.normalize import normalize_posting


@dataclass
class ProbeResult:
    postings: int
    engineering: int
    passing: list[JobPosting] = field(default_factory=list)
    rejected: Counter[str] = field(default_factory=Counter)


def probe_board(settings: Settings, ats_type: AtsType, board_id: str, http_client: httpx.Client) -> ProbeResult:
    """Raises AdapterError when the board can't be fetched; KeyError when ``ats_type`` has no adapter."""
    adapter = ADAPTERS[ats_type](http_client)
    raw = list(adapter.fetch_postings(board_id))
    # A stand-in registry row, never stored: normalize_posting wants a company id, and a product
    # classification keeps the consulting filter from firing on a company nobody has classified.
    company = Company(name=board_id, domain=None, careers_url=None, ats_type=ats_type, board_id=board_id, id=0,
                      classification=CompanyClassification.PRODUCT)
    result = ProbeResult(postings=len(raw), engineering=0)
    for posting in raw:
        if not is_engineering_title(posting.title):
            continue
        result.engineering += 1
        job = normalize_posting(posting, company, settings)
        verdict = apply_factual_filters(job, company, settings.min_base_salary)
        if verdict.passed:
            result.passing.append(job)
        else:
            result.rejected[verdict.reason] += 1
    return result
