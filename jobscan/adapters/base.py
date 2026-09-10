"""Common interface every applicant-tracking-system adapter implements.

Adding a new source (SmartRecruiters, Workday, a custom careers page) means writing one class
with a ``fetch_postings`` method — nothing else in the pipeline needs to change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Iterable

import httpx

from jobscan.models import AtsType, RawPosting


class AdapterError(RuntimeError):
    """Raised when a board can't be fetched (network error, unexpected schema, HTTP error)."""


class SourceAdapter(ABC):
    ats_type: ClassVar[AtsType]

    def __init__(self, client: httpx.Client):
        self.client = client

    @abstractmethod
    def fetch_postings(self, board_id: str) -> Iterable[RawPosting]:
        """Return every current posting for the given board. Must raise AdapterError (not a
        bare exception) on failure so the crawl orchestrator can record a clean per-board
        failure and continue with the remaining boards."""
        raise NotImplementedError
