"""Esri careers adapter.

Esri is the one company in the registry with a genuinely one-off but crawlable careers platform
(see AtsType's docstring for why this gets its own ats_type instead of being folded into
CUSTOM). List: ``POST https://esearchapi.esri.com/search`` — Esri's own Elasticsearch-backed
search API, paginated via offset/len — returns every posting's title, URL, and a short (~250
char) snippet, not the full description.

The full description is client-rendered (Angular) on each posting's own page
(``https://www.esri.com/careers/{id}``): a plain HTTP GET only returns that same short snippet
duplicated in meta tags and JSON-LD. This adapter renders each engineering-titled posting's page
with Playwright and reads the real text from its ``<main>`` content area — costs roughly
0.2-0.5s per posting once the browser is warm (first render pays browser-launch overhead), since
the job-family title gate (applied before this render, as in the Workday/Jobvite adapters) keeps
the actual render count to the handful of engineering-titled postings, not all ~450 company-wide.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

import httpx
from pydantic import ValidationError

from jobscan.adapters.base import AdapterError, SourceAdapter
from jobscan.adapters.schemas import EsriSearchHit, EsriSearchResponse
from jobscan.job_family import is_engineering_title
from jobscan.logging_setup import get_logger, log_extra
from jobscan.models import AtsType, RawPosting, SalarySource

SEARCH_URL = "https://esearchapi.esri.com/search"
PAGE_SIZE = 50
MAX_PAGES = 50  # backstop against a runaway loop if `count` is ever wrong — see jobscan.adapters.workday
_TITLE_SUFFIX_RE = re.compile(r"\s*Job\s*\|\s*Esri Career Opportunity\s*$", re.IGNORECASE)

logger = get_logger("adapters.esri")


class _PlaywrightRenderer:
    """Real detail-page renderer: lazily launches one headless Chromium instance and reuses it
    for every posting in a single fetch_postings() call, rather than paying browser-launch
    overhead per posting."""

    def __init__(self) -> None:
        self._playwright = None
        self._browser = None
        self._page = None

    def render(self, url: str) -> str:
        if self._page is None:
            from playwright.sync_api import sync_playwright  # deferred: only needed if this adapter runs

            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch()
            self._page = self._browser.new_page()
        self._page.goto(url, wait_until="domcontentloaded", timeout=20000)
        self._page.wait_for_selector("main", timeout=10000)
        return self._page.locator("main").inner_text()

    def close(self) -> None:
        if self._browser is not None:
            self._browser.close()
        if self._playwright is not None:
            self._playwright.stop()


class EsriAdapter(SourceAdapter):
    ats_type = AtsType.ESRI

    def __init__(self, client: httpx.Client, renderer: Any = None):
        super().__init__(client)
        # Injectable so tests can supply a fake with a `.render(url) -> str` method instead of
        # launching a real headless browser — same pattern as AnthropicClient.__init__.
        self._renderer = renderer or _PlaywrightRenderer()
        self._owns_renderer = renderer is None

    def fetch_postings(self, board_id: str) -> Iterable[RawPosting]:
        hits = self._fetch_all_hits(board_id)
        engineering_hits = [h for h in hits if is_engineering_title(_clean_title(h))]

        postings = []
        try:
            for hit in engineering_hits:
                posting = self._fetch_detail(hit, board_id)
                if posting is not None:
                    postings.append(posting)
        finally:
            if self._owns_renderer:
                self._renderer.close()
        return postings

    def _fetch_all_hits(self, board_id: str) -> list[EsriSearchHit]:
        hits: list[EsriSearchHit] = []
        offset = 0
        count: int | None = None
        for _ in range(MAX_PAGES):
            body = {
                "lr": "en", "client": "esri_explore", "site": "esri_careers", "format": "json",
                "q": "", "num": PAGE_SIZE, "len": PAGE_SIZE, "offset": offset, "start": offset,
            }
            try:
                response = self.client.post(SEARCH_URL, json=body)
                response.raise_for_status()
                payload = response.json()
            except httpx.HTTPError as exc:
                raise AdapterError(f"esri board '{board_id}': request failed: {exc}") from exc
            except ValueError as exc:
                raise AdapterError(f"esri board '{board_id}': invalid JSON: {exc}") from exc

            try:
                parsed = EsriSearchResponse.model_validate(payload)
            except ValidationError as exc:
                raise AdapterError(f"esri board '{board_id}': unexpected response shape: {exc}") from exc

            page_hits = parsed.search.hits
            if not page_hits:
                break
            hits.extend(page_hits)
            if count is None and parsed.search.count > 0:
                count = parsed.search.count
            offset += PAGE_SIZE
            if len(page_hits) < PAGE_SIZE or (count is not None and offset >= count):
                break
        else:
            logger.warning(
                "esri pagination hit MAX_PAGES without reaching count — stopping early",
                extra=log_extra(board_id=board_id, fetched=len(hits), reported_count=count),
            )
        return hits

    def _fetch_detail(self, hit: EsriSearchHit, board_id: str) -> RawPosting | None:
        url = hit.doc.displayurl
        if not url:
            return None
        try:
            description = self._renderer.render(url)
        except Exception as exc:  # noqa: BLE001 - Playwright raises its own exception types
            logger.warning("esri job detail render failed", extra=log_extra(board_id=board_id, url=url, error=str(exc)))
            return None

        meta = hit.doc.metaFields
        locations = _as_str(meta.get("locations"))
        employment_type = _as_str(meta.get("employmentType"))
        if employment_type == "None":
            employment_type = None

        source_job_id = url.rstrip("/").rsplit("/", 1)[-1]
        return RawPosting(
            source_job_id=source_job_id,
            title=_clean_title(hit),
            location_raw=locations.replace("-", ", ") if locations else None,
            employment_type_raw=employment_type,
            description_html=None,
            description_text=description,
            posting_url=url,
            apply_url=url,
            salary_source=SalarySource.NONE,
        )


def _clean_title(hit: EsriSearchHit) -> str:
    job_title = _as_str(hit.doc.metaFields.get("JobTitle"))
    if job_title:
        return job_title
    return _TITLE_SUFFIX_RE.sub("", hit.doc.title).strip()


def _as_str(value: Any) -> str:
    """metaFields values are occasionally a list (a multi-valued field) instead of a bare
    string — take the first entry rather than failing on it."""
    if isinstance(value, list):
        value = value[0] if value else None
    return value.strip() if isinstance(value, str) else ""
