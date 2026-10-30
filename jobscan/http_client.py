"""The HTTP client every board fetch goes through, with one retry policy for transient failures.

Every request the crawl makes is a read — the POSTs are Workable's and Workday's job searches — so
any of them can be sent again safely. Only failures that can clear up on their own are retried: a
429, a 5xx, a timeout or a dropped connection. Over the first 64 crawls (3,687 board fetches), 5 of
the 83 board failures were of that kind (FMX's Workable board answering 429 four times, Akamai's
Oracle board a 503 once), and Siemens' Avature board times out on ~5% of its parallel detail
requests, a different handful each run. The other 78 board failures were permanent (Workday boards
that moved, board IDs that no longer exist), and retrying those would only slow the crawl down.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx

from jobscan.config import Settings
from jobscan.logging_setup import get_logger, log_extra

MAX_RETRIES = 2
# The waits before the first and second retry when the server doesn't name one. Each is stretched
# by up to half again at random, so parallel requests that failed together don't retry together.
BACKOFF_SECONDS = (2.0, 10.0)
# The longest Retry-After worth honouring. A server asking for more is left until the next run
# instead: a longer sleep would hold one of the crawl's few worker threads.
MAX_RETRY_AFTER_SECONDS = 30.0

logger = get_logger("http")


def make_client(settings: Settings) -> httpx.Client:
    """The client the crawl, `companies probe` and evaluate's homepage reads use: the configured
    timeout and User-Agent, with transient failures retried."""
    return httpx.Client(
        timeout=settings.http_timeout_seconds,
        headers={"User-Agent": settings.http_user_agent},
        transport=RetryTransport(),
    )


def _pause(seconds: float) -> None:
    time.sleep(seconds)


class RetryTransport(httpx.BaseTransport):
    def __init__(
        self, inner: httpx.BaseTransport | None = None, sleep: Callable[[float], None] | None = None
    ):
        """``inner`` sends the requests (default: httpx's own transport); ``sleep`` waits between
        attempts (default: a real sleep — tests pass a recorder)."""
        self._inner = inner or httpx.HTTPTransport()
        self._sleep = sleep or _pause

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        retries = 0
        while True:
            try:
                response = self._inner.handle_request(request)
            except httpx.TransportError as exc:  # timeouts and dropped connections
                if retries == MAX_RETRIES:
                    raise
                problem, wait = type(exc).__name__, _backoff(retries)
            else:
                if response.status_code != 429 and response.status_code < 500:
                    return response
                retry_after = response.headers.get("retry-after")
                asked = _retry_after_seconds(retry_after)
                if retries == MAX_RETRIES or (
                    asked is not None and asked > MAX_RETRY_AFTER_SECONDS
                ):
                    # Retry-After and Cloudflare's ray ID say how long a block lasts and let the
                    # board's operator find it; the caller only sees the status.
                    logger.warning(
                        "http request gave up",
                        extra=log_extra(
                            url=str(request.url),
                            status=response.status_code,
                            retries=retries,
                            retry_after=retry_after,
                            cf_ray=response.headers.get("cf-ray"),
                        ),
                    )
                    return response
                problem, wait = (
                    str(response.status_code),
                    asked if asked is not None else _backoff(retries),
                )
                response.close()
            retries += 1
            logger.info(
                "http retry",
                extra=log_extra(
                    url=str(request.url),
                    problem=problem,
                    attempt=retries,
                    wait_seconds=round(wait, 1),
                ),
            )
            self._sleep(wait)

    def close(self) -> None:
        self._inner.close()


def _backoff(retries: int) -> float:
    return BACKOFF_SECONDS[retries] * (1 + random.random() / 2)


def _retry_after_seconds(value: str | None) -> float | None:
    """A Retry-After header as seconds to wait. It is either a number of seconds or an HTTP date;
    None when absent or unreadable."""
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())
