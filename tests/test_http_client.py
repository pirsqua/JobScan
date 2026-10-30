from __future__ import annotations

from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx
import pytest

from jobscan.http_client import (
    BACKOFF_SECONDS,
    MAX_RETRY_AFTER_SECONDS,
    RetryTransport,
    make_client,
)


def client_answering(*answers, waits: list[float] | None = None):
    """A client whose server gives these answers in turn (a response, or an exception to raise),
    and the requests it received."""
    queue, received = list(answers), []

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(request)
        answer = queue.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    sleep = waits.append if waits is not None else (lambda seconds: None)
    client = httpx.Client(transport=RetryTransport(httpx.MockTransport(handler), sleep=sleep))
    return client, received


class TestRetryTransport:
    def test_a_429_waits_as_long_as_retry_after_asks(self):
        waits: list[float] = []
        client, received = client_answering(
            httpx.Response(429, headers={"Retry-After": "3"}), httpx.Response(200, text="ok"), waits=waits)

        response = client.get("https://boards.example.com/jobs")

        assert (response.status_code, response.text) == (200, "ok")
        assert len(received) == 2
        assert waits == [3.0]

    def test_retry_after_may_be_an_http_date(self):
        waits: list[float] = []
        soon = format_datetime(datetime.now(UTC) + timedelta(seconds=5), usegmt=True)
        client, _ = client_answering(
            httpx.Response(503, headers={"Retry-After": soon}), httpx.Response(200), waits=waits)

        assert client.get("https://boards.example.com/jobs").status_code == 200
        assert 3 < waits[0] <= 5

    def test_without_retry_after_it_backs_off_with_jitter(self):
        waits: list[float] = []
        client, received = client_answering(
            httpx.Response(502), httpx.Response(500), httpx.Response(200), waits=waits)

        assert client.get("https://boards.example.com/jobs").status_code == 200
        assert len(received) == 3
        assert BACKOFF_SECONDS[0] <= waits[0] <= BACKOFF_SECONDS[0] * 1.5
        assert BACKOFF_SECONDS[1] <= waits[1] <= BACKOFF_SECONDS[1] * 1.5

    def test_a_failure_that_persists_is_returned_after_two_retries(self):
        client, received = client_answering(httpx.Response(503), httpx.Response(503), httpx.Response(503))

        assert client.get("https://boards.example.com/jobs").status_code == 503
        assert len(received) == 3

    def test_a_long_retry_after_is_not_waited_on(self):
        # Waiting would hold up one of the crawl's worker threads; the next run tries again.
        waits: list[float] = []
        client, received = client_answering(
            httpx.Response(429, headers={"Retry-After": str(int(MAX_RETRY_AFTER_SECONDS) + 1)}), waits=waits)

        assert client.get("https://boards.example.com/jobs").status_code == 429
        assert len(received) == 1
        assert waits == []

    @pytest.mark.parametrize("status", [301, 400, 403, 404])
    def test_permanent_answers_are_not_retried(self, status):
        # Moved Workday boards and removed board IDs were 78 of the first 83 board failures.
        client, received = client_answering(httpx.Response(status))

        assert client.get("https://boards.example.com/jobs").status_code == status
        assert len(received) == 1

    def test_a_timeout_is_retried(self):
        client, received = client_answering(httpx.ReadTimeout("timed out"), httpx.Response(200))

        assert client.get("https://boards.example.com/jobs").status_code == 200
        assert len(received) == 2

    def test_a_dropped_connection_on_every_attempt_is_raised(self):
        client, received = client_answering(*[httpx.ConnectError("refused")] * 3)

        with pytest.raises(httpx.ConnectError):
            client.get("https://boards.example.com/jobs")
        assert len(received) == 3

    def test_a_post_search_is_resent_with_its_body(self):
        # Workable and Workday list their jobs through POSTed searches, which are reads.
        client, received = client_answering(httpx.Response(429), httpx.Response(200))

        client.post("https://boards.example.com/search", json={"token": "page-2"})

        assert [r.method for r in received] == ["POST", "POST"]
        assert received[0].content == received[1].content == b'{"token":"page-2"}'


def test_the_shared_client_carries_the_settings_and_the_retry_layer(settings):
    with make_client(settings) as client:
        assert client.headers["User-Agent"] == settings.http_user_agent
        assert client.timeout.read == settings.http_timeout_seconds
        assert isinstance(client._transport, RetryTransport)
