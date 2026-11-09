"""The Messages calls of a group of threads, sent as Message Batches: half price, asynchronous.

``MessageBatcher`` stands in for the SDK inside an ``AnthropicClient`` (it exposes
``.messages.create``), so the code that makes the calls — triage, the full evaluation, its second
look — runs unchanged, in a thread per posting. A call blocks its thread; once every participant
is either waiting on a call or done, the waiting calls go out as one batch, and when that batch
ends each thread gets its own message back. A posting's calls depend on each other (the full
evaluation only after triage, a second look only after the first verdict), so a run goes out in
rounds, one batch each. Most batches end within an hour; the API gives up on one after 24.

Cache hits inside a batch are best-effort (the docs give 30-98%), so a batched call can pay a
cache write a direct call wouldn't — measure a run's real cost rather than assume half.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import anthropic

from jobscan.llm.client import LlmCallError
from jobscan.logging_setup import get_logger, log_extra

logger = get_logger("batch")

_POLL_SECONDS = 30.0
_PROGRESS_LOG_SECONDS = 300.0


@dataclass
class _Call:
    params: dict
    done: threading.Event = field(default_factory=threading.Event)
    message: Any = None
    error: str | None = None


def _error_text(error: Any) -> str:
    inner = getattr(error, "error", None)
    message = getattr(inner, "message", None)
    return f"{getattr(inner, 'type', 'error')}: {message}" if message else str(error)


class MessageBatcher:
    def __init__(
        self,
        api_key: str | None = None,
        max_retries: int = 3,
        timeout_seconds: int = 60,
        sdk: Any = None,
        poll_seconds: float = _POLL_SECONDS,
    ):
        """``sdk`` is normally left as None, which creates a real ``anthropic.Anthropic`` from
        ``api_key``. Tests pass a stub exposing ``.messages.batches.create/retrieve/results/cancel``."""
        if sdk is None:
            if not api_key:
                raise LlmCallError("ANTHROPIC_API_KEY is not set")
            sdk = anthropic.Anthropic(
                api_key=api_key, max_retries=max_retries, timeout=timeout_seconds
            )
        self._sdk = sdk
        self._poll_seconds = poll_seconds
        self._lock = threading.Lock()
        self._participants = 0
        self._waiting: list[_Call] = []
        self._cancelled = threading.Event()
        self.batches_sent = 0
        # What AnthropicClient calls: self._client.messages.create(...).
        self.messages = self

    def join(self, count: int = 1) -> None:
        """Count ``count`` more participants: threads that will make calls, then ``leave()``.
        Join them all before any starts, or the first to call goes out in a batch of its own."""
        with self._lock:
            self._participants += count

    def leave(self) -> None:
        """A participant has made its last call; the rest may now be all a round waits for."""
        with self._lock:
            self._participants -= 1
            calls = self._take_round()
        if calls:
            self._send(calls)

    @contextmanager
    def participant(self) -> Iterator[None]:
        self.join()
        try:
            yield
        finally:
            self.leave()

    def cancel(self) -> None:
        """Stop: the batch in flight is cancelled and its calls fail, as does any later call."""
        self._cancelled.set()

    def create(self, **params: Any) -> Any:
        """One Messages call, made as part of the next batch; blocks until that batch ends."""
        call = _Call(params)
        with self._lock:
            if self._cancelled.is_set():
                raise LlmCallError("message batch cancelled")
            self._waiting.append(call)
            calls = self._take_round()
        if calls:
            self._send(calls)
        call.done.wait()
        if call.error is not None:
            raise LlmCallError(call.error)
        return call.message

    def _take_round(self) -> list[_Call]:
        """Under the lock: the waiting calls, once every participant still calling waits on one."""
        if self._waiting and len(self._waiting) >= self._participants:
            calls, self._waiting = self._waiting, []
            return calls
        return []

    def _send(self, calls: list[_Call]) -> None:
        """Runs on the thread that completed the round; every other participant is blocked on a
        call meanwhile. Whatever happens, every call is answered — no thread waits forever."""
        try:
            self._run(calls)
        except Exception as exc:  # noqa: BLE001 — any failure must still answer every call
            for call in calls:
                if call.message is None and call.error is None:
                    call.error = f"message batch failed: {exc}"
        finally:
            for call in calls:
                if call.message is None and call.error is None:
                    call.error = "message batch returned no result for this call"
                call.done.set()

    def _run(self, calls: list[_Call]) -> None:
        if self._cancelled.is_set():
            for call in calls:
                call.error = "message batch cancelled"
            return
        by_id = {f"call-{i}": call for i, call in enumerate(calls)}
        batch = self._sdk.messages.batches.create(
            requests=[{"custom_id": key, "params": call.params} for key, call in by_id.items()]
        )
        self.batches_sent += 1
        started = last_logged = time.monotonic()
        logger.info(
            "message batch sent; waiting for it to end (most take under an hour)",
            extra=log_extra(batch_id=batch.id, requests=len(calls)),
        )
        while batch.processing_status != "ended":
            if self._cancelled.wait(self._poll_seconds):
                self._sdk.messages.batches.cancel(batch.id)
                logger.warning("message batch cancelled", extra=log_extra(batch_id=batch.id))
                for call in calls:
                    call.error = "message batch cancelled"
                return
            batch = self._sdk.messages.batches.retrieve(batch.id)
            if time.monotonic() - last_logged >= _PROGRESS_LOG_SECONDS:
                last_logged = time.monotonic()
                logger.info(
                    "message batch still running",
                    extra=log_extra(
                        batch_id=batch.id,
                        processing=batch.request_counts.processing,
                        requests=len(calls),
                        minutes=round((last_logged - started) / 60),
                    ),
                )
        for entry in self._sdk.messages.batches.results(batch.id):
            call = by_id.get(entry.custom_id)
            if call is None:
                continue
            result = entry.result
            if result.type == "succeeded":
                call.message = result.message
            elif result.type == "errored":
                call.error = f"batch request errored: {_error_text(result.error)}"
            else:
                call.error = f"batch request {result.type}"
        logger.info(
            "message batch ended",
            extra=log_extra(
                batch_id=batch.id,
                requests=len(calls),
                minutes=round((time.monotonic() - started) / 60, 1),
            ),
        )
