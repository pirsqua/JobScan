from __future__ import annotations

import threading

import pytest

from jobscan.llm.batch import MessageBatcher
from jobscan.llm.client import AnthropicClient, LlmCallError
from tests.fakes import FakeBatchSdk, errored_result, json_response


def echo(params):
    """Answers each request with the marker it was sent with, so a test can tell whose it is."""
    return json_response({"marker": params["marker"]})


def run_participants(batcher: MessageBatcher, plans: dict[str, int]) -> dict[str, list]:
    """One thread per participant, each making ``plans[name]`` calls in turn; what each got back."""
    batcher.join(len(plans))
    got: dict[str, list] = {name: [] for name in plans}

    def participant(name: str, calls: int) -> None:
        try:
            for i in range(calls):
                try:
                    got[name].append(batcher.create(marker=f"{name}{i}").content[0].text)
                except LlmCallError as exc:
                    got[name].append(exc)
        finally:
            batcher.leave()

    threads = [threading.Thread(target=participant, args=item) for item in plans.items()]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert not any(thread.is_alive() for thread in threads), "a participant never got its result"
    return got


def markers(batch: list[dict]) -> set[str]:
    return {request["params"]["marker"] for request in batch}


def test_each_round_of_calls_goes_out_as_one_batch():
    sdk = FakeBatchSdk(echo)

    got = run_participants(MessageBatcher(sdk=sdk, poll_seconds=0), {"a": 2, "b": 2, "c": 2})

    assert [markers(batch) for batch in sdk.sent] == [{"a0", "b0", "c0"}, {"a1", "b1", "c1"}]
    assert got["b"] == ['{"marker": "b0"}', '{"marker": "b1"}']


def test_a_participant_that_has_finished_does_not_hold_up_the_next_round():
    sdk = FakeBatchSdk(echo)

    run_participants(MessageBatcher(sdk=sdk, poll_seconds=0), {"a": 1, "b": 3})

    assert [markers(batch) for batch in sdk.sent] == [{"a0", "b0"}, {"b1"}, {"b2"}]


def test_a_failed_request_fails_only_its_own_call():
    sdk = FakeBatchSdk(lambda p: errored_result("Overloaded") if p["marker"] == "a0" else echo(p))

    got = run_participants(MessageBatcher(sdk=sdk, poll_seconds=0), {"a": 1, "b": 1})

    assert isinstance(got["a"][0], LlmCallError)
    assert "overloaded_error: Overloaded" in str(got["a"][0])
    assert got["b"] == ['{"marker": "b0"}']


def test_a_batch_that_cannot_be_sent_fails_every_call_without_hanging():
    sdk = FakeBatchSdk(echo, fail_create=True)

    got = run_participants(MessageBatcher(sdk=sdk, poll_seconds=0), {"a": 1, "b": 1})

    assert all("batch create rejected" in str(results[0]) for results in got.values())


def test_cancel_cancels_the_batch_in_flight_and_every_later_call():
    sdk = FakeBatchSdk(echo, polls=None)  # never ends on its own
    batcher = MessageBatcher(sdk=sdk, poll_seconds=0.01)
    errors: list[Exception] = []

    def participant():
        with batcher.participant():
            try:
                batcher.create(marker="a0")
            except LlmCallError as exc:
                errors.append(exc)

    thread = threading.Thread(target=participant)
    thread.start()
    while not sdk.sent:
        thread.join(timeout=0.01)
    batcher.cancel()
    thread.join(timeout=10)

    assert not thread.is_alive()
    assert sdk.cancelled == ["batch-0"]
    assert "cancelled" in str(errors[0])
    with pytest.raises(LlmCallError, match="cancelled"):
        batcher.create(marker="b0")


def test_a_client_sends_its_usual_request_through_the_batcher():
    sdk = FakeBatchSdk(lambda params: json_response({"answer": 42}, input_tokens=7))
    batcher = MessageBatcher(sdk=sdk, poll_seconds=0)
    client = AnthropicClient(api_key=None, model="m", thinking="disabled", client=object())

    with batcher.participant():
        result = client.through(batcher).call_structured(
            system="sys", user_message="hello", schema={"type": "object", "properties": {}}
        )

    assert (result.data, result.input_tokens) == ({"answer": 42}, 7)
    params = sdk.sent[0][0]["params"]
    assert (params["model"], params["thinking"]) == ("m", {"type": "disabled"})
    assert params["output_config"]["format"]["type"] == "json_schema"
