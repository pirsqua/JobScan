"""Thin wrapper around the Anthropic SDK that asks for one JSON object matching a schema.

Kept deliberately small and injectable: tests pass in a stub with a ``messages.create`` method
instead of hitting the network, and the rest of the codebase only ever sees ``StructuredResult``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import anthropic


class LlmCallError(RuntimeError):
    """The model call failed, declined, or didn't return the requested JSON object."""


@dataclass
class StructuredResult:
    data: dict
    input_tokens: int
    output_tokens: int
    # Separate from input_tokens (which is only the non-cached portion) because they're billed at
    # different rates — see EvaluateStats.estimated_cost_usd.
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


# Keywords structured outputs can't enforce (numeric/length bounds) or that conflict with every
# property being required (defaults). Pydantic still enforces the bounds on the way back in.
_UNSUPPORTED_STRICT_KEYWORDS = frozenset({
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "minLength", "maxLength", "minItems", "maxItems", "default",
})


def strict_input_schema(schema: Any) -> Any:
    """A copy of a Pydantic-generated JSON schema in the form structured outputs accept: every
    object closed (``additionalProperties: false``) with all its properties required (optional
    ones are already nullable via ``anyOf ... null``), unsupported keywords dropped. The API then
    guarantees the response matches — observed live before the schema was enforced, the model once
    wrapped a whole evaluation in an extra ``{"evaluation": {...}}`` object (Lumos)."""
    if isinstance(schema, list):
        return [strict_input_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema
    converted = {
        key: strict_input_schema(value) for key, value in schema.items() if key not in _UNSUPPORTED_STRICT_KEYWORDS
    }
    if converted.get("type") == "object" and "properties" in converted:
        converted["additionalProperties"] = False
        converted["required"] = list(converted["properties"])
    return converted


class AnthropicClient:
    def __init__(
        self,
        api_key: str | None,
        model: str,
        max_retries: int = 3,
        timeout_seconds: int = 60,
        client: Any = None,
        thinking: str | None = None,
    ):
        """``thinking`` is the ``thinking.type`` sent with every call (``"disabled"``,
        ``"between_tools"``, ``"adaptive"``), or None to leave the model's default — which value
        turns thinking off depends on the model, so it's configured beside the model name.

        ``client`` is normally left as None, which creates a real ``anthropic.Anthropic``
        from ``api_key``. Tests pass in a stub object exposing ``.messages.create(...)`` instead
        of hitting the network."""
        self.model = model
        self.thinking = thinking
        if client is not None:
            self._client = client
        else:
            if not api_key:
                raise LlmCallError("ANTHROPIC_API_KEY is not set")
            self._client = anthropic.Anthropic(
                api_key=api_key, max_retries=max_retries, timeout=timeout_seconds
            )

    def call_structured(
        self,
        system: str,
        user_message: str,
        schema: dict,
        max_tokens: int = 2048,
        cacheable_prefix: str | None = None,
    ) -> StructuredResult:
        """One JSON object matching ``schema`` (structured outputs). The model writes the
        properties in the schema's order — verified live on Sonnet 5.5 and Haiku 5.5 — so with
        thinking off, property order is the order it reasons in.

        ``system`` is identical on every call a given caller makes, so it's always marked
        cacheable — a batch of calls pays full input price for it once, then ~10% of it on every
        subsequent call within the cache window. ``cacheable_prefix`` is for content that's
        static within a run but caller-specific (e.g. the rendered candidate profile, identical
        across every job in an evaluate run but not shared with company classification calls) —
        pass it to cache that too, ahead of the per-call dynamic ``user_message``."""
        content: str | list[dict]
        if cacheable_prefix is not None:
            content = [
                {"type": "text", "text": cacheable_prefix, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": user_message},
            ]
        else:
            content = user_message

        request: dict[str, Any] = {}
        if self.thinking is not None:
            request["thinking"] = {"type": self.thinking}
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": content}],
                output_config={"format": {"type": "json_schema", "schema": strict_input_schema(schema)}},
                **request,
            )
        except anthropic.APIError as exc:
            raise LlmCallError(f"Anthropic API call failed: {exc}") from exc

        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason == "refusal":
            # A safety decline (HTTP 200), not an answer — it must never be read as a verdict.
            category = getattr(getattr(response, "stop_details", None), "category", None) or "unspecified"
            raise LlmCallError(f"model declined to answer (refusal category: {category})")
        if stop_reason == "max_tokens":
            raise LlmCallError(f"response was cut off at max_tokens={max_tokens}")

        # Read by block type: with thinking on, thinking blocks come before the answer.
        text = next((block.text for block in response.content if getattr(block, "type", None) == "text"), None)
        if text is None:
            raise LlmCallError("model response did not include a text block")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LlmCallError(f"model response was not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise LlmCallError("model response was not a JSON object")

        usage = response.usage
        return StructuredResult(
            data=data,
            input_tokens=getattr(usage, "input_tokens", 0),
            output_tokens=getattr(usage, "output_tokens", 0),
            cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        )
