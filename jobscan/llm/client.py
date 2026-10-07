"""Thin wrapper around the Anthropic SDK that forces a single structured tool call.

Kept deliberately small and injectable: tests pass in a stub with a ``messages.create`` method
instead of hitting the network, and the rest of the codebase only ever sees ``ToolCallResult``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import anthropic


class LlmCallError(RuntimeError):
    """The model call failed, or didn't return the requested structured tool call."""


@dataclass
class ToolCallResult:
    input: dict
    input_tokens: int
    output_tokens: int
    # Separate from input_tokens (which is only the non-cached portion) because they're billed at
    # different rates — see EvaluateStats.estimated_cost_usd.
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


# Keywords strict tool use can't enforce (numeric/length bounds) or that conflict with every
# property being required (defaults). Pydantic still enforces the bounds on the way back in.
_UNSUPPORTED_STRICT_KEYWORDS = frozenset({
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "minLength", "maxLength", "minItems", "maxItems", "default",
})


def strict_input_schema(schema: Any) -> Any:
    """A copy of a Pydantic-generated JSON schema in the form strict tool use accepts: every
    object closed (``additionalProperties: false``) with all its properties required (optional
    ones are already nullable via ``anyOf ... null``), unsupported keywords dropped. With
    ``strict: true`` the API guarantees the tool input matches — observed live without it, the
    model once wrapped a whole evaluation in an extra ``{"evaluation": {...}}`` object (Lumos)."""
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
    ):
        """``client`` is normally left as None, which creates a real ``anthropic.Anthropic``
        from ``api_key``. Tests pass in a stub object exposing ``.messages.create(...)`` instead
        of hitting the network."""
        self.model = model
        if client is not None:
            self._client = client
        else:
            if not api_key:
                raise LlmCallError("ANTHROPIC_API_KEY is not set")
            self._client = anthropic.Anthropic(
                api_key=api_key, max_retries=max_retries, timeout=timeout_seconds
            )

    def call_tool(
        self,
        system: str,
        user_message: str,
        tool_schema: dict,
        tool_name: str,
        max_tokens: int = 2048,
        cacheable_prefix: str | None = None,
    ) -> ToolCallResult:
        """``system`` and ``tool_schema`` are identical on every call a given caller makes (the
        same evaluation prompt/schema every time), so they're always marked cacheable — a batch
        of calls pays full input price for that ~3,000-token prefix once, then ~10% of it on
        every subsequent call within the cache window. ``cacheable_prefix`` is for content that's
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

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": content}],
                tools=[{
                    **tool_schema,
                    "input_schema": strict_input_schema(tool_schema["input_schema"]),
                    "strict": True,
                    "cache_control": {"type": "ephemeral"},
                }],
                tool_choice={"type": "tool", "name": tool_name},
            )
        except anthropic.APIError as exc:
            raise LlmCallError(f"Anthropic API call failed: {exc}") from exc

        if getattr(response, "stop_reason", None) == "max_tokens":
            raise LlmCallError(f"'{tool_name}' response was cut off at max_tokens={max_tokens}")

        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "tool_use" and getattr(block, "name", None) == tool_name:
                usage = response.usage
                return ToolCallResult(
                    input=block.input,
                    input_tokens=getattr(usage, "input_tokens", 0),
                    output_tokens=getattr(usage, "output_tokens", 0),
                    cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
                    cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
                )

        raise LlmCallError(f"model response did not include a '{tool_name}' tool_use block")
