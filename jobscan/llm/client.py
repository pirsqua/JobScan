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
    ) -> ToolCallResult:
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user_message}],
                tools=[tool_schema],
                tool_choice={"type": "tool", "name": tool_name},
            )
        except anthropic.APIError as exc:
            raise LlmCallError(f"Anthropic API call failed: {exc}") from exc

        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "tool_use" and getattr(block, "name", None) == tool_name:
                usage = response.usage
                return ToolCallResult(
                    input=block.input,
                    input_tokens=getattr(usage, "input_tokens", 0),
                    output_tokens=getattr(usage, "output_tokens", 0),
                )

        raise LlmCallError(f"model response did not include a '{tool_name}' tool_use block")
