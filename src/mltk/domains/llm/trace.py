"""Agent execution trace — dataclasses for representing tool calls and traces."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A single tool/function call within an agent trace.

    Captures the tool name, arguments passed, optional result or error,
    and execution duration.  Used as the building block of :class:`AgentTrace`.

    Attributes:
        name: Tool name (e.g., ``"search"``, ``"calculator"``).
        arguments: Arguments passed to the tool.
        result: Tool output, if available.
        error: Error message if the call failed.
        duration_ms: Execution time in milliseconds.
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str | None = None
    error: str | None = None
    duration_ms: float = 0.0


def _parse_tool_call(raw: dict[str, Any]) -> ToolCall:
    """Build a :class:`ToolCall` from a raw dictionary.

    Handles two layouts:

    1. **Direct** — ``{"name": "search", "arguments": {...}, ...}``
    2. **OpenAI function-calling** — ``{"function": {"name": "search",
       "arguments": "{...}"}, "type": "function"}``

    JSON-encoded ``arguments`` strings are decoded automatically.
    """
    # OpenAI function-calling layout: unwrap the nested "function" key.
    if "function" in raw and isinstance(raw["function"], dict):
        func = raw["function"]
        name = func.get("name", "")
        arguments = func.get("arguments", {})
    else:
        name = raw.get("name", "")
        arguments = raw.get("arguments", {})

    # Arguments may arrive as a JSON string (common in OpenAI responses).
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            arguments = {}

    return ToolCall(
        name=name,
        arguments=arguments if isinstance(arguments, dict) else {},
        result=raw.get("result"),
        error=raw.get("error"),
        duration_ms=float(raw.get("duration_ms", 0.0)),
    )


@dataclass
class AgentTrace:
    """A complete execution trace of an AI agent.

    Collects all :class:`ToolCall` instances made during an agent run along
    with aggregate token counts, wall-clock duration, and arbitrary metadata.

    Attributes:
        tool_calls: Ordered list of tool calls performed by the agent.
        total_tokens: Total tokens consumed across the trace.
        total_duration_ms: Total wall-clock time in milliseconds.
        metadata: Arbitrary metadata attached to the trace.
    """

    tool_calls: list[ToolCall] = field(default_factory=list)
    total_tokens: int = 0
    total_duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any] | list[Any]) -> AgentTrace:
        """Construct an :class:`AgentTrace` from a plain dict (or list).

        Accepts various dict formats — handles both flat lists of tool calls
        and nested structures.  Designed to work with JSON output from
        LangChain, OpenAI function calling, and generic agent frameworks.

        Supported formats:

        1. **Simple dict** with ``"tool_calls"`` key and optional top-level
           ``"total_tokens"`` / ``"total_duration_ms"`` / ``"metadata"``.
        2. **OpenAI function-calling style** where each entry in
           ``"tool_calls"`` has a nested ``"function"`` dict whose
           ``"arguments"`` may be a JSON string.
        3. **Flat list** — a bare ``list`` of tool-call dicts (treated as
           if wrapped in ``{"tool_calls": [...]}``.

        Args:
            data: A dictionary (or list) representing a serialised trace.

        Returns:
            A fully-populated :class:`AgentTrace`.

        Example:
            >>> trace = AgentTrace.from_dict({
            ...     "tool_calls": [
            ...         {"name": "search", "arguments": {"query": "weather"}},
            ...     ],
            ...     "total_tokens": 150,
            ... })
            >>> trace.tool_names
            ['search']
        """
        # Flat list: wrap into the canonical dict shape.
        if isinstance(data, list):
            data = {"tool_calls": data}

        raw_calls: list[dict[str, Any]] = data.get("tool_calls", [])
        tool_calls = [_parse_tool_call(rc) for rc in raw_calls]

        return cls(
            tool_calls=tool_calls,
            total_tokens=int(data.get("total_tokens", 0)),
            total_duration_ms=float(data.get("total_duration_ms", 0.0)),
            metadata=dict(data.get("metadata", {})),
        )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def tool_names(self) -> list[str]:
        """List of tool names in call order."""
        return [tc.name for tc in self.tool_calls]

    @property
    def step_count(self) -> int:
        """Number of tool calls."""
        return len(self.tool_calls)

    @property
    def failed_calls(self) -> list[ToolCall]:
        """Tool calls that resulted in errors."""
        return [tc for tc in self.tool_calls if tc.error is not None]
