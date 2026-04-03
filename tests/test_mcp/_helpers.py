"""Shared helpers for MCP server tests.

Provides mock FastMCP infrastructure, tool registry,
and response validation helpers. All test files in
this directory import from here.
"""
from __future__ import annotations

import json
import sys
import types
from typing import Any

# Global registry populated by mock @mcp.tool() decorator
registered_tools: dict[str, Any] = {}


# ----------------------------------------------------------
# Mock FastMCP infrastructure
# ----------------------------------------------------------

def make_fastmcp_mock():
    """Build a mock FastMCP class that captures tools."""

    class _FastMCP:
        def __init__(self, name: str, **kwargs: Any) -> None:
            self.name = name
            self._tools: dict[str, Any] = {}
            self._kwargs = kwargs

        def tool(self, **kwargs: Any):  # noqa: ARG002
            """Decorator that registers a tool function."""
            def decorator(fn):
                self._tools[fn.__name__] = fn
                registered_tools[fn.__name__] = fn
                return fn
            return decorator

        def run(self, transport: str = "stdio") -> None:
            pass  # no-op for tests

    return _FastMCP


def make_mcp_modules() -> dict[str, types.ModuleType]:
    """Build mock mcp package hierarchy."""
    mcp = types.ModuleType("mcp")
    server = types.ModuleType("mcp.server")
    fastmcp_mod = types.ModuleType("mcp.server.fastmcp")
    fastmcp_mod.FastMCP = make_fastmcp_mock()  # type: ignore[attr-defined]
    mcp.server = server  # type: ignore[attr-defined]
    server.fastmcp = fastmcp_mod  # type: ignore[attr-defined]
    return {
        "mcp": mcp,
        "mcp.server": server,
        "mcp.server.fastmcp": fastmcp_mod,
    }


def import_server():
    """Force-reimport mltk.mcp.server with mocked mcp."""
    mod_name = "mltk.mcp.server"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    from mltk.mcp import server  # noqa: PLC0415
    return server


# ----------------------------------------------------------
# Tool invocation helpers
# ----------------------------------------------------------

def call_tool(name: str, **kwargs: Any) -> dict[str, Any]:
    """Call a registered tool by name, return parsed JSON."""
    fn = registered_tools[name]
    raw = fn(**kwargs)
    return json.loads(raw)


def call_tool_raw(name: str, **kwargs: Any) -> str:
    """Call a registered tool by name, return raw JSON string."""
    fn = registered_tools[name]
    return fn(**kwargs)


# ----------------------------------------------------------
# Response validation helpers
# ----------------------------------------------------------

def assert_ok(data: dict[str, Any]) -> None:
    """Assert response has status=ok."""
    assert data["status"] == "ok", (
        f"Expected status=ok, got {data.get('status')!r}"
    )


def assert_error(data: dict[str, Any]) -> None:
    """Assert response is a well-formed error."""
    assert data["status"] == "error"
    assert "error" in data
    assert isinstance(data["error"], str)
    assert "recoverable" in data
    assert isinstance(data["recoverable"], bool)
    assert "suggested_action" in data
    assert isinstance(data["suggested_action"], str)


def assert_valid_json(raw: str) -> dict[str, Any]:
    """Assert string is valid JSON with status field."""
    data = json.loads(raw)
    assert isinstance(data, dict)
    assert "status" in data
    return data
