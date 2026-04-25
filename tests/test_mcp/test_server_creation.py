"""Tests for MCP server factory and tool registration."""
from __future__ import annotations

from ._helpers import import_server, registered_tools

EXPECTED_TOOLS = {
    "mltk_scan",
    "mltk_test",
    "mltk_list",
    "mltk_eval",
    "mltk_dataset",
    "mltk_report",
    "mltk_suggest",
    "mltk_experiment",
    "mltk_create_pr",
    "mltk_create_issue",
    "mltk_workflow",
    "mltk_container_scan",
}


class TestServerCreation:
    """Verify create_server() produces a correctly wired FastMCP instance."""

    def test_create_server_returns_instance(self, mcp_server):
        # SCENARIO: Call create_server() via the fixture.
        # WHY: Smoke-test that the factory doesn't crash.
        # EXPECTED: A non-None server object.
        assert mcp_server is not None

    def test_server_name_is_mltk(self, mcp_server):
        # SCENARIO: Inspect server identity.
        # WHY: MCP clients discover servers by name.
        # EXPECTED: name == "mltk".
        assert mcp_server.name == "mltk"

    def test_server_has_version(self, mcp_server):
        # SCENARIO: Check version metadata.
        # WHY: Clients may negotiate by version.
        # EXPECTED: A non-empty version string in _kwargs.
        version = mcp_server._kwargs.get("version", "")
        assert isinstance(version, str)
        assert version, "version must be a non-empty string"

    def test_all_twelve_tools_registered(self, mcp_server):
        # SCENARIO: Verify the full tool set.
        # WHY: Missing tools break agent workflows.
        # EXPECTED: Exactly the 12 expected tool names.
        assert set(mcp_server._tools.keys()) == EXPECTED_TOOLS

    def test_tool_names_have_mltk_prefix(self, mcp_server):
        # SCENARIO: Check naming convention.
        # WHY: Prefix prevents collisions with other MCP servers.
        # EXPECTED: Every tool name starts with "mltk_".
        for name in mcp_server._tools:
            assert name.startswith("mltk_"), f"{name!r} missing prefix"

    def test_tool_functions_are_callable(self, mcp_server):  # noqa: ARG002
        # SCENARIO: Verify registered values are real functions.
        # WHY: Non-callable entries would fail at invocation time.
        # EXPECTED: All values in registered_tools are callable.
        for name, fn in registered_tools.items():
            assert callable(fn), f"{name!r} is not callable"

    def test_run_server_exists(self):
        # SCENARIO: Check the public entry-point function.
        # WHY: mltk CLI calls run_server() to start MCP mode.
        # EXPECTED: server module exposes a callable run_server.
        server = import_server()
        assert hasattr(server, "run_server")
        assert callable(server.run_server)
