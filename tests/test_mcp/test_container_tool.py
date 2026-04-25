"""Tests for MCP tool ``mltk_container_scan`` (tool #12).

The tool itself is wired in :mod:`mltk.mcp.server` by the orchestrator
for sprint S93. These tests validate registration, success paths,
and error handling against a mocked :mod:`mltk.container.assertions`
module (Trivy is never invoked).

Patch rule
----------
Per the MCP test infrastructure, MCP tools use lazy imports
inside the tool function body. Tests therefore patch at the
SOURCE module -- ``mltk.container.assertions`` -- never at
``mltk.mcp.server``.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from ._helpers import (
    assert_error,
    assert_ok,
    call_tool,
    registered_tools,
)


def _mk_result(
    name: str,
    passed: bool,
    message: str = "",
    details: dict | None = None,
) -> SimpleNamespace:
    """Build a minimal ``AssertionResult``-like stand-in."""
    return SimpleNamespace(
        name=name,
        passed=passed,
        message=message,
        details=details or {},
    )


class TestMltkContainerScan:
    """Tests for the ``mltk_container_scan`` MCP tool."""

    def test_tool_registered(self):
        # SCENARIO: conftest built the server, tools registered
        # WHY: Tool #12 must be present in the registry
        # EXPECTED: "mltk_container_scan" is in registered_tools
        assert "mltk_container_scan" in registered_tools

    def test_success(self):
        # SCENARIO: Both assertions pass for a clean image
        # WHY: Happy path -- return ok + passed=True payload
        # EXPECTED: status=ok, passed=True, details propagated
        vuln = _mk_result(
            "container.vulnerabilities",
            passed=True,
            message="No vulnerabilities found",
            details={
                "critical_count": 0,
                "high_count": 0,
                "cves": [],
            },
        )
        secret = _mk_result(
            "container.secrets",
            passed=True,
            message="No secrets found",
            details={},
        )
        with patch(
            "mltk.container.assertions"
            ".assert_container_vulnerabilities",
            return_value=vuln,
        ), patch(
            "mltk.container.assertions"
            ".assert_no_secrets_in_image",
            return_value=secret,
        ):
            data = call_tool(
                "mltk_container_scan", image="alpine:3.18",
            )

        assert_ok(data)
        assert data.get("passed") is True

    def test_vulnerability_failure(self):
        # SCENARIO: Vulnerability assertion fails (1 CRITICAL CVE)
        # WHY: Failed scans should return a structured result
        #      (status=ok, passed=False), not raise an error
        # EXPECTED: status=ok, passed=False, CVE details present
        vuln = _mk_result(
            "container.vulnerabilities",
            passed=False,
            message="1 CRITICAL CVE found",
            details={
                "critical_count": 1,
                "cves": [
                    {
                        "id": "CVE-2024-1234",
                        "severity": "CRITICAL",
                    },
                ],
            },
        )
        secret = _mk_result(
            "container.secrets", passed=True,
        )
        with patch(
            "mltk.container.assertions"
            ".assert_container_vulnerabilities",
            return_value=vuln,
        ), patch(
            "mltk.container.assertions"
            ".assert_no_secrets_in_image",
            return_value=secret,
        ):
            data = call_tool(
                "mltk_container_scan",
                image="alpine:3.18",
                max_critical=0,
            )

        assert_ok(data)
        assert data.get("passed") is False

    def test_secret_failure(self):
        # SCENARIO: Secret scan fails (embedded AWS key)
        # WHY: Mirror the vuln failure path but via the secret
        #      assertion so both branches are exercised
        # EXPECTED: status=ok, passed=False
        vuln = _mk_result(
            "container.vulnerabilities", passed=True,
        )
        secret = _mk_result(
            "container.secrets",
            passed=False,
            message="1 secret detected",
            details={"secret_count": 1},
        )
        with patch(
            "mltk.container.assertions"
            ".assert_container_vulnerabilities",
            return_value=vuln,
        ), patch(
            "mltk.container.assertions"
            ".assert_no_secrets_in_image",
            return_value=secret,
        ):
            data = call_tool(
                "mltk_container_scan", image="alpine:3.18",
            )

        assert_ok(data)
        assert data.get("passed") is False

    def test_error_handling(self):
        # SCENARIO: Trivy / container extras not installed
        # WHY: Tool must catch ImportError and return an
        #      error envelope instead of propagating
        # EXPECTED: status=error with recoverable + suggested_action
        with patch(
            "mltk.container.assertions"
            ".assert_container_vulnerabilities",
            side_effect=ImportError("trivy not found"),
        ):
            data = call_tool(
                "mltk_container_scan", image="bad:image",
            )

        assert_error(data)
