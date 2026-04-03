"""Tests for the mltk_report MCP tool.

Covers report generation, result parsing, error handling,
and response structure. No mocking needed -- mltk_report
is a pure function with no lazy imports.
"""
from __future__ import annotations

from ._helpers import (
    assert_error,
    assert_ok,
    assert_valid_json,
    call_tool,
    call_tool_raw,
)


class TestMltkReport:
    """Tests for the mltk_report tool."""

    def test_report_with_title(self):
        # SCENARIO: Generate a report with only a title
        # WHY: Minimal happy path -- title is the only required arg
        # EXPECTED: status=ok, report_text contains the title
        result = call_tool("mltk_report", title="Model Quality")

        assert_ok(result)
        assert "Model Quality" in result["report_text"]

    def test_report_with_results_json(self):
        # SCENARIO: Provide valid JSON results array
        # WHY: Results drive the summary table and pass/fail counts
        # EXPECTED: status=ok, summary mentions counts
        results = '[{"name": "t1", "passed": true}, {"name": "t2", "status": "ok"}]'
        result = call_tool(
            "mltk_report", title="Run", results_json=results,
        )

        assert_ok(result)
        assert "2" in result["summary"]
        assert "passed" in result["summary"]

    def test_report_empty_results(self):
        # SCENARIO: No results_json provided
        # WHY: Empty results should produce a valid report with placeholder
        # EXPECTED: status=ok, report_text contains "No results"
        result = call_tool("mltk_report", title="Empty Run")

        assert_ok(result)
        assert "No results" in result["report_text"]

    def test_report_invalid_results_json(self):
        # SCENARIO: Pass malformed JSON as results_json
        # WHY: Bad input must produce a clear error, not crash
        # EXPECTED: status=error, error mentions "Invalid results_json"
        result = call_tool(
            "mltk_report",
            title="Bad Input",
            results_json="{not valid json",
        )

        assert_error(result)
        assert "Invalid results_json" in result["error"]

    def test_response_has_report_text(self):
        # SCENARIO: Verify report_text is present and non-empty
        # WHY: report_text is the primary output consumers display
        # EXPECTED: report_text is a non-empty string
        result = call_tool("mltk_report", title="Check")

        assert_ok(result)
        assert isinstance(result["report_text"], str)
        assert len(result["report_text"]) > 0

    def test_response_has_summary(self):
        # SCENARIO: Verify summary is present and non-empty
        # WHY: Summary is used for quick status display by agents
        # EXPECTED: summary is a non-empty string
        result = call_tool("mltk_report", title="Check")

        assert_ok(result)
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0

    def test_report_includes_timestamp(self):
        # SCENARIO: Report text should include a timestamp
        # WHY: Timestamps anchor reports in time for auditing
        # EXPECTED: report_text contains "202" (current decade year prefix)
        result = call_tool("mltk_report", title="Timestamped")

        assert_ok(result)
        assert "202" in result["report_text"]

    def test_returns_valid_json(self):
        # SCENARIO: Raw output format validation
        # WHY: MCP tools must always return well-formed JSON
        # EXPECTED: Raw string parses as JSON with status key
        raw = call_tool_raw("mltk_report", title="JSON Check")

        data = assert_valid_json(raw)
        assert data["status"] == "ok"

    def test_report_with_description(self):
        # SCENARIO: Provide a description alongside the title
        # WHY: Description adds context to the generated report
        # EXPECTED: status=ok, report_text contains the description
        result = call_tool(
            "mltk_report",
            title="Full Run",
            description="Full run",
        )

        assert_ok(result)
        assert "Full run" in result["report_text"]

    def test_report_error_has_recoverable(self):
        # SCENARIO: Invalid JSON error recoverable field type check
        # WHY: Agents branch on recoverable -- must be a real bool
        # EXPECTED: recoverable is a bool
        result = call_tool(
            "mltk_report",
            title="Bad",
            results_json="[broken",
        )

        assert_error(result)
        assert isinstance(result["recoverable"], bool)

    def test_report_dict_results_json(self):
        # SCENARIO: results_json is a JSON object (not array)
        # WHY: Server wraps dict in list: [parsed]
        # EXPECTED: status=ok, summary mentions "1 results"
        result = call_tool(
            "mltk_report",
            title="Single",
            results_json='{"name": "single", "passed": true}',
        )

        assert_ok(result)
        assert "1" in result["summary"]

    def test_report_fail_item_shows_fail(self):
        # SCENARIO: Result item without passed/status=ok
        # WHY: _is_pass returns False → [FAIL] tag in report
        # EXPECTED: report_text contains [FAIL]
        result = call_tool(
            "mltk_report",
            title="Mixed",
            results_json='[{"name": "bad", "error": "broken"}]',
        )

        assert_ok(result)
        assert "FAIL" in result["report_text"]
