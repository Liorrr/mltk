"""Tests for mltk_list MCP tool.

Validates assertion discovery, filtering by text and domain,
response schema, and edge cases (empty results).
"""
from __future__ import annotations

from unittest.mock import patch

from ._helpers import (
    assert_error,
    assert_ok,
    assert_valid_json,
    call_tool,
    call_tool_raw,
)

# ----------------------------------------------------------
# Shared mock data
# ----------------------------------------------------------

MOCK_ALL = {
    "data": [
        {"name": "assert_no_drift", "doc": "Detect distribution drift"},
        {"name": "assert_no_nulls", "doc": "Check for null values"},
    ],
    "model": [
        {"name": "assert_metric", "doc": "Validate model metric"},
    ],
    "llm": [
        {"name": "assert_faithfulness", "doc": "Check RAG faithfulness"},
    ],
}

MOCK_DRIFT_ONLY = {
    "data": [
        {"name": "assert_no_drift", "doc": "Detect distribution drift"},
    ],
}

MOCK_MODEL_ONLY = {
    "model": [
        {"name": "assert_metric", "doc": "Validate model metric"},
    ],
}

MOCK_LLM_ONLY = {
    "llm": [
        {"name": "assert_faithfulness", "doc": "Check RAG faithfulness"},
    ],
}

MOCK_EMPTY: dict = {}

_PATCH_TARGET = "mltk.cli._discovery.discover_assertions"


class TestMltkList:
    """mltk_list tool — assertion discovery and filtering."""

    def test_list_all(self) -> None:
        # SCENARIO: Call with no filters.
        # WHY: Default invocation should return every assertion.
        # EXPECTED: status=ok, assertions list is non-empty.
        with patch(_PATCH_TARGET, return_value=MOCK_ALL):
            result = call_tool("mltk_list")
        assert_ok(result)
        assert len(result["assertions"]) > 0

    def test_list_filter_text(self) -> None:
        # SCENARIO: Pass filter_text="drift".
        # WHY: Text filter narrows results to matching entries.
        # EXPECTED: status=ok, only drift-related assertions returned.
        with patch(_PATCH_TARGET, return_value=MOCK_DRIFT_ONLY):
            result = call_tool("mltk_list", filter_text="drift")
        assert_ok(result)
        assert all(
            "drift" in a["name"] or "drift" in a["description"].lower()
            for a in result["assertions"]
        )

    def test_list_filter_domain(self) -> None:
        # SCENARIO: Pass domain="model" with no text filter.
        # WHY: Domain filter restricts to a single category.
        # EXPECTED: status=ok, only model-domain assertions returned.
        with patch(_PATCH_TARGET, return_value=MOCK_MODEL_ONLY):
            result = call_tool("mltk_list", domain="model")
        assert_ok(result)
        assert all(
            a["domain"] == "model" for a in result["assertions"]
        )

    def test_list_empty_filter_returns_all(self) -> None:
        # SCENARIO: Explicitly pass filter_text="".
        # WHY: Empty string should behave like no filter.
        # EXPECTED: status=ok, full set returned.
        with patch(_PATCH_TARGET, return_value=MOCK_ALL):
            result = call_tool("mltk_list", filter_text="")
        assert_ok(result)
        assert result["total"] == 4

    def test_response_has_total_count(self) -> None:
        # SCENARIO: Inspect the total field.
        # WHY: Agents use total to decide if results are complete.
        # EXPECTED: total matches len(assertions).
        with patch(_PATCH_TARGET, return_value=MOCK_ALL):
            result = call_tool("mltk_list")
        assert_ok(result)
        assert result["total"] == len(result["assertions"])

    def test_response_has_assertions_array(self) -> None:
        # SCENARIO: Verify assertions is a list.
        # WHY: Downstream code iterates over assertions.
        # EXPECTED: assertions key exists and is a list.
        with patch(_PATCH_TARGET, return_value=MOCK_ALL):
            result = call_tool("mltk_list")
        assert_ok(result)
        assert isinstance(result["assertions"], list)

    def test_each_assertion_has_name(self) -> None:
        # SCENARIO: Check every assertion entry has a name.
        # WHY: Name is required for referencing assertions in tests.
        # EXPECTED: All entries have a non-empty "name" string.
        with patch(_PATCH_TARGET, return_value=MOCK_ALL):
            result = call_tool("mltk_list")
        assert_ok(result)
        for a in result["assertions"]:
            assert "name" in a
            assert isinstance(a["name"], str)
            assert len(a["name"]) > 0

    def test_each_assertion_has_description(self) -> None:
        # SCENARIO: Check every assertion entry has a description.
        # WHY: Descriptions help agents pick the right assertion.
        # EXPECTED: All entries have a "description" string.
        with patch(_PATCH_TARGET, return_value=MOCK_ALL):
            result = call_tool("mltk_list")
        assert_ok(result)
        for a in result["assertions"]:
            assert "description" in a
            assert isinstance(a["description"], str)

    def test_filter_drift_matches(self) -> None:
        # SCENARIO: Filter for "drift" and inspect returned names.
        # WHY: Confirms text filter surfaces the right assertions.
        # EXPECTED: Every returned name contains "drift".
        with patch(_PATCH_TARGET, return_value=MOCK_DRIFT_ONLY):
            result = call_tool("mltk_list", filter_text="drift")
        assert_ok(result)
        names = [a["name"] for a in result["assertions"]]
        assert len(names) > 0
        assert all("drift" in n for n in names)

    def test_domain_llm(self) -> None:
        # SCENARIO: Filter by domain="llm".
        # WHY: LLM assertions are a distinct category.
        # EXPECTED: All returned entries have domain="llm".
        with patch(_PATCH_TARGET, return_value=MOCK_LLM_ONLY):
            result = call_tool("mltk_list", domain="llm")
        assert_ok(result)
        assert all(
            a["domain"] == "llm" for a in result["assertions"]
        )

    def test_returns_valid_json(self) -> None:
        # SCENARIO: Validate the raw JSON output string.
        # WHY: MCP transport requires well-formed JSON.
        # EXPECTED: Parseable JSON with a "status" field.
        with patch(_PATCH_TARGET, return_value=MOCK_ALL):
            raw = call_tool_raw("mltk_list")
        assert_valid_json(raw)

    def test_no_results_returns_empty(self) -> None:
        # SCENARIO: Mock returns empty dict (no matches).
        # WHY: Edge case — no assertions match the filter.
        # EXPECTED: status=ok, total=0, assertions=[].
        with patch(_PATCH_TARGET, return_value=MOCK_EMPTY):
            result = call_tool("mltk_list", filter_text="nonexistent")
        assert_ok(result)
        assert result["total"] == 0
        assert result["assertions"] == []

    def test_discover_raises_error(self) -> None:
        # SCENARIO: discover_assertions raises an exception.
        # WHY: Server must catch and return a JSON error.
        # EXPECTED: status=error, not an unhandled exception.
        with patch(
            _PATCH_TARGET,
            side_effect=RuntimeError("discovery failed"),
        ):
            result = call_tool("mltk_list")
        assert_error(result)
