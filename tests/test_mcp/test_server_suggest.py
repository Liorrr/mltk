"""Tests for the mltk_suggest MCP tool.

Covers fix suggestion retrieval, category filtering, result
limiting, error handling, and response structure. No mocking
needed -- mltk_suggest is a pure function operating on JSON.
"""
from __future__ import annotations

import json

from ._helpers import (
    assert_error,
    assert_ok,
    assert_valid_json,
    call_tool,
    call_tool_raw,
)

# ----------------------------------------------------------
# Reusable test data
# ----------------------------------------------------------

_FINDING_WITH_FIXES = json.dumps({
    "name": "high_loss",
    "severity": "high",
    "suggested_fixes": [
        {
            "category": "code",
            "title": "Reduce learning rate",
            "description": "Lower the LR to stabilize training.",
            "confidence": "high",
            "code_snippet": "lr = 1e-4",
        },
        {
            "category": "config",
            "title": "Enable gradient clipping",
            "description": "Clip gradients to prevent explosions.",
            "confidence": "medium",
            "code_snippet": "",
        },
        {
            "category": "data",
            "title": "Remove outliers",
            "description": "Filter extreme values from training set.",
            "confidence": "low",
            "code_snippet": "df = df[df.z_score < 3]",
        },
        {
            "category": "process",
            "title": "Add validation monitoring",
            "description": "Track val loss to detect divergence early.",
            "confidence": "medium",
            "code_snippet": "",
        },
        {
            "category": "code",
            "title": "Use weight decay",
            "description": "Add L2 regularization to optimizer.",
            "confidence": "high",
            "code_snippet": "weight_decay=0.01",
        },
        {
            "category": "config",
            "title": "Increase batch size",
            "description": "Larger batches smooth gradient noise.",
            "confidence": "low",
            "code_snippet": "",
        },
    ],
})

_FINDING_NO_FIXES = json.dumps({
    "name": "minor_warning",
    "severity": "low",
})

_FIX_FIELDS = {"category", "title", "description", "confidence", "code_snippet"}


class TestMltkSuggest:
    """Tests for the mltk_suggest tool."""

    def test_valid_finding_with_fixes(self):
        # SCENARIO: Finding has suggested_fixes array
        # WHY: Happy path -- fixes should be returned as suggestions
        # EXPECTED: status=ok, suggestions list is non-empty
        result = call_tool(
            "mltk_suggest",
            finding_json=_FINDING_WITH_FIXES,
        )

        assert_ok(result)
        assert len(result["suggestions"]) > 0

    def test_valid_finding_no_fixes(self):
        # SCENARIO: Finding has no suggested_fixes key
        # WHY: Not all findings have fixes -- should return empty list
        # EXPECTED: status=ok, suggestions=[], helpful message
        result = call_tool(
            "mltk_suggest",
            finding_json=_FINDING_NO_FIXES,
        )

        assert_ok(result)
        assert result["suggestions"] == []
        assert result["total"] == 0
        assert "No suggestions" in result["suggested_next_step"]

    def test_invalid_json(self):
        # SCENARIO: finding_json is not valid JSON
        # WHY: Must produce clear error, not crash
        # EXPECTED: status=error, error mentions "Invalid finding_json"
        result = call_tool(
            "mltk_suggest",
            finding_json="{not valid json",
        )

        assert_error(result)
        assert "Invalid finding_json" in result["error"]

    def test_filter_by_category(self):
        # SCENARIO: Filter suggestions by category "code"
        # WHY: Agents may only want code-level fixes
        # EXPECTED: All returned suggestions have category "code"
        result = call_tool(
            "mltk_suggest",
            finding_json=_FINDING_WITH_FIXES,
            category="code",
        )

        assert_ok(result)
        assert len(result["suggestions"]) > 0
        for s in result["suggestions"]:
            assert s["category"] == "code"

    def test_filter_unknown_category(self):
        # SCENARIO: Filter by a category that no fix matches
        # WHY: Should return empty list, not error
        # EXPECTED: status=ok, suggestions=[], total=0
        result = call_tool(
            "mltk_suggest",
            finding_json=_FINDING_WITH_FIXES,
            category="nonexistent",
        )

        assert_ok(result)
        assert result["suggestions"] == []
        assert result["total"] == 0

    def test_max_results_limits_output(self):
        # SCENARIO: Set max_results=2 on a finding with 6 fixes
        # WHY: Agents may want only top suggestions to save tokens
        # EXPECTED: At most 2 suggestions returned
        result = call_tool(
            "mltk_suggest",
            finding_json=_FINDING_WITH_FIXES,
            max_results=2,
        )

        assert_ok(result)
        assert len(result["suggestions"]) == 2
        assert result["total"] == 2

    def test_response_has_total_count(self):
        # SCENARIO: Verify total field matches suggestions length
        # WHY: Agents use total to decide if truncation occurred
        # EXPECTED: total == len(suggestions)
        result = call_tool(
            "mltk_suggest",
            finding_json=_FINDING_WITH_FIXES,
            max_results=3,
        )

        assert_ok(result)
        assert result["total"] == len(result["suggestions"])

    def test_response_has_suggested_next_step(self):
        # SCENARIO: Verify suggested_next_step is present
        # WHY: Agents use this to decide what to do after reading fixes
        # EXPECTED: suggested_next_step is a non-empty string
        result = call_tool(
            "mltk_suggest",
            finding_json=_FINDING_WITH_FIXES,
        )

        assert_ok(result)
        assert isinstance(result["suggested_next_step"], str)
        assert len(result["suggested_next_step"]) > 0

    def test_suggestion_has_all_fields(self):
        # SCENARIO: Each suggestion dict has all 5 required fields
        # WHY: Consumers rely on consistent schema across suggestions
        # EXPECTED: Every suggestion has category, title, description,
        #           confidence, and code_snippet keys
        result = call_tool(
            "mltk_suggest",
            finding_json=_FINDING_WITH_FIXES,
        )

        assert_ok(result)
        for s in result["suggestions"]:
            assert set(s.keys()) == _FIX_FIELDS, (
                f"Missing fields: {_FIX_FIELDS - set(s.keys())}"
            )

    def test_returns_valid_json(self):
        # SCENARIO: Raw output format validation
        # WHY: MCP tools must always return well-formed JSON
        # EXPECTED: Raw string parses as JSON with status key
        raw = call_tool_raw(
            "mltk_suggest",
            finding_json=_FINDING_WITH_FIXES,
        )

        data = assert_valid_json(raw)
        assert data["status"] == "ok"

    def test_error_has_recoverable_field(self):
        # SCENARIO: Error response contains recoverable bool
        # WHY: Agents branch on recoverable to decide retry logic
        # EXPECTED: recoverable is a bool
        result = call_tool(
            "mltk_suggest",
            finding_json="[not valid",
        )

        assert_error(result)
        assert isinstance(result["recoverable"], bool)

    def test_array_input_rejected(self):
        # SCENARIO: Pass a JSON array instead of a single object
        # WHY: Tool expects one finding, not a list of findings
        # EXPECTED: status=error, error mentions "single object"
        array_json = json.dumps([
            {"name": "f1", "suggested_fixes": []},
            {"name": "f2", "suggested_fixes": []},
        ])
        result = call_tool(
            "mltk_suggest",
            finding_json=array_json,
        )

        assert_error(result)
        assert "single object" in result["error"]

    def test_category_filter_case_insensitive(self):
        # SCENARIO: Pass category in mixed case ("Code")
        # WHY: Case should not matter for category filtering
        # EXPECTED: Matches "code" suggestions despite uppercase input
        result = call_tool(
            "mltk_suggest",
            finding_json=_FINDING_WITH_FIXES,
            category="Code",
        )

        assert_ok(result)
        assert len(result["suggestions"]) > 0
        for s in result["suggestions"]:
            assert s["category"] == "code"

    def test_empty_finding_json_string(self):
        # SCENARIO: Pass an empty string as finding_json
        # WHY: Edge case -- must produce error, not crash
        # EXPECTED: status=error, error mentions "Empty"
        result = call_tool(
            "mltk_suggest",
            finding_json="",
        )

        assert_error(result)
        assert "Empty" in result["error"]
