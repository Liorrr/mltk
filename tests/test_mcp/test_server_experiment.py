"""Tests for the mltk_experiment MCP tool.

Covers heuristic fix ranking, scoring strategies, result
limiting, error handling, and response structure. No mocking
needed -- mltk_experiment is a pure function operating on JSON.
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
            "category": "process",
            "title": "Add validation monitoring",
            "description": "Track val loss to detect divergence early.",
            "confidence": "low",
            "code_snippet": "",
        },
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
            "confidence": "medium",
            "code_snippet": "batch_size=64",
        },
    ],
})

_FINDING_NO_FIXES = json.dumps({
    "name": "minor_warning",
    "severity": "low",
})


class TestMltkExperiment:
    """Tests for the mltk_experiment tool."""

    def test_valid_finding_with_fixes(self):
        # SCENARIO: Finding has suggested_fixes array
        # WHY: Happy path -- ranked fixes should be returned
        # EXPECTED: status=ok, ranked_fixes list is non-empty
        result = call_tool(
            "mltk_experiment",
            finding_json=_FINDING_WITH_FIXES,
        )

        assert_ok(result)
        assert len(result["ranked_fixes"]) > 0

    def test_ranked_fixes_have_score_and_rank(self):
        # SCENARIO: Each ranked fix must have score and rank fields
        # WHY: Consumers rely on these fields for ordering
        # EXPECTED: Every fix dict has score (float) and rank (int)
        result = call_tool(
            "mltk_experiment",
            finding_json=_FINDING_WITH_FIXES,
        )

        assert_ok(result)
        for fix in result["ranked_fixes"]:
            assert "score" in fix, "Missing 'score' field"
            assert "rank" in fix, "Missing 'rank' field"
            assert isinstance(fix["score"], (int, float))
            assert isinstance(fix["rank"], int)

    def test_rank_one_has_highest_score(self):
        # SCENARIO: The fix with rank=1 should have the highest score
        # WHY: Rank ordering must be consistent with scoring
        # EXPECTED: rank=1 fix has score >= all other fixes
        result = call_tool(
            "mltk_experiment",
            finding_json=_FINDING_WITH_FIXES,
        )

        assert_ok(result)
        fixes = result["ranked_fixes"]
        assert len(fixes) >= 2
        top = fixes[0]
        assert top["rank"] == 1
        for fix in fixes[1:]:
            assert top["score"] >= fix["score"]

    def test_strategy_passed_confidence_first(self):
        # SCENARIO: Strategy "passed" ranks by confidence first
        # WHY: Confidence-first helps agents pick most certain fix
        # EXPECTED: High-confidence fixes appear before medium/low
        result = call_tool(
            "mltk_experiment",
            finding_json=_FINDING_WITH_FIXES,
            rank_by="passed",
        )

        assert_ok(result)
        assert result["strategy"] == "passed"
        fixes = result["ranked_fixes"]
        # First fixes should be high confidence (code category)
        assert fixes[0]["confidence"] == "high"
        # Last fixes should be low confidence
        assert fixes[-1]["confidence"] == "low"

    def test_strategy_delta_category_first(self):
        # SCENARIO: Strategy "delta" ranks by category actionability
        # WHY: Actionability-first helps agents pick easiest fix
        # EXPECTED: code (4) fixes appear before config (3), data (2), process (1)
        result = call_tool(
            "mltk_experiment",
            finding_json=_FINDING_WITH_FIXES,
            rank_by="delta",
            max_results=50,
        )

        assert_ok(result)
        assert result["strategy"] == "delta"
        fixes = result["ranked_fixes"]
        # First fixes should be "code" category (highest actionability)
        assert fixes[0]["category"] == "code"
        # Last fix should be "process" (lowest actionability)
        assert fixes[-1]["category"] == "process"

    def test_strategy_composite_balanced(self):
        # SCENARIO: Strategy "composite" uses weighted scoring
        # WHY: Balanced approach considers multiple dimensions
        # EXPECTED: Uses composite strategy, scores reflect weighted sum
        result = call_tool(
            "mltk_experiment",
            finding_json=_FINDING_WITH_FIXES,
            rank_by="composite",
        )

        assert_ok(result)
        assert result["strategy"] == "composite"
        # The top fix should be the one with best composite score
        # (high confidence=3, code category=4, with snippet=1)
        # score = 3*0.4 + 4*0.3 + 1*0.3 = 1.2 + 1.2 + 0.3 = 2.7
        top = result["ranked_fixes"][0]
        assert top["score"] == 2.7

    def test_invalid_json(self):
        # SCENARIO: finding_json is not valid JSON
        # WHY: Must produce clear error, not crash
        # EXPECTED: status=error, error mentions "Invalid finding_json"
        result = call_tool(
            "mltk_experiment",
            finding_json="{not valid json",
        )

        assert_error(result)
        assert "Invalid finding_json" in result["error"]

    def test_empty_finding_json(self):
        # SCENARIO: Pass an empty string as finding_json
        # WHY: Edge case -- must produce error, not crash
        # EXPECTED: status=error, error mentions "Empty"
        result = call_tool(
            "mltk_experiment",
            finding_json="",
        )

        assert_error(result)
        assert "Empty" in result["error"]

    def test_array_input_rejected(self):
        # SCENARIO: Pass a JSON array instead of a single object
        # WHY: Tool expects one finding, not a list
        # EXPECTED: status=error, error mentions "single object"
        array_json = json.dumps([
            {"name": "f1", "suggested_fixes": []},
            {"name": "f2", "suggested_fixes": []},
        ])
        result = call_tool(
            "mltk_experiment",
            finding_json=array_json,
        )

        assert_error(result)
        assert "single object" in result["error"]

    def test_no_suggested_fixes(self):
        # SCENARIO: Finding has no suggested_fixes key
        # WHY: Not all findings have fixes -- should return empty list
        # EXPECTED: status=ok, ranked_fixes=[], helpful message
        result = call_tool(
            "mltk_experiment",
            finding_json=_FINDING_NO_FIXES,
        )

        assert_ok(result)
        assert result["ranked_fixes"] == []
        assert result["total"] == 0
        assert "No fixes" in result["suggested_next_step"]

    def test_max_results_limits_output(self):
        # SCENARIO: Set max_results=2 on a finding with 6 fixes
        # WHY: Agents may want only top suggestions to save tokens
        # EXPECTED: At most 2 ranked fixes returned
        result = call_tool(
            "mltk_experiment",
            finding_json=_FINDING_WITH_FIXES,
            max_results=2,
        )

        assert_ok(result)
        assert len(result["ranked_fixes"]) == 2
        assert result["total"] == 2

    def test_response_has_suggested_next_step(self):
        # SCENARIO: Verify suggested_next_step is present
        # WHY: Agents use this to decide what to do after ranking
        # EXPECTED: suggested_next_step is a non-empty string
        result = call_tool(
            "mltk_experiment",
            finding_json=_FINDING_WITH_FIXES,
        )

        assert_ok(result)
        assert isinstance(result["suggested_next_step"], str)
        assert len(result["suggested_next_step"]) > 0

    def test_returns_valid_json(self):
        # SCENARIO: Raw output format validation
        # WHY: MCP tools must always return well-formed JSON
        # EXPECTED: Raw string parses as JSON with status key
        raw = call_tool_raw(
            "mltk_experiment",
            finding_json=_FINDING_WITH_FIXES,
        )

        data = assert_valid_json(raw)
        assert data["status"] == "ok"

    def test_invalid_strategy_falls_back_to_passed(self):
        # SCENARIO: Pass an unrecognised rank_by strategy
        # WHY: Should gracefully fall back, not error
        # EXPECTED: status=ok, strategy reported as "passed"
        result = call_tool(
            "mltk_experiment",
            finding_json=_FINDING_WITH_FIXES,
            rank_by="unknown_strategy",
        )

        assert_ok(result)
        assert result["strategy"] == "passed"
        assert len(result["ranked_fixes"]) > 0
