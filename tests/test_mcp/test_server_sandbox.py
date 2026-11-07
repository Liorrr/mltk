"""Tests for unsupported JSON finding sandbox mode."""

from __future__ import annotations

import json
from unittest.mock import patch

from ._helpers import assert_error, assert_ok, call_tool

_FINDING = json.dumps({
    "name": "high_loss",
    "suggested_fixes": [{"title": "Fix", "code_snippet": "x = 1"}],
})


class TestMltkExperimentSandbox:
    """Tests for mltk_experiment sandbox behavior."""

    def test_sandbox_true_returns_unsupported_error(self):
        result = call_tool("mltk_experiment", finding_json=_FINDING, sandbox=True)
        assert_error(result)
        assert "sandbox=True is not supported" in result["error"]
        assert result["suggested_action"] == (
            "Call mltk_experiment with sandbox=False for heuristic ranking."
        )

    def test_sandbox_true_runs_no_snippet(self):
        with (
            patch("mltk.experiment.sandbox.SandboxedExperimentRunner") as runner,
            patch("mltk.experiment.worktree.GitWorktree") as worktree,
        ):
            result = call_tool(
                "mltk_experiment", finding_json=_FINDING, sandbox=True,
            )
        assert_error(result)
        runner.assert_not_called()
        worktree.assert_not_called()

    def test_sandbox_false_unchanged(self):
        result = call_tool(
            "mltk_experiment", finding_json=_FINDING, sandbox=False,
        )
        assert_ok(result)
        assert result["ranked_fixes"]
        assert "sandbox" not in result

    # Input validation runs before the sandbox check, so these still apply.

    def test_sandbox_true_empty_finding_json(self):
        """sandbox=True with empty finding_json returns error."""
        # SCENARIO: Empty string input
        # WHY: Input validation before sandbox path
        # EXPECTED: status=error, mentions "Empty"
        result = call_tool(
            "mltk_experiment",
            finding_json="",
            sandbox=True,
        )

        assert_error(result)
        assert "Empty" in result["error"]

    def test_sandbox_true_invalid_json(self):
        """sandbox=True with invalid JSON returns error."""
        # SCENARIO: Malformed JSON input
        # WHY: Input validation before sandbox path
        # EXPECTED: status=error, mentions "Invalid finding_json"
        result = call_tool(
            "mltk_experiment",
            finding_json="{bad json",
            sandbox=True,
        )

        assert_error(result)
        assert "Invalid finding_json" in result["error"]

    def test_sandbox_true_array_input(self):
        """sandbox=True with array input returns error."""
        # SCENARIO: JSON array instead of object
        # WHY: Input validation before sandbox path
        # EXPECTED: status=error, mentions "single object"
        array_json = json.dumps([{"name": "f1"}])
        result = call_tool(
            "mltk_experiment",
            finding_json=array_json,
            sandbox=True,
        )

        assert_error(result)
        assert "single object" in result["error"]
