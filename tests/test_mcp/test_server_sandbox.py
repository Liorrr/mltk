"""Tests for mltk_experiment MCP tool sandbox mode.

Covers sandboxed experiment execution via git worktrees,
git availability checks, error handling, and response
structure.  All external dependencies are mocked.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from ._helpers import (
    assert_error,
    assert_ok,
    call_tool,
)

# ----------------------------------------------------------
# Reusable test data
# ----------------------------------------------------------

_FINDING_WITH_FIXES = json.dumps({
    "name": "high_loss",
    "severity": "high",
    "scanner_name": "overfit",
    "suggested_fixes": [
        {
            "category": "code",
            "title": "Reduce learning rate",
            "description": "Lower the LR to stabilize.",
            "confidence": "high",
            "code_snippet": "lr = 1e-4",
        },
        {
            "category": "config",
            "title": "Enable gradient clipping",
            "description": "Clip gradients.",
            "confidence": "medium",
            "code_snippet": "",
        },
        {
            "category": "data",
            "title": "Remove outliers",
            "description": "Filter extreme values.",
            "confidence": "low",
            "code_snippet": "df = df[df.z < 3]",
        },
    ],
})

_FINDING_NO_FIXES = json.dumps({
    "name": "minor_warning",
    "severity": "low",
    "scanner_name": "drift",
})


# ----------------------------------------------------------
# Mock helpers
# ----------------------------------------------------------

def _mock_hypothesis_result(
    *,
    fix_title: str = "Reduce learning rate",
    fix_category: str = "code",
    fix_confidence: str = "high",
    fix_description: str = "Lower the LR.",
    fix_snippet: str = "lr = 1e-4",
    passed: bool = True,
    improvement: float = 1.0,
    rank: int = 1,
) -> MagicMock:
    """Build a mock HypothesisResult for testing."""
    hr = MagicMock()
    hr.hypothesis.fix.category = fix_category
    hr.hypothesis.fix.title = fix_title
    hr.hypothesis.fix.description = fix_description
    hr.hypothesis.fix.confidence = fix_confidence
    hr.hypothesis.fix.code_snippet = fix_snippet
    hr.rank = rank
    hr.improvement = improvement
    hr.fixed_result.passed = passed
    hr.is_winning = passed and improvement > 0
    return hr


def _mock_experiment_result(
    any_fix: bool = True,
    num_hypotheses: int = 2,
    selected_title: str | None = "Reduce learning rate",
) -> MagicMock:
    """Build a mock ExperimentResult for testing."""
    result = MagicMock()
    result.any_fix_works = any_fix
    result.duration_ms = 42.0

    hrs = []
    for i in range(num_hypotheses):
        hrs.append(_mock_hypothesis_result(
            fix_title=f"Fix {i + 1}",
            rank=i + 1,
            passed=any_fix if i == 0 else False,
            improvement=1.0 if (any_fix and i == 0) else 0.0,
        ))
    result.hypothesis_results = hrs

    if any_fix and selected_title is not None:
        result.selected_fix = MagicMock()
        result.selected_fix.title = selected_title
    else:
        result.selected_fix = None

    return result


def _sandbox_patches(
    *,
    git_ok: bool = True,
    git_root: Path | None = None,
    run_result: MagicMock | None = None,
    runner_raises: Exception | None = None,
):
    """Return a dict of patches for sandbox dependencies.

    All patches target the SOURCE modules, not the MCP module.
    """
    if git_root is None:
        git_root = Path("/fake/repo")
    if run_result is None:
        run_result = _mock_experiment_result()

    mock_runner_cls = MagicMock()
    if runner_raises is not None:
        mock_runner_cls.return_value.run.side_effect = (
            runner_raises
        )
    else:
        mock_runner_cls.return_value.run.return_value = (
            run_result
        )

    patches = {
        "git_available": patch(
            "mltk.experiment.worktree.git_available",
            return_value=git_ok,
        ),
        "find_git_root": patch(
            "mltk.experiment.worktree.find_git_root",
            return_value=git_root,
        ),
        "runner": patch(
            "mltk.experiment.sandbox.SandboxedExperimentRunner",
            mock_runner_cls,
        ),
    }
    return patches, mock_runner_cls


# ----------------------------------------------------------
# Tests
# ----------------------------------------------------------


class TestMltkExperimentSandbox:
    """Tests for mltk_experiment with sandbox=True."""

    def test_sandbox_false_unchanged(self):
        """sandbox=False behaves identically to current."""
        # SCENARIO: Explicit sandbox=False
        # WHY: Regression -- must not change existing behavior
        # EXPECTED: status=ok, ranked_fixes present, no sandbox key
        result = call_tool(
            "mltk_experiment",
            finding_json=_FINDING_WITH_FIXES,
            sandbox=False,
        )

        assert_ok(result)
        assert len(result["ranked_fixes"]) > 0
        assert "sandbox" not in result

    def test_sandbox_true_valid_finding(self):
        """sandbox=True with valid finding returns status=ok."""
        # SCENARIO: Happy path sandbox execution
        # WHY: Core functionality must work end-to-end
        # EXPECTED: status=ok, ranked_fixes, sandbox=True
        patches, _ = _sandbox_patches()
        with (
            patches["git_available"],
            patches["find_git_root"],
            patches["runner"],
        ):
            result = call_tool(
                "mltk_experiment",
                finding_json=_FINDING_WITH_FIXES,
                sandbox=True,
            )

        assert_ok(result)
        assert len(result["ranked_fixes"]) > 0

    def test_sandbox_true_no_git(self):
        """sandbox=True with no git returns error."""
        # SCENARIO: Git CLI not installed
        # WHY: Must produce clear error, not crash
        # EXPECTED: status=error, mentions "Git CLI not found"
        patches, _ = _sandbox_patches(git_ok=False)
        with patches["git_available"]:
            result = call_tool(
                "mltk_experiment",
                finding_json=_FINDING_WITH_FIXES,
                sandbox=True,
            )

        assert_error(result)
        assert "Git CLI not found" in result["error"]

    def test_sandbox_true_no_git_repo(self):
        """sandbox=True with no git repo returns error."""
        # SCENARIO: Not inside a git repository
        # WHY: Must produce clear error
        # EXPECTED: status=error, mentions "Not in a git repository"
        with (
            patch(
                "mltk.experiment.worktree.git_available",
                return_value=True,
            ),
            patch(
                "mltk.experiment.worktree.find_git_root",
                side_effect=FileNotFoundError("no repo"),
            ),
        ):
            result = call_tool(
                "mltk_experiment",
                finding_json=_FINDING_WITH_FIXES,
                sandbox=True,
            )

        assert_error(result)
        assert "Not in a git repository" in result["error"]

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

    def test_sandbox_true_no_suggested_fixes(self):
        """sandbox=True with no suggested_fixes returns empty."""
        # SCENARIO: Finding has no fixes
        # WHY: Should return empty list, not crash
        # EXPECTED: status=ok, ranked_fixes=[], sandbox=True
        patches, _ = _sandbox_patches()
        with (
            patches["git_available"],
            patches["find_git_root"],
            patches["runner"],
        ):
            result = call_tool(
                "mltk_experiment",
                finding_json=_FINDING_NO_FIXES,
                sandbox=True,
            )

        assert_ok(result)
        assert result["ranked_fixes"] == []
        assert result["total"] == 0
        assert result["sandbox"] is True

    def test_sandbox_true_returns_ranked_fixes(self):
        """sandbox=True returns ranked_fixes in response."""
        # SCENARIO: Runner returns hypothesis results
        # WHY: Response must contain ranked fix data
        # EXPECTED: ranked_fixes has rank, title, category fields
        patches, _ = _sandbox_patches()
        with (
            patches["git_available"],
            patches["find_git_root"],
            patches["runner"],
        ):
            result = call_tool(
                "mltk_experiment",
                finding_json=_FINDING_WITH_FIXES,
                sandbox=True,
            )

        assert_ok(result)
        fixes = result["ranked_fixes"]
        assert len(fixes) > 0
        first = fixes[0]
        assert "rank" in first
        assert "title" in first
        assert "category" in first
        assert "improvement" in first
        assert "passed" in first

    def test_sandbox_true_response_includes_sandbox_flag(self):
        """sandbox=True response includes 'sandbox': true."""
        # SCENARIO: Sandbox flag in response
        # WHY: Consumers need to distinguish sandbox vs heuristic
        # EXPECTED: "sandbox" key is True in response
        patches, _ = _sandbox_patches()
        with (
            patches["git_available"],
            patches["find_git_root"],
            patches["runner"],
        ):
            result = call_tool(
                "mltk_experiment",
                finding_json=_FINDING_WITH_FIXES,
                sandbox=True,
            )

        assert_ok(result)
        assert result["sandbox"] is True

    def test_sandbox_true_respects_max_results(self):
        """sandbox=True respects max_results parameter."""
        # SCENARIO: Limit output to 1 result
        # WHY: Agents may want only the top suggestion
        # EXPECTED: At most 1 ranked fix returned
        mock_result = _mock_experiment_result(
            num_hypotheses=3,
        )
        patches, _ = _sandbox_patches(
            run_result=mock_result,
        )
        with (
            patches["git_available"],
            patches["find_git_root"],
            patches["runner"],
        ):
            result = call_tool(
                "mltk_experiment",
                finding_json=_FINDING_WITH_FIXES,
                max_results=1,
                sandbox=True,
            )

        assert_ok(result)
        assert len(result["ranked_fixes"]) == 1
        assert result["total"] == 1

    def test_sandbox_true_respects_rank_by_strategy(self):
        """sandbox=True passes rank_by strategy to runner."""
        # SCENARIO: Custom strategy "delta"
        # WHY: Strategy must propagate to SandboxedExperimentRunner
        # EXPECTED: strategy appears in response, runner created
        #   with that strategy
        patches, mock_cls = _sandbox_patches()
        with (
            patches["git_available"],
            patches["find_git_root"],
            patches["runner"],
        ):
            result = call_tool(
                "mltk_experiment",
                finding_json=_FINDING_WITH_FIXES,
                rank_by="delta",
                sandbox=True,
            )

        assert_ok(result)
        assert result["strategy"] == "delta"
        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args
        assert call_kwargs.kwargs["strategy"] == "delta"

    def test_sandbox_true_returns_suggested_next_step(self):
        """sandbox=True returns suggested_next_step."""
        # SCENARIO: Verify next step is present
        # WHY: Agents use this to decide what to do next
        # EXPECTED: suggested_next_step is a non-empty string
        patches, _ = _sandbox_patches()
        with (
            patches["git_available"],
            patches["find_git_root"],
            patches["runner"],
        ):
            result = call_tool(
                "mltk_experiment",
                finding_json=_FINDING_WITH_FIXES,
                sandbox=True,
            )

        assert_ok(result)
        assert isinstance(result["suggested_next_step"], str)
        assert len(result["suggested_next_step"]) > 0

    def test_sandbox_true_runner_exception_graceful(self):
        """sandbox=True handles runner exception gracefully."""
        # SCENARIO: SandboxedExperimentRunner.run() raises
        # WHY: Must not crash, must return structured error
        # EXPECTED: status=error with the exception message
        patches, _ = _sandbox_patches(
            runner_raises=RuntimeError("worktree failed"),
        )
        with (
            patches["git_available"],
            patches["find_git_root"],
            patches["runner"],
        ):
            result = call_tool(
                "mltk_experiment",
                finding_json=_FINDING_WITH_FIXES,
                sandbox=True,
            )

        assert_error(result)
        assert "worktree failed" in result["error"]

    def test_sandbox_true_no_winning_fix(self):
        """sandbox=True with no winning fix reports correctly."""
        # SCENARIO: Runner finds no fix that resolves the finding
        # WHY: Must report absence of winning fix clearly
        # EXPECTED: status=ok, no selected_fix, appropriate message
        mock_result = _mock_experiment_result(
            any_fix=False, selected_title=None,
        )
        patches, _ = _sandbox_patches(
            run_result=mock_result,
        )
        with (
            patches["git_available"],
            patches["find_git_root"],
            patches["runner"],
        ):
            result = call_tool(
                "mltk_experiment",
                finding_json=_FINDING_WITH_FIXES,
                sandbox=True,
            )

        assert_ok(result)
        assert "selected_fix" not in result
        assert "No fix resolved" in result["suggested_next_step"]

    def test_sandbox_true_winning_fix_includes_selected(self):
        """sandbox=True with winning fix includes selected_fix."""
        # SCENARIO: Runner selects a winning fix
        # WHY: Must include selected_fix title in response
        # EXPECTED: selected_fix key present with fix title
        mock_result = _mock_experiment_result(
            any_fix=True,
            selected_title="Reduce learning rate",
        )
        patches, _ = _sandbox_patches(
            run_result=mock_result,
        )
        with (
            patches["git_available"],
            patches["find_git_root"],
            patches["runner"],
        ):
            result = call_tool(
                "mltk_experiment",
                finding_json=_FINDING_WITH_FIXES,
                sandbox=True,
            )

        assert_ok(result)
        assert result["selected_fix"] == "Reduce learning rate"

    def test_sandbox_true_includes_duration(self):
        """sandbox=True response includes duration_ms."""
        # SCENARIO: Timing info in response
        # WHY: Agents need to know how long the experiment took
        # EXPECTED: duration_ms is a positive number
        patches, _ = _sandbox_patches()
        with (
            patches["git_available"],
            patches["find_git_root"],
            patches["runner"],
        ):
            result = call_tool(
                "mltk_experiment",
                finding_json=_FINDING_WITH_FIXES,
                sandbox=True,
            )

        assert_ok(result)
        assert "duration_ms" in result
        assert isinstance(result["duration_ms"], (int, float))
        assert result["duration_ms"] >= 0

    def test_sandbox_true_passes_repo_root_to_runner(self):
        """sandbox=True passes found repo_root to runner."""
        # SCENARIO: Verify repo_root propagation
        # WHY: Runner needs the correct repo root path
        # EXPECTED: Runner constructed with repo_root kwarg
        fake_root = Path("/my/project")
        patches, mock_cls = _sandbox_patches(
            git_root=fake_root,
        )
        with (
            patches["git_available"],
            patches["find_git_root"],
            patches["runner"],
        ):
            call_tool(
                "mltk_experiment",
                finding_json=_FINDING_WITH_FIXES,
                sandbox=True,
            )

        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args
        assert call_kwargs.kwargs["repo_root"] == fake_root
