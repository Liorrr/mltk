"""Tests for mltk.experiment.sandbox -- SandboxedExperimentRunner.

The SandboxedExperimentRunner overrides _run_hypothesis() to execute
each hypothesis in an isolated git worktree via subprocess.  These
tests mock GitWorktree entirely -- no real git operations.

Tests:
    1.  Subclass of ExperimentRunner.
    2.  Constructor stores repo_root.
    3.  Constructor calls find_git_root when repo_root is None.
    4.  _run_hypothesis creates GitWorktree and enters context.
    5.  _run_hypothesis calls write_file with code_snippet.
    6.  _run_hypothesis calls run_in_worktree with python + script.
    7.  _run_hypothesis returns passed TestResult on success.
    8.  _run_hypothesis returns failed TestResult on worktree error.
    9.  _run_hypothesis returns failed TestResult on subprocess timeout.
    10. _run_hypothesis returns failed on CalledProcessError.
    11. _run_hypothesis returns failed on invalid JSON stdout.
    12. _run_hypothesis includes duration_ms in result.
    13. _run_hypothesis includes sandboxed=True in details.
    14. _apply_fix writes code_snippet when present.
    15. _apply_fix does nothing when code_snippet is empty.
    16. _parse_test_result parses valid JSON.
    17. _parse_test_result raises ValueError on garbage input.
    18. _build_assertion_script returns valid Python string.
    19. run() integration: produces ExperimentResult via sandbox.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.core.result import Severity, TestResult
from mltk.experiment.hypothesis import Hypothesis
from mltk.experiment.result import ExperimentResult
from mltk.experiment.runner import ExperimentRunner
from mltk.experiment.sandbox import (
    SandboxedExperimentRunner,
    _build_assertion_script,
    _parse_test_result,
)
from mltk.scan.finding import FixSuggestion, ScanFinding

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _make_fix(
    title: str = "test fix",
    snippet: str = "x = 1",
) -> FixSuggestion:
    """Create a minimal FixSuggestion for testing."""
    return FixSuggestion(
        category="code",
        title=title,
        description=f"Apply {title}",
        confidence="high",
        code_snippet=snippet,
    )


def _make_finding(
    scanner: str = "data",
) -> ScanFinding:
    """Create a minimal ScanFinding with a failing assertion."""

    def _failing_assertion() -> TestResult:
        result = TestResult(
            name="baseline_check",
            passed=False,
            severity=Severity.CRITICAL,
            message="Original failure",
        )
        raise MltkAssertionError(result)

    return ScanFinding(
        result=TestResult(
            name="scan_result",
            passed=False,
            severity=Severity.CRITICAL,
            message="Scan found issue",
        ),
        assertion_fn=_failing_assertion,
        scanner_name=scanner,
        suggested_fixes=[_make_fix()],
    )


def _make_hypothesis(
    fix: FixSuggestion | None = None,
    apply_fn: object | None = None,
) -> Hypothesis:
    """Create a Hypothesis with a dummy apply_fn."""
    if fix is None:
        fix = _make_fix()
    if apply_fn is None:

        def _default_apply() -> TestResult:
            return TestResult(
                name="fixed",
                passed=True,
                severity=Severity.INFO,
                message="Fixed",
            )

        apply_fn = _default_apply
    return Hypothesis(
        fix=fix,
        apply_fn=apply_fn,
        description="Test hypothesis",
    )


def _mock_worktree(
    stdout: str = ('{"name":"test","passed":true,"severity":"info","message":"ok"}'),
    tmp_path: Path | None = None,
) -> MagicMock:
    """Return a mock GitWorktree context manager.

    The mock supports ``__enter__``/``__exit__``, ``path``,
    ``branch``, ``run_in_worktree``, and ``write_file``.
    """
    wt = MagicMock()
    wt.path = tmp_path or Path("/tmp/mock-worktree")
    wt.branch = "mltk-sandbox-test"
    wt.run_in_worktree.return_value = subprocess.CompletedProcess(
        args=[sys.executable, "_mltk_assert.py"],
        returncode=0,
        stdout=stdout,
        stderr="",
    )
    wt.write_file = MagicMock()

    # Make it a working context manager.
    wt.__enter__ = MagicMock(return_value=wt)
    wt.__exit__ = MagicMock(return_value=False)
    return wt


def _patch_worktree(
    wt_mock: MagicMock,
) -> object:
    """Patch GitWorktree at its import location in sandbox.py."""
    return patch(
        "mltk.experiment.worktree.GitWorktree",
        return_value=wt_mock,
    )


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------


class TestSandboxedExperimentRunnerClass:
    """Verify class structure and constructor."""

    def test_is_subclass_of_experiment_runner(self) -> None:
        """SandboxedExperimentRunner inherits ExperimentRunner."""
        assert issubclass(
            SandboxedExperimentRunner,
            ExperimentRunner,
        )

    def test_constructor_stores_repo_root(
        self,
        tmp_path: Path,
    ) -> None:
        """Constructor stores the provided repo_root path."""
        runner = SandboxedExperimentRunner(
            repo_root=tmp_path,
        )
        assert runner._repo_root == tmp_path

    @patch("mltk.experiment.worktree.find_git_root")
    def test_constructor_calls_find_git_root_when_none(
        self,
        mock_find: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Constructor calls find_git_root when repo_root is None."""
        mock_find.return_value = tmp_path
        runner = SandboxedExperimentRunner(repo_root=None)
        mock_find.assert_called_once()
        assert runner._repo_root == tmp_path


class TestRunHypothesis:
    """Verify _run_hypothesis method behavior."""

    def test_creates_worktree_and_enters_context(
        self,
        tmp_path: Path,
    ) -> None:
        """_run_hypothesis creates GitWorktree and enters it."""
        wt = _mock_worktree()
        runner = SandboxedExperimentRunner(
            repo_root=tmp_path,
        )
        hyp = _make_hypothesis()
        finding = _make_finding()

        with _patch_worktree(wt):
            runner._run_hypothesis(hyp, finding)

        wt.__enter__.assert_called_once()
        wt.__exit__.assert_called_once()

    def test_calls_write_file_with_snippet(
        self,
        tmp_path: Path,
    ) -> None:
        """_run_hypothesis writes fix.code_snippet to worktree."""
        wt = _mock_worktree()
        fix = _make_fix(snippet="y = 42")
        hyp = _make_hypothesis(fix=fix)
        finding = _make_finding()
        runner = SandboxedExperimentRunner(
            repo_root=tmp_path,
        )

        with _patch_worktree(wt):
            runner._run_hypothesis(hyp, finding)

        # write_file called at least once with the fix snippet
        fix_calls = [c for c in wt.write_file.call_args_list if c.args[0] == "_mltk_fix.py"]
        assert len(fix_calls) == 1
        assert fix_calls[0].args[1] == "y = 42"

    def test_calls_run_in_worktree_with_python(
        self,
        tmp_path: Path,
    ) -> None:
        """_run_hypothesis runs python + assertion script."""
        wt = _mock_worktree()
        runner = SandboxedExperimentRunner(
            repo_root=tmp_path,
        )
        hyp = _make_hypothesis()
        finding = _make_finding()

        with _patch_worktree(wt):
            runner._run_hypothesis(hyp, finding)

        wt.run_in_worktree.assert_called_once()
        call_args = wt.run_in_worktree.call_args
        assert call_args.args[0] == sys.executable
        assert call_args.args[1] == "_mltk_assert.py"

    def test_returns_passed_test_result_on_success(
        self,
        tmp_path: Path,
    ) -> None:
        """Successful subprocess returns passed TestResult."""
        stdout = json.dumps(
            {
                "name": "sandbox.assertion",
                "passed": True,
                "severity": "info",
                "message": "Fix applied",
            }
        )
        wt = _mock_worktree(stdout=stdout)
        runner = SandboxedExperimentRunner(
            repo_root=tmp_path,
        )
        hyp = _make_hypothesis()
        finding = _make_finding()

        with _patch_worktree(wt):
            result = runner._run_hypothesis(hyp, finding)

        assert result.passed is True
        assert result.name == "sandbox.assertion"

    def test_returns_failed_on_worktree_creation_error(
        self,
        tmp_path: Path,
    ) -> None:
        """Worktree creation error returns failed TestResult."""
        wt = _mock_worktree()
        wt.__enter__.side_effect = RuntimeError(
            "git worktree add failed",
        )
        runner = SandboxedExperimentRunner(
            repo_root=tmp_path,
        )
        hyp = _make_hypothesis()
        finding = _make_finding()

        with _patch_worktree(wt):
            result = runner._run_hypothesis(hyp, finding)

        assert result.passed is False
        assert result.severity == Severity.CRITICAL
        assert "RuntimeError" in result.message
        assert result.details["sandboxed"] is True

    def test_returns_failed_on_subprocess_timeout(
        self,
        tmp_path: Path,
    ) -> None:
        """Subprocess timeout returns failed TestResult."""
        wt = _mock_worktree()
        wt.run_in_worktree.side_effect = subprocess.TimeoutExpired(cmd="python", timeout=5.0)
        runner = SandboxedExperimentRunner(
            repo_root=tmp_path,
            timeout=5.0,
        )
        hyp = _make_hypothesis()
        finding = _make_finding(scanner="drift")

        with _patch_worktree(wt):
            result = runner._run_hypothesis(hyp, finding)

        assert result.passed is False
        assert "timed out" in result.message.lower()
        assert result.name == "sandbox.timeout.drift"
        assert result.details["sandboxed"] is True

    def test_returns_failed_on_called_process_error(
        self,
        tmp_path: Path,
    ) -> None:
        """CalledProcessError returns failed TestResult."""
        wt = _mock_worktree()
        wt.run_in_worktree.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=["python", "_mltk_assert.py"],
            stderr="SyntaxError: invalid",
        )
        runner = SandboxedExperimentRunner(
            repo_root=tmp_path,
        )
        hyp = _make_hypothesis()
        finding = _make_finding(scanner="bias")

        with _patch_worktree(wt):
            result = runner._run_hypothesis(hyp, finding)

        assert result.passed is False
        assert "exit 1" in result.message
        assert result.name == "sandbox.process_error.bias"
        assert result.details["returncode"] == 1

    def test_returns_failed_on_invalid_json_stdout(
        self,
        tmp_path: Path,
    ) -> None:
        """Invalid JSON in stdout returns failed TestResult."""
        wt = _mock_worktree(stdout="not json at all\n")
        runner = SandboxedExperimentRunner(
            repo_root=tmp_path,
        )
        hyp = _make_hypothesis()
        finding = _make_finding()

        with _patch_worktree(wt):
            result = runner._run_hypothesis(hyp, finding)

        # The ValueError from _parse_test_result is caught by the
        # outer exception handler, producing a sandbox.error result.
        assert result.passed is False
        assert "ValueError" in result.message
        assert result.details["sandboxed"] is True

    def test_includes_duration_ms(
        self,
        tmp_path: Path,
    ) -> None:
        """Result includes non-zero duration_ms."""
        wt = _mock_worktree()
        runner = SandboxedExperimentRunner(
            repo_root=tmp_path,
        )
        hyp = _make_hypothesis()
        finding = _make_finding()

        with _patch_worktree(wt):
            result = runner._run_hypothesis(hyp, finding)

        assert result.duration_ms >= 0.0

    def test_includes_sandboxed_true_in_details(
        self,
        tmp_path: Path,
    ) -> None:
        """Successful result has sandboxed=True in details."""
        stdout = json.dumps(
            {
                "name": "sandbox.assertion",
                "passed": True,
                "severity": "info",
                "message": "ok",
            }
        )
        wt = _mock_worktree(stdout=stdout)
        runner = SandboxedExperimentRunner(
            repo_root=tmp_path,
        )
        hyp = _make_hypothesis()
        finding = _make_finding()

        with _patch_worktree(wt):
            result = runner._run_hypothesis(hyp, finding)

        assert result.details.get("sandboxed") is True


class TestApplyFix:
    """Verify _apply_fix behavior."""

    def test_writes_snippet_when_present(self) -> None:
        """_apply_fix writes code_snippet to _mltk_fix.py."""
        wt = MagicMock()
        fix = _make_fix(snippet="import os\nprint(os.getcwd())")
        hyp = _make_hypothesis(fix=fix)

        SandboxedExperimentRunner._apply_fix(wt, hyp)

        wt.write_file.assert_called_once_with(
            "_mltk_fix.py",
            "import os\nprint(os.getcwd())",
        )

    def test_does_nothing_when_snippet_empty(self) -> None:
        """_apply_fix is a no-op when code_snippet is empty."""
        wt = MagicMock()
        fix = _make_fix(snippet="")
        hyp = _make_hypothesis(fix=fix)

        SandboxedExperimentRunner._apply_fix(wt, hyp)

        wt.write_file.assert_not_called()


class TestParseTestResult:
    """Verify _parse_test_result parsing."""

    def test_parses_valid_json(self) -> None:
        """_parse_test_result parses a well-formed JSON line."""
        data = {
            "name": "sandbox.assertion",
            "passed": True,
            "severity": "info",
            "message": "All good",
            "duration_ms": 42.5,
        }
        result = _parse_test_result(json.dumps(data))

        assert result.name == "sandbox.assertion"
        assert result.passed is True
        assert result.severity == Severity.INFO
        assert result.message == "All good"
        assert result.duration_ms == 42.5
        assert result.details["sandboxed"] is True

    def test_parses_last_json_line(self) -> None:
        """_parse_test_result finds JSON even after debug output."""
        stdout = (
            "DEBUG: loading modules...\n"
            "WARNING: something\n"
            '{"name":"test","passed":false,'
            '"severity":"warning","message":"fail"}\n'
        )
        result = _parse_test_result(stdout)

        assert result.name == "test"
        assert result.passed is False
        assert result.severity == Severity.WARNING

    def test_raises_on_garbage(self) -> None:
        """_parse_test_result raises ValueError on non-JSON."""
        with pytest.raises(ValueError, match="No valid TestResult"):
            _parse_test_result("this is not json")

    def test_raises_on_empty_string(self) -> None:
        """_parse_test_result raises ValueError on empty stdout."""
        with pytest.raises(ValueError, match="No valid TestResult"):
            _parse_test_result("")

    def test_defaults_severity_to_critical(self) -> None:
        """Unknown severity string maps to CRITICAL."""
        data = {
            "name": "test",
            "passed": False,
            "severity": "unknown_level",
            "message": "bad",
        }
        result = _parse_test_result(json.dumps(data))
        assert result.severity == Severity.CRITICAL

    def test_preserves_extra_details(self) -> None:
        """Extra fields in details dict are preserved."""
        data = {
            "name": "test",
            "passed": True,
            "severity": "info",
            "message": "ok",
            "details": {"custom_key": "custom_value"},
        }
        result = _parse_test_result(json.dumps(data))
        assert result.details["custom_key"] == "custom_value"
        assert result.details["sandboxed"] is True


class TestBuildAssertionScript:
    """Verify _build_assertion_script output."""

    def test_returns_valid_python(self) -> None:
        """Script output is syntactically valid Python."""
        finding = _make_finding(scanner="leakage")
        script = _build_assertion_script(finding)
        # ast.parse will raise SyntaxError if invalid
        ast.parse(script)

    def test_includes_scanner_name(self) -> None:
        """Script references the scanner name in the message."""
        finding = _make_finding(scanner="drift")
        script = _build_assertion_script(finding)
        assert "drift" in script

    def test_produces_json_output(self) -> None:
        """Script ends with a json.dumps print statement."""
        finding = _make_finding(scanner="data")
        script = _build_assertion_script(finding)
        assert "json.dumps" in script
        assert "print(" in script


class TestIntegration:
    """Integration-level tests using run() with mocked worktree."""

    def test_run_produces_experiment_result(
        self,
        tmp_path: Path,
    ) -> None:
        """run() with SandboxedExperimentRunner yields ExperimentResult."""
        stdout = json.dumps(
            {
                "name": "sandbox.assertion",
                "passed": True,
                "severity": "info",
                "message": "Fix applied",
            }
        )
        wt = _mock_worktree(stdout=stdout)
        fix = _make_fix(title="integration-fix")
        finding = _make_finding()
        hyp = _make_hypothesis(fix=fix)

        runner = SandboxedExperimentRunner(
            repo_root=tmp_path,
        )

        with _patch_worktree(wt):
            result = runner.run(finding, hypotheses=[hyp])

        assert isinstance(result, ExperimentResult)
        assert result.hypotheses_tested == 1
        assert result.selected_fix is fix
        assert result.any_fix_works is True
        assert result.duration_ms >= 0.0

    def test_run_with_no_hypotheses(
        self,
        tmp_path: Path,
    ) -> None:
        """run() with no hypotheses produces empty result."""
        finding = _make_finding()
        runner = SandboxedExperimentRunner(
            repo_root=tmp_path,
        )

        # No worktree created when there are no hypotheses
        result = runner.run(finding)

        assert result.selected_fix is None
        assert result.hypotheses_tested == 0

    def test_run_failing_hypothesis_no_selected_fix(
        self,
        tmp_path: Path,
    ) -> None:
        """run() where hypothesis fails selects no fix."""
        stdout = json.dumps(
            {
                "name": "sandbox.assertion",
                "passed": False,
                "severity": "critical",
                "message": "Still broken",
            }
        )
        wt = _mock_worktree(stdout=stdout)
        fix = _make_fix(title="bad-fix")
        finding = _make_finding()
        hyp = _make_hypothesis(fix=fix)

        runner = SandboxedExperimentRunner(
            repo_root=tmp_path,
        )

        with _patch_worktree(wt):
            result = runner.run(finding, hypotheses=[hyp])

        assert result.selected_fix is None
        assert result.any_fix_works is False
