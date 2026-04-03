"""Integration tests for sandboxed experiment execution pipeline.

Verifies end-to-end wiring between ``SandboxedExperimentRunner``,
``GitWorktree``, ``ExperimentResult``, and the package ``__init__.py``
exports.  Mocks ``subprocess.run`` at the lowest level so no real git
repo is needed, but lets all class code run for real.

Tests:
    1.  ``from mltk.experiment import SandboxedExperimentRunner`` works.
    2.  ``from mltk.experiment import GitWorktree`` works.
    3.  ``SandboxedExperimentRunner`` in ``__all__``.
    4.  ``GitWorktree`` in ``__all__``.
    5.  SandboxedExperimentRunner is subclass of ExperimentRunner.
    6.  Full pipeline: run() -> GitWorktree -> subprocess -> result.
    7.  Result has selected_fix when hypothesis passes.
    8.  Result has no selected_fix when hypothesis fails.
    9.  Multiple hypotheses each get their own worktree branch.
    10. Worktree cleanup happens even when hypothesis raises.
    11. run_batch() processes multiple findings.
    12. Timeout in subprocess produces failed TestResult.
    13. ExperimentResult.any_fix_works property.
    14. ExperimentResult.best_result property.
    15. ExperimentResult.hypotheses_tested property.
    16. Non-sandboxed ExperimentRunner API unchanged (regression).
    17. GitWorktree context manager lifecycle (enter -> use -> exit).
    18. Worktree cleanup on failed hypothesis does not mask error.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.core.result import Severity, TestResult
from mltk.experiment import (
    ExperimentResult,
    ExperimentRunner,
    GitWorktree,
    Hypothesis,
    HypothesisResult,
    SandboxedExperimentRunner,
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
        description="A test fix",
        confidence="high",
        code_snippet=snippet,
    )


def _make_finding(
    scanner_name: str = "data",
    fixes: list[FixSuggestion] | None = None,
) -> ScanFinding:
    """Create a ScanFinding whose assertion always fails."""

    def _assertion_fn() -> TestResult:
        result = TestResult(
            name="baseline_check",
            passed=False,
            severity=Severity.HIGH if hasattr(Severity, "HIGH") else Severity.CRITICAL,
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
        assertion_fn=_assertion_fn,
        scanner_name=scanner_name,
        suggested_fixes=fixes or [_make_fix()],
    )


def _ok(
    stdout: str = "",
    args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Build a successful CompletedProcess."""
    return subprocess.CompletedProcess(
        args=args or ["git"],
        returncode=0,
        stdout=stdout,
        stderr="",
    )


def _passed_json() -> str:
    """JSON stdout that represents a passing TestResult."""
    return json.dumps({
        "name": "sandbox.assertion",
        "passed": True,
        "severity": "info",
        "message": "Fix applied",
    })


def _failed_json() -> str:
    """JSON stdout that represents a failing TestResult."""
    return json.dumps({
        "name": "sandbox.assertion",
        "passed": False,
        "severity": "critical",
        "message": "Still broken",
    })


def _mock_subprocess_factory(
    assertion_stdout: str | None = None,
) -> object:
    """Return a side_effect function for subprocess.run mocking.

    Handles git commands (rev-parse, worktree add/remove, branch -D)
    and python execution (assertion script) in a single mock.
    """
    stdout = assertion_stdout or _passed_json()

    def _side_effect(
        args: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        cmd = args if isinstance(args, list) else [str(args)]

        # --- git commands ---
        if cmd[0] == "git":
            if "rev-parse" in cmd:
                return _ok(stdout="/fake/repo\n")
            if "worktree" in cmd and "add" in cmd:
                return _ok()
            if "worktree" in cmd and "remove" in cmd:
                return _ok()
            if "branch" in cmd and "-D" in cmd:
                return _ok()
            return _ok()

        # --- python execution (assertion script) ---
        if cmd[0] == sys.executable or "python" in str(cmd[0]):
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout=stdout + "\n",
                stderr="",
            )

        return _ok()

    return _side_effect


def _mock_worktree(
    stdout: str | None = None,
    tmp_path: Path | None = None,
) -> MagicMock:
    """Return a mock GitWorktree context manager."""
    wt = MagicMock()
    wt.path = tmp_path or Path("/tmp/mock-worktree")
    wt.branch = "mltk-sandbox-test"
    wt.run_in_worktree.return_value = subprocess.CompletedProcess(
        args=[sys.executable, "_mltk_assert.py"],
        returncode=0,
        stdout=stdout or _passed_json(),
        stderr="",
    )
    wt.write_file = MagicMock()
    wt.__enter__ = MagicMock(return_value=wt)
    wt.__exit__ = MagicMock(return_value=False)
    return wt


# -------------------------------------------------------------------
# Package export tests
# -------------------------------------------------------------------


class TestPackageExports:
    """Verify __init__.py wiring for new public names."""

    def test_import_sandboxed_runner(self) -> None:
        """SandboxedExperimentRunner is importable from package.

        WHY: Users must be able to import the sandboxed runner
        from the top-level experiment package without knowing
        the internal module path.
        """
        from mltk.experiment import SandboxedExperimentRunner as cls

        assert cls is not None
        assert cls.__name__ == "SandboxedExperimentRunner"

    def test_import_git_worktree(self) -> None:
        """GitWorktree is importable from the experiment package.

        WHY: Advanced users may construct worktrees directly
        for custom sandbox workflows.
        """
        from mltk.experiment import GitWorktree as cls

        assert cls is not None
        assert cls.__name__ == "GitWorktree"

    def test_sandboxed_runner_in_all(self) -> None:
        """SandboxedExperimentRunner is listed in __all__.

        WHY: ``from mltk.experiment import *`` must include
        the sandboxed runner for notebook convenience.
        """
        import mltk.experiment as mod

        assert "SandboxedExperimentRunner" in mod.__all__

    def test_git_worktree_in_all(self) -> None:
        """GitWorktree is listed in __all__.

        WHY: Ensures wildcard import exposes the worktree class.
        """
        import mltk.experiment as mod

        assert "GitWorktree" in mod.__all__

    def test_all_is_sorted_alphabetically(self) -> None:
        """__all__ entries are in alphabetical order.

        WHY: Convention in the codebase -- keeps exports
        predictable and diff-friendly.
        """
        import mltk.experiment as mod

        assert mod.__all__ == sorted(mod.__all__)

    def test_all_has_seven_entries(self) -> None:
        """__all__ contains exactly 7 public names.

        WHY: Guards against accidental additions or removals.
        """
        import mltk.experiment as mod

        assert len(mod.__all__) == 7


class TestSubclassRelationship:
    """Verify SandboxedExperimentRunner inherits ExperimentRunner."""

    def test_is_subclass(self) -> None:
        """SandboxedExperimentRunner is a subclass of ExperimentRunner.

        WHY: The sandboxed runner must honour the same public API
        (run, run_batch) so it can be used as a drop-in replacement.
        """
        assert issubclass(
            SandboxedExperimentRunner, ExperimentRunner,
        )

    def test_instance_is_experiment_runner(
        self,
        tmp_path: Path,
    ) -> None:
        """An instance passes isinstance check for ExperimentRunner.

        WHY: Polymorphic code that type-checks the runner must
        accept the sandboxed variant without special-casing.
        """
        runner = SandboxedExperimentRunner(repo_root=tmp_path)
        assert isinstance(runner, ExperimentRunner)


# -------------------------------------------------------------------
# Full pipeline integration tests
# -------------------------------------------------------------------


class TestFullPipeline:
    """End-to-end tests: runner.run() through to ExperimentResult."""

    def test_run_produces_experiment_result(
        self,
        tmp_path: Path,
    ) -> None:
        """Full pipeline produces a valid ExperimentResult.

        WHY: This is the core integration test -- every layer
        (runner -> worktree -> subprocess -> parse -> rank) must
        cooperate without errors.
        """
        wt = _mock_worktree(stdout=_passed_json())
        fix = _make_fix(title="pipeline-fix")
        finding = _make_finding(fixes=[fix])
        hyp = Hypothesis(
            fix=fix,
            apply_fn=lambda: None,
            description="Pipeline test",
        )
        runner = SandboxedExperimentRunner(
            repo_root=tmp_path,
        )

        with patch(
            "mltk.experiment.worktree.GitWorktree",
            return_value=wt,
        ):
            result = runner.run(finding, hypotheses=[hyp])

        assert isinstance(result, ExperimentResult)
        assert result.hypotheses_tested == 1
        assert result.duration_ms >= 0.0

    def test_selected_fix_when_hypothesis_passes(
        self,
        tmp_path: Path,
    ) -> None:
        """selected_fix is set when hypothesis turns failure to pass.

        WHY: The primary value proposition -- a passing fix must
        be surfaced to the caller.
        """
        wt = _mock_worktree(stdout=_passed_json())
        fix = _make_fix(title="good-fix")
        finding = _make_finding(fixes=[fix])
        hyp = Hypothesis(
            fix=fix,
            apply_fn=lambda: None,
            description="Good fix",
        )
        runner = SandboxedExperimentRunner(
            repo_root=tmp_path,
        )

        with patch(
            "mltk.experiment.worktree.GitWorktree",
            return_value=wt,
        ):
            result = runner.run(finding, hypotheses=[hyp])

        assert result.selected_fix is fix
        assert result.any_fix_works is True

    def test_no_selected_fix_when_hypothesis_fails(
        self,
        tmp_path: Path,
    ) -> None:
        """selected_fix is None when hypothesis still fails.

        WHY: Honest reporting -- a fix that does not resolve the
        issue must not be recommended.
        """
        wt = _mock_worktree(stdout=_failed_json())
        fix = _make_fix(title="bad-fix")
        finding = _make_finding(fixes=[fix])
        hyp = Hypothesis(
            fix=fix,
            apply_fn=lambda: None,
            description="Bad fix",
        )
        runner = SandboxedExperimentRunner(
            repo_root=tmp_path,
        )

        with patch(
            "mltk.experiment.worktree.GitWorktree",
            return_value=wt,
        ):
            result = runner.run(finding, hypotheses=[hyp])

        assert result.selected_fix is None
        assert result.any_fix_works is False

    def test_multiple_hypotheses_get_own_worktree(
        self,
        tmp_path: Path,
    ) -> None:
        """Each hypothesis triggers its own GitWorktree creation.

        WHY: Isolation -- hypotheses must not share state.  Each
        must get a fresh worktree so side effects do not leak.
        """
        worktree_cls = MagicMock()
        wt_a = _mock_worktree(stdout=_passed_json())
        wt_b = _mock_worktree(stdout=_failed_json())
        worktree_cls.side_effect = [wt_a, wt_b]

        fix_a = _make_fix(title="fix-a", snippet="a = 1")
        fix_b = _make_fix(title="fix-b", snippet="b = 2")
        finding = _make_finding(fixes=[fix_a, fix_b])
        hyp_a = Hypothesis(
            fix=fix_a, apply_fn=lambda: None,
            description="Hypothesis A",
        )
        hyp_b = Hypothesis(
            fix=fix_b, apply_fn=lambda: None,
            description="Hypothesis B",
        )
        runner = SandboxedExperimentRunner(
            repo_root=tmp_path,
        )

        with patch(
            "mltk.experiment.worktree.GitWorktree",
            worktree_cls,
        ):
            result = runner.run(
                finding, hypotheses=[hyp_a, hyp_b],
            )

        assert worktree_cls.call_count == 2
        assert result.hypotheses_tested == 2

    def test_worktree_cleanup_on_hypothesis_exception(
        self,
        tmp_path: Path,
    ) -> None:
        """Worktree __exit__ is called even when hypothesis raises.

        WHY: Resource safety -- leaked worktrees waste disk and
        can leave stale git branches.
        """
        wt = _mock_worktree()
        wt.run_in_worktree.side_effect = RuntimeError(
            "subprocess crashed",
        )
        runner = SandboxedExperimentRunner(
            repo_root=tmp_path,
        )
        fix = _make_fix(title="crashy-fix")
        finding = _make_finding(fixes=[fix])
        hyp = Hypothesis(
            fix=fix, apply_fn=lambda: None,
            description="Crashy",
        )

        with patch(
            "mltk.experiment.worktree.GitWorktree",
            return_value=wt,
        ):
            result = runner.run(finding, hypotheses=[hyp])

        # __exit__ must have been called for cleanup
        wt.__exit__.assert_called_once()
        # Result should be a failure, not a crash
        assert result.hypotheses_tested == 1
        assert result.selected_fix is None
        hr = result.hypothesis_results[0]
        assert hr.fixed_result.passed is False

    def test_cleanup_does_not_mask_original_error(
        self,
        tmp_path: Path,
    ) -> None:
        """Worktree cleanup failure does not mask hypothesis error.

        WHY: If both the hypothesis AND the cleanup fail, the
        hypothesis error is the one the user cares about.
        Cleanup errors must be swallowed.
        """
        wt = _mock_worktree()
        wt.run_in_worktree.side_effect = ValueError(
            "assertion parse failed",
        )
        # Exit also fails
        wt.__exit__.side_effect = OSError("cleanup failed")

        runner = SandboxedExperimentRunner(
            repo_root=tmp_path,
        )
        fix = _make_fix(title="double-fail-fix")
        finding = _make_finding(fixes=[fix])
        hyp = Hypothesis(
            fix=fix, apply_fn=lambda: None,
            description="Double fail",
        )

        with patch(
            "mltk.experiment.worktree.GitWorktree",
            return_value=wt,
        ):
            # Should not raise -- errors are captured
            result = runner.run(finding, hypotheses=[hyp])

        assert result.hypotheses_tested == 1
        hr = result.hypothesis_results[0]
        assert hr.fixed_result.passed is False


# -------------------------------------------------------------------
# run_batch integration
# -------------------------------------------------------------------


class TestRunBatch:
    """Verify run_batch processes multiple findings end-to-end."""

    def test_batch_processes_multiple_findings(
        self,
        tmp_path: Path,
    ) -> None:
        """run_batch returns one ExperimentResult per finding.

        WHY: Real scans produce many findings.  The batch path
        must iterate correctly and wire each finding to its
        runner.run() call.
        """
        wt_pass = _mock_worktree(stdout=_passed_json())
        wt_fail = _mock_worktree(stdout=_failed_json())
        worktree_cls = MagicMock()
        worktree_cls.side_effect = [wt_pass, wt_fail]

        fix_a = _make_fix(title="fix-a")
        fix_b = _make_fix(title="fix-b")
        finding_a = _make_finding(
            scanner_name="scanner_a", fixes=[fix_a],
        )
        finding_b = _make_finding(
            scanner_name="scanner_b", fixes=[fix_b],
        )

        def _pass_fn() -> TestResult:
            return TestResult(
                name="fixed",
                passed=True,
                severity=Severity.INFO,
                message="Fixed",
            )

        def _fail_fn() -> TestResult:
            return TestResult(
                name="still_bad",
                passed=False,
                severity=Severity.WARNING,
                message="Not fixed",
            )

        runner = SandboxedExperimentRunner(
            repo_root=tmp_path,
        )

        with patch(
            "mltk.experiment.worktree.GitWorktree",
            worktree_cls,
        ):
            results = runner.run_batch(
                [finding_a, finding_b],
                apply_fns_map={
                    "scanner_a": {
                        "fix-a": _pass_fn,
                    },
                    "scanner_b": {
                        "fix-b": _fail_fn,
                    },
                },
            )

        assert len(results) == 2
        assert all(isinstance(r, ExperimentResult) for r in results)

    def test_batch_empty_findings_returns_empty(
        self,
        tmp_path: Path,
    ) -> None:
        """run_batch with no findings returns empty list.

        WHY: Edge case -- empty input must not crash or produce
        phantom results.
        """
        runner = SandboxedExperimentRunner(
            repo_root=tmp_path,
        )
        results = runner.run_batch([])
        assert results == []


# -------------------------------------------------------------------
# Subprocess timeout integration
# -------------------------------------------------------------------


class TestSubprocessTimeout:
    """Verify timeout handling through the full stack."""

    def test_timeout_produces_failed_result(
        self,
        tmp_path: Path,
    ) -> None:
        """Subprocess timeout yields a failed TestResult.

        WHY: Hypotheses that hang (infinite loops, stuck I/O)
        must be safely aborted and reported as failures.
        """
        wt = _mock_worktree()
        wt.run_in_worktree.side_effect = (
            subprocess.TimeoutExpired(
                cmd="python", timeout=5.0,
            )
        )
        runner = SandboxedExperimentRunner(
            repo_root=tmp_path, timeout=5.0,
        )
        fix = _make_fix(title="slow-fix")
        finding = _make_finding(
            scanner_name="drift", fixes=[fix],
        )
        hyp = Hypothesis(
            fix=fix, apply_fn=lambda: None,
            description="Slow hypothesis",
        )

        with patch(
            "mltk.experiment.worktree.GitWorktree",
            return_value=wt,
        ):
            result = runner.run(finding, hypotheses=[hyp])

        assert result.selected_fix is None
        hr = result.hypothesis_results[0]
        assert hr.fixed_result.passed is False
        assert "timed out" in hr.fixed_result.message.lower()
        assert hr.fixed_result.details.get("sandboxed") is True

    def test_called_process_error_produces_failed_result(
        self,
        tmp_path: Path,
    ) -> None:
        """Non-zero exit code yields a failed TestResult.

        WHY: Assertion scripts may crash with syntax errors or
        import failures.  The error must be captured, not raised.
        """
        wt = _mock_worktree()
        wt.run_in_worktree.side_effect = (
            subprocess.CalledProcessError(
                returncode=1,
                cmd=["python", "_mltk_assert.py"],
                stderr="ImportError: no module",
            )
        )
        runner = SandboxedExperimentRunner(
            repo_root=tmp_path,
        )
        fix = _make_fix(title="crash-fix")
        finding = _make_finding(
            scanner_name="bias", fixes=[fix],
        )
        hyp = Hypothesis(
            fix=fix, apply_fn=lambda: None,
            description="Crash hypothesis",
        )

        with patch(
            "mltk.experiment.worktree.GitWorktree",
            return_value=wt,
        ):
            result = runner.run(finding, hypotheses=[hyp])

        assert result.selected_fix is None
        hr = result.hypothesis_results[0]
        assert hr.fixed_result.passed is False
        assert "exit 1" in hr.fixed_result.message


# -------------------------------------------------------------------
# ExperimentResult properties
# -------------------------------------------------------------------


class TestExperimentResultProperties:
    """Verify convenience properties on ExperimentResult."""

    def test_any_fix_works_true(
        self,
        tmp_path: Path,
    ) -> None:
        """any_fix_works is True when at least one hypothesis passes.

        WHY: Quick boolean check for callers who only need to know
        if any remediation is available.
        """
        wt = _mock_worktree(stdout=_passed_json())
        fix = _make_fix(title="good-fix")
        finding = _make_finding(fixes=[fix])
        hyp = Hypothesis(
            fix=fix, apply_fn=lambda: None,
            description="Good",
        )
        runner = SandboxedExperimentRunner(
            repo_root=tmp_path,
        )

        with patch(
            "mltk.experiment.worktree.GitWorktree",
            return_value=wt,
        ):
            result = runner.run(finding, hypotheses=[hyp])

        assert result.any_fix_works is True

    def test_any_fix_works_false(
        self,
        tmp_path: Path,
    ) -> None:
        """any_fix_works is False when no hypothesis passes.

        WHY: Must accurately reflect that no remediation worked.
        """
        wt = _mock_worktree(stdout=_failed_json())
        fix = _make_fix(title="bad-fix")
        finding = _make_finding(fixes=[fix])
        hyp = Hypothesis(
            fix=fix, apply_fn=lambda: None,
            description="Bad",
        )
        runner = SandboxedExperimentRunner(
            repo_root=tmp_path,
        )

        with patch(
            "mltk.experiment.worktree.GitWorktree",
            return_value=wt,
        ):
            result = runner.run(finding, hypotheses=[hyp])

        assert result.any_fix_works is False

    def test_best_result_returns_rank_one(
        self,
        tmp_path: Path,
    ) -> None:
        """best_result returns the rank-1 hypothesis result.

        WHY: Callers need a single best answer without iterating
        through all hypothesis results manually.
        """
        wt = _mock_worktree(stdout=_passed_json())
        fix = _make_fix(title="best-fix")
        finding = _make_finding(fixes=[fix])
        hyp = Hypothesis(
            fix=fix, apply_fn=lambda: None,
            description="Best",
        )
        runner = SandboxedExperimentRunner(
            repo_root=tmp_path,
        )

        with patch(
            "mltk.experiment.worktree.GitWorktree",
            return_value=wt,
        ):
            result = runner.run(finding, hypotheses=[hyp])

        best = result.best_result
        assert best is not None
        assert isinstance(best, HypothesisResult)

    def test_best_result_none_when_empty(
        self,
        tmp_path: Path,
    ) -> None:
        """best_result is None when no hypotheses are tested.

        WHY: Edge case -- a finding with no fixes should return
        None, not raise.
        """
        finding = _make_finding()
        runner = SandboxedExperimentRunner(
            repo_root=tmp_path,
        )

        result = runner.run(finding, hypotheses=[])

        assert result.best_result is None

    def test_hypotheses_tested_count(
        self,
        tmp_path: Path,
    ) -> None:
        """hypotheses_tested returns the correct count.

        WHY: Callers need to know how thorough the experiment was.
        """
        wt_a = _mock_worktree(stdout=_passed_json())
        wt_b = _mock_worktree(stdout=_failed_json())
        worktree_cls = MagicMock()
        worktree_cls.side_effect = [wt_a, wt_b]

        fix_a = _make_fix(title="fix-a")
        fix_b = _make_fix(title="fix-b")
        finding = _make_finding(fixes=[fix_a, fix_b])
        hyps = [
            Hypothesis(
                fix=fix_a, apply_fn=lambda: None,
                description="A",
            ),
            Hypothesis(
                fix=fix_b, apply_fn=lambda: None,
                description="B",
            ),
        ]
        runner = SandboxedExperimentRunner(
            repo_root=tmp_path,
        )

        with patch(
            "mltk.experiment.worktree.GitWorktree",
            worktree_cls,
        ):
            result = runner.run(finding, hypotheses=hyps)

        assert result.hypotheses_tested == 2


# -------------------------------------------------------------------
# Non-sandboxed ExperimentRunner regression
# -------------------------------------------------------------------


class TestBaseRunnerRegression:
    """Verify the base ExperimentRunner still works unchanged."""

    def test_base_runner_run_still_works(self) -> None:
        """ExperimentRunner.run() operates identically to before.

        WHY: Adding the sandboxed subclass must not break the
        original runner.  This is a regression guard.
        """
        fix = _make_fix(title="base-fix", snippet="")
        finding = _make_finding(fixes=[fix])

        def _passing_apply() -> TestResult:
            return TestResult(
                name="fixed",
                passed=True,
                severity=Severity.INFO,
                message="Fixed",
            )

        hyp = Hypothesis(
            fix=fix,
            apply_fn=_passing_apply,
            description="Base test",
        )
        runner = ExperimentRunner(strategy="passed")
        result = runner.run(finding, hypotheses=[hyp])

        assert isinstance(result, ExperimentResult)
        assert result.selected_fix is fix
        assert result.any_fix_works is True

    def test_base_runner_run_batch_still_works(self) -> None:
        """ExperimentRunner.run_batch() still works.

        WHY: Batch processing is critical for scan workflows.
        """
        runner = ExperimentRunner(strategy="passed")
        results = runner.run_batch([])
        assert results == []

    def test_base_runner_accepts_no_hypotheses(self) -> None:
        """ExperimentRunner with no hypotheses returns empty result.

        WHY: Findings with no fixes should produce valid results.
        """
        finding = _make_finding()
        runner = ExperimentRunner(strategy="passed")
        result = runner.run(finding, hypotheses=[])

        assert result.selected_fix is None
        assert result.hypotheses_tested == 0
        assert result.hypothesis_results == []


# -------------------------------------------------------------------
# GitWorktree lifecycle integration
# -------------------------------------------------------------------


class TestWorktreeLifecycle:
    """Verify GitWorktree context manager lifecycle end-to-end."""

    @patch("mltk.experiment.worktree.shutil.rmtree")
    @patch("mltk.experiment.worktree.tempfile.mkdtemp")
    @patch("mltk.experiment.worktree.subprocess.run")
    def test_enter_use_exit_lifecycle(
        self,
        mock_run: MagicMock,
        mock_mkdtemp: MagicMock,
        mock_rmtree: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Full lifecycle: enter -> write_file -> run -> exit.

        WHY: The worktree must be usable throughout the with block
        and cleaned up afterward.  This tests the real class code
        against mocked subprocess.
        """
        wt_dir = tmp_path / "worktree"
        wt_dir.mkdir()
        mock_mkdtemp.return_value = str(wt_dir)
        mock_run.return_value = _ok()

        wt = GitWorktree(
            repo_root=tmp_path,
            branch_name="lifecycle-test",
            base_ref="HEAD",
        )

        with wt:
            assert wt.path == wt_dir
            assert wt.branch == "lifecycle-test"

            # Write a file into the worktree
            written = wt.write_file("test.py", "print('hello')")
            assert written.exists()
            assert written.read_text(encoding="utf-8") == (
                "print('hello')"
            )

        # After exit, worktree path is cleared
        assert wt._worktree_path is None

    @patch("mltk.experiment.worktree.shutil.rmtree")
    @patch("mltk.experiment.worktree.tempfile.mkdtemp")
    @patch("mltk.experiment.worktree.subprocess.run")
    def test_exit_called_on_exception(
        self,
        mock_run: MagicMock,
        mock_mkdtemp: MagicMock,
        mock_rmtree: MagicMock,
        tmp_path: Path,
    ) -> None:
        """__exit__ runs even when the with block raises.

        WHY: Exception safety -- resources must be cleaned up
        regardless of what happens inside the with block.
        """
        wt_dir = tmp_path / "worktree"
        wt_dir.mkdir()
        mock_mkdtemp.return_value = str(wt_dir)
        mock_run.return_value = _ok()

        wt = GitWorktree(
            repo_root=tmp_path,
            branch_name="exception-test",
        )

        with pytest.raises(ValueError, match="intentional"):
            with wt:
                raise ValueError("intentional error")

        # Cleanup git commands were attempted
        # Enter = 1 call (worktree add)
        # Exit  = 2 calls (worktree remove, branch -D)
        assert mock_run.call_count == 3
        assert wt._worktree_path is None

    def test_path_raises_before_enter(self) -> None:
        """Accessing .path before __enter__ raises RuntimeError.

        WHY: The worktree directory does not exist until
        __enter__ creates it.  Early access is a bug.
        """
        wt = GitWorktree(repo_root=Path("/repo"))
        with pytest.raises(RuntimeError, match="not been entered"):
            _ = wt.path

    def test_auto_branch_name_format(self) -> None:
        """Auto-generated branch names follow mltk-sandbox-<hex>.

        WHY: Branch naming must be predictable for cleanup
        scripts and log parsing.
        """
        wt = GitWorktree(repo_root=Path("/repo"))
        assert wt.branch.startswith("mltk-sandbox-")
        assert len(wt.branch) == len("mltk-sandbox-") + 12
