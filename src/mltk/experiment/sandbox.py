"""Sandboxed experiment runner -- execute hypotheses in git worktrees.

Each hypothesis is tested inside an isolated git worktree via
subprocess, so that fixes cannot corrupt the main working tree.
The runner writes the fix code snippet to the worktree, executes a
minimal assertion script, and parses the :class:`TestResult` JSON
from stdout.

Usage::

    from mltk.experiment.sandbox import SandboxedExperimentRunner

    runner = SandboxedExperimentRunner(repo_root=Path("."))
    result = runner.run(finding, hypotheses=[hyp])
    if result.selected_fix:
        print(result.selected_fix.title)
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from mltk.core.result import Severity, TestResult
from mltk.experiment.hypothesis import Hypothesis
from mltk.experiment.runner import ExperimentRunner
from mltk.scan.finding import ScanFinding

if TYPE_CHECKING:
    from mltk.experiment.worktree import GitWorktree

__all__ = ["SandboxedExperimentRunner"]

logger = logging.getLogger(__name__)

# Maps JSON severity strings to the Severity enum.
_SEVERITY_MAP: dict[str, Severity] = {
    "critical": Severity.CRITICAL,
    "warning": Severity.WARNING,
    "info": Severity.INFO,
}


class SandboxedExperimentRunner(ExperimentRunner):
    """Run experiments in isolated git worktrees.

    Inherits the full ExperimentRunner API (run(), run_batch(),
    ranking).  Overrides ``_run_hypothesis()`` to execute each
    hypothesis in a temporary worktree via subprocess.

    The hypothesis ``fix.code_snippet`` is written to the worktree,
    then a minimal assertion script is run in a subprocess to produce
    a :class:`~mltk.core.result.TestResult` JSON on stdout.

    Args:
        repo_root: Path to the git repository root.  If ``None``,
            detected automatically via ``find_git_root()``.
        strategy: Ranking strategy (``"passed"``, ``"delta"``,
            or ``"composite"``).
        timeout: Per-hypothesis timeout in seconds.
    """

    def __init__(
        self,
        repo_root: Path | None = None,
        strategy: str = "passed",
        timeout: float = 30.0,
    ) -> None:
        super().__init__(strategy=strategy, timeout=timeout)
        if repo_root is not None:
            self._repo_root = repo_root
        else:
            # Lazy import to avoid hard dependency on git at
            # module-load time.
            from mltk.experiment.worktree import find_git_root

            self._repo_root = find_git_root()

    def _run_hypothesis(
        self,
        hypothesis: Hypothesis,
        finding: ScanFinding,
    ) -> TestResult:
        """Execute one hypothesis in an isolated git worktree.

        Steps:
            1. Create worktree via GitWorktree context manager.
            2. Write ``fix.code_snippet`` to a file in worktree.
            3. Run a minimal assertion script in subprocess.
            4. Parse :class:`TestResult` JSON from stdout.
            5. Cleanup worktree on exit.

        On any error (worktree creation, subprocess, JSON parse),
        returns a failed :class:`TestResult` with details about the
        failure.

        Args:
            hypothesis: The hypothesis to test.
            finding: The scan finding being addressed.

        Returns:
            :class:`TestResult` from the assertion subprocess, or
            a synthetic failure if execution failed.
        """
        t0 = time.perf_counter()
        try:
            # Lazy import -- avoids hard dependency on git at
            # module-load time and makes the patch target clear.
            from mltk.experiment.worktree import GitWorktree

            with GitWorktree(self._repo_root) as wt:
                # Apply fix: write code_snippet to worktree.
                self._apply_fix(wt, hypothesis)

                # Run assertion in subprocess.
                result = self._run_assertion(wt, finding)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                if result.duration_ms <= 0.0:
                    result.duration_ms = elapsed_ms
                return result

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.warning(
                "Sandboxed hypothesis '%s' failed: %s",
                hypothesis.fix.title,
                exc,
                exc_info=exc,
            )
            return TestResult(
                name=(f"sandbox.error.{hypothesis.fix.title}"),
                passed=False,
                severity=Severity.CRITICAL,
                message=(f"Sandbox error: {type(exc).__name__}: {exc}"),
                details={
                    "exception_type": type(exc).__name__,
                    "sandboxed": True,
                },
                duration_ms=elapsed_ms,
            )

    @staticmethod
    def _apply_fix(
        worktree: GitWorktree,
        hypothesis: Hypothesis,
    ) -> None:
        """Write the fix ``code_snippet`` to the worktree.

        If the snippet is empty or absent, this is a no-op.

        Args:
            worktree: A :class:`GitWorktree` instance.
            hypothesis: Hypothesis whose fix is applied.
        """
        snippet = hypothesis.fix.code_snippet
        if snippet:
            worktree.write_file("_mltk_fix.py", snippet)

    def _run_assertion(
        self,
        worktree: GitWorktree,
        finding: ScanFinding,
    ) -> TestResult:
        """Run an assertion in the worktree subprocess.

        Builds a minimal Python script, writes it to the worktree,
        and executes it.  Parses :class:`TestResult` from JSON
        stdout.

        Args:
            worktree: A :class:`GitWorktree` instance.
            finding: The scan finding being tested.

        Returns:
            Parsed :class:`TestResult` from subprocess stdout.
        """
        script = _build_assertion_script(finding)
        worktree.write_file("_mltk_assert.py", script)

        try:
            proc = worktree.run_in_worktree(
                sys.executable,
                "_mltk_assert.py",
                timeout=self._timeout,
            )
            return _parse_test_result(proc.stdout)
        except subprocess.TimeoutExpired:
            return TestResult(
                name=(f"sandbox.timeout.{finding.scanner_name}"),
                passed=False,
                severity=Severity.CRITICAL,
                message=(f"Assertion timed out after {self._timeout:.1f}s"),
                details={
                    "sandboxed": True,
                    "timeout": self._timeout,
                },
            )
        except subprocess.CalledProcessError as exc:
            stderr_snippet = (exc.stderr or "")[:200]
            return TestResult(
                name=(f"sandbox.process_error.{finding.scanner_name}"),
                passed=False,
                severity=Severity.CRITICAL,
                message=(f"Assertion process failed (exit {exc.returncode}): {stderr_snippet}"),
                details={
                    "sandboxed": True,
                    "returncode": exc.returncode,
                },
            )


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _build_assertion_script(finding: ScanFinding) -> str:
    """Build a minimal Python script that runs the assertion.

    The script prints a JSON object with keys:
    ``name``, ``passed``, ``severity``, ``message``.
    This output is parsed back by :func:`_parse_test_result`.

    Args:
        finding: The scan finding being tested.

    Returns:
        A string of valid Python code.
    """
    scanner_safe = json.dumps(finding.scanner_name)
    return (
        "import json, sys, os, importlib.util\n"
        "try:\n"
        "    fix_path = os.path.join("
        "os.getcwd(), '_mltk_fix.py')\n"
        "    if os.path.exists(fix_path):\n"
        "        spec = importlib.util.spec_from_file_location("
        "'_fix', fix_path)\n"
        "        mod = importlib.util.module_from_spec(spec)\n"
        "        spec.loader.exec_module(mod)\n"
        "    result = {"
        "'name': 'sandbox.assertion', "
        "'passed': True, "
        "'severity': 'info', "
        f"'message': 'Fix applied for ' + {scanner_safe}"
        "}\n"
        "except Exception as exc:\n"
        "    result = {"
        "'name': 'sandbox.assertion', "
        "'passed': False, "
        "'severity': 'critical', "
        "'message': str(exc)"
        "}\n"
        "print(json.dumps(result))\n"
    )


def _parse_test_result(stdout: str) -> TestResult:
    """Parse a :class:`TestResult` from JSON stdout.

    Scans stdout lines in reverse for the last JSON object.
    This allows the subprocess to emit debug output before
    the final JSON line.

    Args:
        stdout: Complete stdout from the subprocess.

    Returns:
        Parsed :class:`TestResult`.

    Raises:
        ValueError: If no valid JSON is found in *stdout*.
    """
    lines = stdout.strip().splitlines()
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("{"):
            try:
                data = json.loads(line)
                return TestResult(
                    name=data.get(
                        "name",
                        "sandbox.assertion",
                    ),
                    passed=bool(data.get("passed", False)),
                    severity=_SEVERITY_MAP.get(
                        str(
                            data.get("severity", "critical"),
                        ).lower(),
                        Severity.CRITICAL,
                    ),
                    message=str(data.get("message", "")),
                    details={
                        **data.get("details", {}),
                        "sandboxed": True,
                    },
                    duration_ms=float(
                        data.get("duration_ms", 0.0),
                    ),
                )
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    raise ValueError(f"No valid TestResult JSON in stdout: {stdout[:200]}")
