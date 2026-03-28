"""Composable test suite for running mltk assertions without pytest.

This module provides :class:`MltkSuite`, a programmatic runner that wraps
mltk assertion functions so they **never raise** -- failures are captured as
:class:`~mltk.core.result.TestResult` objects.  This enables mltk usage in
notebooks, scripts, CI pipelines, and monitoring jobs that do not use pytest.

Typical usage::

    from mltk.core.suite import MltkSuite
    from mltk.data.drift import assert_no_drift
    from mltk.model.metrics import assert_metric

    suite = MltkSuite("nightly-validation")
    suite.add(assert_no_drift, train_col, prod_col)
    suite.add(assert_metric, y_true, y_pred, metric="f1", threshold=0.85)

    result = suite.run()
    print(result.passed, result.pass_rate)
    suite.to_json("results.json")
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from mltk.core.assertion import MltkAssertionError
from mltk.core.result import Severity, TestResult

# ---------------------------------------------------------------------------
# SuiteResult -- aggregated outcome of a suite run
# ---------------------------------------------------------------------------

@dataclass
class SuiteResult:
    """Aggregated outcome of running an :class:`MltkSuite`.

    Holds every :class:`~mltk.core.result.TestResult` produced by the run,
    plus convenience counters and timing information.

    Attributes:
        name: Suite name (inherited from the parent MltkSuite).
        results: Ordered list of TestResult objects.
        total: Number of assertions that were executed.
        passed_count: Number of assertions that passed.
        failed_count: Number of assertions that failed.
        duration_ms: Wall-clock time for the entire run in milliseconds.
    """

    name: str
    results: list[TestResult] = field(default_factory=list)
    total: int = 0
    passed_count: int = 0
    failed_count: int = 0
    duration_ms: float = 0.0

    @property
    def passed(self) -> bool:
        """True when every assertion in the suite passed."""
        return self.failed_count == 0

    @property
    def pass_rate(self) -> float:
        """Fraction of assertions that passed (0.0 -- 1.0).

        Returns 0.0 for an empty suite rather than raising
        ``ZeroDivisionError``.
        """
        if self.total == 0:
            return 0.0
        return self.passed_count / self.total


# ---------------------------------------------------------------------------
# _PendingAssertion -- internal bookkeeping for deferred calls
# ---------------------------------------------------------------------------

@dataclass
class _PendingAssertion:
    """Stores a deferred assertion call (function + arguments).

    This is an internal type -- callers interact with :class:`MltkSuite`
    and never see ``_PendingAssertion`` directly.
    """

    fn: Callable[..., TestResult]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


# ---------------------------------------------------------------------------
# MltkSuite -- the public API
# ---------------------------------------------------------------------------

class MltkSuite:
    """Composable test suite for running mltk assertions without pytest.

    **Why this exists:**

    mltk assertion functions are designed for pytest -- they raise
    :class:`~mltk.core.assertion.MltkAssertionError` on failure, which
    pytest catches and reports.  But not everyone uses pytest:

    * Data scientists run assertions in Jupyter notebooks.
    * CI scripts need programmatic pass/fail without pytest overhead.
    * Monitoring jobs validate models on a schedule.
    * Integration tests in non-Python frameworks need results as data.

    ``MltkSuite`` wraps assertion functions so they **never raise**.
    Failures are captured as :class:`~mltk.core.result.TestResult`
    objects.  You run all assertions, then inspect results
    programmatically or export to JSON / HTML / JUnit XML.

    Usage::

        suite = MltkSuite("My Model Tests")
        suite.add(assert_no_drift, train_col, prod_col)
        suite.add(assert_metric, y_true, y_pred, metric="f1",
                  threshold=0.85)

        results = suite.run()
        print(f"{results.passed_count}/{results.total} passed")

        suite.to_json("results.json")
        suite.to_html("report.html")
        suite.to_junit("results.xml")

    Method chaining is supported::

        results = (
            MltkSuite("quick")
            .add(assert_true, True, name="a", message="ok")
            .add(assert_true, True, name="b", message="ok")
            .run()
        )

    Args:
        name: Human-readable suite name.  Appears in exported reports
            and in the :attr:`SuiteResult.name` field.
    """

    def __init__(self, name: str = "mltk") -> None:
        self._name = name
        self._pending: list[_PendingAssertion] = []
        self._results: list[TestResult] = []
        self._suite_result: SuiteResult | None = None

    # -- building the suite ------------------------------------------------

    def add(
        self,
        assertion_fn: Callable[..., TestResult],
        *args: Any,
        **kwargs: Any,
    ) -> MltkSuite:
        """Register an assertion to be executed when :meth:`run` is called.

        The assertion is **not** executed immediately -- it is stored as
        a deferred call.  This lets you build up a full suite before
        executing anything.

        Args:
            assertion_fn: Any mltk assertion function that returns a
                :class:`~mltk.core.result.TestResult` (and may raise
                :class:`~mltk.core.assertion.MltkAssertionError`).
            *args: Positional arguments forwarded to *assertion_fn*.
            **kwargs: Keyword arguments forwarded to *assertion_fn*.

        Returns:
            ``self`` so calls can be chained:
            ``suite.add(fn1, ...).add(fn2, ...)``.
        """
        self._pending.append(
            _PendingAssertion(fn=assertion_fn, args=args, kwargs=kwargs)
        )
        return self

    # -- execution ---------------------------------------------------------

    def run(self) -> SuiteResult:
        """Execute every registered assertion and collect results.

        Each assertion is called in registration order.  If the function
        raises :class:`~mltk.core.assertion.MltkAssertionError`, the
        embedded :attr:`~MltkAssertionError.result` is captured as a
        failed :class:`~mltk.core.result.TestResult`.  Any other
        exception is converted into a ``CRITICAL`` failure result so
        the suite **never raises**.

        Calling ``run()`` multiple times re-executes all assertions and
        produces a fresh :class:`SuiteResult` each time -- previous
        results are replaced.

        Returns:
            :class:`SuiteResult` with aggregated counts, timing, and
            the full list of :class:`~mltk.core.result.TestResult`
            objects.
        """
        collected: list[TestResult] = []
        start = time.perf_counter()

        for pending in self._pending:
            result = self._execute_one(pending)
            collected.append(result)

        elapsed_ms = (time.perf_counter() - start) * 1000

        passed_count = sum(1 for r in collected if r.passed)
        failed_count = len(collected) - passed_count

        suite_result = SuiteResult(
            name=self._name,
            results=list(collected),
            total=len(collected),
            passed_count=passed_count,
            failed_count=failed_count,
            duration_ms=elapsed_ms,
        )

        self._results = list(collected)
        self._suite_result = suite_result
        return suite_result

    @staticmethod
    def _execute_one(pending: _PendingAssertion) -> TestResult:
        """Run a single deferred assertion, catching all exceptions.

        Returns a :class:`~mltk.core.result.TestResult` in every case:

        * **Normal return** -- the TestResult from the assertion.
        * **MltkAssertionError** -- the embedded ``result`` attribute.
        * **Any other exception** -- a synthetic CRITICAL-failure
          TestResult describing the unexpected error.
        """
        t0 = time.perf_counter()
        try:
            result = pending.fn(*pending.args, **pending.kwargs)
            elapsed = (time.perf_counter() - t0) * 1000
            if result.duration_ms == 0.0:
                result.duration_ms = elapsed
            return result
        except MltkAssertionError as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            result = exc.result
            if result.duration_ms == 0.0:
                result.duration_ms = elapsed
            return result
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            fn_name = getattr(
                pending.fn, "__name__", str(pending.fn)
            )
            return TestResult(
                name=f"suite.error.{fn_name}",
                passed=False,
                severity=Severity.CRITICAL,
                message=f"Unexpected error: {exc!r}",
                details={"exception_type": type(exc).__name__},
                duration_ms=elapsed,
            )

    # -- export ------------------------------------------------------------

    def _results_as_dicts(self) -> list[dict[str, Any]]:
        """Serialize collected results to plain dicts."""
        records: list[dict[str, Any]] = []
        for r in self._results:
            records.append({
                "name": r.name,
                "passed": r.passed,
                "severity": r.severity.value,
                "message": r.message,
                "details": r.details,
                "duration_ms": r.duration_ms,
                "timestamp": r.timestamp.isoformat(),
            })
        return records

    def to_json(self, path: str) -> str:
        """Export results as a JSON file.

        Writes an array of result objects matching the schema from
        :meth:`TestResult.json_schema`.

        Args:
            path: Destination file path (e.g. ``"results.json"``).

        Returns:
            Absolute path to the written file.

        Raises:
            RuntimeError: If :meth:`run` has not been called yet.
        """
        self._ensure_has_results("to_json")
        from pathlib import Path

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "suite": self._name,
            "total": len(self._results),
            "passed": sum(1 for r in self._results if r.passed),
            "failed": sum(1 for r in self._results if not r.passed),
            "results": self._results_as_dicts(),
        }
        out.write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
        return str(out.resolve())

    def to_html(self, path: str) -> str:
        """Export results as a self-contained HTML report.

        Delegates to :func:`mltk.report.generator.generate_report`.
        Requires the ``jinja2`` package (install via
        ``pip install mltk[report]``).

        Args:
            path: Destination file path (e.g. ``"report.html"``).
                The parent directory is used as *output_dir* and the
                file is written inside it.

        Returns:
            Absolute path to the generated HTML file.

        Raises:
            RuntimeError: If :meth:`run` has not been called yet.
            ImportError: If ``jinja2`` is not installed.
        """
        self._ensure_has_results("to_html")
        from pathlib import Path

        from mltk.report.generator import generate_report

        out = Path(path)
        # generate_report expects list-of-dicts with nodeid/outcome keys
        report_results: list[dict[str, Any]] = []
        for r in self._results:
            report_results.append({
                "nodeid": r.name,
                "outcome": "passed" if r.passed else "failed",
                "duration": r.duration_ms / 1000.0,
                "ml_result": r,
            })

        generated = generate_report(
            results=report_results,
            output_dir=str(out.parent),
            title=f"MLTK Suite: {self._name}",
        )
        return str(generated.resolve())

    def to_junit(self, path: str) -> str:
        """Export results as JUnit XML for CI/CD dashboards.

        Delegates to :func:`mltk.report.junit.export_junit_xml`.
        The output is compatible with Jenkins, GitLab CI, Azure DevOps,
        and CircleCI.

        Args:
            path: Destination file path (e.g. ``"results.xml"``).

        Returns:
            Absolute path to the written XML file.

        Raises:
            RuntimeError: If :meth:`run` has not been called yet.
        """
        self._ensure_has_results("to_junit")
        from mltk.report.junit import export_junit_xml

        return export_junit_xml(
            results=self._results_as_dicts(),
            output_path=path,
            suite_name=self._name,
        )

    # -- introspection -----------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable one-line summary of the last run.

        Returns:
            A string like ``"my-suite: 8/10 passed (80.0%) in 42.5 ms"``
            or ``"my-suite: not yet run"`` if :meth:`run` has not been
            called.
        """
        if self._suite_result is None:
            return f"{self._name}: not yet run"
        sr = self._suite_result
        pct = sr.pass_rate * 100
        return (
            f"{sr.name}: {sr.passed_count}/{sr.total} passed "
            f"({pct:.1f}%) in {sr.duration_ms:.1f} ms"
        )

    @property
    def passed(self) -> bool:
        """True if the last :meth:`run` had zero failures.

        Raises:
            RuntimeError: If :meth:`run` has not been called yet.
        """
        self._ensure_has_results("passed")
        assert self._suite_result is not None  # for type checker
        return self._suite_result.passed

    @property
    def results(self) -> list[TestResult]:
        """List of :class:`~mltk.core.result.TestResult` from the last run.

        Returns an empty list if :meth:`run` has not been called.
        """
        return list(self._results)

    @property
    def name(self) -> str:
        """Suite name (read-only)."""
        return self._name

    # -- private helpers ---------------------------------------------------

    def _ensure_has_results(self, method: str) -> None:
        """Raise RuntimeError if run() has not been called."""
        if self._suite_result is None:
            raise RuntimeError(
                f"Cannot call {method}() before run(). "
                f"Call suite.run() first."
            )

    # -- dunder ------------------------------------------------------------

    def __repr__(self) -> str:
        n_pending = len(self._pending)
        n_results = len(self._results)
        return (
            f"MltkSuite(name={self._name!r}, "
            f"pending={n_pending}, results={n_results})"
        )

    def __len__(self) -> int:
        """Number of pending assertions (before run) or results (after)."""
        if self._suite_result is not None:
            return len(self._results)
        return len(self._pending)
