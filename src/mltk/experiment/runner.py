"""Experiment runner -- test fix hypotheses against scan findings.

The :class:`ExperimentRunner` takes a
:class:`~mltk.scan.finding.ScanFinding`, applies each fix hypothesis,
re-runs the original assertion, and collects results.  This is the
core loop of the experiment module:

1. Record the **baseline** (re-run the original failing assertion).
2. For each hypothesis: call the apply function, capture the
   :class:`~mltk.core.result.TestResult`.
3. **Rank** results using the configured strategy.
4. Select the best fix (if any hypothesis turned a failure into a
   pass).

Usage::

    from mltk.experiment import ExperimentRunner

    runner = ExperimentRunner(strategy="passed", timeout=10.0)
    result = runner.run(finding, apply_fns=[
        (finding.suggested_fixes[0], my_fix_fn),
    ])
    if result.selected_fix:
        print(f"Best fix: {result.selected_fix.title}")
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from mltk.core.assertion import MltkAssertionError
from mltk.core.result import Severity, TestResult
from mltk.experiment.hypothesis import Hypothesis, HypothesisResult
from mltk.experiment.ranking import rank_hypotheses
from mltk.experiment.result import ExperimentResult
from mltk.scan.finding import FixSuggestion, ScanFinding

__all__ = ["ExperimentRunner"]

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Test fix hypotheses against scan findings and rank results.

    The runner:

    1. Records the baseline (original failing assertion).
    2. For each hypothesis: applies the fix, re-runs the assertion.
    3. Ranks results using the configured strategy.
    4. Selects the best fix.

    Args:
        strategy: Ranking strategy -- ``"passed"``, ``"delta"``,
            or ``"composite"``.  See :func:`rank_hypotheses`.
        timeout: Per-hypothesis timeout in seconds.  If a
            hypothesis exceeds this, it is captured as a failed
            :class:`~mltk.core.result.TestResult`.
    """

    def __init__(
        self,
        strategy: str = "passed",
        timeout: float = 30.0,
    ) -> None:
        self._strategy = strategy
        self._timeout = timeout

    # -- public API --------------------------------------------------------

    def run(
        self,
        finding: ScanFinding,
        apply_fns: (
            list[tuple[FixSuggestion, Callable[..., TestResult]]]
            | None
        ) = None,
        hypotheses: list[Hypothesis] | None = None,
    ) -> ExperimentResult:
        """Test all fix hypotheses for a single finding.

        Provide EITHER *apply_fns* (list of fix/callable pairs) OR
        pre-built *hypotheses*.  If *apply_fns* is provided,
        :class:`Hypothesis` objects are built automatically.

        Each apply_fn signature: ``() -> TestResult``.  The caller is
        responsible for setting up modified model/data and calling the
        assertion internally.

        Args:
            finding: :class:`ScanFinding` with ``suggested_fixes``.
            apply_fns: List of ``(FixSuggestion, callable)`` pairs.
                Each callable returns a :class:`TestResult`.
            hypotheses: Pre-built :class:`Hypothesis` objects
                (alternative to *apply_fns*).

        Returns:
            :class:`ExperimentResult` with ranked hypothesis results
            and the selected fix (if any).
        """
        start = time.perf_counter()

        # Build hypothesis list from whichever source is provided.
        hyps = self._resolve_hypotheses(apply_fns, hypotheses)

        # Capture baseline by re-running the original assertion.
        baseline = self._get_baseline(finding)

        # Execute each hypothesis and collect results.
        hypothesis_results: list[HypothesisResult] = []
        for hyp in hyps:
            fixed_result = self._run_hypothesis(hyp, finding)
            improvement = (
                1.0
                if (fixed_result.passed and not baseline.passed)
                else 0.0
            )
            hypothesis_results.append(
                HypothesisResult(
                    hypothesis=hyp,
                    baseline_result=baseline,
                    fixed_result=fixed_result,
                    improvement=improvement,
                )
            )

        # Rank and select best fix.
        if hypothesis_results:
            hypothesis_results = rank_hypotheses(
                hypothesis_results, self._strategy,
            )

        selected_fix = self._select_fix(hypothesis_results)

        elapsed_ms = (time.perf_counter() - start) * 1000
        return ExperimentResult(
            finding=finding,
            baseline_result=baseline,
            hypothesis_results=hypothesis_results,
            selected_fix=selected_fix,
            duration_ms=elapsed_ms,
        )

    def run_batch(
        self,
        findings: list[ScanFinding],
        apply_fns_map: (
            dict[str, dict[str, Callable[..., TestResult]]] | None
        ) = None,
    ) -> list[ExperimentResult]:
        """Test fixes across multiple findings.

        Args:
            findings: List of :class:`ScanFinding` objects.
            apply_fns_map: Nested dict mapping
                ``scanner_name -> fix_title -> apply_fn``.
                Used to look up the correct apply function for
                each finding's suggested fixes.

        Returns:
            List of :class:`ExperimentResult`, one per finding.
        """
        results: list[ExperimentResult] = []
        for finding in findings:
            apply_fns: (
                list[
                    tuple[
                        FixSuggestion,
                        Callable[..., TestResult],
                    ]
                ]
                | None
            ) = None

            if apply_fns_map is not None:
                scanner_fns = apply_fns_map.get(
                    finding.scanner_name, {},
                )
                if scanner_fns:
                    pairs: list[
                        tuple[
                            FixSuggestion,
                            Callable[..., TestResult],
                        ]
                    ] = []
                    for fix in finding.suggested_fixes:
                        fn = scanner_fns.get(fix.title)
                        if fn is not None:
                            pairs.append((fix, fn))
                    if pairs:
                        apply_fns = pairs

            results.append(self.run(finding, apply_fns=apply_fns))
        return results

    # -- internal ----------------------------------------------------------

    def _resolve_hypotheses(
        self,
        apply_fns: (
            list[tuple[FixSuggestion, Callable[..., TestResult]]]
            | None
        ),
        hypotheses: list[Hypothesis] | None,
    ) -> list[Hypothesis]:
        """Build the hypothesis list from whichever source."""
        if hypotheses is not None:
            return list(hypotheses)
        if apply_fns is not None:
            return [
                Hypothesis(
                    fix=fix,
                    apply_fn=fn,
                    description=fix.title,
                )
                for fix, fn in apply_fns
            ]
        return []

    def _run_hypothesis(
        self,
        hypothesis: Hypothesis,
        finding: ScanFinding,
    ) -> TestResult:
        """Execute one hypothesis with timeout.

        Runs ``hypothesis.apply_fn()`` in a daemon thread.  If the
        thread exceeds :attr:`_timeout` or raises, a failed
        :class:`TestResult` is returned.

        .. note:: Timed-out threads continue running in the background
           (daemon threads cannot be forcefully stopped in Python).
           If apply_fn holds large model/data references, memory will
           be pinned until the thread finishes or the process exits.

        Args:
            hypothesis: The hypothesis to test.
            finding: The scan finding being addressed.

        Returns:
            :class:`TestResult` from the apply function, or a
            synthetic failure if execution timed out or crashed.
        """
        result_holder: list[TestResult | None] = [None]
        error_holder: list[BaseException | None] = [None]
        t0 = time.perf_counter()

        def _target() -> None:
            try:
                result_holder[0] = hypothesis.apply_fn()
            except MltkAssertionError as exc:
                result_holder[0] = exc.result
            except Exception as exc:
                error_holder[0] = exc

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        thread.join(timeout=self._timeout)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if thread.is_alive():
            logger.warning(
                "Hypothesis '%s' timed out after %.1fs",
                hypothesis.fix.title,
                self._timeout,
            )
            return TestResult(
                name=f"experiment.timeout.{hypothesis.fix.title}",
                passed=False,
                severity=Severity.CRITICAL,
                message=(
                    f"Hypothesis '{hypothesis.fix.title}' timed out "
                    f"after {self._timeout:.1f}s"
                ),
                details={"timeout_seconds": self._timeout},
                duration_ms=elapsed_ms,
            )

        if error_holder[0] is not None:
            exc = error_holder[0]
            logger.warning(
                "Hypothesis '%s' raised: %s",
                hypothesis.fix.title,
                exc,
                exc_info=exc,
            )
            return TestResult(
                name=f"experiment.error.{hypothesis.fix.title}",
                passed=False,
                severity=Severity.CRITICAL,
                message=(
                    f"Hypothesis '{hypothesis.fix.title}' raised "
                    f"{type(exc).__name__}: {exc}"
                ),
                details={"exception_type": type(exc).__name__},
                duration_ms=elapsed_ms,
            )

        if result_holder[0] is not None:
            if result_holder[0].duration_ms == 0.0:
                result_holder[0].duration_ms = elapsed_ms
            return result_holder[0]

        # apply_fn returned None -- treat as failure.
        return TestResult(
            name=f"experiment.no_result.{hypothesis.fix.title}",
            passed=False,
            severity=Severity.CRITICAL,
            message=(
                f"Hypothesis '{hypothesis.fix.title}' returned None "
                f"instead of a TestResult"
            ),
        )

    def _get_baseline(self, finding: ScanFinding) -> TestResult:
        """Re-run the original assertion to get baseline result.

        Uses the same error-isolation pattern as
        :class:`~mltk.core.suite.MltkSuite`: catches
        :class:`~mltk.core.assertion.MltkAssertionError` for
        expected failures and wraps unexpected exceptions in a
        CRITICAL :class:`TestResult`.

        Args:
            finding: The scan finding whose assertion is re-run.

        Returns:
            :class:`TestResult` from the original assertion.
        """
        t0 = time.perf_counter()
        try:
            result = finding.assertion_fn(
                *finding.assertion_args,
                **finding.assertion_kwargs,
            )
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
                finding.assertion_fn,
                "__name__",
                str(finding.assertion_fn),
            )
            return TestResult(
                name=f"experiment.baseline.{fn_name}",
                passed=False,
                severity=Severity.CRITICAL,
                message=f"Baseline assertion raised: {exc!r}",
                details={"exception_type": type(exc).__name__},
                duration_ms=elapsed,
            )

    @staticmethod
    def _select_fix(
        hypothesis_results: list[HypothesisResult],
    ) -> FixSuggestion | None:
        """Pick the best fix from ranked results.

        Returns the fix from the top-ranked hypothesis (rank == 1)
        if it turned a failure into a pass.  Otherwise returns
        ``None``.
        """
        if not hypothesis_results:
            return None
        # Find the rank-1 hypothesis; fall back to first if
        # no ranking was applied.
        best = hypothesis_results[0]
        for hr in hypothesis_results:
            if hr.rank == 1:
                best = hr
                break
        if best.is_winning:
            return best.hypothesis.fix
        return None
