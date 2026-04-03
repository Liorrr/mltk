"""Scan finding -- the unit of output from a scanner.

A ScanFinding carries **two things**:

1. **Evidence** -- a :class:`~mltk.core.result.TestResult` that
   records what happened (metric value, threshold, pass/fail).
2. **Reproduction recipe** -- the assertion function plus its exact
   arguments, so the finding can be replayed later in an
   :class:`~mltk.core.suite.MltkSuite` or exported as pytest code.

This dual nature is what makes ``mltk scan`` unique: it does not
just *report* issues -- it gives you runnable code to reproduce
them.  Every finding can be turned into a pytest test, added to a
suite, or rendered in HTML/JUnit reports.

Example::

    finding = ScanFinding(
        result=test_result,
        assertion_fn=assert_metric,
        assertion_args=(y_true_slice, y_pred_slice),
        assertion_kwargs={"metric": "accuracy", "threshold": 0.7},
        suggested_test='def test_age_gt55(): ...',
        scanner_name="slice",
    )

    # Replay in a suite
    fn, args, kwargs = finding.to_pending()
    suite.add(fn, *args, **kwargs)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from mltk.core.result import TestResult

__all__ = ["FixSuggestion", "ScanFinding"]


@dataclass
class FixSuggestion:
    """A concrete remediation step for a scan finding.

    Each suggestion carries a category (code/config/data/process),
    a confidence level, and optionally a runnable code snippet.
    Scanners attach 1-3 suggestions per finding, ranked by
    confidence.

    Attributes:
        category: Type of fix -- "code", "config", "data",
            or "process".
        title: One-line summary shown in console/reports.
        description: 1-2 sentence explanation of why this
            fix applies and what it does.
        confidence: How certain this fix is correct --
            "high", "medium", or "low".
        code_snippet: Optional Python code implementing the
            fix.  Empty string if the fix is non-code.
    """

    _VALID_CATEGORIES = {"code", "config", "data", "process"}
    _VALID_CONFIDENCES = {"high", "medium", "low"}

    category: str   # "code" | "config" | "data" | "process"
    title: str
    description: str
    confidence: str  # "high" | "medium" | "low"
    code_snippet: str = ""

    def __post_init__(self) -> None:
        if self.category not in self._VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category {self.category!r}, "
                f"expected one of {self._VALID_CATEGORIES}"
            )
        if self.confidence not in self._VALID_CONFIDENCES:
            raise ValueError(
                f"Invalid confidence {self.confidence!r}, "
                f"expected one of {self._VALID_CONFIDENCES}"
            )


@dataclass
class ScanFinding:
    """A single issue discovered by a scanner.

    Carries both the EVIDENCE (TestResult -- what happened) and
    the REPRODUCTION recipe (assertion_fn + args -- how to
    reproduce it).  This dual nature enables:

    - ``to_suite()`` on ScanReport builds a runnable MltkSuite
      by calling ``assertion_fn(*args, **kwargs)`` for each
      finding.
    - ``to_html()`` / ``to_junit()`` use ``.result`` (the
      TestResult) to render reports.
    - ``to_test_file()`` uses ``.suggested_test`` to emit
      self-contained pytest code.

    Attributes:
        result: The TestResult that records what the scanner
            found -- metric values, thresholds, pass/fail
            status, severity, and timing.
        assertion_fn: The mltk assertion function that
            detected this issue (e.g., ``assert_metric``,
            ``assert_no_bias``).  Stored so the finding can
            be replayed without re-scanning.
        assertion_args: Positional arguments to pass to
            ``assertion_fn`` for reproduction.
        assertion_kwargs: Keyword arguments to pass to
            ``assertion_fn`` for reproduction.
        suggested_test: A string of valid pytest code that
            reproduces this finding.  Validated with
            ``ast.parse()`` before being written to disk.
        scanner_name: Which scanner found this issue (e.g.,
            ``"slice"``, ``"bias"``, ``"leakage"``).  Used
            for grouping in reports and console output.
    """

    result: TestResult
    assertion_fn: Callable[..., TestResult]
    assertion_args: tuple[Any, ...] = ()
    assertion_kwargs: dict[str, Any] = field(
        default_factory=dict,
    )
    suggested_test: str = ""
    scanner_name: str = ""
    suggested_fixes: list[FixSuggestion] = field(default_factory=list)

    def to_pending(
        self,
    ) -> tuple[
        Callable[..., TestResult],
        tuple[Any, ...],
        dict[str, Any],
    ]:
        """Return the triple needed by MltkSuite.add().

        Returns:
            ``(assertion_fn, assertion_args, assertion_kwargs)``
            ready to be unpacked into
            ``suite.add(fn, *args, **kwargs)``.
        """
        return (
            self.assertion_fn,
            self.assertion_args,
            self.assertion_kwargs,
        )
