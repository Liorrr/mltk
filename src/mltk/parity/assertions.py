"""Parity gates integrated with mltk's result recorder."""

from __future__ import annotations

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import TestResult
from mltk.parity.models import ParityReport
from mltk.parity.output import ArtifactPaths, report_summary


@timed_assertion
def assert_parity(report: ParityReport, *, fail_on_negative_flip: bool = False,
                  artifacts: ArtifactPaths | None = None) -> TestResult:
    """Require a demonstrated passing gate; optionally reject every negative flip.

    Raises MltkAssertionError on failed or incomplete verification. Execution
    regressions on oracle-passing baseline items always fail product parity.
    """
    passed = report.passed
    if report.mode == "product":
        required = [r for r in report.rates if r.get("role") != "diagnostic"]
        passed = passed and not any(r.get("from_failure_count", 0) for r in required)
        passed = passed and not any(r.get("execution_regression_count", 0) for r in required)
        if fail_on_negative_flip:
            passed = passed and not any(r.get("negative_flip_count", 0) for r in required)
    suffix = " (strict gate failed)" if report.passed and not passed else ""
    return assert_true(passed, "parity." + report.mode,
                       f"Parity {report.mode}: {report.status}" + suffix,
                       **report_summary(report, artifacts))
