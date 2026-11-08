"""Parity gates integrated with mltk's result recorder."""

from __future__ import annotations

import warnings

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import TestResult
from mltk.parity.models import ParityReport
from mltk.parity.output import ArtifactPaths, artifact_names, report_summary
from mltk.parity.presentation import finalize_report


@timed_assertion
def assert_parity(
    report: ParityReport,
    *,
    fail_on_negative_flip: bool | None = None,
    artifacts: ArtifactPaths | None = None,
) -> TestResult:
    """Assert the engine's report status without recomputing its gate.

    Configure negative-flip rejection through ParityConfig before comparison.
    The deprecated argument, if supplied, must agree with the report config.
    A mismatch raises ValueError as configuration misuse, not a failed gate.
    """
    if fail_on_negative_flip is not None:
        warnings.warn(
            "Set fail_on_negative_flip in ParityConfig before comparison; "
            "the assertion argument is deprecated.",
            DeprecationWarning,
            stacklevel=2,
        )
        if type(fail_on_negative_flip) is not bool or fail_on_negative_flip != report.config.get(
            "fail_on_negative_flip", False
        ):
            raise ValueError(
                "fail_on_negative_flip must equal report.config; "
                "configure the comparison engine before creating the report"
            )
    if artifacts is not None:
        report.artifacts = artifact_names(artifacts)
        finalize_report(report)
    elif not report.message:
        finalize_report(report)
    return assert_true(
        report.status == "PASS",
        "parity." + report.mode,
        report.message,
        **report_summary(report, artifacts),
    )
