"""Regression tests for HTML exports that pass legacy ml_result records."""

from __future__ import annotations

from pathlib import Path

import pytest

from mltk.core.assertion import assert_true
from mltk.core.result import Severity, TestResult
from mltk.core.suite import MltkSuite
from mltk.scan.engine import ScanReport
from mltk.scan.finding import ScanFinding


def _suite_assertion() -> TestResult:
    """Return a passing assertion with recorded evidence."""
    return assert_true(
        True,
        name="suite.assertion",
        message="suite passed",
        recorded="suite-value",
    )


def _scan_assertion() -> TestResult:
    """Return a passing scan assertion with recorded evidence."""
    return TestResult(
        name="scan.assertion",
        passed=True,
        severity=Severity.INFO,
        message="scan passed",
        details={"recorded": "scan-value"},
    )


class TestHtmlExportApi:
    """Verify high-level HTML exporters preserve assertion details."""

    def test_mltk_suite_to_html_contains_recorded_detail(self, tmp_path: Path) -> None:
        """SUITE: MltkSuite HTML output includes the assertion's detail value.

        WHY: Suite export still supplies only ml_result, so the generator's legacy
        fallback must expose recorded evidence in the public HTML API.
        """
        pytest.importorskip("jinja2")
        suite = MltkSuite("detail-suite")
        suite.add(_suite_assertion)
        suite.run()

        html_path = suite.to_html(str(tmp_path / "suite.html"))
        html = Path(html_path).read_text(encoding="utf-8")

        assert "recorded = suite-value" in html

    def test_scan_report_to_html_contains_recorded_detail(self, tmp_path: Path) -> None:
        """SCAN: ScanReport HTML output includes the finding's detail value.

        WHY: Scan export also supplies only ml_result; its findings must retain the
        evidence captured by the scanner when rendered as HTML.
        """
        pytest.importorskip("jinja2")
        result = _scan_assertion()
        finding = ScanFinding(
            result=result,
            assertion_fn=_scan_assertion,
            scanner_name="test",
        )
        report = ScanReport(findings=[finding])

        html_path = report.to_html(str(tmp_path / "scan.html"))
        html = Path(html_path).read_text(encoding="utf-8")

        assert "recorded = scan-value" in html
