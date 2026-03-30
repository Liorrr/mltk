"""Snapshot / structural tests for mltk report output formats.

Complements ``mltk.testing.golden`` (numeric data comparison) by verifying
the *structure* of rendered outputs -- HTML reports and JUnit XML.  Uses
syrupy for snapshot assertions so that any structural drift is caught
automatically and can be reviewed via ``pytest --snapshot-update``.

Classes:
    TestHtmlReportStructure   -- TestResult._repr_html_() output
    TestHtmlSuiteStructure    -- TestSuite._repr_html_() output
    TestJunitXmlStructure     -- JUnit XML element tree
    TestHtmlSnapshots         -- syrupy snapshot assertions
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from mltk.core.result import Severity, TestResult, TestSuite
from mltk.report.junit import export_junit_xml, format_result_to_junit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pass_result(**overrides) -> TestResult:
    """Create a passing TestResult with sensible defaults."""
    kwargs = dict(
        name="test.example",
        passed=True,
        message="All good",
        severity=Severity.INFO,
        details={"score": 0.95},
        duration_ms=12.3,
    )
    kwargs.update(overrides)
    return TestResult(**kwargs)


def _make_fail_result(**overrides) -> TestResult:
    """Create a failing TestResult with sensible defaults."""
    kwargs = dict(
        name="model.accuracy.check",
        passed=False,
        message="accuracy 0.72 < 0.80 threshold",
        severity=Severity.CRITICAL,
        details={"actual": 0.72, "threshold": 0.80, "delta": -0.08},
        duration_ms=45.7,
    )
    kwargs.update(overrides)
    return TestResult(**kwargs)


def _make_suite() -> TestSuite:
    """Create a TestSuite with mixed pass/fail results."""
    suite = TestSuite()
    suite.add(_make_pass_result())
    suite.add(_make_fail_result())
    suite.add(TestResult(
        name="data.drift.psi",
        passed=True,
        message="PSI=0.03 within tolerance",
        severity=Severity.WARNING,
        details={"psi": 0.03, "limit": 0.10},
        duration_ms=8.1,
    ))
    return suite


# ---------------------------------------------------------------------------
# HTML structure tests -- TestResult._repr_html_()
# ---------------------------------------------------------------------------

class TestHtmlReportStructure:
    """Verify TestResult._repr_html_() output has expected elements."""

    def test_repr_html_pass_structure(self) -> None:
        """PASS result HTML contains badge, name, message, details table."""
        result = _make_pass_result()
        html = result._repr_html_()

        assert "PASS" in html
        assert "test.example" in html
        assert "All good" in html
        assert "score" in html
        assert "<table" in html

    def test_repr_html_fail_structure(self) -> None:
        """FAIL result HTML has red badge and all detail keys."""
        result = _make_fail_result()
        html = result._repr_html_()

        assert "FAIL" in html
        assert "model.accuracy.check" in html
        assert "accuracy 0.72" in html
        # All detail keys must be present
        for key in ("actual", "threshold", "delta"):
            assert key in html
        # Red color for failure badge
        assert "#ef4444" in html

    def test_repr_html_pass_has_green_badge(self) -> None:
        """PASS result uses green (#22c55e) for the status badge."""
        result = _make_pass_result()
        html = result._repr_html_()

        assert "#22c55e" in html

    def test_repr_html_severity_shown(self) -> None:
        """Severity value appears in the rendered HTML."""
        for sev in Severity:
            result = _make_pass_result(severity=sev)
            html = result._repr_html_()
            assert sev.value in html

    def test_repr_html_duration_formatted(self) -> None:
        """Duration is formatted with two decimal places."""
        result = _make_pass_result(duration_ms=123.456)
        html = result._repr_html_()

        assert "123.46" in html

    def test_repr_html_xss_escaped(self) -> None:
        """html.escape() prevents XSS in name, message, and details."""
        result = TestResult(
            name="<script>alert('xss')</script>",
            passed=False,
            message="<b>bold</b>",
            severity=Severity.CRITICAL,
            details={"key": "<img onerror=alert(1)>"},
            duration_ms=0.0,
        )
        html = result._repr_html_()

        # Raw script tags must NOT appear
        assert "<script>" not in html
        assert "<b>bold</b>" not in html.split("style")[0]  # not as raw HTML
        assert "<img onerror" not in html

        # Escaped versions must appear
        assert "&lt;script&gt;" in html
        assert "&lt;b&gt;" in html
        assert "&lt;img onerror" in html

    def test_repr_html_empty_details(self) -> None:
        """Result with no details omits the table element entirely."""
        result = _make_pass_result(details={})
        html = result._repr_html_()

        assert "<table" not in html

    def test_repr_html_is_valid_div(self) -> None:
        """Output is wrapped in a single <div> element."""
        result = _make_pass_result()
        html = result._repr_html_()

        assert html.strip().startswith("<div")
        assert html.strip().endswith("</div>")


# ---------------------------------------------------------------------------
# HTML structure tests -- TestSuite._repr_html_()
# ---------------------------------------------------------------------------

class TestHtmlSuiteStructure:
    """Verify TestSuite._repr_html_() output has expected elements."""

    def test_suite_html_has_score(self) -> None:
        """Suite HTML shows pass count, total, and percentage."""
        suite = _make_suite()
        html = suite._repr_html_()

        assert "2/3" in html  # 2 passed out of 3
        assert "66.7%" in html

    def test_suite_html_has_badges(self) -> None:
        """Suite HTML contains pass and fail count badges."""
        suite = _make_suite()
        html = suite._repr_html_()

        assert "2 passed" in html
        assert "1 failed" in html

    def test_suite_html_has_table_headers(self) -> None:
        """Suite HTML table has Test, Status, Message, Duration headers."""
        suite = _make_suite()
        html = suite._repr_html_()

        for header in ("Test", "Status", "Message", "Duration"):
            assert header in html

    def test_suite_html_all_test_names_present(self) -> None:
        """Every test name from the suite appears in the rendered HTML."""
        suite = _make_suite()
        html = suite._repr_html_()

        assert "test.example" in html
        assert "model.accuracy.check" in html
        assert "data.drift.psi" in html

    def test_suite_html_total_duration(self) -> None:
        """Suite HTML shows the summed duration of all tests."""
        suite = _make_suite()
        total = sum(r.duration_ms for r in suite.results)
        html = suite._repr_html_()

        assert f"{total:.2f}" in html

    def test_suite_html_empty_suite(self) -> None:
        """Empty suite renders without crashing and shows 0/0."""
        suite = TestSuite()
        html = suite._repr_html_()

        assert "0/0" in html
        assert "0.0%" in html


# ---------------------------------------------------------------------------
# JUnit XML structure tests
# ---------------------------------------------------------------------------

class TestJunitXmlStructure:
    """Verify JUnit XML output structure and element attributes."""

    def test_junit_has_testsuites_root(self, tmp_path: Path) -> None:
        """Root element is <testsuites>."""
        results = [
            {"name": "a.b", "passed": True, "duration_ms": 10.0, "message": "ok"},
        ]
        out = tmp_path / "root.xml"
        path = export_junit_xml(results, output_path=str(out))

        tree = ET.parse(path)
        assert tree.getroot().tag == "testsuites"

    def test_junit_testsuite_has_required_attrs(
        self, tmp_path: Path,
    ) -> None:
        """<testsuite> element has name, tests, failures, errors, time."""
        results = [
            {"name": "a.b", "passed": True, "duration_ms": 10.0, "message": "ok"},
            {"name": "c.d", "passed": False, "duration_ms": 20.0, "message": "bad"},
        ]
        out = tmp_path / "attrs.xml"
        export_junit_xml(results, output_path=str(out))

        tree = ET.parse(str(out))
        suite = tree.getroot().find("testsuite")
        assert suite is not None

        required_attrs = {"name", "tests", "failures", "errors", "time"}
        assert required_attrs.issubset(set(suite.attrib.keys()))

    def test_junit_testcase_attributes(self, tmp_path: Path) -> None:
        """Each <testcase> has name, classname, and time attributes."""
        results = [
            {"name": "data.schema.check", "passed": True,
             "duration_ms": 50.0, "message": "ok"},
        ]
        out = tmp_path / "tc_attrs.xml"
        export_junit_xml(results, output_path=str(out))

        tree = ET.parse(str(out))
        tc = tree.getroot().find(".//testcase")
        assert tc is not None
        assert "name" in tc.attrib
        assert "classname" in tc.attrib
        assert "time" in tc.attrib

    def test_junit_failure_element(self, tmp_path: Path) -> None:
        """Failed test has <failure> with message, type, and text."""
        results = [
            {"name": "model.metric.f1", "passed": False,
             "duration_ms": 100.0, "message": "F1 too low"},
        ]
        out = tmp_path / "failure.xml"
        export_junit_xml(results, output_path=str(out))

        tree = ET.parse(str(out))
        failure = tree.getroot().find(".//failure")
        assert failure is not None
        assert failure.get("message") == "F1 too low"
        assert failure.get("type") == "MltkAssertionError"
        assert failure.text == "F1 too low"

    def test_junit_xml_declaration(self, tmp_path: Path) -> None:
        """Output starts with an XML declaration."""
        results = [
            {"name": "x", "passed": True, "duration_ms": 1.0, "message": "ok"},
        ]
        out = tmp_path / "decl.xml"
        path = export_junit_xml(results, output_path=str(out))

        content = Path(path).read_text(encoding="utf-8")
        assert content.startswith("<?xml")

    def test_junit_indented_output(self, tmp_path: Path) -> None:
        """Output is human-readable with indentation."""
        results = [
            {"name": "a.b", "passed": True, "duration_ms": 10.0, "message": "ok"},
        ]
        out = tmp_path / "indent.xml"
        path = export_junit_xml(results, output_path=str(out))

        content = Path(path).read_text(encoding="utf-8")
        # ET.indent uses spaces; there should be indented lines
        lines = content.strip().splitlines()
        indented = [ln for ln in lines if ln.startswith("  ")]
        assert len(indented) > 0


# ---------------------------------------------------------------------------
# Syrupy snapshot tests
# ---------------------------------------------------------------------------

class TestHtmlSnapshots:
    """Syrupy snapshot assertions for HTML output.

    Run ``pytest --snapshot-update`` to create/update snapshots.
    Subsequent runs will fail if the HTML structure changes.
    """

    def test_pass_result_snapshot(self, snapshot) -> None:
        """Snapshot of a passing TestResult HTML."""
        result = _make_pass_result()
        assert result._repr_html_() == snapshot

    def test_fail_result_snapshot(self, snapshot) -> None:
        """Snapshot of a failing TestResult HTML."""
        result = _make_fail_result()
        assert result._repr_html_() == snapshot

    def test_empty_details_snapshot(self, snapshot) -> None:
        """Snapshot of a result with no details."""
        result = _make_pass_result(details={})
        assert result._repr_html_() == snapshot

    def test_suite_snapshot(self, snapshot) -> None:
        """Snapshot of a mixed-result TestSuite HTML."""
        suite = _make_suite()
        assert suite._repr_html_() == snapshot


class TestJunitSnapshots:
    """Syrupy snapshot assertions for JUnit XML output."""

    def test_single_pass_xml_snapshot(self, snapshot) -> None:
        """Snapshot of XML for a single passing testcase."""
        elem = format_result_to_junit(
            {"name": "data.schema.check", "passed": True,
             "duration_ms": 50.0, "message": "ok"},
        )
        xml_str = ET.tostring(elem, encoding="unicode")
        assert xml_str == snapshot

    def test_single_fail_xml_snapshot(self, snapshot) -> None:
        """Snapshot of XML for a single failing testcase."""
        elem = format_result_to_junit(
            {"name": "model.metric.accuracy", "passed": False,
             "duration_ms": 120.0, "message": "accuracy 0.75 < 0.80"},
        )
        xml_str = ET.tostring(elem, encoding="unicode")
        assert xml_str == snapshot
