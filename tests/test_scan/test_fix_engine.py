"""Tests for engine-level fix suggestion integration.

Verifies that FixSuggestion objects attached to ScanFindings
are correctly serialized in to_json(), reflected in summary(),
and formatted for console output via format_fixes().
"""

from __future__ import annotations

import json

from mltk.core.result import Severity, TestResult
from mltk.scan.console import format_fixes
from mltk.scan.engine import ScanReport
from mltk.scan.finding import FixSuggestion, ScanFinding

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _make_result(**overrides):
    """Build a TestResult with sensible defaults."""
    defaults = {
        "name": "test",
        "passed": False,
        "severity": Severity.WARNING,
        "message": "drift detected",
    }
    defaults.update(overrides)
    return TestResult(**defaults)


def _make_fix(**overrides):
    """Build a FixSuggestion with sensible defaults."""
    defaults = {
        "category": "data",
        "title": "Retrain model",
        "description": "Retrain on recent data to reduce drift.",
        "confidence": "high",
    }
    defaults.update(overrides)
    return FixSuggestion(**defaults)


def _make_finding(fixes=None, **overrides):
    """Build a ScanFinding with sensible defaults."""
    defaults = {
        "result": _make_result(),
        "assertion_fn": lambda: None,
        "suggested_test": "def test_x(): pass",
        "scanner_name": "drift",
        "suggested_fixes": fixes if fixes is not None else [],
    }
    defaults.update(overrides)
    return ScanFinding(**defaults)


def _make_report(findings=None, scanners_run=None):
    """Build a ScanReport with sensible defaults."""
    return ScanReport(
        findings=findings or [],
        scanners_run=scanners_run or ["drift"],
    )


# ---------------------------------------------------------------
# to_json tests
# ---------------------------------------------------------------


class TestToJsonSuggestedFixes:
    """to_json() correctly serializes suggested_fixes."""

    def test_to_json_includes_suggested_fixes(self) -> None:
        """A finding with fixes includes suggested_fixes array in JSON."""
        fix = _make_fix()
        finding = _make_finding(fixes=[fix])
        report = _make_report(findings=[finding])

        data = json.loads(report.to_json())
        assert "suggested_fixes" in data["findings"][0]
        assert len(data["findings"][0]["suggested_fixes"]) == 1

    def test_to_json_fix_has_all_fields(self) -> None:
        """Each fix in JSON has category, title, description, confidence, code_snippet."""
        fix = FixSuggestion(
            category="config",
            title="Lower threshold",
            description="Reduce accuracy threshold to 0.7.",
            confidence="medium",
            code_snippet="threshold = 0.7",
        )
        finding = _make_finding(fixes=[fix])
        report = _make_report(findings=[finding])

        data = json.loads(report.to_json())
        fix_json = data["findings"][0]["suggested_fixes"][0]

        assert fix_json["category"] == "config"
        assert fix_json["title"] == "Lower threshold"
        assert fix_json["description"] == "Reduce accuracy threshold to 0.7."
        assert fix_json["confidence"] == "medium"
        assert fix_json["code_snippet"] == "threshold = 0.7"

    def test_to_json_empty_fixes_is_empty_list(self) -> None:
        """A finding with no fixes produces an empty suggested_fixes array."""
        finding = _make_finding(fixes=[])
        report = _make_report(findings=[finding])

        data = json.loads(report.to_json())
        assert data["findings"][0]["suggested_fixes"] == []

    def test_to_json_multiple_fixes(self) -> None:
        """A finding with 3 fixes produces 3 items in the JSON array."""
        fixes = [
            _make_fix(title="Fix A", confidence="high"),
            _make_fix(title="Fix B", confidence="medium"),
            _make_fix(title="Fix C", confidence="low"),
        ]
        finding = _make_finding(fixes=fixes)
        report = _make_report(findings=[finding])

        data = json.loads(report.to_json())
        fix_list = data["findings"][0]["suggested_fixes"]
        assert len(fix_list) == 3
        assert fix_list[0]["title"] == "Fix A"
        assert fix_list[1]["title"] == "Fix B"
        assert fix_list[2]["title"] == "Fix C"


# ---------------------------------------------------------------
# summary tests
# ---------------------------------------------------------------


class TestSummaryFixes:
    """summary() mentions fix counts when fixes exist."""

    def test_summary_mentions_fixes(self) -> None:
        """Summary text includes fix count when findings have fixes."""
        fixes = [_make_fix(), _make_fix(title="Second fix")]
        finding = _make_finding(fixes=fixes)
        report = _make_report(findings=[finding])

        text = report.summary()
        assert "Fix suggestions: 2" in text
        assert "--verbose" in text

    def test_summary_no_fix_mention_when_none(self) -> None:
        """Summary does not mention fixes when no findings have them."""
        finding = _make_finding(fixes=[])
        report = _make_report(findings=[finding])

        text = report.summary()
        assert "Fix suggestions" not in text


# ---------------------------------------------------------------
# format_fixes tests
# ---------------------------------------------------------------


class TestFormatFixes:
    """format_fixes() renders fix suggestions for the console."""

    def test_format_fixes_empty(self) -> None:
        """format_fixes([]) returns an empty string."""
        assert format_fixes([]) == ""

    def test_format_fixes_with_code(self) -> None:
        """format_fixes renders the code snippet indented below the title."""
        fix = _make_fix(
            title="Add validation",
            code_snippet="if x > 0:\n    return x\nelse:\n    raise ValueError",
        )
        text = format_fixes([fix])

        assert "Add validation" in text
        assert "if x > 0:" in text
        assert "    return x" in text
        # Only first 3 lines of snippet are shown
        lines = text.split("\n")
        code_lines = [
            ln for ln in lines if ln.startswith("       ")
        ]
        assert len(code_lines) == 3

    def test_format_fixes_confidence_tags(self) -> None:
        """high=+++, medium=++, low=+ tags appear correctly."""
        fixes = [
            _make_fix(title="High fix", confidence="high"),
            _make_fix(title="Med fix", confidence="medium"),
            _make_fix(title="Low fix", confidence="low"),
        ]
        text = format_fixes(fixes)

        assert "[+++] High fix" in text
        assert "[++] Med fix" in text
        assert "[+] Low fix" in text


# ---------------------------------------------------------------
# Roundtrip test
# ---------------------------------------------------------------


class TestToJsonRoundtrip:
    """to_json output can be parsed back and matches structure."""

    def test_to_json_roundtrip(self) -> None:
        """to_json -> json.loads -> verify full structure."""
        fix = FixSuggestion(
            category="process",
            title="Add monitoring",
            description="Set up drift monitoring in production.",
            confidence="high",
            code_snippet="monitor.track(model)",
        )
        result = _make_result(
            name="drift_check",
            passed=False,
            severity=Severity.CRITICAL,
            message="Distribution shift detected",
        )
        finding = _make_finding(
            fixes=[fix],
            result=result,
            scanner_name="drift",
        )
        report = ScanReport(
            findings=[finding],
            scanners_run=["drift"],
            scanners_skipped=["bias"],
            model_type="classifier",
            n_samples=1000,
            n_features=10,
            duration_ms=42.5,
        )

        text = report.to_json()
        data = json.loads(text)

        # Top-level structure
        assert data["scanners_run"] == ["drift"]
        assert data["scanners_skipped"] == ["bias"]
        assert data["model_type"] == "classifier"
        assert data["n_samples"] == 1000
        assert data["n_features"] == 10
        assert data["duration_ms"] == 42.5

        # Finding structure
        assert len(data["findings"]) == 1
        f = data["findings"][0]
        assert f["name"] == "drift_check"
        assert f["passed"] is False
        assert f["severity"] == "critical"
        assert f["message"] == "Distribution shift detected"
        assert f["scanner_name"] == "drift"

        # Fix structure
        assert len(f["suggested_fixes"]) == 1
        fx = f["suggested_fixes"][0]
        assert fx["category"] == "process"
        assert fx["title"] == "Add monitoring"
        assert fx["description"] == "Set up drift monitoring in production."
        assert fx["confidence"] == "high"
        assert fx["code_snippet"] == "monitor.track(model)"
