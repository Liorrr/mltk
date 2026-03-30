from __future__ import annotations

"""Tests for ScanReport.to_json() JSON export.

Verifies wire format, field completeness, enum
serialisation, numpy type handling, and file output.
"""

import json

import numpy as np
import pytest

from mltk.core.result import Severity, TestResult
from mltk.scan.config import ScanConfig
from mltk.scan.engine import ScanReport
from mltk.scan.finding import ScanFinding


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _dummy_assertion(*_a, **_kw):
    """No-op assertion for test findings."""
    return TestResult(
        name="noop",
        passed=True,
        severity=Severity.INFO,
        message="noop",
    )


def _make_finding(
    name: str = "scan.test.dummy",
    passed: bool = False,
    severity: Severity = Severity.CRITICAL,
    message: str = "something failed",
    scanner: str = "test_scanner",
    duration_ms: float = 1.5,
    details: dict | None = None,
    suggested_test: str = "",
) -> ScanFinding:
    """Build a ScanFinding with sensible defaults."""
    result = TestResult(
        name=name,
        passed=passed,
        severity=severity,
        message=message,
        details=details or {},
        duration_ms=duration_ms,
    )
    return ScanFinding(
        result=result,
        assertion_fn=_dummy_assertion,
        assertion_args=(),
        assertion_kwargs={},
        suggested_test=suggested_test,
        scanner_name=scanner,
    )


def _make_report(
    n_findings: int = 2,
) -> ScanReport:
    """Build a ScanReport with *n_findings* findings."""
    findings = [
        _make_finding(
            name=f"scan.test.f{i}",
            passed=(i % 2 == 0),
            severity=(
                Severity.CRITICAL
                if i % 2 else Severity.WARNING
            ),
            message=f"finding {i}",
            scanner=f"scanner_{i}",
            duration_ms=float(i),
            suggested_test=f"def test_f{i}(): ...",
        )
        for i in range(n_findings)
    ]
    return ScanReport(
        findings=findings,
        scanners_run=["scanner_0", "scanner_1"],
        scanners_skipped=["skipped_one"],
        scanners_errored={"bad_scanner": "boom"},
        duration_ms=99.9,
        model_type="classifier",
        n_samples=500,
        n_features=10,
        config=ScanConfig(),
    )


# ---------------------------------------------------------------
# Test: to_json returns valid JSON
# ---------------------------------------------------------------


class TestToJsonBasic:
    """Core serialisation behaviour."""

    def test_returns_valid_json(self):
        """to_json output parses as valid JSON."""
        report = _make_report()
        raw = report.to_json()
        data = json.loads(raw)
        assert isinstance(data, dict)

    def test_writes_to_file(self, tmp_path):
        """to_json writes file when path given."""
        report = _make_report()
        dest = str(tmp_path / "out.json")
        returned = report.to_json(dest)

        # File exists and contains valid JSON
        with open(dest, encoding="utf-8") as fh:
            from_disk = json.load(fh)
        assert from_disk == json.loads(returned)

    def test_creates_parent_dirs(self, tmp_path):
        """to_json creates missing parent dirs."""
        report = _make_report(n_findings=0)
        nested = tmp_path / "a" / "b" / "report.json"
        report.to_json(str(nested))
        assert nested.exists()


# ---------------------------------------------------------------
# Test: top-level fields
# ---------------------------------------------------------------

_TOP_LEVEL_KEYS = {
    "findings",
    "scanners_run",
    "scanners_skipped",
    "scanners_errored",
    "model_type",
    "n_samples",
    "n_features",
    "duration_ms",
    "exit_code",
}


class TestTopLevelFields:
    """Verify the JSON wire format top-level keys."""

    def test_all_required_keys_present(self):
        """JSON contains every required top-level key."""
        report = _make_report()
        data = json.loads(report.to_json())
        assert _TOP_LEVEL_KEYS <= set(data.keys())

    def test_metadata_values(self):
        """Metadata fields carry correct values."""
        report = _make_report()
        data = json.loads(report.to_json())
        assert data["model_type"] == "classifier"
        assert data["n_samples"] == 500
        assert data["n_features"] == 10
        assert data["duration_ms"] == pytest.approx(
            99.9
        )

    def test_exit_code_with_findings(self):
        """exit_code is 1 when findings exist."""
        report = ScanReport(
            findings=[_make_finding()],
            scanners_run=["s"],
        )
        data = json.loads(report.to_json())
        assert data["exit_code"] == 1

    def test_exit_code_with_errors(self):
        """exit_code is 2 when scanners errored."""
        report = _make_report()
        data = json.loads(report.to_json())
        # _make_report adds a scanner error
        assert data["exit_code"] == 2

    def test_exit_code_clean(self):
        """exit_code is 0 for clean scan."""
        report = ScanReport()
        data = json.loads(report.to_json())
        assert data["exit_code"] == 0


# ---------------------------------------------------------------
# Test: finding field names
# ---------------------------------------------------------------

_FINDING_KEYS = {
    "name",
    "passed",
    "severity",
    "message",
    "scanner_name",
    "duration_ms",
    "details",
    "suggested_test",
}


class TestFindingFields:
    """Verify individual finding serialisation."""

    def test_finding_has_all_keys(self):
        """Each finding dict has every required key."""
        report = _make_report()
        data = json.loads(report.to_json())
        for finding in data["findings"]:
            assert _FINDING_KEYS <= set(
                finding.keys()
            )

    def test_no_callable_fields(self):
        """assertion_fn/args/kwargs are excluded."""
        report = _make_report()
        raw = report.to_json()
        assert "assertion_fn" not in raw
        assert "assertion_args" not in raw
        assert "assertion_kwargs" not in raw


# ---------------------------------------------------------------
# Test: Severity serialised as string
# ---------------------------------------------------------------


class TestSeveritySerialisation:
    """Severity enum must become its string value."""

    def test_severity_is_string(self):
        """severity field is a plain string."""
        report = _make_report()
        data = json.loads(report.to_json())
        for finding in data["findings"]:
            assert isinstance(
                finding["severity"], str
            )
            assert finding["severity"] in {
                "critical",
                "warning",
                "info",
            }


# ---------------------------------------------------------------
# Test: numpy type serialisation
# ---------------------------------------------------------------


class TestNumpySerialisation:
    """Numpy scalars and arrays must become native."""

    def test_numpy_float_in_details(self):
        """np.float64 in details becomes float."""
        finding = _make_finding(
            details={
                "metric": np.float64(0.95),
            },
        )
        report = ScanReport(
            findings=[finding],
            scanners_run=["test"],
        )
        data = json.loads(report.to_json())
        val = data["findings"][0]["details"]["metric"]
        assert isinstance(val, float)
        assert val == pytest.approx(0.95)

    def test_numpy_int_in_details(self):
        """np.int64 in details becomes int."""
        finding = _make_finding(
            details={"count": np.int64(42)},
        )
        report = ScanReport(
            findings=[finding],
            scanners_run=["test"],
        )
        data = json.loads(report.to_json())
        val = data["findings"][0]["details"]["count"]
        assert isinstance(val, int)
        assert val == 42

    def test_numpy_array_in_details(self):
        """np.ndarray in details becomes list."""
        arr = np.array([1.0, 2.0, 3.0])
        finding = _make_finding(
            details={"values": arr},
        )
        report = ScanReport(
            findings=[finding],
            scanners_run=["test"],
        )
        data = json.loads(report.to_json())
        val = data["findings"][0]["details"]["values"]
        assert isinstance(val, list)
        assert val == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------
# Test: empty report
# ---------------------------------------------------------------


class TestEmptyReport:
    """Edge case: report with zero findings."""

    def test_empty_findings_list(self):
        """Empty report produces empty findings array."""
        report = ScanReport()
        data = json.loads(report.to_json())
        assert data["findings"] == []
        assert data["scanners_run"] == []
        assert data["exit_code"] == 0
