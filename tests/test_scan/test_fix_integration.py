"""Integration tests for the Fix Suggestion Engine pipeline.

End-to-end tests verifying scanner _gen_fix() -> ScanFinding ->
ScanReport -> JSON/console/suite output.
"""
from __future__ import annotations

import ast
import json

from mltk.core.result import Severity, TestResult
from mltk.scan.console import format_console_output
from mltk.scan.engine import ScanReport
from mltk.scan.finding import FixSuggestion, ScanFinding
from mltk.scan.scanners.bias import BiasScanner
from mltk.scan.scanners.calibration import CalibrationScanner
from mltk.scan.scanners.data import DataScanner
from mltk.scan.scanners.drift import DriftScanner
from mltk.scan.scanners.leakage import LeakageScanner
from mltk.scan.scanners.overfit import OverfitScanner
from mltk.scan.scanners.robustness import RobustnessScanner
from mltk.scan.scanners.slice import SliceScanner

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _make_result(**overrides):
    """Build a TestResult with sensible defaults."""
    defaults = {
        "name": "test_check",
        "passed": False,
        "severity": Severity.WARNING,
        "message": "issue detected",
    }
    defaults.update(overrides)
    return TestResult(**defaults)


def _make_fix(**overrides):
    """Build a FixSuggestion with sensible defaults."""
    defaults = {
        "category": "code",
        "title": "Apply fix",
        "description": "A concrete remediation step.",
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
        "scanner_name": "test_scanner",
        "suggested_fixes": fixes if fixes is not None else [],
    }
    defaults.update(overrides)
    return ScanFinding(**defaults)


def _make_report(findings=None, **overrides):
    """Build a ScanReport with sensible defaults."""
    defaults = {
        "findings": findings or [],
        "scanners_run": ["drift"],
        "model_type": "classifier",
        "n_samples": 500,
        "n_features": 8,
        "duration_ms": 25.0,
    }
    defaults.update(overrides)
    return ScanReport(**defaults)


# ---------------------------------------------------------------
# Scanner _gen_fix() smoke tests
# ---------------------------------------------------------------


class TestScannerGenFixSmoke:
    """Each scanner's _gen_fix() returns valid FixSuggestion lists."""

    def test_drift_gen_fix_returns_valid_fixes(self) -> None:
        # SCENARIO: Call DriftScanner._gen_fix() with typical args.
        # WHY: Verify the static method returns non-empty list of
        #   FixSuggestion objects with valid fields.
        # EXPECTED: list[FixSuggestion] with >= 1 item, each having
        #   non-empty title, description, and valid category/confidence.
        fixes = DriftScanner._gen_fix(
            col="age", method="ks", details={},
        )
        assert len(fixes) >= 1
        for fix in fixes:
            assert isinstance(fix, FixSuggestion)
            assert fix.title
            assert fix.description
            assert fix.category in (
                "code", "config", "data", "process",
            )
            assert fix.confidence in (
                "high", "medium", "low",
            )

    def test_bias_gen_fix_returns_valid_fixes(self) -> None:
        # SCENARIO: Call BiasScanner._gen_fix() with typical args.
        # WHY: Confirm bias scanner fix generation works end-to-end.
        # EXPECTED: Non-empty list with valid FixSuggestion objects.
        fixes = BiasScanner._gen_fix(
            col="gender", method="chi2", details={},
        )
        assert len(fixes) >= 1
        for fix in fixes:
            assert isinstance(fix, FixSuggestion)
            assert fix.title
            assert fix.category in (
                "code", "config", "data", "process",
            )

    def test_overfit_gen_fix_returns_valid_fixes(self) -> None:
        # SCENARIO: Call OverfitScanner._gen_fix() with a gap value.
        # WHY: Confirm overfit scanner fix generation works.
        # EXPECTED: Non-empty list with valid FixSuggestion objects.
        fixes = OverfitScanner._gen_fix(gap=0.15)
        assert len(fixes) >= 1
        for fix in fixes:
            assert isinstance(fix, FixSuggestion)
            assert fix.title
            assert fix.confidence in (
                "high", "medium", "low",
            )

    def test_data_gen_fix_returns_valid_fixes(self) -> None:
        # SCENARIO: Call DataScanner._gen_null_fix() with column info.
        # WHY: Confirm data quality scanner fix generation works.
        # EXPECTED: Non-empty list with valid FixSuggestion objects.
        fixes = DataScanner._gen_null_fix(
            col="income",
            details={"null_count": 42, "total_rows": 1000},
        )
        assert len(fixes) >= 1
        for fix in fixes:
            assert isinstance(fix, FixSuggestion)
            assert fix.title
            assert fix.description


# ---------------------------------------------------------------
# Full pipeline: finding + fixes -> to_json -> parse back
# ---------------------------------------------------------------


class TestFullPipelineJsonRoundtrip:
    """Finding with fixes round-trips through to_json correctly."""

    def test_finding_fixes_survive_json_roundtrip(self) -> None:
        # SCENARIO: Create a ScanReport with a finding that has 2
        #   fixes, serialize to JSON, parse back, and verify
        #   the full structure.
        # WHY: Ensures the entire pipeline from dataclass to JSON
        #   preserves fix data without loss.
        # EXPECTED: Parsed JSON has correct fix count, categories,
        #   titles, confidence levels, and code snippets.
        fix_a = FixSuggestion(
            category="code",
            title="Add regularization",
            description="Reduce overfitting via L2.",
            confidence="high",
            code_snippet="model.l2 = 0.01",
        )
        fix_b = FixSuggestion(
            category="process",
            title="Add cross-validation",
            description="Validate with k-fold CV.",
            confidence="medium",
        )
        result = _make_result(
            name="overfit_check",
            passed=False,
            severity=Severity.CRITICAL,
            message="Train/test gap too large",
        )
        finding = _make_finding(
            fixes=[fix_a, fix_b],
            result=result,
            scanner_name="overfit",
            suggested_test="def test_overfit(): pass",
        )
        report = _make_report(
            findings=[finding],
            scanners_run=["overfit"],
            model_type="classifier",
            n_samples=2000,
            n_features=12,
            duration_ms=55.0,
        )

        raw = report.to_json()
        data = json.loads(raw)

        # Top-level metadata
        assert data["model_type"] == "classifier"
        assert data["n_samples"] == 2000
        assert data["n_features"] == 12

        # Finding round-trip
        assert len(data["findings"]) == 1
        f = data["findings"][0]
        assert f["name"] == "overfit_check"
        assert f["passed"] is False
        assert f["severity"] == "critical"
        assert f["scanner_name"] == "overfit"

        # Fix round-trip
        assert len(f["suggested_fixes"]) == 2
        fx0 = f["suggested_fixes"][0]
        assert fx0["category"] == "code"
        assert fx0["title"] == "Add regularization"
        assert fx0["confidence"] == "high"
        assert fx0["code_snippet"] == "model.l2 = 0.01"

        fx1 = f["suggested_fixes"][1]
        assert fx1["category"] == "process"
        assert fx1["title"] == "Add cross-validation"
        assert fx1["confidence"] == "medium"
        assert fx1["code_snippet"] == ""


# ---------------------------------------------------------------
# Console output tests
# ---------------------------------------------------------------


class TestConsoleOutputFixes:
    """format_console_output() renders fixes correctly."""

    def test_verbose_true_shows_fixes(self) -> None:
        # SCENARIO: Report with finding+fixes, verbose=True.
        # WHY: Users expect to see fix details in verbose mode.
        # EXPECTED: Output contains fix titles and "Suggested fixes"
        #   header.
        fix = _make_fix(
            title="Retrain on recent data",
            confidence="high",
        )
        finding = _make_finding(
            fixes=[fix],
            scanner_name="drift",
        )
        report = _make_report(findings=[finding])

        output = format_console_output(
            report, verbose=True,
        )
        assert "Suggested fixes" in output
        assert "Retrain on recent data" in output

    def test_verbose_false_hides_inline_fixes(self) -> None:
        # SCENARIO: Same report but verbose=False.
        # WHY: Non-verbose mode should suppress inline fix details
        #   but still show the summary fix count.
        # EXPECTED: "Suggested fixes:" header does NOT appear inline,
        #   but summary line still works.
        fix = _make_fix(title="Some inline fix")
        finding = _make_finding(
            fixes=[fix],
            scanner_name="drift",
        )
        report = _make_report(findings=[finding])

        output = format_console_output(
            report, verbose=False,
        )
        assert "Suggested fixes:" not in output
        # The format_console_output does not show the summary()
        # fix count line, so just verify fix title is absent.
        assert "Some inline fix" not in output


# ---------------------------------------------------------------
# Multiple findings with varying fix counts
# ---------------------------------------------------------------


class TestMultipleFindingsFixCounts:
    """Report with mixed fix counts serializes correctly."""

    def test_mixed_fix_counts_in_json(self) -> None:
        # SCENARIO: 3 findings with 0, 2, and 1 fix respectively.
        # WHY: Ensures to_json handles variable fix counts per
        #   finding without off-by-one errors.
        # EXPECTED: JSON has correct fix array lengths for each
        #   finding.
        f0 = _make_finding(
            fixes=[],
            result=_make_result(
                name="clean_check", severity=Severity.INFO,
            ),
            scanner_name="data",
        )
        f1 = _make_finding(
            fixes=[
                _make_fix(title="Fix A"),
                _make_fix(title="Fix B"),
            ],
            result=_make_result(
                name="bias_check",
                severity=Severity.WARNING,
            ),
            scanner_name="bias",
        )
        f2 = _make_finding(
            fixes=[_make_fix(title="Fix C")],
            result=_make_result(
                name="drift_check",
                severity=Severity.CRITICAL,
            ),
            scanner_name="drift",
        )
        report = _make_report(
            findings=[f0, f1, f2],
            scanners_run=["data", "bias", "drift"],
        )

        data = json.loads(report.to_json())
        fix_counts = [
            len(f["suggested_fixes"])
            for f in data["findings"]
        ]
        assert fix_counts == [0, 2, 1]

    def test_mixed_fix_counts_in_summary(self) -> None:
        # SCENARIO: Same 3 findings (0 + 2 + 1 = 3 total fixes).
        # WHY: summary() should aggregate fix counts across all
        #   findings.
        # EXPECTED: "Fix suggestions: 3" appears in summary text.
        f0 = _make_finding(fixes=[])
        f1 = _make_finding(
            fixes=[_make_fix(), _make_fix()],
        )
        f2 = _make_finding(fixes=[_make_fix()])
        report = _make_report(findings=[f0, f1, f2])

        text = report.summary()
        assert "Fix suggestions: 3" in text


# ---------------------------------------------------------------
# Suite and test file generation survive fixes
# ---------------------------------------------------------------


class TestFixesSurviveOutputFormats:
    """Fixes don't break to_suite() or to_test_file()."""

    def test_fixes_dont_break_to_suite(self) -> None:
        # SCENARIO: Report with findings+fixes -> to_suite().
        # WHY: The new suggested_fixes field must not interfere
        #   with to_pending() / suite.add() which only use
        #   assertion_fn, assertion_args, assertion_kwargs.
        # EXPECTED: to_suite() returns an MltkSuite without error.
        fix = _make_fix(title="Regularize model")
        finding = _make_finding(
            fixes=[fix],
            assertion_fn=lambda: None,
            assertion_args=(),
            assertion_kwargs={},
        )
        report = _make_report(findings=[finding])

        suite = report.to_suite()
        # Suite created successfully -- it has pending tests
        assert suite is not None

    def test_fixes_dont_break_to_test_file(
        self, tmp_path,
    ) -> None:
        # SCENARIO: Report with findings+fixes+suggested_test ->
        #   to_test_file().
        # WHY: The new field must not corrupt the generated Python
        #   file.
        # EXPECTED: Written file is valid Python (ast.parse
        #   succeeds).
        fix = _make_fix(title="Add threshold check")
        suggested_test = (
            "def test_drift_age():\n"
            "    \"\"\"Age column must not drift.\"\"\"\n"
            "    assert True\n"
        )
        finding = _make_finding(
            fixes=[fix],
            suggested_test=suggested_test,
            scanner_name="drift",
        )
        report = _make_report(findings=[finding])

        out_path = str(tmp_path / "test_generated.py")
        written = report.to_test_file(out_path)

        # File was written
        assert written

        # File is valid Python
        with open(written, encoding="utf-8") as fh:
            code = fh.read()
        ast.parse(code)

        # File contains the test function
        assert "def test_drift_age" in code


# ---------------------------------------------------------------
# Category and confidence coverage across scanners
# ---------------------------------------------------------------


class TestFixCategoryCoverage:
    """All fix categories appear across the built-in scanners."""

    def test_all_four_categories_present(self) -> None:
        # SCENARIO: Collect _gen_fix() output from all 8 scanners.
        # WHY: The engine supports 4 categories (code/config/data/
        #   process). We should verify the built-in scanners
        #   collectively produce all four.
        # EXPECTED: The union of all categories == the full set.
        all_fixes: list[FixSuggestion] = []
        all_fixes.extend(DriftScanner._gen_fix(
            col="x", method="ks", details={},
        ))
        all_fixes.extend(BiasScanner._gen_fix(
            col="gender", method="chi2", details={},
        ))
        all_fixes.extend(OverfitScanner._gen_fix(gap=0.1))
        all_fixes.extend(DataScanner._gen_null_fix(
            col="income",
            details={"null_count": 5, "total_rows": 100},
        ))
        all_fixes.extend(LeakageScanner._gen_fix(
            feature="target_proxy",
            details={"correlation": 0.95},
        ))
        all_fixes.extend(CalibrationScanner._gen_fix(
            details={},
        ))
        all_fixes.extend(RobustnessScanner._gen_fix())
        all_fixes.extend(SliceScanner._gen_fix(
            slice_col="region",
            slice_desc="region=West",
            metric_val=0.65,
            threshold=0.8,
        ))

        categories = {fix.category for fix in all_fixes}
        assert categories == {
            "code", "config", "data", "process",
        }

    def test_all_three_confidence_levels_present(self) -> None:
        # SCENARIO: Collect _gen_fix() output from all 8 scanners.
        # WHY: The engine supports 3 confidence levels
        #   (high/medium/low). We should verify the built-in
        #   scanners collectively produce all three.
        # EXPECTED: The union of all confidence levels == full set.
        all_fixes: list[FixSuggestion] = []
        all_fixes.extend(DriftScanner._gen_fix(
            col="x", method="ks", details={},
        ))
        all_fixes.extend(BiasScanner._gen_fix(
            col="gender", method="chi2", details={},
        ))
        all_fixes.extend(OverfitScanner._gen_fix(gap=0.1))
        all_fixes.extend(DataScanner._gen_null_fix(
            col="income",
            details={"null_count": 5, "total_rows": 100},
        ))
        all_fixes.extend(LeakageScanner._gen_fix(
            feature="target_proxy",
            details={"correlation": 0.95},
        ))
        all_fixes.extend(CalibrationScanner._gen_fix(
            details={},
        ))
        all_fixes.extend(RobustnessScanner._gen_fix())
        all_fixes.extend(SliceScanner._gen_fix(
            slice_col="region",
            slice_desc="region=West",
            metric_val=0.65,
            threshold=0.8,
        ))

        confidences = {fix.confidence for fix in all_fixes}
        assert confidences == {"high", "medium", "low"}


# ---------------------------------------------------------------
# Summary fix count aggregation
# ---------------------------------------------------------------


class TestSummaryFixAggregation:
    """summary() aggregates fix counts across findings."""

    def test_summary_aggregates_total_fix_count(self) -> None:
        # SCENARIO: 2 findings with 3 fixes each = 6 total.
        # WHY: The summary footer must correctly sum across all
        #   findings, not just report the first finding's count.
        # EXPECTED: "Fix suggestions: 6" in summary text.
        fixes_a = [
            _make_fix(title="A1"),
            _make_fix(title="A2"),
            _make_fix(title="A3"),
        ]
        fixes_b = [
            _make_fix(title="B1"),
            _make_fix(title="B2"),
            _make_fix(title="B3"),
        ]
        f0 = _make_finding(
            fixes=fixes_a, scanner_name="drift",
        )
        f1 = _make_finding(
            fixes=fixes_b, scanner_name="bias",
        )
        report = _make_report(
            findings=[f0, f1],
            scanners_run=["drift", "bias"],
        )

        text = report.summary()
        assert "Fix suggestions: 6" in text

    def test_summary_no_fix_line_when_zero_fixes(self) -> None:
        # SCENARIO: Report where no findings have fixes.
        # WHY: The "Fix suggestions" line should be suppressed
        #   entirely when there are zero fixes, not show "0".
        # EXPECTED: "Fix suggestions" does NOT appear in summary.
        f0 = _make_finding(fixes=[])
        f1 = _make_finding(fixes=[])
        report = _make_report(findings=[f0, f1])

        text = report.summary()
        assert "Fix suggestions" not in text
