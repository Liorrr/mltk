"""Tests for Jupyter notebook integration — _repr_html_ and display_report."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from mltk.core.result import Severity, TestResult, TestSuite

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    name: str = "test_example",
    passed: bool = True,
    message: str = "All good",
    severity: Severity = Severity.CRITICAL,
    duration_ms: float = 12.5,
    details: dict | None = None,
) -> TestResult:
    return TestResult(
        name=name,
        passed=passed,
        severity=severity,
        message=message,
        duration_ms=duration_ms,
        details=details or {},
    )


def _make_suite(*results: TestResult) -> TestSuite:
    suite = TestSuite()
    for r in results:
        suite.add(r)
    return suite


# ---------------------------------------------------------------------------
# TestResult._repr_html_
# ---------------------------------------------------------------------------

class TestResultReprHtmlPass:
    # SCENARIO: A passing TestResult renders its HTML representation.
    # WHY: Jupyter calls _repr_html_ automatically; we must confirm the badge
    #      colour and label reflect a passing state.
    # EXPECTED: HTML contains the text "PASS" and the green colour hex #22c55e.

    def test_result_repr_html_pass(self):
        result = _make_result(passed=True, name="accuracy_check", message="Accuracy within bounds")
        html = result._repr_html_()
        assert "PASS" in html
        assert "#22c55e" in html

    def test_result_repr_html_pass_contains_name(self):
        # SCENARIO: The test name must appear in the output.
        # WHY: Users need to identify which assertion they are looking at.
        # EXPECTED: The name string is present in the rendered HTML.
        result = _make_result(passed=True, name="my_special_test")
        html = result._repr_html_()
        assert "my_special_test" in html

    def test_result_repr_html_pass_contains_duration(self):
        # SCENARIO: Duration in milliseconds is displayed.
        # WHY: Performance diagnostics depend on duration visibility.
        # EXPECTED: The formatted duration value appears in the HTML.
        result = _make_result(passed=True, duration_ms=42.75)
        html = result._repr_html_()
        assert "42.75" in html

    def test_result_repr_html_pass_contains_severity(self):
        # SCENARIO: Severity level is shown on the result card.
        # WHY: Users need severity at a glance to triage failures.
        # EXPECTED: Severity value string is present.
        result = _make_result(passed=True, severity=Severity.WARNING)
        html = result._repr_html_()
        assert "warning" in html


class TestResultReprHtmlFail:
    # SCENARIO: A failing TestResult renders its HTML representation.
    # WHY: Failed assertions must be visually distinct — red badge and colour.
    # EXPECTED: HTML contains the text "FAIL" and the red colour hex #ef4444.

    def test_result_repr_html_fail(self):
        result = _make_result(passed=False, name="drift_check", message="Drift detected")
        html = result._repr_html_()
        assert "FAIL" in html
        assert "#ef4444" in html

    def test_result_repr_html_fail_does_not_contain_pass_badge(self):
        # SCENARIO: A failing result must not show a PASS badge.
        # WHY: Showing the wrong status would mislead notebook users.
        # EXPECTED: The word "PASS" does not appear as a badge label.
        #           (The dark theme bg colour #22c55e should also be absent
        #           from the badge span — we check the badge label directly.)
        result = _make_result(passed=False)
        html = result._repr_html_()
        # "PASS" must not appear as a status word in the badge
        # We look for the badge pattern specifically
        assert ">PASS<" not in html

    def test_result_repr_html_fail_contains_message(self):
        # SCENARIO: The failure message is embedded in the HTML.
        # WHY: The message is the primary human-readable failure description.
        # EXPECTED: The exact message string appears in the output.
        result = _make_result(passed=False, message="KL divergence too high")
        html = result._repr_html_()
        assert "KL divergence too high" in html


class TestResultReprHtmlDetails:
    # SCENARIO: A TestResult with a details dict renders a key-value table.
    # WHY: Assertion details (thresholds, actual values) aid debugging.
    # EXPECTED: Both key and value strings appear in the rendered HTML.

    def test_result_repr_html_details_keys_and_values(self):
        result = _make_result(
            passed=False,
            details={"threshold": 0.05, "actual": 0.12, "metric": "psi"},
        )
        html = result._repr_html_()
        assert "threshold" in html
        assert "0.05" in html
        assert "actual" in html
        assert "0.12" in html
        assert "metric" in html
        assert "psi" in html

    def test_result_repr_html_no_details_no_table(self):
        # SCENARIO: A result with no details dict should not emit a table.
        # WHY: An empty table wastes space and looks broken.
        # EXPECTED: The <table> tag is absent when details is empty.
        result = _make_result(details={})
        html = result._repr_html_()
        assert "<table" not in html

    def test_result_repr_html_returns_string(self):
        # SCENARIO: _repr_html_ always returns a str, not None or bytes.
        # WHY: IPython's display machinery expects a str from _repr_html_.
        # EXPECTED: Return type is str.
        result = _make_result()
        assert isinstance(result._repr_html_(), str)


# ---------------------------------------------------------------------------
# TestSuite._repr_html_
# ---------------------------------------------------------------------------

class TestSuiteReprHtmlSummary:
    # SCENARIO: A mixed suite renders a summary header with pass/fail counts.
    # WHY: The headline metric is the most important signal in the report.
    # EXPECTED: The fraction "2/3" and percentage "66.7%" appear in the HTML.

    def test_suite_repr_html_summary(self):
        suite = _make_suite(
            _make_result(name="a", passed=True),
            _make_result(name="b", passed=True),
            _make_result(name="c", passed=False),
        )
        html = suite._repr_html_()
        assert "2/3" in html
        assert "66.7" in html

    def test_suite_repr_html_pass_fail_badges(self):
        # SCENARIO: Individual pass and fail count badges are shown.
        # WHY: Badges give a quick colour-coded tally at a glance.
        # EXPECTED: "2 passed" and "1 failed" appear in the HTML.
        suite = _make_suite(
            _make_result(name="x", passed=True),
            _make_result(name="y", passed=True),
            _make_result(name="z", passed=False),
        )
        html = suite._repr_html_()
        assert "2 passed" in html
        assert "1 failed" in html

    def test_suite_repr_html_total_duration(self):
        # SCENARIO: The sum of all result durations is displayed.
        # WHY: Total runtime helps identify slow test suites.
        # EXPECTED: The summed duration value is present in the HTML.
        suite = _make_suite(
            _make_result(name="p", passed=True, duration_ms=10.0),
            _make_result(name="q", passed=False, duration_ms=20.0),
        )
        html = suite._repr_html_()
        assert "30.00" in html

    def test_suite_repr_html_all_passed_green_border(self):
        # SCENARIO: An all-pass suite uses the green accent on the header border.
        # WHY: The border colour signals overall suite health at a glance.
        # EXPECTED: The green hex #22c55e appears as the border colour.
        suite = _make_suite(
            _make_result(name="ok1", passed=True),
            _make_result(name="ok2", passed=True),
        )
        html = suite._repr_html_()
        assert "#22c55e" in html

    def test_suite_repr_html_any_failed_red_border(self):
        # SCENARIO: A suite with at least one critical failure uses a red border.
        # WHY: Red immediately flags the suite as unhealthy.
        # EXPECTED: The red hex #ef4444 appears as the border colour.
        suite = _make_suite(
            _make_result(name="ok", passed=True),
            _make_result(name="bad", passed=False, severity=Severity.CRITICAL),
        )
        html = suite._repr_html_()
        assert "#ef4444" in html


class TestSuiteReprHtmlTable:
    # SCENARIO: The suite table contains one row per result, with names visible.
    # WHY: The table is the main body of the report; missing rows = missing data.
    # EXPECTED: All test names appear somewhere in the rendered HTML.

    def test_suite_repr_html_table(self):
        names = ["drift_psi", "accuracy_check", "bias_score"]
        results = [_make_result(name=n, passed=(i % 2 == 0)) for i, n in enumerate(names)]
        suite = _make_suite(*results)
        html = suite._repr_html_()
        for name in names:
            assert name in html, f"Expected '{name}' in suite HTML"

    def test_suite_repr_html_table_contains_pass_and_fail(self):
        # SCENARIO: Mixed suite table rows carry both PASS and FAIL labels.
        # WHY: Each row must reflect its own status, not the aggregate.
        # EXPECTED: Both "PASS" and "FAIL" text appear in the table section.
        suite = _make_suite(
            _make_result(name="a", passed=True),
            _make_result(name="b", passed=False),
        )
        html = suite._repr_html_()
        assert "PASS" in html
        assert "FAIL" in html

    def test_suite_repr_html_empty_suite(self):
        # SCENARIO: An empty suite produces valid HTML without errors.
        # WHY: Edge case — suites can be empty before any assertions run.
        # EXPECTED: Returns a string containing the "0/0" summary and no crash.
        suite = TestSuite()
        html = suite._repr_html_()
        assert isinstance(html, str)
        assert "0/0" in html

    def test_suite_repr_html_returns_string(self):
        # SCENARIO: _repr_html_ always returns a str.
        # WHY: IPython requires a str; None or other types break display.
        # EXPECTED: Return type is str.
        suite = _make_suite(_make_result())
        assert isinstance(suite._repr_html_(), str)


# ---------------------------------------------------------------------------
# display_report — fallback without IPython
# ---------------------------------------------------------------------------

class TestDisplayReportNoIPython:
    # SCENARIO: display_report is called in an environment without IPython.
    # WHY: mltk must work in plain Python scripts, not just Jupyter.
    # EXPECTED: Falls back to print output without raising an exception.

    def test_display_report_no_ipython(self, capsys):
        suite = _make_suite(
            _make_result(name="plain_test", passed=True, message="OK"),
        )
        # Temporarily hide IPython from the import system
        with patch.dict(sys.modules, {"IPython": None, "IPython.display": None}):
            from mltk.jupyter import display_report  # noqa: PLC0415
            display_report(suite)

        captured = capsys.readouterr()
        assert "1/1" in captured.out
        assert "plain_test" in captured.out

    def test_display_report_no_ipython_fail_listed(self, capsys):
        # SCENARIO: Fallback output lists failing tests with [FAIL] prefix.
        # WHY: Users running outside Jupyter must still see which tests failed.
        # EXPECTED: "[FAIL]" and the failed test name appear in stdout.
        suite = _make_suite(
            _make_result(name="bad_test", passed=False, message="Drift too high"),
        )
        with patch.dict(sys.modules, {"IPython": None, "IPython.display": None}):
            from mltk.jupyter import display_report  # noqa: PLC0415
            display_report(suite)

        captured = capsys.readouterr()
        assert "[FAIL]" in captured.out
        assert "bad_test" in captured.out

    def test_display_report_no_ipython_no_crash_empty_suite(self, capsys):
        # SCENARIO: Fallback with an empty suite does not crash.
        # WHY: Edge-case robustness — empty suites must not raise exceptions.
        # EXPECTED: Function returns without error; stdout contains "0/0".
        suite = TestSuite()
        with patch.dict(sys.modules, {"IPython": None, "IPython.display": None}):
            from mltk.jupyter import display_report  # noqa: PLC0415
            display_report(suite)

        captured = capsys.readouterr()
        assert "0/0" in captured.out

    def test_display_report_with_ipython_calls_display(self):
        # SCENARIO: When IPython is available, display(HTML(...)) is called.
        # WHY: In a real notebook the rich HTML path must be exercised.
        # EXPECTED: IPython.display.display is called exactly once.
        suite = _make_suite(_make_result(name="nb_test", passed=True))

        mock_display = MagicMock()
        mock_html = MagicMock(side_effect=lambda x: x)  # HTML(x) returns x
        mock_ipython_display = MagicMock()
        mock_ipython_display.display = mock_display
        mock_ipython_display.HTML = mock_html

        mock_ipython = MagicMock()
        mock_ipython.display = mock_ipython_display

        modules = {"IPython": mock_ipython, "IPython.display": mock_ipython_display}
        with patch.dict(sys.modules, modules):
            # Re-import to pick up the patched module
            import importlib

            import mltk.jupyter as jup_mod
            importlib.reload(jup_mod)
            jup_mod.display_report(suite)

        mock_display.assert_called_once()

    def test_display_report_no_ipython_pass_listed(self, capsys):
        # SCENARIO: Fallback output shows [PASS] for passing tests.
        # WHY: Users need to confirm which tests passed in plain-text mode.
        # EXPECTED: "[PASS]" appears in stdout alongside the test name.
        suite = _make_suite(
            _make_result(name="good_test", passed=True, message="Within limits"),
        )
        with patch.dict(sys.modules, {"IPython": None, "IPython.display": None}):
            from mltk.jupyter import display_report  # noqa: PLC0415
            display_report(suite)

        captured = capsys.readouterr()
        assert "[PASS]" in captured.out
        assert "good_test" in captured.out
