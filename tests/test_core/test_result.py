"""Tests for mltk.core.result."""

from mltk.core.result import Severity, TestResult, TestSuite


def test_test_result_creation() -> None:
    result = TestResult(
        name="test_example",
        passed=True,
        severity=Severity.CRITICAL,
        message="All good",
    )
    assert result.passed is True
    assert result.severity == Severity.CRITICAL
    assert result.name == "test_example"


def test_test_suite_passed() -> None:
    suite = TestSuite()
    suite.add(
        TestResult(
            name="t1", passed=True, severity=Severity.CRITICAL, message="ok"
        )
    )
    suite.add(
        TestResult(
            name="t2", passed=True, severity=Severity.WARNING, message="ok"
        )
    )
    assert suite.passed is True
    assert suite.total == 2
    assert suite.score == 100.0


def test_test_suite_failed_critical() -> None:
    suite = TestSuite()
    suite.add(
        TestResult(
            name="t1", passed=False, severity=Severity.CRITICAL, message="fail"
        )
    )
    suite.add(
        TestResult(
            name="t2", passed=True, severity=Severity.WARNING, message="ok"
        )
    )
    assert suite.passed is False
    assert suite.failed_count == 1


def test_test_suite_warning_only_still_passes() -> None:
    suite = TestSuite()
    suite.add(
        TestResult(
            name="t1", passed=True, severity=Severity.CRITICAL, message="ok"
        )
    )
    suite.add(
        TestResult(
            name="t2", passed=False, severity=Severity.WARNING, message="warn"
        )
    )
    assert suite.passed is True


def test_empty_suite_score() -> None:
    suite = TestSuite()
    assert suite.score == 0.0
