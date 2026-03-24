"""Tests for mltk.core.assertion."""

import pytest

from mltk.core.assertion import MltkAssertionError, assert_true
from mltk.core.result import Severity


def test_assert_true_passes() -> None:
    result = assert_true(True, name="check", message="ok")
    assert result.passed is True


def test_assert_true_fails_critical() -> None:
    with pytest.raises(MltkAssertionError) as exc_info:
        assert_true(False, name="check", message="failed")
    assert exc_info.value.result.passed is False
    assert exc_info.value.result.severity == Severity.CRITICAL


def test_assert_true_warning_does_not_raise() -> None:
    result = assert_true(False, name="check", message="warn", severity=Severity.WARNING)
    assert result.passed is False
    assert result.severity == Severity.WARNING


def test_assert_true_carries_details() -> None:
    result = assert_true(True, name="check", message="ok", score=0.95, threshold=0.9)
    assert result.details["score"] == 0.95
    assert result.details["threshold"] == 0.9
