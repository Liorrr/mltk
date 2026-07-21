"""Tests for the shared on_empty policy helpers (S102).

These helpers replace six drifted per-module copies. The load-bearing property
is that the ``severity`` default (CRITICAL) reproduces BOTH prior variants:
the four copies that hardcoded CRITICAL, and the two that parameterised it.
"""
from __future__ import annotations

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.core.empty import (
    ON_EMPTY_OPTIONS,
    empty_input_result,
    unknown_on_empty_result,
)
from mltk.core.result import Severity


class TestOnEmptyOptions:
    def test_options_match_documented_policy(self):
        assert ON_EMPTY_OPTIONS == ("fail", "skip", "pass")


class TestUnknownOnEmptyResult:
    def test_default_severity_critical_raises(self):
        # Reproduces the five copies that hardcoded Severity.CRITICAL.
        with pytest.raises(MltkAssertionError):
            unknown_on_empty_result("x.test", "bogus")

    def test_non_critical_severity_returns_failed_result(self):
        # Reproduces data/synthetic.py, which passed severity through.
        result = unknown_on_empty_result("x.test", "bogus", Severity.WARNING)
        assert result.passed is False
        assert result.severity is Severity.WARNING

    def test_message_lists_supported_options(self):
        result = unknown_on_empty_result("x.test", "bogus", Severity.WARNING)
        assert "bogus" in result.message
        for option in ON_EMPTY_OPTIONS:
            assert option in result.message

    def test_records_on_empty_detail(self):
        result = unknown_on_empty_result("x.test", "bogus", Severity.WARNING)
        assert result.details["on_empty"] == "bogus"


class TestEmptyInputResultFail:
    def test_fail_with_default_critical_raises(self):
        with pytest.raises(MltkAssertionError):
            empty_input_result(
                name="x.test",
                reason="no samples",
                on_empty="fail",
                legacy_message="legacy",
            )

    def test_fail_with_warning_returns_failed_result(self):
        result = empty_input_result(
            name="x.test",
            reason="no samples",
            on_empty="fail",
            legacy_message="legacy",
            severity=Severity.WARNING,
        )
        assert result.passed is False
        assert result.severity is Severity.WARNING
        assert "no samples" in result.message
        assert "empty input is not allowed" in result.message


class TestEmptyInputResultSkip:
    def test_skip_passes_as_info(self):
        result = empty_input_result(
            name="x.test",
            reason="no samples",
            on_empty="skip",
            legacy_message="legacy",
        )
        assert result.passed is True
        assert result.severity is Severity.INFO
        assert result.details["skipped"] is True
        assert result.details["reason"] == "no samples"
        assert "Skipped" in result.message

    def test_skip_ignores_caller_severity(self):
        # skip is always INFO regardless of the configured severity.
        result = empty_input_result(
            name="x.test",
            reason="no samples",
            on_empty="skip",
            legacy_message="legacy",
            severity=Severity.WARNING,
        )
        assert result.severity is Severity.INFO


class TestEmptyInputResultLegacyPass:
    def test_pass_uses_legacy_message_and_details(self):
        result = empty_input_result(
            name="x.test",
            reason="no samples",
            on_empty="pass",
            legacy_message="legacy message",
            score=0.0,
            count=0,
        )
        assert result.passed is True
        assert result.message == "legacy message"
        assert result.details["score"] == 0.0
        assert result.details["count"] == 0

    def test_pass_default_severity_is_critical(self):
        # Passing assertions never raise, so CRITICAL is observable here.
        result = empty_input_result(
            name="x.test",
            reason="no samples",
            on_empty="pass",
            legacy_message="legacy",
        )
        assert result.severity is Severity.CRITICAL

    def test_pass_honours_explicit_severity(self):
        result = empty_input_result(
            name="x.test",
            reason="no samples",
            on_empty="pass",
            legacy_message="legacy",
            severity=Severity.WARNING,
        )
        assert result.severity is Severity.WARNING
