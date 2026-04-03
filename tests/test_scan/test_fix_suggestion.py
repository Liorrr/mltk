"""Tests for FixSuggestion and its integration with ScanFinding.

FixSuggestion is a concrete remediation step attached to scan
findings.  These tests verify field storage, defaults, dict
conversion, and integration with ScanFinding (including that
to_pending() is unaffected by the new field).
"""
from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock

from mltk.core.result import Severity, TestResult
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
        "message": "msg",
    }
    defaults.update(overrides)
    return TestResult(**defaults)


def _make_fix(**overrides):
    """Build a FixSuggestion with sensible defaults."""
    defaults = {
        "category": "code",
        "title": "Add threshold check",
        "description": "Insert a validation step before inference.",
        "confidence": "high",
    }
    defaults.update(overrides)
    return FixSuggestion(**defaults)


def _make_finding(**overrides):
    """Build a ScanFinding with sensible defaults."""
    defaults = {
        "result": _make_result(),
        "assertion_fn": lambda y, p: None,
        "assertion_args": (1, 2),
        "assertion_kwargs": {"metric": "accuracy"},
        "suggested_test": "def test_x(): pass",
        "scanner_name": "slice",
    }
    defaults.update(overrides)
    return ScanFinding(**defaults)


# ---------------------------------------------------------------
# FixSuggestion unit tests
# ---------------------------------------------------------------


class TestFixSuggestionCreation:
    """FixSuggestion stores all fields correctly."""

    def test_fix_suggestion_creation(self) -> None:
        """Basic creation with all fields populates attributes."""
        fix = FixSuggestion(
            category="code",
            title="Increase threshold",
            description="Raise the accuracy threshold to 0.8.",
            confidence="high",
            code_snippet="threshold = 0.8",
        )
        assert fix.category == "code"
        assert fix.title == "Increase threshold"
        assert fix.description == "Raise the accuracy threshold to 0.8."
        assert fix.confidence == "high"
        assert fix.code_snippet == "threshold = 0.8"

    def test_fix_suggestion_code_snippet_default(self) -> None:
        """code_snippet defaults to empty string when omitted."""
        fix = _make_fix()
        assert fix.code_snippet == ""

    def test_fix_suggestion_category_values(self) -> None:
        """All four category values are accepted."""
        for cat in ("code", "config", "data", "process"):
            fix = _make_fix(category=cat)
            assert fix.category == cat

    def test_fix_suggestion_confidence_values(self) -> None:
        """All three confidence levels are accepted."""
        for level in ("high", "medium", "low"):
            fix = _make_fix(confidence=level)
            assert fix.confidence == level


class TestFixSuggestionSerialization:
    """FixSuggestion converts to dict and has useful repr."""

    def test_fix_suggestion_to_dict(self) -> None:
        """dataclasses.asdict produces the expected dict."""
        fix = FixSuggestion(
            category="config",
            title="Set batch size",
            description="Lower batch size to reduce memory.",
            confidence="medium",
            code_snippet="batch_size = 16",
        )
        d = dataclasses.asdict(fix)
        assert d == {
            "category": "config",
            "title": "Set batch size",
            "description": "Lower batch size to reduce memory.",
            "confidence": "medium",
            "code_snippet": "batch_size = 16",
        }

    def test_fix_suggestion_with_code(self) -> None:
        """code_snippet field stores multi-line code."""
        snippet = "model.compile(\n    optimizer='adam',\n    loss='mse',\n)"
        fix = _make_fix(code_snippet=snippet)
        assert fix.code_snippet == snippet
        assert "optimizer" in fix.code_snippet

    def test_fix_suggestion_repr(self) -> None:
        """repr includes the title for easy identification."""
        fix = _make_fix(title="Resample training data")
        r = repr(fix)
        assert "Resample training data" in r


# ---------------------------------------------------------------
# ScanFinding integration tests
# ---------------------------------------------------------------


class TestScanFindingWithFixes:
    """ScanFinding.suggested_fixes integrates correctly."""

    def test_scan_finding_suggested_fixes_default(self) -> None:
        """suggested_fixes defaults to an empty list."""
        finding = _make_finding()
        assert finding.suggested_fixes == []
        assert isinstance(finding.suggested_fixes, list)

    def test_scan_finding_with_fixes(self) -> None:
        """Attaching two fixes to a finding preserves them."""
        fix_a = _make_fix(title="Fix A", confidence="high")
        fix_b = _make_fix(title="Fix B", confidence="low")
        finding = _make_finding(suggested_fixes=[fix_a, fix_b])
        assert len(finding.suggested_fixes) == 2
        assert finding.suggested_fixes[0].title == "Fix A"
        assert finding.suggested_fixes[1].title == "Fix B"

    def test_scan_finding_fixes_preserved_in_to_pending(self) -> None:
        """to_pending() still returns (fn, args, kwargs) with fixes present."""
        fn = MagicMock()
        args = (10, 20)
        kwargs = {"threshold": 0.9}
        fix = _make_fix(title="Some fix")
        finding = _make_finding(
            assertion_fn=fn,
            assertion_args=args,
            assertion_kwargs=kwargs,
            suggested_fixes=[fix],
        )
        got_fn, got_args, got_kwargs = finding.to_pending()
        assert got_fn is fn
        assert got_args == args
        assert got_kwargs == kwargs

    def test_multiple_fixes_different_categories(self) -> None:
        """A finding can hold fixes across all four categories."""
        fixes = [
            _make_fix(category="code", title="Code fix"),
            _make_fix(category="config", title="Config fix"),
            _make_fix(category="data", title="Data fix"),
            _make_fix(category="process", title="Process fix"),
        ]
        finding = _make_finding(suggested_fixes=fixes)
        categories = [f.category for f in finding.suggested_fixes]
        assert categories == ["code", "config", "data", "process"]

    def test_fix_confidence_ordering(self) -> None:
        """Fixes can be sorted by confidence (high first)."""
        order = {"high": 0, "medium": 1, "low": 2}
        fixes = [
            _make_fix(confidence="low", title="Low"),
            _make_fix(confidence="high", title="High"),
            _make_fix(confidence="medium", title="Medium"),
        ]
        sorted_fixes = sorted(fixes, key=lambda f: order[f.confidence])
        assert sorted_fixes[0].confidence == "high"
        assert sorted_fixes[1].confidence == "medium"
        assert sorted_fixes[2].confidence == "low"
