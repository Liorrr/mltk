from __future__ import annotations

"""Tests for mltk.scan.finding -- ScanFinding dataclass.

ScanFinding carries both the evidence (TestResult) and the
reproduction recipe (assertion_fn + args).  These tests
verify field storage and the to_pending() conversion used
by MltkSuite.add().
"""

from dataclasses import fields
from unittest.mock import MagicMock

import pytest

try:
    from mltk.scan.finding import ScanFinding
except ImportError:
    ScanFinding = None  # type: ignore[assignment,misc]

pytestmark = pytest.mark.skipif(
    ScanFinding is None,
    reason="mltk.scan.finding not yet implemented",
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _make_finding(**overrides):
    """Build a ScanFinding with sensible defaults."""
    defaults = {
        "result": MagicMock(
            name="mock_result",
            passed=False,
            message="test issue",
        ),
        "assertion_fn": lambda y, p: None,
        "assertion_args": (1, 2),
        "assertion_kwargs": {"metric": "accuracy"},
        "suggested_test": "def test_x(): pass",
        "scanner_name": "slice",
    }
    defaults.update(overrides)
    return ScanFinding(**defaults)


# ---------------------------------------------------------------
# Tests
# ---------------------------------------------------------------


class TestScanFindingFields:
    """ScanFinding stores all required attributes."""

    def test_has_required_fields(self) -> None:
        """All six fields declared in the plan exist."""
        names = {f.name for f in fields(ScanFinding)}
        expected = {
            "result",
            "assertion_fn",
            "assertion_args",
            "assertion_kwargs",
            "suggested_test",
            "scanner_name",
        }
        assert expected.issubset(names)

    def test_stores_result(self) -> None:
        """Result field is preserved on construction."""
        mock_result = MagicMock(passed=False)
        finding = _make_finding(result=mock_result)
        assert finding.result is mock_result

    def test_stores_scanner_name(self) -> None:
        """Scanner name is stored verbatim."""
        finding = _make_finding(scanner_name="bias")
        assert finding.scanner_name == "bias"

    def test_stores_suggested_test(self) -> None:
        """Suggested test code string is preserved."""
        code = "def test_generated(): assert True"
        finding = _make_finding(suggested_test=code)
        assert finding.suggested_test == code


class TestScanFindingToPending:
    """to_pending() returns the tuple MltkSuite.add() needs."""

    def test_returns_tuple(self) -> None:
        """to_pending() gives (fn, args, kwargs)."""
        fn = MagicMock()
        finding = _make_finding(
            assertion_fn=fn,
            assertion_args=(10, 20),
            assertion_kwargs={"k": "v"},
        )
        result = finding.to_pending()
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_tuple_contents(self) -> None:
        """Tuple carries the correct fn, args, kwargs."""
        fn = MagicMock()
        args = (1, 2, 3)
        kwargs = {"threshold": 0.8}
        finding = _make_finding(
            assertion_fn=fn,
            assertion_args=args,
            assertion_kwargs=kwargs,
        )
        got_fn, got_args, got_kwargs = finding.to_pending()
        assert got_fn is fn
        assert got_args == args
        assert got_kwargs == kwargs
