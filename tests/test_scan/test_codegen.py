from __future__ import annotations

"""Tests for mltk.scan.codegen -- test file generation.

codegen.py takes ScanFindings and produces a self-contained
pytest file with fixtures, imports, and a test class.  These
tests verify the generated code is syntactically valid Python
and contains the expected structural elements.
"""

import ast
from unittest.mock import MagicMock

import pytest

from mltk.core.result import Severity

try:
    from mltk.scan.codegen import generate_test_file
    _HAS_CODEGEN = True
except ImportError:
    _HAS_CODEGEN = False

pytestmark = pytest.mark.skipif(
    not _HAS_CODEGEN,
    reason="mltk.scan.codegen not yet implemented",
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _mock_finding(
    test_code: str = "def test_x(): assert True",
    scanner: str = "slice",
):
    """Build a mock ScanFinding with suggested_test."""
    f = MagicMock()
    f.suggested_test = test_code
    f.scanner_name = scanner
    result_mock = MagicMock()
    result_mock.name = "scan.test"
    result_mock.passed = False
    result_mock.message = "issue found"
    result_mock.severity = Severity.CRITICAL
    result_mock.details = {}
    result_mock.duration_ms = 0.0
    f.result = result_mock
    return f


# ---------------------------------------------------------------
# Tests
# ---------------------------------------------------------------


class TestGenerateTestFile:
    """generate_test_file() produces valid pytest code."""

    def test_output_parses_as_python(self) -> None:
        """Generated code passes ast.parse() -- no syntax
        errors."""
        findings = [_mock_finding(), _mock_finding()]
        code = generate_test_file(findings)
        # Must not raise SyntaxError
        ast.parse(code)

    def test_contains_import(self) -> None:
        """Generated file includes mltk imports."""
        findings = [_mock_finding()]
        code = generate_test_file(findings)
        assert "import" in code

    def test_contains_test_class_or_function(self) -> None:
        """Generated file has at least one test function
        or class."""
        findings = [_mock_finding()]
        code = generate_test_file(findings)
        has_class = "class Test" in code
        has_func = "def test_" in code
        assert has_class or has_func

    def test_empty_findings_still_valid(self) -> None:
        """Empty findings list produces valid (possibly
        minimal) Python."""
        code = generate_test_file([])
        ast.parse(code)

    def test_multiple_scanners_in_output(self) -> None:
        """Findings from different scanners appear in the
        generated code."""
        findings = [
            _mock_finding(
                "def test_slice(): pass",
                scanner="slice",
            ),
            _mock_finding(
                "def test_bias(): pass",
                scanner="bias",
            ),
        ]
        code = generate_test_file(findings)
        ast.parse(code)
        # Both test functions should be in the output
        assert "test_slice" in code
        assert "test_bias" in code
