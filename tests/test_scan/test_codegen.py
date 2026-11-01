"""Tests for mltk.scan.codegen -- test file generation.

codegen.py takes ScanFindings and produces a self-contained
pytest file with fixtures, imports, and a test class.  These
tests verify the generated code is syntactically valid Python
and contains the expected structural elements.
"""

from __future__ import annotations

import ast
from unittest.mock import MagicMock

import pytest

from mltk.core.result import Severity
from mltk.scan.engine import ScanReport

try:
    from mltk.scan.codegen import (
        SNIPPET_PROVIDED_NAMES,
        check_snippet,
        generate_test_file,
    )
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
    test_code: str = "assert True",
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
                "assert True",
                scanner="slice",
            ),
            _mock_finding(
                "assert True",
                scanner="bias",
            ),
        ]
        code = generate_test_file(findings)
        ast.parse(code)
        # Both scanner findings should be represented in the output
        assert "Scanner: slice" in code
        assert "Scanner: bias" in code

    def test_prelude_is_conditional_and_imports_are_exact(self) -> None:
        finding = _mock_finding(
            "assert_metric(y, y_pred, metric='accuracy')\n"
            "assert model_fn(X) is not None\n"
            "assert X_train is not None"
        )
        code = generate_test_file([finding], target_col="label")
        assert "X, y = data" in code
        assert "y_pred = model.predict(X)" in code
        assert "model_fn = model.predict" in code
        assert "X_train, y_train = train_data" in code
        assert "def test_001_scan_test(self, model, data, train_data):" in code
        assert "from mltk import assert_metric" in code
        assert SNIPPET_PROVIDED_NAMES

        plain = generate_test_file([_mock_finding("assert True")])
        assert "y_pred = model.predict(X)" not in plain
        assert "model_fn = model.predict" not in plain
        assert "train_data" not in plain
        assert "from mltk import" not in plain

    @pytest.mark.parametrize(
        ("body", "message"),
        [
            ("def test_nested(): pass", "top-level"),
            ("import pathlib", "import"),
            ("assert (", "syntax"),
            ("assert mystery", "mystery"),
        ],
    )
    def test_check_snippet_rejects_contract_violations(
        self, body: str, message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            check_snippet(body)

    def test_check_snippet_accepts_provided_names(self) -> None:
        check_snippet(
            "X, y = data\n"
            "y_pred = model.predict(X)\n"
            "model_fn(X)\n"
            "assert X_train is not None\n"
            "np.asarray(y)\n"
            "pd.DataFrame(X)\n"
            "pytest.skip('ok')"
        )

    def test_bad_scanner_snippet_fails_before_writing(self, tmp_path) -> None:
        output = tmp_path / "generated.py"
        with pytest.raises(ValueError, match="top-level"):
            generate_test_file(
                [_mock_finding("def test_broken(): pass")],
                output_path=str(output),
            )
        assert not output.exists()

    def test_report_output_matches_direct_codegen(self, tmp_path) -> None:
        findings = [_mock_finding("assert True")]
        expected = generate_test_file(
            findings,
            model_path="model.pkl",
            data_path="data.csv",
            target_col="label",
        )
        output = tmp_path / "generated.py"
        ScanReport(findings=findings).to_test_file(
            str(output),
            model_path="model.pkl",
            data_path="data.csv",
            target_col="label",
        )
        assert output.read_text(encoding="utf-8") == expected
