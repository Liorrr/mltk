"""Tests for the mltk_test MCP tool.

Covers YAML suite parsing, pytest subprocess execution,
error handling, and response structure validation.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from ._helpers import (
    assert_error,
    assert_ok,
    assert_valid_json,
    call_tool,
    call_tool_raw,
)


def _mock_subprocess(returncode=0, stdout="", stderr=""):
    """Build a MagicMock for subprocess.run return value."""
    proc = MagicMock()
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = stderr
    return proc


class TestMltkTest:
    """Tests for the mltk_test tool."""

    def test_yaml_suite_ok(self, tmp_path):
        # SCENARIO: Parse a valid YAML test suite
        # WHY: Core happy path for YAML suites (parse-only)
        # EXPECTED: status=ok, has total and suite name
        yml = tmp_path / "suite.yaml"
        yml.write_text("name: demo\ntests:\n  - name: t1\n")

        result = call_tool("mltk_test", suite_path=str(yml))

        assert_ok(result)
        assert "total" in result
        assert result["total"] == 1
        assert result["suite"] == "demo"

    def test_py_file_ok(self, tmp_path):
        # SCENARIO: Run a .py test file via mocked subprocess
        # WHY: Core happy path for pytest execution
        # EXPECTED: status=ok, has passed and failed counts
        py_file = tmp_path / "test_model.py"
        py_file.write_text("def test_one(): pass\n")

        mock_proc = _mock_subprocess(
            returncode=0,
            stdout="1 passed\n",
        )
        with patch("subprocess.run", return_value=mock_proc):
            result = call_tool(
                "mltk_test", suite_path=str(py_file),
            )

        assert_ok(result)
        assert "passed" in result
        assert "failed" in result

    def test_nonexistent_file_error(self):
        # SCENARIO: Test file does not exist
        # WHY: Should fail before any import with a clear error
        # EXPECTED: status=error, no mock needed
        result = call_tool(
            "mltk_test",
            suite_path="/does/not/exist/test.py",
        )
        assert_error(result)
        assert "not found" in result["error"].lower()

    def test_response_has_pass_fail_counts_yaml(self, tmp_path):
        # SCENARIO: YAML parse-only returns zero counts
        # WHY: YAML path parses but does not execute; counts are 0
        # EXPECTED: passed=0, failed=0
        yml = tmp_path / "counts.yaml"
        yml.write_text("name: countsuite\ntests:\n  - name: a\n")

        result = call_tool("mltk_test", suite_path=str(yml))

        assert_ok(result)
        assert result["passed"] == 0
        assert result["failed"] == 0

    def test_response_has_pass_fail_counts_py(self, tmp_path):
        # SCENARIO: .py execution with mocked subprocess
        # WHY: Parsed stdout determines pass/fail numbers
        # EXPECTED: passed and failed are integers matching stdout
        py_file = tmp_path / "test_counts.py"
        py_file.write_text("pass\n")

        mock_proc = _mock_subprocess(
            returncode=0,
            stdout="3 passed, 1 failed\n",
        )
        with patch("subprocess.run", return_value=mock_proc):
            result = call_tool(
                "mltk_test", suite_path=str(py_file),
            )

        assert_ok(result)
        assert result["passed"] == 3
        assert result["failed"] == 1

    def test_verbose_yaml(self, tmp_path):
        # SCENARIO: Parse YAML with verbose=True
        # WHY: verbose=True should include per-test results list
        # EXPECTED: results is a non-empty list
        yml = tmp_path / "verbose.yaml"
        yml.write_text(
            "name: vsuite\ntests:\n"
            "  - name: t1\n  - name: t2\n"
        )

        result = call_tool(
            "mltk_test",
            suite_path=str(yml),
            verbose=True,
        )

        assert_ok(result)
        assert isinstance(result["results"], list)
        assert len(result["results"]) == 2

    def test_has_suggested_next_step(self, tmp_path):
        # SCENARIO: Any successful test response
        # WHY: All ok responses must include suggested_next_step
        # EXPECTED: suggested_next_step is a non-empty string
        yml = tmp_path / "hint.yaml"
        yml.write_text("name: hint\ntests:\n  - name: x\n")

        result = call_tool("mltk_test", suite_path=str(yml))

        assert_ok(result)
        assert "suggested_next_step" in result
        assert isinstance(result["suggested_next_step"], str)
        assert len(result["suggested_next_step"]) > 0

    def test_all_passing_suggests_scan(self, tmp_path):
        # SCENARIO: All tests pass in .py execution
        # WHY: When nothing fails, agent should be guided to scan
        # EXPECTED: suggested_next_step mentions "scan"
        py_file = tmp_path / "test_pass.py"
        py_file.write_text("pass\n")

        mock_proc = _mock_subprocess(
            returncode=0,
            stdout="5 passed\n",
        )
        with patch("subprocess.run", return_value=mock_proc):
            result = call_tool(
                "mltk_test", suite_path=str(py_file),
            )

        assert_ok(result)
        hint = result["suggested_next_step"].lower()
        assert "scan" in hint

    def test_some_failing_suggests_fix(self, tmp_path):
        # SCENARIO: Some tests fail in .py execution
        # WHY: When failures exist, agent should be guided to fix
        # EXPECTED: suggested_next_step mentions "fix" or "fail"
        py_file = tmp_path / "test_fail.py"
        py_file.write_text("pass\n")

        mock_proc = _mock_subprocess(
            returncode=1,
            stdout="2 passed, 3 failed\n",
        )
        with patch("subprocess.run", return_value=mock_proc):
            result = call_tool(
                "mltk_test", suite_path=str(py_file),
            )

        assert_ok(result)
        hint = result["suggested_next_step"].lower()
        assert "fix" in hint or "fail" in hint

    def test_returns_valid_json(self, tmp_path):
        # SCENARIO: Raw output format validation
        # WHY: MCP tools must always return well-formed JSON
        # EXPECTED: Raw string parses as JSON with status key
        yml = tmp_path / "rawjson.yaml"
        yml.write_text("name: raw\ntests:\n  - name: r1\n")

        raw = call_tool_raw("mltk_test", suite_path=str(yml))

        data = assert_valid_json(raw)
        assert data["status"] == "ok"

    def test_error_has_recoverable(self):
        # SCENARIO: Error response structure check
        # WHY: Every error must declare whether it is recoverable
        # EXPECTED: recoverable is a bool
        result = call_tool(
            "mltk_test",
            suite_path="/no/such/suite.py",
        )
        assert_error(result)
        assert isinstance(result["recoverable"], bool)

    def test_unsupported_suffix(self, tmp_path):
        # SCENARIO: Pass a .txt file instead of .yaml or .py
        # WHY: Only .yaml/.yml/.py are supported
        # EXPECTED: status=error mentioning "Unsupported type"
        txt = tmp_path / "data.txt"
        txt.write_text("hello\n")

        result = call_tool("mltk_test", suite_path=str(txt))

        assert_error(result)
        assert "unsupported" in result["error"].lower()

    def test_yml_extension(self, tmp_path):
        # SCENARIO: YAML file with .yml extension (not .yaml)
        # WHY: Server accepts both .yaml and .yml
        # EXPECTED: status=ok, parses correctly
        yml = tmp_path / "suite.yml"
        yml.write_text("name: alt\ntests:\n  - name: t1\n")

        result = call_tool("mltk_test", suite_path=str(yml))

        assert_ok(result)
        assert result["suite"] == "alt"

    def test_yaml_not_a_dict(self, tmp_path):
        # SCENARIO: YAML file that parses to a list, not a dict
        # WHY: Server requires a mapping (dict) at top level
        # EXPECTED: status=error mentioning "mapping"
        yml = tmp_path / "bad.yaml"
        yml.write_text("- item1\n- item2\n")

        result = call_tool("mltk_test", suite_path=str(yml))

        assert_error(result)
        assert "mapping" in result["error"].lower()

    def test_verbose_py_file(self, tmp_path):
        # SCENARIO: Run .py file with verbose=True
        # WHY: Verbose adds -v flag and returns full output
        # EXPECTED: status=ok, output is full (not truncated)
        py = tmp_path / "test_v.py"
        py.write_text("def test_ok(): pass\n")
        mock = _mock_subprocess(
            returncode=0,
            stdout="test_v.py::test_ok PASSED\n1 passed in 0.1s\n",
        )

        with patch("subprocess.run", return_value=mock):
            result = call_tool(
                "mltk_test",
                suite_path=str(py),
                verbose=True,
            )

        assert_ok(result)
        assert result["passed"] == 1
