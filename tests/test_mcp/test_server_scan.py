"""Tests for the mltk_scan MCP tool.

Covers file/directory scanning, JSON report parsing,
scanner selection, error responses, and response structure.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from ._helpers import (
    assert_error,
    assert_ok,
    assert_valid_json,
    call_tool,
    call_tool_raw,
)


def _patch_scan():
    """Context manager that mocks mltk.scan.ScanConfig and ScanEngine."""
    mock_config = MagicMock(name="ScanConfig")
    mock_engine = MagicMock(name="ScanEngine")
    return patch.multiple(
        "mltk.scan",
        ScanConfig=mock_config,
        ScanEngine=mock_engine,
        create=True,
    )


class TestMltkScan:
    """Tests for the mltk_scan tool."""

    def test_scan_valid_file(self, tmp_path):
        # SCENARIO: Scan a real .py file
        # WHY: Core happy path -- single Python file scan
        # EXPECTED: status=ok, python_files contains the filename
        py_file = tmp_path / "model.py"
        py_file.write_text("x = 1\n")

        with _patch_scan():
            result = call_tool("mltk_scan", path=str(py_file))

        assert_ok(result)
        assert "python_files" in result
        assert "model.py" in result["python_files"]

    def test_scan_nonexistent_path(self):
        # SCENARIO: Scan a path that does not exist
        # WHY: Should fail before lazy import with a clear error
        # EXPECTED: status=error, no mock needed
        result = call_tool(
            "mltk_scan",
            path="/does/not/exist/model.py",
        )
        assert_error(result)
        assert "not found" in result["error"].lower()

    def test_scan_specific_scanners(self, tmp_path):
        # SCENARIO: Pass specific scanner names
        # WHY: Verify user-selected scanners are split and forwarded
        # EXPECTED: status=ok, enabled is a list of strings
        py_file = tmp_path / "app.py"
        py_file.write_text("pass\n")

        with _patch_scan():
            result = call_tool(
                "mltk_scan",
                path=str(py_file),
                scanners="drift,bias",
            )

        assert_ok(result)
        assert isinstance(result["enabled"], list)
        assert "drift" in result["enabled"]
        assert "bias" in result["enabled"]

    def test_scan_all_scanners(self, tmp_path):
        # SCENARIO: Pass scanners="all" explicitly
        # WHY: "all" is a special sentinel, not a scanner name
        # EXPECTED: status=ok, enabled is the string "all"
        py_file = tmp_path / "train.py"
        py_file.write_text("pass\n")

        with _patch_scan():
            result = call_tool(
                "mltk_scan",
                path=str(py_file),
                scanners="all",
            )

        assert_ok(result)
        assert result["enabled"] == "all"

    def test_scan_directory(self, tmp_path):
        # SCENARIO: Scan a directory with .py files
        # WHY: Directory mode lists Python files in the tree
        # EXPECTED: status=ok, file_count > 0
        (tmp_path / "a.py").write_text("pass\n")
        (tmp_path / "b.py").write_text("pass\n")

        with _patch_scan():
            result = call_tool("mltk_scan", path=str(tmp_path))

        assert_ok(result)
        assert result["file_count"] > 0
        assert len(result["python_files"]) >= 2

    def test_scan_json_report(self, tmp_path):
        # SCENARIO: Scan a .json report file
        # WHY: JSON mode parses existing findings instead of scanning
        # EXPECTED: status=ok, response has findings key
        report = tmp_path / "report.json"
        report.write_text(json.dumps({
            "findings": [{"type": "drift"}],
            "scanners_run": ["drift"],
            "duration_ms": 42,
        }))

        with _patch_scan():
            result = call_tool("mltk_scan", path=str(report))

        assert_ok(result)
        assert "findings" in result
        assert len(result["findings"]) == 1
        assert result["findings"][0]["type"] == "drift"

    def test_scan_has_suggested_next_step(self, tmp_path):
        # SCENARIO: Any successful scan response
        # WHY: All ok responses must include suggested_next_step
        # EXPECTED: suggested_next_step is a non-empty string
        py_file = tmp_path / "check.py"
        py_file.write_text("pass\n")

        with _patch_scan():
            result = call_tool("mltk_scan", path=str(py_file))

        assert_ok(result)
        assert "suggested_next_step" in result
        assert isinstance(result["suggested_next_step"], str)
        assert len(result["suggested_next_step"]) > 0

    def test_scan_error_has_recoverable(self):
        # SCENARIO: Error response structure check
        # WHY: Every error must declare whether it is recoverable
        # EXPECTED: recoverable is a bool
        result = call_tool(
            "mltk_scan",
            path="/no/such/path.py",
        )
        assert_error(result)
        assert isinstance(result["recoverable"], bool)

    def test_scan_error_has_suggested_action(self):
        # SCENARIO: Error response structure check
        # WHY: Every error must give the agent a suggested action
        # EXPECTED: suggested_action is a non-empty string
        result = call_tool(
            "mltk_scan",
            path="/no/such/path.py",
        )
        assert_error(result)
        assert isinstance(result["suggested_action"], str)
        assert len(result["suggested_action"]) > 0

    def test_scan_returns_valid_json(self, tmp_path):
        # SCENARIO: Raw output format validation
        # WHY: MCP tools must always return well-formed JSON
        # EXPECTED: Raw string parses as JSON with status key
        py_file = tmp_path / "valid.py"
        py_file.write_text("pass\n")

        with _patch_scan():
            raw = call_tool_raw("mltk_scan", path=str(py_file))

        data = assert_valid_json(raw)
        assert data["status"] == "ok"

    def test_scan_default_scanners_is_all(self, tmp_path):
        # SCENARIO: Omit scanners parameter entirely
        # WHY: Default should be "all", not empty or error
        # EXPECTED: enabled is "all"
        py_file = tmp_path / "default.py"
        py_file.write_text("pass\n")

        with _patch_scan():
            result = call_tool("mltk_scan", path=str(py_file))

        assert_ok(result)
        assert result["enabled"] == "all"

    def test_scan_empty_directory(self, tmp_path):
        # SCENARIO: Scan a directory with no .py files
        # WHY: Empty dirs should succeed with file_count=0
        # EXPECTED: status=ok, file_count=0, python_files is empty
        (tmp_path / "readme.txt").write_text("nothing here\n")

        with _patch_scan():
            result = call_tool("mltk_scan", path=str(tmp_path))

        assert_ok(result)
        assert result["file_count"] == 0
        assert result["python_files"] == []

    def test_scan_file_cap_50(self, tmp_path):
        # SCENARIO: Directory with >50 .py files
        # WHY: Server caps python_files at [:50]
        # EXPECTED: file_count == 50 even with 55 files
        for i in range(55):
            (tmp_path / f"mod_{i:03d}.py").write_text(f"x = {i}\n")

        with _patch_scan():
            result = call_tool("mltk_scan", path=str(tmp_path))

        assert_ok(result)
        assert result["file_count"] == 50
