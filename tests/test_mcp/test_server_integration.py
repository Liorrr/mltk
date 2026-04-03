"""Cross-tool integration and contract tests for the mltk MCP server.

Validates multi-tool workflows, JSON round-trips, lazy imports,
error isolation, and response-contract invariants across all tools.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ._helpers import (
    assert_error,
    assert_ok,
    assert_valid_json,
    call_tool,
    call_tool_raw,
    import_server,
    registered_tools,
)

# ----------------------------------------------------------
# Shared mock builders
# ----------------------------------------------------------

def _patch_scan():
    """Context manager that mocks mltk.scan.ScanConfig and ScanEngine."""
    return patch.multiple(
        "mltk.scan",
        ScanConfig=MagicMock(name="ScanConfig"),
        ScanEngine=MagicMock(name="ScanEngine"),
        create=True,
    )


MOCK_LIST_ALL = {
    "data": [
        {"name": "assert_no_drift", "doc": "Detect distribution drift"},
    ],
    "model": [
        {"name": "assert_metric", "doc": "Validate model metric"},
    ],
}

_LIST_PATCH = "mltk.cli._discovery.discover_assertions"


def _patch_eval():
    """Context manager that mocks mltk.eval.task.load_dataset and EvalTask."""
    mock_result = MagicMock()
    mock_result.metrics = {"accuracy": 1.0}
    mock_result.total_samples = 2
    mock_result.duration_ms = 10

    mock_task_instance = MagicMock()
    mock_task_instance.run.return_value = mock_result

    mock_task_cls = MagicMock(return_value=mock_task_instance)

    mock_load = MagicMock(return_value=[
        MagicMock(input="q1", target="a1"),
        MagicMock(input="q2", target="a2"),
    ])

    return patch.multiple(
        "mltk.eval.task",
        load_dataset=mock_load,
        EvalTask=mock_task_cls,
        create=True,
    ), patch.multiple(
        "mltk.eval",
        GenerateSolver=MagicMock(),
        ChainOfThoughtSolver=MagicMock(),
        FewShotSolver=MagicMock(),
        ExactMatchScorer=MagicMock(),
        IncludesScorer=MagicMock(),
        PatternScorer=MagicMock(),
        create=True,
    )


class TestIntegration:
    """Cross-tool integration and contract tests."""

    def test_full_workflow_scan_test_report(self, tmp_path):
        # SCENARIO: Chain scan -> test yaml -> report.
        # WHY: End-to-end workflow that agents commonly follow.
        # EXPECTED: All three tool calls return status=ok.
        py_file = tmp_path / "model.py"
        py_file.write_text("x = 1\n")
        yml_file = tmp_path / "suite.yaml"
        yml_file.write_text("name: demo\ntests:\n  - name: t1\n")

        # Step 1: scan
        with _patch_scan():
            scan_result = call_tool("mltk_scan", path=str(py_file))
        assert_ok(scan_result)

        # Step 2: test yaml
        test_result = call_tool("mltk_test", suite_path=str(yml_file))
        assert_ok(test_result)

        # Step 3: report (pure function, no mock)
        report_result = call_tool(
            "mltk_report",
            title="Workflow Report",
            description="Scan + test results",
            results_json=json.dumps([
                {"name": "scan", "status": "ok"},
                {"name": "test", "passed": True},
            ]),
        )
        assert_ok(report_result)
        assert "report_text" in report_result

    def test_json_roundtrip_all_tools(self, tmp_path):
        # SCENARIO: Every tool returns parseable JSON with "status".
        # WHY: MCP transport demands well-formed JSON from all tools.
        # EXPECTED: Raw outputs parse as JSON with "status" key.
        py_file = tmp_path / "check.py"
        py_file.write_text("pass\n")

        # mltk_scan
        with _patch_scan():
            raw_scan = call_tool_raw("mltk_scan", path=str(py_file))
        assert_valid_json(raw_scan)

        # mltk_list
        with patch(_LIST_PATCH, return_value=MOCK_LIST_ALL):
            raw_list = call_tool_raw("mltk_list")
        assert_valid_json(raw_list)

        # mltk_report (pure)
        raw_report = call_tool_raw(
            "mltk_report",
            title="Roundtrip Test",
        )
        assert_valid_json(raw_report)

    def test_error_does_not_crash_server(self, tmp_path):
        # SCENARIO: mltk_scan's lazy import raises RuntimeError.
        # WHY: Tool errors must be caught and returned as JSON, not exceptions.
        # EXPECTED: status=error with an error message, no unhandled exception.
        py_file = tmp_path / "boom.py"
        py_file.write_text("pass\n")

        with patch(
            "mltk.scan.ScanConfig",
            side_effect=RuntimeError("boom"),
            create=True,
        ), patch(
            "mltk.scan.ScanEngine",
            MagicMock(),
            create=True,
        ):
            result = call_tool("mltk_scan", path=str(py_file))

        assert result["status"] == "error"
        assert "boom" in result["error"]

    def test_all_tools_import_lazily(self):
        # SCENARIO: After server creation, no heavy deps are loaded BY the server.
        # WHY: MCP servers must start fast; heavy deps are only needed at call time.
        # EXPECTED: registered_tools has 7 tools; server module itself does not
        #   import torch/tensorflow (numpy/pandas may be present from the
        #   broader mltk package, so we verify the server module's own imports).
        assert len(registered_tools) == 7
        expected = {
            "mltk_scan", "mltk_test", "mltk_list",
            "mltk_eval", "mltk_dataset", "mltk_report",
            "mltk_suggest",
        }
        assert set(registered_tools.keys()) == expected

        # The server module itself only imports json, re, subprocess, sys,
        # traceback, datetime, pathlib, typing, and (mocked) mcp.
        # Verify it does NOT directly import torch or tensorflow.
        import mltk.mcp.server as srv_mod  # noqa: PLC0415
        source = srv_mod.__file__
        assert source is not None
        src = Path(source).read_text(encoding="utf-8")
        # Top-level imports should not include heavy ML deps
        for dep in ("torch", "tensorflow"):
            assert f"import {dep}" not in src, (
                f"server.py has a top-level 'import {dep}'"
            )

    def test_server_without_mcp_package(self):
        # SCENARIO: FastMCP is None (mcp package not installed).
        # WHY: create_server must raise ImportError with install hint.
        # EXPECTED: ImportError raised.
        server = import_server()
        with patch.object(server, "FastMCP", None), pytest.raises(ImportError):
            server.create_server()

    def test_main_module_entry_point(self):
        # SCENARIO: Server module exposes a callable run_server.
        # WHY: CLI entry point calls run_server() to start MCP mode.
        # EXPECTED: run_server is a callable attribute.
        server = import_server()
        assert hasattr(server, "run_server")
        assert callable(server.run_server)

    def test_tool_descriptions_start_with_verb(self):
        # SCENARIO: Every tool's docstring starts with an action verb.
        # WHY: MCP tool descriptions are shown to agents; verb-first is clearer.
        # EXPECTED: First word of each __doc__ is a recognized verb.
        verbs = {
            "scan", "run", "list", "evaluate", "get", "generate",
            "retrieve", "check", "detect", "find", "compute",
            "fetch", "create", "build", "produce", "return",
            "search", "validate", "execute",
        }
        for name, fn in registered_tools.items():
            doc = (fn.__doc__ or "").strip()
            assert doc, f"{name} has no docstring"
            first_word = doc.split()[0].lower().rstrip("s")
            assert first_word in verbs, (
                f"{name} docstring starts with {first_word!r}, "
                f"expected one of {sorted(verbs)}"
            )

    def test_all_tools_return_status(self, tmp_path):
        # SCENARIO: Multiple tools all have "status" in response.
        # WHY: Agents key off the "status" field to decide next action.
        # EXPECTED: Every response dict has a "status" key.
        py_file = tmp_path / "status.py"
        py_file.write_text("pass\n")

        # scan
        with _patch_scan():
            scan = call_tool("mltk_scan", path=str(py_file))
        assert "status" in scan

        # list
        with patch(_LIST_PATCH, return_value=MOCK_LIST_ALL):
            lst = call_tool("mltk_list")
        assert "status" in lst

        # report
        report = call_tool("mltk_report", title="Status Check")
        assert "status" in report

    def test_all_error_responses_have_recoverable(self, tmp_path):  # noqa: ARG002
        # SCENARIO: Error responses from scan and test have "recoverable".
        # WHY: Agents use "recoverable" to decide whether to retry or abort.
        # EXPECTED: Both error responses have a bool "recoverable" field.

        # scan with nonexistent path
        scan_err = call_tool(
            "mltk_scan", path="/nonexistent/path/file.py",
        )
        assert_error(scan_err)
        assert isinstance(scan_err["recoverable"], bool)

        # test with nonexistent path
        test_err = call_tool(
            "mltk_test", suite_path="/nonexistent/path/test.py",
        )
        assert_error(test_err)
        assert isinstance(test_err["recoverable"], bool)

    def test_eval_then_report_workflow(self, tmp_path):
        # SCENARIO: Chain eval -> report with eval output as report input.
        # WHY: Agents commonly evaluate, then generate a report from results.
        # EXPECTED: Both tools return status=ok; report includes eval data.
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("input,target\nq1,a1\nq2,a2\n")

        eval_task_patch, eval_imports_patch = _patch_eval()
        with eval_task_patch, eval_imports_patch:
            eval_result = call_tool(
                "mltk_eval", dataset_path=str(csv_file),
            )
        assert_ok(eval_result)
        assert "metrics" in eval_result

        # Feed eval results into report
        report_result = call_tool(
            "mltk_report",
            title="Eval Report",
            description="Evaluation pipeline results",
            results_json=json.dumps({
                "name": "eval-run",
                "status": "ok",
                "metrics": eval_result["metrics"],
            }),
        )
        assert_ok(report_result)
        assert "report_text" in report_result

    def test_list_then_eval_workflow(self, tmp_path):
        # SCENARIO: Chain list assertions -> eval dataset.
        # WHY: Agents discover assertions then run evaluations.
        # EXPECTED: Both tools return status=ok.
        csv_file = tmp_path / "eval.csv"
        csv_file.write_text("input,target\nq1,a1\n")

        # Step 1: list assertions
        with patch(_LIST_PATCH, return_value=MOCK_LIST_ALL):
            list_result = call_tool("mltk_list")
        assert_ok(list_result)
        assert list_result["total"] > 0

        # Step 2: eval dataset
        eval_task_patch, eval_imports_patch = _patch_eval()
        with eval_task_patch, eval_imports_patch:
            eval_result = call_tool(
                "mltk_eval", dataset_path=str(csv_file),
            )
        assert_ok(eval_result)
        assert "metrics" in eval_result
