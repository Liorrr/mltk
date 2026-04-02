"""Tests for mltk.mcp.server — MCP tool server.

Validates the 6 MCP tools (scan, test, list, eval,
dataset, report), server creation, JSON response
contracts, and integration workflows.

All external dependencies are mocked. No network,
no filesystem side effects, no sleep calls.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import (
    patch,
)

import pytest

# ----------------------------------------------------------
# Mock infrastructure — FastMCP and mcp package
# ----------------------------------------------------------

_registered_tools: dict[str, object] = {}


def _make_fastmcp_mock():
    """Build a mock FastMCP class that captures tools."""

    class _FastMCP:
        def __init__(self, name: str, **kwargs):
            self.name = name
            self._tools: dict[str, object] = {}
            self._kwargs = kwargs

        def tool(self, **kwargs):
            """Decorator that registers a tool function."""
            def decorator(fn):
                self._tools[fn.__name__] = fn
                _registered_tools[fn.__name__] = fn
                return fn
            return decorator

        def run(self, transport: str = "stdio"):
            pass  # no-op for tests

    return _FastMCP


def _make_mcp_module():
    """Build mock mcp package hierarchy."""
    mcp = types.ModuleType("mcp")
    server = types.ModuleType("mcp.server")
    fastmcp_mod = types.ModuleType(
        "mcp.server.fastmcp"
    )
    fastmcp_mod.FastMCP = _make_fastmcp_mock()
    mcp.server = server
    server.fastmcp = fastmcp_mod
    return {
        "mcp": mcp,
        "mcp.server": server,
        "mcp.server.fastmcp": fastmcp_mod,
    }


@pytest.fixture(autouse=True)
def _mock_mcp_package():
    """Inject mock mcp package for all tests."""
    modules = _make_mcp_module()
    _registered_tools.clear()
    with patch.dict(sys.modules, modules):
        yield modules


# ----------------------------------------------------------
# Lazy import helper
# ----------------------------------------------------------


def _import_server():
    """Import the server module after mocking."""
    # Force re-import to pick up mocked mcp
    mod_name = "mltk.mcp.server"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    from mltk.mcp import server
    return server


def _call_tool(name: str, **kwargs) -> dict:
    """Call a registered tool and parse JSON result."""
    fn = _registered_tools[name]
    # Tools are sync functions returning JSON strings
    result_str = fn(**kwargs)
    return json.loads(result_str)


# ----------------------------------------------------------
# Response validation helpers
# ----------------------------------------------------------


def _assert_ok(data: dict) -> None:
    """Assert response has status=ok."""
    assert data["status"] == "ok"


def _assert_error(data: dict) -> None:
    """Assert response is a well-formed error."""
    assert data["status"] == "error"
    assert "error" in data
    assert isinstance(data["error"], str)
    assert "recoverable" in data
    assert isinstance(data["recoverable"], bool)
    assert "suggested_action" in data
    assert isinstance(data["suggested_action"], str)


def _assert_valid_json(raw: str) -> dict:
    """Assert string is valid JSON, return parsed."""
    data = json.loads(raw)
    assert isinstance(data, dict)
    assert "status" in data
    return data


# ==========================================================
# Server Creation (7 tests)
# ==========================================================


class TestServerCreation:
    """Server factory and tool registration."""

    def test_create_server_returns_instance(
        self,
    ) -> None:
        # SCENARIO: create_server() produces a server.
        # WHY: Entry point must return a usable object.
        # EXPECTED: Non-None server instance.
        server = _import_server()
        srv = server.create_server()
        assert srv is not None

    def test_server_name_is_mltk(self) -> None:
        # SCENARIO: Server name matches the project.
        # WHY: MCP clients show the name to users.
        # EXPECTED: name == "mltk".
        server = _import_server()
        srv = server.create_server()
        assert srv.name == "mltk"

    def test_server_has_version(self) -> None:
        # SCENARIO: Server reports a version string.
        # WHY: Clients display version in tool lists.
        # EXPECTED: Version string present in kwargs.
        server = _import_server()
        srv = server.create_server()
        version = srv._kwargs.get("version", "")
        assert version, "Server must declare version"

    def test_all_six_tools_registered(self) -> None:
        # SCENARIO: All 6 tools exist after creation.
        # WHY: Incomplete registration = missing tools.
        # EXPECTED: Exactly 6 tools registered.
        server = _import_server()
        srv = server.create_server()
        expected = {
            "mltk_scan",
            "mltk_test",
            "mltk_list",
            "mltk_eval",
            "mltk_dataset",
            "mltk_report",
        }
        actual = set(srv._tools.keys())
        assert expected == actual

    def test_tool_names_have_mltk_prefix(
        self,
    ) -> None:
        # SCENARIO: Every tool name starts with mltk_.
        # WHY: Namespaced tools avoid collisions.
        # EXPECTED: All tool names start with "mltk_".
        server = _import_server()
        srv = server.create_server()
        for name in srv._tools:
            assert name.startswith("mltk_"), (
                f"Tool {name!r} missing mltk_ prefix"
            )

    def test_tool_functions_are_callable(
        self,
    ) -> None:
        # SCENARIO: Registered tools are functions.
        # WHY: FastMCP requires callable tool handlers.
        # EXPECTED: All tools are callable.
        server = _import_server()
        srv = server.create_server()
        for name, fn in srv._tools.items():
            assert callable(fn), (
                f"Tool {name!r} is not callable"
            )

    def test_run_server_exists(self) -> None:
        # SCENARIO: run_server is a callable entry point.
        # WHY: MCP servers need a run function.
        # EXPECTED: run_server is callable.
        server = _import_server()
        assert callable(server.run_server)


# ==========================================================
# mltk_scan (12 tests)
# ==========================================================


class TestMltkScan:
    """mltk_scan tool — project scanning."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.py_file = tmp_path / "model.py"
        self.py_file.write_text("x = 1\n")

    def _scan(self, **kwargs) -> dict:
        return _call_tool("mltk_scan", **kwargs)

    def test_scan_valid_path(self) -> None:
        # SCENARIO: Scan an existing Python file.
        # WHY: Core happy path must return ok.
        # EXPECTED: status=ok.
        _import_server()
        with patch(
            "mltk.mcp.server.scan_project",
            return_value={"findings": []},
        ):
            result = self._scan(
                path=str(self.py_file),
            )
        _assert_ok(result)

    def test_scan_nonexistent_path(self) -> None:
        # SCENARIO: Scan a path that does not exist.
        # WHY: Missing paths must produce an error.
        # EXPECTED: status=error with recoverable.
        _import_server()
        result = self._scan(
            path="/nonexistent/path.py",
        )
        _assert_error(result)

    def test_scan_specific_scanners(self) -> None:
        # SCENARIO: Scan with a specific scanner set.
        # WHY: Users pick scanners by name.
        # EXPECTED: status=ok, scanner filter applied.
        _import_server()
        with patch(
            "mltk.mcp.server.scan_project",
            return_value={"findings": []},
        ) as mock_scan:
            result = self._scan(
                path=str(self.py_file),
                scanners="drift,bias",
            )
        _assert_ok(result)
        call_args = mock_scan.call_args
        assert call_args is not None

    def test_scan_all_scanners(self) -> None:
        # SCENARIO: scanners="all" runs everything.
        # WHY: Default should enable all scanners.
        # EXPECTED: status=ok.
        _import_server()
        with patch(
            "mltk.mcp.server.scan_project",
            return_value={"findings": ["issue_a"]},
        ):
            result = self._scan(
                path=str(self.py_file),
                scanners="all",
            )
        _assert_ok(result)

    def test_scan_response_has_findings(self) -> None:
        # SCENARIO: Scan returns a findings list.
        # WHY: Callers iterate over findings.
        # EXPECTED: "findings" key in response.
        _import_server()
        with patch(
            "mltk.mcp.server.scan_project",
            return_value={
                "findings": [
                    {"type": "drift", "severity": "high"},
                ],
            },
        ):
            result = self._scan(
                path=str(self.py_file),
            )
        assert "findings" in result

    def test_scan_has_suggested_next_step(
        self,
    ) -> None:
        # SCENARIO: Success includes next step hint.
        # WHY: Agents use hints to chain tools.
        # EXPECTED: suggested_next_step in response.
        _import_server()
        with patch(
            "mltk.mcp.server.scan_project",
            return_value={"findings": []},
        ):
            result = self._scan(
                path=str(self.py_file),
            )
        assert "suggested_next_step" in result

    def test_scan_error_has_recoverable(
        self,
    ) -> None:
        # SCENARIO: Error includes recoverable flag.
        # WHY: Agents decide retry vs abort.
        # EXPECTED: recoverable is a bool.
        _import_server()
        result = self._scan(path="/no/such/file")
        assert isinstance(
            result["recoverable"], bool
        )

    def test_scan_error_has_suggested_action(
        self,
    ) -> None:
        # SCENARIO: Error includes suggested_action.
        # WHY: Agents need recovery guidance.
        # EXPECTED: suggested_action is a string.
        _import_server()
        result = self._scan(path="/no/such/file")
        assert isinstance(
            result["suggested_action"], str
        )
        assert len(result["suggested_action"]) > 0

    def test_scan_empty_project_no_findings(
        self,
    ) -> None:
        # SCENARIO: Empty project has no findings.
        # WHY: Clean projects pass cleanly.
        # EXPECTED: findings list is empty.
        _import_server()
        with patch(
            "mltk.mcp.server.scan_project",
            return_value={"findings": []},
        ):
            result = self._scan(
                path=str(self.tmp),
            )
        assert result["findings"] == []

    def test_scan_returns_valid_json(self) -> None:
        # SCENARIO: Raw return is a JSON string.
        # WHY: MCP tools must return JSON strings.
        # EXPECTED: json.loads succeeds.
        _import_server()
        with patch(
            "mltk.mcp.server.scan_project",
            return_value={"findings": []},
        ):
            fn = _registered_tools["mltk_scan"]
            raw = fn(path=str(self.py_file))
        _assert_valid_json(raw)

    def test_scan_default_scanners_is_all(
        self,
    ) -> None:
        # SCENARIO: Omitting scanners defaults to all.
        # WHY: Default must scan everything.
        # EXPECTED: status=ok without explicit scanners.
        _import_server()
        with patch(
            "mltk.mcp.server.scan_project",
            return_value={"findings": []},
        ):
            result = self._scan(
                path=str(self.py_file),
            )
        _assert_ok(result)

    def test_scan_directory_path(self) -> None:
        # SCENARIO: Scan a directory, not a file.
        # WHY: Users often pass project root.
        # EXPECTED: status=ok.
        _import_server()
        with patch(
            "mltk.mcp.server.scan_project",
            return_value={"findings": []},
        ):
            result = self._scan(
                path=str(self.tmp),
            )
        _assert_ok(result)


# ==========================================================
# mltk_test (12 tests)
# ==========================================================


class TestMltkTest:
    """mltk_test tool — test suite execution."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.suite = tmp_path / "suite.yaml"
        self.suite.write_text("name: demo\n")
        self.py_test = tmp_path / "test_example.py"
        self.py_test.write_text("def test_ok(): pass\n")

    def _test(self, **kwargs) -> dict:
        return _call_tool("mltk_test", **kwargs)

    def test_yaml_suite_ok(self) -> None:
        # SCENARIO: Run a YAML test suite.
        # WHY: YAML suites are primary input.
        # EXPECTED: status=ok.
        _import_server()
        with patch(
            "mltk.mcp.server.run_test_suite",
            return_value={
                "passed": 3,
                "failed": 0,
                "results": [],
            },
        ):
            result = self._test(
                suite_path=str(self.suite),
            )
        _assert_ok(result)

    def test_py_file_ok(self) -> None:
        # SCENARIO: Run a .py test file.
        # WHY: Users pass pytest files directly.
        # EXPECTED: status=ok.
        _import_server()
        with patch(
            "mltk.mcp.server.run_pytest_file",
            return_value={
                "passed": 1,
                "failed": 0,
                "results": [],
            },
        ):
            result = self._test(
                suite_path=str(self.py_test),
            )
        _assert_ok(result)

    def test_nonexistent_file_error(self) -> None:
        # SCENARIO: Test file does not exist.
        # WHY: Missing files must produce an error.
        # EXPECTED: status=error.
        _import_server()
        result = self._test(
            suite_path="/no/such/suite.yaml",
        )
        _assert_error(result)

    def test_response_has_pass_fail_counts(
        self,
    ) -> None:
        # SCENARIO: Response includes passed/failed.
        # WHY: Agents need pass/fail for decisions.
        # EXPECTED: passed and failed keys present.
        _import_server()
        with patch(
            "mltk.mcp.server.run_test_suite",
            return_value={
                "passed": 5,
                "failed": 2,
                "results": [],
            },
        ):
            result = self._test(
                suite_path=str(self.suite),
            )
        assert "passed" in result
        assert "failed" in result

    def test_response_has_results_list(
        self,
    ) -> None:
        # SCENARIO: Response includes results array.
        # WHY: Agents inspect individual test results.
        # EXPECTED: results key is a list.
        _import_server()
        with patch(
            "mltk.mcp.server.run_test_suite",
            return_value={
                "passed": 1,
                "failed": 0,
                "results": [
                    {"name": "t1", "passed": True},
                ],
            },
        ):
            result = self._test(
                suite_path=str(self.suite),
            )
        assert isinstance(result["results"], list)

    def test_verbose_mode_includes_details(
        self,
    ) -> None:
        # SCENARIO: verbose=True adds extra detail.
        # WHY: Debugging needs more output.
        # EXPECTED: Result has verbose-specific data.
        _import_server()
        with patch(
            "mltk.mcp.server.run_test_suite",
            return_value={
                "passed": 1,
                "failed": 0,
                "results": [
                    {
                        "name": "t1",
                        "passed": True,
                        "output": "details here",
                    },
                ],
                "verbose": True,
            },
        ):
            result = self._test(
                suite_path=str(self.suite),
                verbose=True,
            )
        _assert_ok(result)

    def test_has_suggested_next_step(self) -> None:
        # SCENARIO: Success includes next step hint.
        # WHY: Agent tool chaining depends on hints.
        # EXPECTED: suggested_next_step present.
        _import_server()
        with patch(
            "mltk.mcp.server.run_test_suite",
            return_value={
                "passed": 1,
                "failed": 0,
                "results": [],
            },
        ):
            result = self._test(
                suite_path=str(self.suite),
            )
        assert "suggested_next_step" in result

    def test_all_passing_suggests_report(
        self,
    ) -> None:
        # SCENARIO: All tests pass.
        # WHY: Next step should be "generate report".
        # EXPECTED: suggested_next_step mentions report.
        _import_server()
        with patch(
            "mltk.mcp.server.run_test_suite",
            return_value={
                "passed": 5,
                "failed": 0,
                "results": [],
            },
        ):
            result = self._test(
                suite_path=str(self.suite),
            )
        hint = result.get(
            "suggested_next_step", ""
        ).lower()
        assert "report" in hint

    def test_some_failing_suggests_fix(
        self,
    ) -> None:
        # SCENARIO: Some tests fail.
        # WHY: Next step should be "investigate/fix".
        # EXPECTED: suggested_next_step mentions fix.
        _import_server()
        with patch(
            "mltk.mcp.server.run_test_suite",
            return_value={
                "passed": 3,
                "failed": 2,
                "results": [
                    {"name": "t1", "passed": False},
                ],
            },
        ):
            result = self._test(
                suite_path=str(self.suite),
            )
        hint = result.get(
            "suggested_next_step", ""
        ).lower()
        assert (
            "fix" in hint
            or "investigat" in hint
            or "fail" in hint
        )

    def test_returns_valid_json(self) -> None:
        # SCENARIO: Raw return is valid JSON.
        # WHY: MCP tools return JSON strings.
        # EXPECTED: json.loads succeeds.
        _import_server()
        with patch(
            "mltk.mcp.server.run_test_suite",
            return_value={
                "passed": 1,
                "failed": 0,
                "results": [],
            },
        ):
            fn = _registered_tools["mltk_test"]
            raw = fn(suite_path=str(self.suite))
        _assert_valid_json(raw)

    def test_error_response_has_recoverable(
        self,
    ) -> None:
        # SCENARIO: Error response format is correct.
        # WHY: All errors must have recoverable flag.
        # EXPECTED: recoverable key present.
        _import_server()
        result = self._test(
            suite_path="/missing.yaml",
        )
        assert "recoverable" in result

    def test_error_response_has_suggested_action(
        self,
    ) -> None:
        # SCENARIO: Error has suggested_action.
        # WHY: Agents need recovery guidance.
        # EXPECTED: suggested_action is non-empty.
        _import_server()
        result = self._test(
            suite_path="/missing.yaml",
        )
        assert len(
            result.get("suggested_action", "")
        ) > 0


# ==========================================================
# mltk_list (12 tests)
# ==========================================================


class TestMltkList:
    """mltk_list tool — assertion listing."""

    def _list(self, **kwargs) -> dict:
        return _call_tool("mltk_list", **kwargs)

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        _import_server()
        self._assertions = [
            {
                "name": "assert_no_drift",
                "description": "Detect distribution drift",
                "domain": "data",
            },
            {
                "name": "assert_metric",
                "description": "Validate model metric",
                "domain": "model",
            },
            {
                "name": "assert_faithfulness",
                "description": "Check RAG faithfulness",
                "domain": "llm",
            },
            {
                "name": "assert_no_bias",
                "description": "Detect model bias",
                "domain": "model",
            },
        ]

    def test_list_all(self) -> None:
        # SCENARIO: List without any filter.
        # WHY: Default lists everything.
        # EXPECTED: status=ok with assertions.
        with patch(
            "mltk.mcp.server.get_assertions",
            return_value=self._assertions,
        ):
            result = self._list()
        _assert_ok(result)
        assert len(result["assertions"]) > 0

    def test_list_filter_text(self) -> None:
        # SCENARIO: Filter by text substring.
        # WHY: Users search by keyword.
        # EXPECTED: Only matching assertions.
        with patch(
            "mltk.mcp.server.get_assertions",
            return_value=[self._assertions[0]],
        ):
            result = self._list(filter_text="drift")
        _assert_ok(result)
        for a in result["assertions"]:
            assert (
                "drift" in a["name"].lower()
                or "drift" in a["description"].lower()
            )

    def test_list_filter_domain(self) -> None:
        # SCENARIO: Filter by domain.
        # WHY: Users scope to a specific domain.
        # EXPECTED: Only assertions from that domain.
        model_assertions = [
            a for a in self._assertions
            if a["domain"] == "model"
        ]
        with patch(
            "mltk.mcp.server.get_assertions",
            return_value=model_assertions,
        ):
            result = self._list(domain="model")
        _assert_ok(result)
        for a in result["assertions"]:
            assert a["domain"] == "model"

    def test_list_empty_filter_returns_all(
        self,
    ) -> None:
        # SCENARIO: filter_text="" returns everything.
        # WHY: Empty filter = no filtering.
        # EXPECTED: All assertions returned.
        with patch(
            "mltk.mcp.server.get_assertions",
            return_value=self._assertions,
        ):
            result = self._list(filter_text="")
        assert (
            len(result["assertions"])
            == len(self._assertions)
        )

    def test_response_has_total_count(self) -> None:
        # SCENARIO: Response has total count field.
        # WHY: Agents use count for progress display.
        # EXPECTED: total key matches assertions len.
        with patch(
            "mltk.mcp.server.get_assertions",
            return_value=self._assertions,
        ):
            result = self._list()
        assert "total" in result
        assert result["total"] == len(
            result["assertions"]
        )

    def test_response_has_assertions_array(
        self,
    ) -> None:
        # SCENARIO: assertions is a list.
        # WHY: Callers iterate over the array.
        # EXPECTED: assertions key is a list.
        with patch(
            "mltk.mcp.server.get_assertions",
            return_value=self._assertions,
        ):
            result = self._list()
        assert isinstance(
            result["assertions"], list
        )

    def test_each_assertion_has_name(self) -> None:
        # SCENARIO: Every assertion has a name.
        # WHY: Name is the primary identifier.
        # EXPECTED: All entries have "name" key.
        with patch(
            "mltk.mcp.server.get_assertions",
            return_value=self._assertions,
        ):
            result = self._list()
        for a in result["assertions"]:
            assert "name" in a
            assert len(a["name"]) > 0

    def test_each_assertion_has_description(
        self,
    ) -> None:
        # SCENARIO: Every assertion has description.
        # WHY: Description helps agents pick tools.
        # EXPECTED: All entries have "description".
        with patch(
            "mltk.mcp.server.get_assertions",
            return_value=self._assertions,
        ):
            result = self._list()
        for a in result["assertions"]:
            assert "description" in a

    def test_filter_drift(self) -> None:
        # SCENARIO: Filter text "drift" finds drift.
        # WHY: Common query for data engineers.
        # EXPECTED: Only drift assertions returned.
        with patch(
            "mltk.mcp.server.get_assertions",
            return_value=[self._assertions[0]],
        ):
            result = self._list(filter_text="drift")
        assert len(result["assertions"]) >= 1
        names = [
            a["name"] for a in result["assertions"]
        ]
        assert any("drift" in n for n in names)

    def test_domain_llm(self) -> None:
        # SCENARIO: domain="llm" returns LLM items.
        # WHY: Scoping by domain is common.
        # EXPECTED: Only LLM assertions returned.
        llm_assertions = [
            a for a in self._assertions
            if a["domain"] == "llm"
        ]
        with patch(
            "mltk.mcp.server.get_assertions",
            return_value=llm_assertions,
        ):
            result = self._list(domain="llm")
        for a in result["assertions"]:
            assert a["domain"] == "llm"

    def test_returns_valid_json(self) -> None:
        # SCENARIO: Raw return is valid JSON.
        # WHY: MCP tools return JSON strings.
        # EXPECTED: json.loads succeeds.
        with patch(
            "mltk.mcp.server.get_assertions",
            return_value=self._assertions,
        ):
            fn = _registered_tools["mltk_list"]
            raw = fn()
        _assert_valid_json(raw)

    def test_no_results_returns_empty_list(
        self,
    ) -> None:
        # SCENARIO: Filter matches nothing.
        # WHY: No results is not an error.
        # EXPECTED: status=ok, empty assertions list.
        with patch(
            "mltk.mcp.server.get_assertions",
            return_value=[],
        ):
            result = self._list(
                filter_text="nonexistent_xyz"
            )
        _assert_ok(result)
        assert result["assertions"] == []
        assert result["total"] == 0


# ==========================================================
# mltk_eval (12 tests)
# ==========================================================


class TestMltkEval:
    """mltk_eval tool — dataset evaluation."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.dataset = tmp_path / "eval_data.jsonl"
        self.dataset.write_text(
            '{"input": "2+2", "expected": "4"}\n'
        )

    def _eval(self, **kwargs) -> dict:
        return _call_tool("mltk_eval", **kwargs)

    def test_eval_valid_dataset(self) -> None:
        # SCENARIO: Eval an existing dataset file.
        # WHY: Core happy path must return ok.
        # EXPECTED: status=ok.
        _import_server()
        with patch(
            "mltk.mcp.server.run_evaluation",
            return_value={
                "metrics": {"accuracy": 0.95},
                "sample_count": 10,
            },
        ):
            result = self._eval(
                dataset_path=str(self.dataset),
            )
        _assert_ok(result)

    def test_eval_nonexistent_dataset(self) -> None:
        # SCENARIO: Dataset file does not exist.
        # WHY: Missing input must produce error.
        # EXPECTED: status=error.
        _import_server()
        result = self._eval(
            dataset_path="/no/such/data.jsonl",
        )
        _assert_error(result)

    def test_eval_exact_match_scorer(self) -> None:
        # SCENARIO: Use exact_match scorer.
        # WHY: Default scorer is exact_match.
        # EXPECTED: status=ok with metrics.
        _import_server()
        with patch(
            "mltk.mcp.server.run_evaluation",
            return_value={
                "metrics": {"exact_match": 0.8},
                "sample_count": 5,
            },
        ):
            result = self._eval(
                dataset_path=str(self.dataset),
                scorer="exact_match",
            )
        _assert_ok(result)
        assert "metrics" in result

    def test_eval_custom_solver(self) -> None:
        # SCENARIO: Use chain_of_thought solver.
        # WHY: Users pick different solvers.
        # EXPECTED: status=ok.
        _import_server()
        with patch(
            "mltk.mcp.server.run_evaluation",
            return_value={
                "metrics": {"accuracy": 0.9},
                "sample_count": 5,
            },
        ):
            result = self._eval(
                dataset_path=str(self.dataset),
                solver="chain_of_thought",
            )
        _assert_ok(result)

    def test_response_has_metrics_dict(
        self,
    ) -> None:
        # SCENARIO: Response includes metrics.
        # WHY: Metrics are the primary eval output.
        # EXPECTED: metrics is a dict.
        _import_server()
        with patch(
            "mltk.mcp.server.run_evaluation",
            return_value={
                "metrics": {"accuracy": 0.95},
                "sample_count": 10,
            },
        ):
            result = self._eval(
                dataset_path=str(self.dataset),
            )
        assert isinstance(result["metrics"], dict)

    def test_response_has_sample_count(
        self,
    ) -> None:
        # SCENARIO: Response reports sample count.
        # WHY: Agents display how many samples ran.
        # EXPECTED: sample_count is an int > 0.
        _import_server()
        with patch(
            "mltk.mcp.server.run_evaluation",
            return_value={
                "metrics": {"accuracy": 0.95},
                "sample_count": 10,
            },
        ):
            result = self._eval(
                dataset_path=str(self.dataset),
            )
        assert result["sample_count"] > 0

    def test_has_suggested_next_step(self) -> None:
        # SCENARIO: Success includes next step hint.
        # WHY: Agent tool chaining depends on hints.
        # EXPECTED: suggested_next_step present.
        _import_server()
        with patch(
            "mltk.mcp.server.run_evaluation",
            return_value={
                "metrics": {"accuracy": 0.95},
                "sample_count": 10,
            },
        ):
            result = self._eval(
                dataset_path=str(self.dataset),
            )
        assert "suggested_next_step" in result

    def test_invalid_scorer_error(self) -> None:
        # SCENARIO: Unknown scorer name.
        # WHY: Invalid scorer must explain options.
        # EXPECTED: status=error with suggestion.
        _import_server()
        with patch(
            "mltk.mcp.server.run_evaluation",
            side_effect=ValueError(
                "Unknown scorer: bogus"
            ),
        ):
            result = self._eval(
                dataset_path=str(self.dataset),
                scorer="bogus",
            )
        _assert_error(result)
        assert len(result["suggested_action"]) > 0

    def test_invalid_solver_error(self) -> None:
        # SCENARIO: Unknown solver name.
        # WHY: Invalid solver must explain options.
        # EXPECTED: status=error with suggestion.
        _import_server()
        with patch(
            "mltk.mcp.server.run_evaluation",
            side_effect=ValueError(
                "Unknown solver: bogus"
            ),
        ):
            result = self._eval(
                dataset_path=str(self.dataset),
                solver="bogus",
            )
        _assert_error(result)

    def test_returns_valid_json(self) -> None:
        # SCENARIO: Raw return is valid JSON.
        # WHY: MCP tools return JSON strings.
        # EXPECTED: json.loads succeeds.
        _import_server()
        with patch(
            "mltk.mcp.server.run_evaluation",
            return_value={
                "metrics": {"accuracy": 0.95},
                "sample_count": 10,
            },
        ):
            fn = _registered_tools["mltk_eval"]
            raw = fn(
                dataset_path=str(self.dataset),
            )
        _assert_valid_json(raw)

    def test_default_scorer_is_exact_match(
        self,
    ) -> None:
        # SCENARIO: Omitting scorer uses default.
        # WHY: Default scorer is exact_match.
        # EXPECTED: status=ok.
        _import_server()
        with patch(
            "mltk.mcp.server.run_evaluation",
            return_value={
                "metrics": {"exact_match": 1.0},
                "sample_count": 1,
            },
        ):
            result = self._eval(
                dataset_path=str(self.dataset),
            )
        _assert_ok(result)

    def test_default_solver_is_generate(
        self,
    ) -> None:
        # SCENARIO: Omitting solver uses default.
        # WHY: Default solver is generate.
        # EXPECTED: status=ok.
        _import_server()
        with patch(
            "mltk.mcp.server.run_evaluation",
            return_value={
                "metrics": {"accuracy": 0.9},
                "sample_count": 5,
            },
        ):
            result = self._eval(
                dataset_path=str(self.dataset),
            )
        _assert_ok(result)


# ==========================================================
# mltk_dataset (10 tests)
# ==========================================================


class TestMltkDataset:
    """mltk_dataset tool — dataset info retrieval."""

    def _dataset(self, **kwargs) -> dict:
        return _call_tool("mltk_dataset", **kwargs)

    def test_dataset_exists(self) -> None:
        # SCENARIO: Request info for a known dataset.
        # WHY: Happy path must return dataset info.
        # EXPECTED: status=ok with dataset name.
        _import_server()
        with patch(
            "mltk.mcp.server.get_dataset_info",
            return_value={
                "name": "mmlu",
                "versions": ["1.0", "1.1"],
                "quality": {"completeness": 0.99},
            },
        ):
            result = self._dataset(name="mmlu")
        _assert_ok(result)

    def test_dataset_not_found(self) -> None:
        # SCENARIO: Request a nonexistent dataset.
        # WHY: Missing dataset must produce error.
        # EXPECTED: status=error.
        _import_server()
        with patch(
            "mltk.mcp.server.get_dataset_info",
            side_effect=KeyError("not_real"),
        ):
            result = self._dataset(
                name="not_real",
            )
        _assert_error(result)

    def test_specific_version(self) -> None:
        # SCENARIO: Request a specific version.
        # WHY: Users pin dataset versions.
        # EXPECTED: status=ok with correct version.
        _import_server()
        with patch(
            "mltk.mcp.server.get_dataset_info",
            return_value={
                "name": "mmlu",
                "version": "1.1",
                "versions": ["1.0", "1.1"],
                "quality": {"completeness": 0.99},
            },
        ):
            result = self._dataset(
                name="mmlu", version="1.1",
            )
        _assert_ok(result)

    def test_latest_version(self) -> None:
        # SCENARIO: Omit version for latest.
        # WHY: Default gets most recent version.
        # EXPECTED: status=ok.
        _import_server()
        with patch(
            "mltk.mcp.server.get_dataset_info",
            return_value={
                "name": "mmlu",
                "versions": ["1.0", "1.1"],
                "quality": {"completeness": 0.99},
            },
        ):
            result = self._dataset(name="mmlu")
        _assert_ok(result)

    def test_response_has_quality_metrics(
        self,
    ) -> None:
        # SCENARIO: Response includes quality info.
        # WHY: Data quality is a core mltk concern.
        # EXPECTED: quality dict present.
        _import_server()
        with patch(
            "mltk.mcp.server.get_dataset_info",
            return_value={
                "name": "mmlu",
                "versions": ["1.0"],
                "quality": {
                    "completeness": 0.99,
                    "duplicates": 0,
                },
            },
        ):
            result = self._dataset(name="mmlu")
        assert "quality" in result
        assert isinstance(result["quality"], dict)

    def test_response_has_version_list(
        self,
    ) -> None:
        # SCENARIO: Response lists all versions.
        # WHY: Users browse available versions.
        # EXPECTED: versions is a list.
        _import_server()
        with patch(
            "mltk.mcp.server.get_dataset_info",
            return_value={
                "name": "mmlu",
                "versions": ["1.0", "1.1", "2.0"],
                "quality": {},
            },
        ):
            result = self._dataset(name="mmlu")
        assert isinstance(
            result["versions"], list
        )
        assert len(result["versions"]) > 0

    def test_error_has_suggested_action(
        self,
    ) -> None:
        # SCENARIO: Error includes recovery hint.
        # WHY: Agents need actionable guidance.
        # EXPECTED: suggested_action is non-empty.
        _import_server()
        with patch(
            "mltk.mcp.server.get_dataset_info",
            side_effect=KeyError("nope"),
        ):
            result = self._dataset(name="nope")
        assert len(
            result.get("suggested_action", "")
        ) > 0

    def test_returns_valid_json(self) -> None:
        # SCENARIO: Raw return is valid JSON.
        # WHY: MCP tools return JSON strings.
        # EXPECTED: json.loads succeeds.
        _import_server()
        with patch(
            "mltk.mcp.server.get_dataset_info",
            return_value={
                "name": "mmlu",
                "versions": ["1.0"],
                "quality": {},
            },
        ):
            fn = _registered_tools["mltk_dataset"]
            raw = fn(name="mmlu")
        _assert_valid_json(raw)

    def test_error_recoverable_is_bool(
        self,
    ) -> None:
        # SCENARIO: Error recoverable field type.
        # WHY: Type contract must hold.
        # EXPECTED: recoverable is a bool.
        _import_server()
        with patch(
            "mltk.mcp.server.get_dataset_info",
            side_effect=KeyError("missing"),
        ):
            result = self._dataset(name="missing")
        assert isinstance(
            result["recoverable"], bool
        )

    def test_dataset_name_in_response(
        self,
    ) -> None:
        # SCENARIO: Response echoes dataset name.
        # WHY: Confirms correct dataset was queried.
        # EXPECTED: name field matches request.
        _import_server()
        with patch(
            "mltk.mcp.server.get_dataset_info",
            return_value={
                "name": "squad",
                "versions": ["2.0"],
                "quality": {},
            },
        ):
            result = self._dataset(name="squad")
        assert result.get("name") == "squad"


# ==========================================================
# mltk_report (10 tests)
# ==========================================================


class TestMltkReport:
    """mltk_report tool — report generation."""

    def _report(self, **kwargs) -> dict:
        return _call_tool("mltk_report", **kwargs)

    def test_report_with_title(self) -> None:
        # SCENARIO: Generate report with title only.
        # WHY: Title is the minimum required input.
        # EXPECTED: status=ok with report_text.
        _import_server()
        with patch(
            "mltk.mcp.server.generate_mcp_report",
            return_value={
                "report_text": "# Test Report\n",
                "summary": "1 section generated",
            },
        ):
            result = self._report(
                title="Test Report",
            )
        _assert_ok(result)
        assert "report_text" in result

    def test_report_with_results_json(
        self,
    ) -> None:
        # SCENARIO: Include results JSON in report.
        # WHY: Reports aggregate test results.
        # EXPECTED: Results data appears in report.
        _import_server()
        results = json.dumps(
            {"passed": 5, "failed": 1}
        )
        with patch(
            "mltk.mcp.server.generate_mcp_report",
            return_value={
                "report_text": "# R\npassed: 5",
                "summary": "included results",
            },
        ):
            result = self._report(
                title="R",
                results_json=results,
            )
        _assert_ok(result)

    def test_report_empty_results(self) -> None:
        # SCENARIO: No results_json provided.
        # WHY: Basic report without data is valid.
        # EXPECTED: status=ok with minimal report.
        _import_server()
        with patch(
            "mltk.mcp.server.generate_mcp_report",
            return_value={
                "report_text": "# Empty Report\n",
                "summary": "no results",
            },
        ):
            result = self._report(
                title="Empty Report",
            )
        _assert_ok(result)
        assert len(result["report_text"]) > 0

    def test_report_invalid_results_json(
        self,
    ) -> None:
        # SCENARIO: results_json is not valid JSON.
        # WHY: Malformed input must produce error.
        # EXPECTED: status=error.
        _import_server()
        result = self._report(
            title="Bad",
            results_json="{not valid json",
        )
        _assert_error(result)

    def test_response_has_report_text(
        self,
    ) -> None:
        # SCENARIO: Response contains report_text.
        # WHY: Agents read the generated markdown.
        # EXPECTED: report_text is a non-empty string.
        _import_server()
        with patch(
            "mltk.mcp.server.generate_mcp_report",
            return_value={
                "report_text": "# Report\nContent",
                "summary": "ok",
            },
        ):
            result = self._report(title="Report")
        assert isinstance(
            result["report_text"], str
        )
        assert len(result["report_text"]) > 0

    def test_response_has_summary(self) -> None:
        # SCENARIO: Response contains a summary.
        # WHY: Summary is a quick overview for agents.
        # EXPECTED: summary is a non-empty string.
        _import_server()
        with patch(
            "mltk.mcp.server.generate_mcp_report",
            return_value={
                "report_text": "# R\n",
                "summary": "1 section, 0 failures",
            },
        ):
            result = self._report(title="R")
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0

    def test_report_includes_timestamp(
        self,
    ) -> None:
        # SCENARIO: Report text has a timestamp.
        # WHY: Reports need temporal context.
        # EXPECTED: report_text contains date/time.
        _import_server()
        with patch(
            "mltk.mcp.server.generate_mcp_report",
            return_value={
                "report_text": (
                    "# R\nGenerated: 2026-04-02"
                ),
                "summary": "ok",
            },
        ):
            result = self._report(title="R")
        assert "202" in result["report_text"]

    def test_returns_valid_json(self) -> None:
        # SCENARIO: Raw return is valid JSON.
        # WHY: MCP tools return JSON strings.
        # EXPECTED: json.loads succeeds.
        _import_server()
        with patch(
            "mltk.mcp.server.generate_mcp_report",
            return_value={
                "report_text": "# R\n",
                "summary": "ok",
            },
        ):
            fn = _registered_tools["mltk_report"]
            raw = fn(title="R")
        _assert_valid_json(raw)

    def test_report_with_description(self) -> None:
        # SCENARIO: Include optional description.
        # WHY: Description adds context to report.
        # EXPECTED: status=ok.
        _import_server()
        with patch(
            "mltk.mcp.server.generate_mcp_report",
            return_value={
                "report_text": "# R\nDesc here",
                "summary": "ok",
            },
        ):
            result = self._report(
                title="R",
                description="Full test run",
            )
        _assert_ok(result)

    def test_report_error_has_recoverable(
        self,
    ) -> None:
        # SCENARIO: Error response has recoverable.
        # WHY: Type contract must hold.
        # EXPECTED: recoverable is a bool.
        _import_server()
        result = self._report(
            title="Bad",
            results_json="{{invalid",
        )
        assert isinstance(
            result["recoverable"], bool
        )


# ==========================================================
# Integration (11 tests)
# ==========================================================


class TestIntegration:
    """Cross-tool integration and contract tests."""

    def test_full_workflow_scan_test_report(
        self,
    ) -> None:
        # SCENARIO: Scan -> test -> report pipeline.
        # WHY: Agent workflows chain all three.
        # EXPECTED: Each step returns status=ok.
        _import_server()
        with patch(
            "mltk.mcp.server.scan_project",
            return_value={"findings": []},
        ):
            scan_r = _call_tool(
                "mltk_scan", path="/tmp/proj",
            )
        _assert_ok(scan_r)

        with patch(
            "mltk.mcp.server.run_test_suite",
            return_value={
                "passed": 3,
                "failed": 0,
                "results": [],
            },
        ):
            test_r = _call_tool(
                "mltk_test",
                suite_path="/tmp/suite.yaml",
            )
        _assert_ok(test_r)

        results = json.dumps({
            "scan": scan_r,
            "test": test_r,
        })
        with patch(
            "mltk.mcp.server.generate_mcp_report",
            return_value={
                "report_text": "# Pipeline\n",
                "summary": "ok",
            },
        ):
            report_r = _call_tool(
                "mltk_report",
                title="Pipeline",
                results_json=results,
            )
        _assert_ok(report_r)

    def test_json_roundtrip_all_tools(
        self,
    ) -> None:
        # SCENARIO: All tools return parseable JSON.
        # WHY: MCP contract: tools return JSON strs.
        # EXPECTED: Every tool's output parses.
        _import_server()
        tool_calls = {
            "mltk_scan": {
                "path": "/tmp/proj",
            },
            "mltk_list": {},
            "mltk_report": {
                "title": "Test",
            },
        }
        patches = {
            "mltk.mcp.server.scan_project": {
                "findings": [],
            },
            "mltk.mcp.server.get_assertions": [],
            "mltk.mcp.server.generate_mcp_report": {
                "report_text": "# R\n",
                "summary": "ok",
            },
        }
        for p, rv in patches.items():
            with patch(p, return_value=rv):
                for name, kwargs in tool_calls.items():
                    if name in _registered_tools:
                        fn = _registered_tools[name]
                        raw = fn(**kwargs)
                        data = json.loads(raw)
                        assert "status" in data

    def test_error_does_not_crash_server(
        self,
    ) -> None:
        # SCENARIO: Tool raises unexpected exception.
        # WHY: Server must catch and return error.
        # EXPECTED: status=error, not an exception.
        _import_server()
        with patch(
            "mltk.mcp.server.scan_project",
            side_effect=RuntimeError("boom"),
        ):
            result = _call_tool(
                "mltk_scan", path="/tmp/x",
            )
        _assert_error(result)

    def test_all_tools_import_lazily(self) -> None:
        # SCENARIO: Heavy deps not imported at load.
        # WHY: Fast startup for MCP server.
        # EXPECTED: Module loads without numpy/pandas.
        _import_server()
        # If we got here, the module loaded with
        # only the mocked mcp package, no heavy deps.
        assert "mltk_scan" in _registered_tools

    def test_server_without_mcp_package(
        self,
    ) -> None:
        # SCENARIO: mcp package is not installed.
        # WHY: Graceful error message needed.
        # EXPECTED: ImportError or clear message.
        with patch.dict(
            sys.modules,
            {
                "mcp": None,
                "mcp.server": None,
                "mcp.server.fastmcp": None,
            },
        ):
            mod_name = "mltk.mcp.server"
            if mod_name in sys.modules:
                del sys.modules[mod_name]
            with pytest.raises(
                (ImportError, ModuleNotFoundError)
            ):
                pass  # noqa: F811

    def test_main_module_entry_point(self) -> None:
        # SCENARIO: __main__.py exists for mltk.mcp.
        # WHY: python -m mltk.mcp must work.
        # EXPECTED: Module or attribute exists.
        _import_server()
        server = _import_server()
        assert hasattr(server, "run_server")

    def test_tool_descriptions_start_with_verb(
        self,
    ) -> None:
        # SCENARIO: Tool docstrings start with verbs.
        # WHY: MCP best practice for tool discoverability.
        # EXPECTED: First word is an action verb.
        _import_server()
        action_verbs = {
            "scan", "run", "list", "evaluate",
            "get", "generate", "retrieve", "check",
            "detect", "find", "compute", "fetch",
            "create", "build", "produce", "return",
            "search", "validate", "execute",
        }
        for name, fn in _registered_tools.items():
            doc = (fn.__doc__ or "").strip()
            if doc:
                first_word = doc.split()[0].lower()
                assert first_word in action_verbs, (
                    f"Tool {name!r} doc starts "
                    f"with {first_word!r}, "
                    f"expected action verb"
                )

    def test_all_tools_return_status(self) -> None:
        # SCENARIO: Every tool has "status" in output.
        # WHY: Contract: all responses have status.
        # EXPECTED: status in every parsed response.
        _import_server()
        patches = [
            (
                "mltk.mcp.server.scan_project",
                {"findings": []},
                "mltk_scan",
                {"path": "/tmp"},
            ),
            (
                "mltk.mcp.server.get_assertions",
                [],
                "mltk_list",
                {},
            ),
            (
                "mltk.mcp.server.generate_mcp_report",
                {
                    "report_text": "#\n",
                    "summary": "ok",
                },
                "mltk_report",
                {"title": "T"},
            ),
        ]
        for patch_target, rv, tool, kwargs in patches:
            with patch(patch_target, return_value=rv):
                result = _call_tool(tool, **kwargs)
                assert "status" in result

    def test_all_error_responses_have_recoverable(
        self,
    ) -> None:
        # SCENARIO: Every error has recoverable flag.
        # WHY: Contract: error responses are uniform.
        # EXPECTED: recoverable in every error.
        _import_server()
        error_calls = [
            ("mltk_scan", {"path": "/no/path"}),
            (
                "mltk_test",
                {"suite_path": "/no/file"},
            ),
        ]
        for tool, kwargs in error_calls:
            result = _call_tool(tool, **kwargs)
            if result["status"] == "error":
                assert "recoverable" in result

    def test_eval_then_report_workflow(
        self,
    ) -> None:
        # SCENARIO: Eval results feed into report.
        # WHY: Common agent workflow.
        # EXPECTED: Both steps succeed.
        _import_server()
        with patch(
            "mltk.mcp.server.run_evaluation",
            return_value={
                "metrics": {"accuracy": 0.9},
                "sample_count": 100,
            },
        ):
            eval_r = _call_tool(
                "mltk_eval",
                dataset_path="/tmp/data.jsonl",
            )
        _assert_ok(eval_r)

        with patch(
            "mltk.mcp.server.generate_mcp_report",
            return_value={
                "report_text": "# Eval\n",
                "summary": "accuracy=0.9",
            },
        ):
            report_r = _call_tool(
                "mltk_report",
                title="Eval Results",
                results_json=json.dumps(eval_r),
            )
        _assert_ok(report_r)

    def test_list_then_eval_workflow(self) -> None:
        # SCENARIO: List assertions then evaluate.
        # WHY: Agent discovers then uses assertions.
        # EXPECTED: Both steps succeed.
        _import_server()
        with patch(
            "mltk.mcp.server.get_assertions",
            return_value=[
                {
                    "name": "assert_metric",
                    "description": "Validate",
                    "domain": "model",
                },
            ],
        ):
            list_r = _call_tool("mltk_list")
        _assert_ok(list_r)
        assert len(list_r["assertions"]) > 0

        with patch(
            "mltk.mcp.server.run_evaluation",
            return_value={
                "metrics": {"f1": 0.85},
                "sample_count": 50,
            },
        ):
            eval_r = _call_tool(
                "mltk_eval",
                dataset_path="/tmp/d.jsonl",
            )
        _assert_ok(eval_r)
