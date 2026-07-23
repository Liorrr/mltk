"""Slim claim-integrity canaries — fast gate before agents claim “done”.

These call public APIs with tiny synthetic inputs. They do not re-import
the full domain suites. Target wall time for the package: under 30s.

Run::

    python -m pytest tests/regression -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from mltk.cli._discovery import discover_assertions
from mltk.core.assertion import MltkAssertionError
from mltk.core.result import Severity
from mltk.cost import CostTracker, assert_cost_within
from mltk.domains.llm import assert_valid_json
from mltk.domains.llm.mcp import McpToolCall, McpTrace, assert_mcp_tool_selection
from mltk.domains.recommendation import assert_hit_rate

# Sibling helpers for MCP tool invocation (path inserted by conftest).
_TESTS_ROOT = Path(__file__).resolve().parent.parent
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))

from test_mcp._helpers import assert_ok, call_tool  # noqa: E402

pytestmark = pytest.mark.regression

# Measured 2026-07-21 (this session): discover_assertions() → 21 categories,
# 229 unique assert_* names. Exact equality is brittle across minor API churn;
# use a stable band so claim-integrity still fails on catastrophic discovery loss.
_DISCOVERY_MIN_UNIQUE = 200
_DISCOVERY_MAX_UNIQUE = 300
_DISCOVERY_MIN_CATEGORIES = 10


class TestDiscoveryCountSmoke:
    """Public discovery surface stays populated (claim: 200+ assertions)."""

    def test_discover_assertions_categories_and_count_band(self) -> None:
        # SCENARIO: Call discover_assertions with no filter.
        # WHY: Claim integrity for “200+ assertions” / list wiring.
        # EXPECTED: Non-empty categories; unique names in measured band.
        by_category = discover_assertions()
        assert isinstance(by_category, dict)
        assert len(by_category) >= _DISCOVERY_MIN_CATEGORIES

        names = {entry["name"] for items in by_category.values() for entry in items}
        n_unique = len(names)
        assert n_unique >= _DISCOVERY_MIN_UNIQUE, (
            f"discovery unique names {n_unique} < {_DISCOVERY_MIN_UNIQUE} "
            "(measured 229 on 2026-07-21)"
        )
        assert n_unique <= _DISCOVERY_MAX_UNIQUE, (
            f"discovery unique names {n_unique} > {_DISCOVERY_MAX_UNIQUE} "
            "(unexpected double-count or packaging blow-up; measured 229)"
        )
        assert all(n.startswith("assert_") for n in names)


class TestMcpScanHonesty:
    """MCP mltk_scan must not pretend a .py path alone ran ML scanners."""

    def test_py_file_scan_performed_false(
        self,
        tmp_path: Path,
        mcp_scan_registered: None,
    ) -> None:
        # SCENARIO: mltk_scan on a lone Python source file.
        # WHY: Honesty claim — path-only MCP scan is a static listing.
        # EXPECTED: status=ok, scan_performed=False, file listed.
        py_file = tmp_path / "model.py"
        py_file.write_text("x = 1\n", encoding="utf-8")

        result = call_tool("mltk_scan", path=str(py_file))
        assert_ok(result)
        assert result["scan_performed"] is False
        assert "static file listing" in result["message"]
        assert "model.py" in result["python_files"]


class TestEmptyInputRecommendation:
    """Empty recommendation inputs fail closed by default."""

    def test_assert_hit_rate_empty_users_fails(self) -> None:
        # SCENARIO: No users (empty lists), default on_empty.
        # WHY: Fail-closed policy is a claim-integrity surface (not silent pass).
        # EXPECTED: MltkAssertionError; message mentions no users.
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_hit_rate([], [], min_rate=0.5)
        assert "no users provided" in exc_info.value.result.message.lower()
        assert exc_info.value.result.passed is False


class TestStructuredOutput:
    """JSON structured-output happy path + one fail (stdlib json only)."""

    def test_assert_valid_json_happy(self) -> None:
        # SCENARIO: Well-formed JSON object string.
        # WHY: S96 structured-output surface still wired.
        # EXPECTED: passed TestResult with parsed_type dict.
        result = assert_valid_json('{"key": "value", "n": 1}')
        assert result.passed is True
        assert result.details["parsed_type"] == "dict"

    def test_assert_valid_json_fail(self) -> None:
        # SCENARIO: Malformed JSON with WARNING severity (no raise).
        # WHY: Failure path returns structured TestResult.
        # EXPECTED: passed is False; message mentions invalid JSON.
        result = assert_valid_json("{not-json", severity=Severity.WARNING)
        assert result.passed is False
        assert "Invalid JSON" in result.message


class TestCostSmoke:
    """Cost tracker + budget assert with a known pricing table model."""

    def test_assert_cost_within_cheap_tracker(self) -> None:
        # SCENARIO: One small gpt-4o-mini call under a $1 budget.
        # WHY: mltk.cost public API still prices and asserts.
        # EXPECTED: total_cost > 0 and assert_cost_within passes.
        tracker = CostTracker()
        rec = tracker.record("gpt-4o-mini", input_tokens=1_000, output_tokens=500)
        # gpt-4o-mini: 0.15/1M in + 0.60/1M out → 0.00015 + 0.0003 = 0.00045
        assert rec.cost_usd == pytest.approx(0.00045)
        assert tracker.total_cost_usd == pytest.approx(0.00045)
        result = assert_cost_within(tracker, max_usd=1.0)
        assert result.passed is True
        assert result.details["total_cost_usd"] == pytest.approx(0.00045)


class TestMcpEvalAssertExists:
    """One MCP evaluation assertion still importable and runnable offline."""

    def test_assert_mcp_tool_selection_pass(self) -> None:
        # SCENARIO: Trace with one namespaced tool matching expected_tools.
        # WHY: assert_mcp_* public surface without jsonschema optional dep.
        # EXPECTED: selection passes with perfect precision/recall.
        trace = McpTrace(
            tool_calls=[
                McpToolCall(
                    name="read_file",
                    server="filesystem",
                    arguments={"path": "/data.csv"},
                ),
            ],
        )
        result = assert_mcp_tool_selection(
            trace,
            expected_tools=["filesystem::read_file"],
        )
        assert result.passed is True
        assert result.details["precision"] == 1.0
        assert result.details["recall"] == 1.0
