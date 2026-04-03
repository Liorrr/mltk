"""Tests for mltk_eval MCP tool.

Validates the evaluation pipeline: dataset loading,
solver/scorer selection, metrics response, and error paths.
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

# ----------------------------------------------------------
# Shared mock helpers
# ----------------------------------------------------------

_PATCH_LOAD = "mltk.eval.task.load_dataset"
_PATCH_TASK = "mltk.eval.task.EvalTask"


def _make_sample():
    """Build a mock evaluation sample."""
    s = MagicMock()
    s.input = "2+2"
    s.target = "4"
    return s


def _make_eval_result(
    metrics=None, total_samples=10, duration_ms=50,
):
    """Build a mock EvalTask.run() result."""
    r = MagicMock()
    r.metrics = metrics or {"accuracy": 0.95}
    r.total_samples = total_samples
    r.duration_ms = duration_ms
    return r


def _make_task_cls(result=None):
    """Build a mock EvalTask class whose instances return *result*."""
    mock_cls = MagicMock()
    mock_cls.return_value.run.return_value = (
        result or _make_eval_result()
    )
    return mock_cls


class TestMltkEval:
    """mltk_eval tool — evaluation pipeline."""

    def test_eval_valid_dataset(self, tmp_path) -> None:
        # SCENARIO: Run eval on a valid dataset file.
        # WHY: The happy path must produce metrics.
        # EXPECTED: status=ok, response contains metrics dict.
        ds = tmp_path / "data.csv"
        ds.write_text("input,target\n2+2,4\n", encoding="utf-8")
        with patch(_PATCH_LOAD, return_value=[_make_sample()]), \
             patch(_PATCH_TASK, _make_task_cls()):
            result = call_tool("mltk_eval", dataset_path=str(ds))
        assert_ok(result)
        assert "metrics" in result

    def test_eval_nonexistent_dataset(self) -> None:
        # SCENARIO: Pass a path that does not exist.
        # WHY: Tool must guard against missing files.
        # EXPECTED: status=error mentioning "Not found".
        result = call_tool(
            "mltk_eval", dataset_path="/no/such/file.csv",
        )
        assert_error(result)
        assert "Not found" in result["error"]

    def test_eval_exact_match_scorer(self, tmp_path) -> None:
        # SCENARIO: Explicitly request scorer="exact_match".
        # WHY: Confirms the scorer parameter is accepted.
        # EXPECTED: status=ok, scorer="exact_match" in response.
        ds = tmp_path / "data.csv"
        ds.write_text("input,target\na,b\n", encoding="utf-8")
        with patch(_PATCH_LOAD, return_value=[_make_sample()]), \
             patch(_PATCH_TASK, _make_task_cls()):
            result = call_tool(
                "mltk_eval",
                dataset_path=str(ds),
                scorer="exact_match",
            )
        assert_ok(result)
        assert result["scorer"] == "exact_match"

    def test_eval_custom_solver(self, tmp_path) -> None:
        # SCENARIO: Use solver="chain_of_thought".
        # WHY: Non-default solver must be wired correctly.
        # EXPECTED: status=ok, solver="chain_of_thought" echoed.
        ds = tmp_path / "data.csv"
        ds.write_text("input,target\na,b\n", encoding="utf-8")
        with patch(_PATCH_LOAD, return_value=[_make_sample()]), \
             patch(_PATCH_TASK, _make_task_cls()):
            result = call_tool(
                "mltk_eval",
                dataset_path=str(ds),
                solver="chain_of_thought",
            )
        assert_ok(result)
        assert result["solver"] == "chain_of_thought"

    def test_response_has_metrics_dict(self, tmp_path) -> None:
        # SCENARIO: Inspect the metrics field type.
        # WHY: Agents parse metrics as a dict of floats.
        # EXPECTED: metrics is a dict.
        ds = tmp_path / "data.json"
        ds.write_text('[{"input":"a","target":"b"}]', encoding="utf-8")
        with patch(_PATCH_LOAD, return_value=[_make_sample()]), \
             patch(_PATCH_TASK, _make_task_cls()):
            result = call_tool("mltk_eval", dataset_path=str(ds))
        assert_ok(result)
        assert isinstance(result["metrics"], dict)

    def test_response_has_sample_count(self, tmp_path) -> None:
        # SCENARIO: Verify sample_count is present and positive.
        # WHY: Zero or missing count signals a broken pipeline.
        # EXPECTED: sample_count > 0.
        ds = tmp_path / "data.csv"
        ds.write_text("input,target\na,b\n", encoding="utf-8")
        with patch(_PATCH_LOAD, return_value=[_make_sample()]), \
             patch(_PATCH_TASK, _make_task_cls()):
            result = call_tool("mltk_eval", dataset_path=str(ds))
        assert_ok(result)
        assert result["sample_count"] > 0

    def test_has_suggested_next_step(self, tmp_path) -> None:
        # SCENARIO: Check for the suggested_next_step field.
        # WHY: MCP agents rely on this for workflow guidance.
        # EXPECTED: suggested_next_step is a non-empty string.
        ds = tmp_path / "data.csv"
        ds.write_text("input,target\na,b\n", encoding="utf-8")
        with patch(_PATCH_LOAD, return_value=[_make_sample()]), \
             patch(_PATCH_TASK, _make_task_cls()):
            result = call_tool("mltk_eval", dataset_path=str(ds))
        assert_ok(result)
        assert isinstance(result["suggested_next_step"], str)
        assert len(result["suggested_next_step"]) > 0

    def test_eval_empty_dataset(self, tmp_path) -> None:
        # SCENARIO: load_dataset returns an empty list.
        # WHY: Tool must reject empty datasets gracefully.
        # EXPECTED: status=error, message contains "empty".
        ds = tmp_path / "empty.csv"
        ds.write_text("input,target\n", encoding="utf-8")
        with patch(_PATCH_LOAD, return_value=[]), \
             patch(_PATCH_TASK, _make_task_cls()):
            result = call_tool("mltk_eval", dataset_path=str(ds))
        assert_error(result)
        assert "empty" in result["error"].lower()

    def test_eval_includes_scorer(self, tmp_path) -> None:
        # SCENARIO: Use scorer="includes".
        # WHY: Alternative scorers must work without error.
        # EXPECTED: status=ok, scorer="includes" in response.
        ds = tmp_path / "data.csv"
        ds.write_text("input,target\na,b\n", encoding="utf-8")
        with patch(_PATCH_LOAD, return_value=[_make_sample()]), \
             patch(_PATCH_TASK, _make_task_cls()):
            result = call_tool(
                "mltk_eval",
                dataset_path=str(ds),
                scorer="includes",
            )
        assert_ok(result)
        assert result["scorer"] == "includes"

    def test_returns_valid_json(self, tmp_path) -> None:
        # SCENARIO: Validate raw JSON output string.
        # WHY: MCP transport requires well-formed JSON.
        # EXPECTED: Parseable JSON with "status" field.
        ds = tmp_path / "data.csv"
        ds.write_text("input,target\na,b\n", encoding="utf-8")
        with patch(_PATCH_LOAD, return_value=[_make_sample()]), \
             patch(_PATCH_TASK, _make_task_cls()):
            raw = call_tool_raw(
                "mltk_eval", dataset_path=str(ds),
            )
        assert_valid_json(raw)

    def test_default_scorer_is_exact_match(self, tmp_path) -> None:
        # SCENARIO: Omit the scorer parameter entirely.
        # WHY: Default must be exact_match per the tool signature.
        # EXPECTED: status=ok, scorer="exact_match".
        ds = tmp_path / "data.csv"
        ds.write_text("input,target\na,b\n", encoding="utf-8")
        with patch(_PATCH_LOAD, return_value=[_make_sample()]), \
             patch(_PATCH_TASK, _make_task_cls()):
            result = call_tool("mltk_eval", dataset_path=str(ds))
        assert_ok(result)
        assert result["scorer"] == "exact_match"

    def test_default_solver_is_generate(self, tmp_path) -> None:
        # SCENARIO: Omit the solver parameter entirely.
        # WHY: Default must be generate per the tool signature.
        # EXPECTED: status=ok, solver="generate".
        ds = tmp_path / "data.csv"
        ds.write_text("input,target\na,b\n", encoding="utf-8")
        with patch(_PATCH_LOAD, return_value=[_make_sample()]), \
             patch(_PATCH_TASK, _make_task_cls()):
            result = call_tool("mltk_eval", dataset_path=str(ds))
        assert_ok(result)
        assert result["solver"] == "generate"
