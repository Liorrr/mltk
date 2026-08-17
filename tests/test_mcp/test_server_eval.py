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

    def test_default_model_mode_is_passthrough(self, tmp_path) -> None:
        # SCENARIO: Omit model_mode — honesty fields must still appear.
        # WHY: Agents must not mistake passthrough for a real model.
        # EXPECTED: model_mode=passthrough, model=identity_passthrough.
        ds = tmp_path / "data.csv"
        ds.write_text("input,target\na,b\n", encoding="utf-8")
        with patch(_PATCH_LOAD, return_value=[_make_sample()]), \
             patch(_PATCH_TASK, _make_task_cls()):
            result = call_tool("mltk_eval", dataset_path=str(ds))
        assert_ok(result)
        assert result["model_mode"] == "passthrough"
        assert result["model"] == "identity_passthrough"
        assert result.get("model_ref", "") == ""

    def test_unknown_model_mode_refuses(self, tmp_path) -> None:
        # SCENARIO: model_mode is not supported.
        # WHY: Honest refuse > silent facade.
        # EXPECTED: status=error mentioning model_mode.
        ds = tmp_path / "data.csv"
        ds.write_text("input,target\na,b\n", encoding="utf-8")
        result = call_tool(
            "mltk_eval",
            dataset_path=str(ds),
            model_mode="openai",
        )
        assert_error(result)
        assert "model_mode" in result["error"].lower()

    def test_unknown_scorer_refuses(self, tmp_path) -> None:
        # SCENARIO: scorer name is not supported.
        # WHY: it previously fell back to ExactMatchScorer while
        #   echoing the *requested* name back, so an agent reported
        #   ExactMatch numbers as though the asked-for scorer had run.
        # EXPECTED: recoverable error listing the supported names.
        ds = tmp_path / "data.csv"
        ds.write_text("input,target\na,b\n", encoding="utf-8")
        result = call_tool(
            "mltk_eval", dataset_path=str(ds), scorer="bleu",
        )
        assert_error(result)
        assert "scorer" in result["error"].lower()
        assert "exact_match" in result["error"]

    def test_unknown_solver_refuses(self, tmp_path) -> None:
        # SCENARIO: solver name is not supported.
        # WHY: same silent-fallback shape as scorer — it defaulted to
        #   GenerateSolver while echoing the requested name.
        # EXPECTED: recoverable error listing the supported names.
        ds = tmp_path / "data.csv"
        ds.write_text("input,target\na,b\n", encoding="utf-8")
        result = call_tool(
            "mltk_eval", dataset_path=str(ds), solver="tree_of_thought",
        )
        assert_error(result)
        assert "solver" in result["error"].lower()
        assert "generate" in result["error"]

    def test_llm_judge_scorer_refuses_with_reason(self, tmp_path) -> None:
        # SCENARIO: scorer="llm_judge" — a real scorer that mltk ships,
        #   and which docs/api/mcp-server.md used to advertise here.
        # WHY: LLMJudgeScorer takes a mandatory judge_fn callable and
        #   every MCP parameter is a string, so it cannot be built from
        #   this surface. The caller has not made a typo, so the error
        #   must explain that rather than say "unknown scorer".
        # EXPECTED: error naming judge_fn and pointing to the Python API.
        ds = tmp_path / "data.csv"
        ds.write_text("input,target\na,b\n", encoding="utf-8")
        result = call_tool(
            "mltk_eval", dataset_path=str(ds), scorer="llm_judge",
        )
        assert_error(result)
        assert "judge_fn" in result["error"]
        assert "llm_judge" in result["error"]

    def test_blank_scorer_and_solver_use_documented_defaults(
        self, tmp_path,
    ) -> None:
        # SCENARIO: explicit empty strings for both names.
        # WHY: refusing unknowns must not also refuse "omitted" —
        #   blank means "use the default", matching how model_mode
        #   already treats "".
        # EXPECTED: ok, echoing the documented defaults.
        ds = tmp_path / "data.csv"
        ds.write_text("input,target\na,b\n", encoding="utf-8")
        with patch(_PATCH_LOAD, return_value=[_make_sample()]), \
             patch(_PATCH_TASK, _make_task_cls()):
            result = call_tool(
                "mltk_eval", dataset_path=str(ds),
                scorer="", solver="",
            )
        assert_ok(result)
        assert result["scorer"] == "exact_match"
        assert result["solver"] == "generate"

    def test_echoed_scorer_names_the_scorer_that_ran(
        self, tmp_path,
    ) -> None:
        # SCENARIO: a supported non-default scorer.
        # WHY: the response's "scorer" field is what an agent reports
        #   onward. It must name the class actually constructed, which
        #   is only guaranteed now that unknowns refuse.
        # EXPECTED: echoed name matches the scorer passed to EvalTask.
        from mltk.eval import PatternScorer

        ds = tmp_path / "data.csv"
        ds.write_text("input,target\na,b\n", encoding="utf-8")
        task_cls = _make_task_cls()
        with patch(_PATCH_LOAD, return_value=[_make_sample()]), \
             patch(_PATCH_TASK, task_cls):
            result = call_tool(
                "mltk_eval", dataset_path=str(ds), scorer="pattern",
            )
        assert_ok(result)
        assert result["scorer"] == "pattern"
        passed_scorer = task_cls.call_args.kwargs["scorers"]
        assert isinstance(passed_scorer, PatternScorer)

    def test_module_mode_requires_model_ref(self, tmp_path) -> None:
        # SCENARIO: model_mode=module without model_ref.
        # EXPECTED: recoverable error, no eval run.
        ds = tmp_path / "data.csv"
        ds.write_text("input,target\na,b\n", encoding="utf-8")
        result = call_tool(
            "mltk_eval",
            dataset_path=str(ds),
            model_mode="module",
        )
        assert_error(result)
        assert "model_ref" in result["error"].lower()

    def test_module_mode_blocked_without_allowlist(self, tmp_path) -> None:
        # SCENARIO: module mode with MLTK_MCP_MODEL_MODULES unset.
        # WHY: importlib + getattr resolves any dotted path, and dataset
        #   rows become the callable's argument — so the mode must be
        #   opt-in, not merely documented as "trusted callables only".
        # EXPECTED: recoverable error naming the env var; nothing imported.
        ds = tmp_path / "data.csv"
        ds.write_text("input,target\na,b\n", encoding="utf-8")
        result = call_tool(
            "mltk_eval",
            dataset_path=str(ds),
            model_mode="module",
            model_ref="test_mcp._eval_model_fixture:constant_four",
        )
        assert_error(result)
        assert "MLTK_MCP_MODEL_MODULES" in result["error"]

    def test_module_mode_refuses_module_outside_allowlist(
        self, tmp_path, monkeypatch
    ) -> None:
        # SCENARIO: allowlist is set, but model_ref points outside it.
        # WHY: the prefix check must actually constrain, and must refuse
        #   before import — `os` must never be imported-and-called here.
        # EXPECTED: recoverable error naming the offending module.
        monkeypatch.setenv("MLTK_MCP_MODEL_MODULES", "test_mcp")
        ds = tmp_path / "data.csv"
        ds.write_text("input,target\na,b\n", encoding="utf-8")
        result = call_tool(
            "mltk_eval",
            dataset_path=str(ds),
            model_mode="module",
            model_ref="os:system",
        )
        assert_error(result)
        assert "allowlist" in result["error"].lower()

    def test_module_mode_prefix_match_is_not_substring(
        self, tmp_path, monkeypatch
    ) -> None:
        # SCENARIO: allowlist entry is a prefix of the requested module
        #   name but not a package boundary ("test_mcp" vs "test_mcpevil").
        # WHY: a naive startswith() would admit `test_mcpevil`; the check
        #   must split on a dot boundary.
        # EXPECTED: refused.
        monkeypatch.setenv("MLTK_MCP_MODEL_MODULES", "test_mcp")
        ds = tmp_path / "data.csv"
        ds.write_text("input,target\na,b\n", encoding="utf-8")
        result = call_tool(
            "mltk_eval",
            dataset_path=str(ds),
            model_mode="module",
            model_ref="test_mcpevil.mod:fn",
        )
        assert_error(result)
        assert "allowlist" in result["error"].lower()

    def test_module_mode_non_callable_refuses(
        self, tmp_path, monkeypatch
    ) -> None:
        # SCENARIO: model_ref resolves to an int, not a callable.
        # WHY: honest TypeError refuse, not a silent facade. The fixture
        #   carried `not_a_callable` with a comment claiming it was used
        #   for exactly this, while nothing referenced it and the
        #   `raise TypeError` branch had zero coverage.
        # EXPECTED: recoverable error naming the non-callable resolution.
        monkeypatch.setenv("MLTK_MCP_MODEL_MODULES", "test_mcp")
        ds = tmp_path / "data.csv"
        ds.write_text("input,target\na,b\n", encoding="utf-8")
        result = call_tool(
            "mltk_eval",
            dataset_path=str(ds),
            model_mode="module",
            model_ref="test_mcp._eval_model_fixture:not_a_callable",
        )
        assert_error(result)
        assert "non-callable" in result["error"].lower()

    def test_module_mode_injects_non_identity_model(
        self, tmp_path, monkeypatch
    ) -> None:
        # SCENARIO: reverse_model differs from passthrough on every row.
        # WHY: constant_four returns "4" for a dataset whose targets are
        #   both "4", so that test still scores 1.0 even if the injected
        #   callable were never consulted. reverse_model is
        #   input-dependent, so 1.0 is only reachable if the injected
        #   callable actually produced the output.
        # EXPECTED: ok, 100% accuracy against reversed targets.
        monkeypatch.setenv("MLTK_MCP_MODEL_MODULES", "test_mcp")
        ds = tmp_path / "data.csv"
        ds.write_text("input,target\nabc,cba\nxy,yx\n", encoding="utf-8")
        result = call_tool(
            "mltk_eval",
            dataset_path=str(ds),
            model_mode="module",
            model_ref="test_mcp._eval_model_fixture:reverse_model",
        )
        assert_ok(result)
        acc_keys = [k for k in result["metrics"] if k.endswith("/accuracy")]
        assert acc_keys, f"no accuracy metrics in {result['metrics']}"
        assert result["metrics"][acc_keys[0]] == 1.0

    def test_module_model_injection(self, tmp_path, monkeypatch) -> None:
        # SCENARIO: Inject trusted module callable (constant_four).
        # WHY: D1 real-work path without vendor SDK.
        # EXPECTED: ok, model_mode=module, metrics from injected model.
        monkeypatch.setenv("MLTK_MCP_MODEL_MODULES", "test_mcp")
        ds = tmp_path / "data.csv"
        # GenerateSolver prompt is the input field; constant_four → "4".
        ds.write_text(
            "input,target\n2+2,4\n3+1,4\n",
            encoding="utf-8",
        )
        result = call_tool(
            "mltk_eval",
            dataset_path=str(ds),
            model_mode="module",
            model_ref="test_mcp._eval_model_fixture:constant_four",
        )
        assert_ok(result)
        assert result["model_mode"] == "module"
        assert result["model"] == (
            "test_mcp._eval_model_fixture:constant_four"
        )
        assert result["model_ref"] == (
            "test_mcp._eval_model_fixture:constant_four"
        )
        # Both targets are "4" → perfect exact_match accuracy.
        acc_keys = [
            k for k in result["metrics"] if k.endswith("/accuracy")
        ]
        assert acc_keys, f"no accuracy metrics in {result['metrics']}"
        assert result["metrics"][acc_keys[0]] == 1.0

    def test_passthrough_unmocked_identity_semantics(
        self, tmp_path,
    ) -> None:
        # SCENARIO: Live EvalTask, identity model, mixed match/mismatch.
        # WHY: Pin passthrough semantics without mocking the pipeline.
        # EXPECTED: 50% exact_match accuracy on hello/hello + hello/world.
        ds = tmp_path / "data.csv"
        ds.write_text(
            "input,target\nhello,hello\nhello,world\n",
            encoding="utf-8",
        )
        result = call_tool("mltk_eval", dataset_path=str(ds))
        assert_ok(result)
        assert result["model_mode"] == "passthrough"
        acc_keys = [
            k for k in result["metrics"] if k.endswith("/accuracy")
        ]
        assert acc_keys
        assert result["metrics"][acc_keys[0]] == 0.5
        assert result["sample_count"] == 2
