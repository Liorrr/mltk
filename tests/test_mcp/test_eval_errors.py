"""MCP eval error telemetry regression tests."""

from __future__ import annotations

from ._helpers import assert_ok, call_tool


class TestMcpEvalErrors:
    """The real mltk_eval response exposes per-scorer error counts."""

    def test_response_contains_errors_per_scorer(self, tmp_path):
        # SCENARIO: Run mltk_eval through a real passthrough pipeline.
        # WHY: MCP consumers need scorer-level error telemetry.
        # EXPECTED: errors has the required fields for each scorer.
        dataset = tmp_path / "eval.csv"
        dataset.write_text("input,target\nquestion,answer\n", encoding="utf-8")

        result = call_tool("mltk_eval", dataset_path=str(dataset))

        assert_ok(result)
        assert len(result["errors"]) == 1
        for info in result["errors"].values():
            assert set(info) == {"errored", "n_valid", "error_rate"}
