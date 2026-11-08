"""Stored-output scoring preserves live scoring and assertion error contracts."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.eval import EvalSample, EvalTask, ExactMatchScorer, GenerateSolver
from mltk.eval.scorers import IncludesScorer, Scorer


class RaisingScorer(Scorer):
    def score(self, state):
        raise RuntimeError("judge unavailable")


def make_task(scorers=None, samples=None, solver=None):
    return EvalTask(
        "stored",
        solver if solver is not None else GenerateSolver(),
        scorers if scorers is not None else ExactMatchScorer(),
        samples if samples is not None else [
            EvalSample("first", "yes", sample_id="a"),
            EvalSample("second", "no", sample_id="b"),
        ],
    )


def test_live_stored_equality_including_scorer_errors():
    task = make_task([ExactMatchScorer(), IncludesScorer(), RaisingScorer()])
    live = task.run({"first": "yes", "second": "wrong"}.__getitem__)
    stored = task.score({"b": "wrong", "a": "yes"})
    assert stored.metrics == live.metrics
    assert stored.errors == live.errors
    assert stored.scores == live.scores
    assert [s.sample.sample_id for s in stored.samples] == ["a", "b"]
    assert [s.output for s in stored.samples] == [s.output for s in live.samples]


def test_stored_never_calls_solver_and_missing_output_skips_scorers():
    solver = Mock()
    solver.solve.side_effect = AssertionError("must not generate")
    scorer = ExactMatchScorer()
    scorer.score = Mock(wraps=scorer.score)
    task = make_task([scorer, RaisingScorer()], solver=solver)
    result = task.score({"a": "yes"})
    solver.solve.assert_not_called()
    assert scorer.score.call_count == 1
    assert result.total_samples == 2
    assert result.errors["ExactMatchScorer"] == {
        "errored": 1, "n_valid": 1, "error_rate": 0.5,
    }
    assert result.metrics["ExactMatchScorer/mean"] == 1.0
    for scores in result.scores.values():
        assert scores[1].errored
        assert scores[1].metadata["sample_id"] == "b"
        assert "Missing stored output" in scores[1].metadata["error"]


@pytest.mark.parametrize("ids", [(None, "b"), ("a", "a"), ("", "b"), (" ", "b"), (1, "b")])
def test_invalid_or_duplicate_ids_rejected_before_scoring(ids):
    scorer = ExactMatchScorer()
    scorer.score = Mock(side_effect=AssertionError("must validate first"))
    task = make_task(scorer, [EvalSample("q", "yes", sample_id=i) for i in ids])
    with pytest.raises(ValueError, match="sample_id"):
        task.score({"a": "yes"})
    scorer.score.assert_not_called()


def test_live_run_still_allows_absent_or_duplicate_ids():
    for ids in [(None, None), ("same", "same")]:
        task = make_task(samples=[EvalSample("q", "yes", sample_id=i) for i in ids])
        assert task.run(lambda _: "yes").metrics["ExactMatchScorer/mean"] == 1.0


def test_unknown_ids_rejected_and_mapping_required():
    task = make_task()
    with pytest.raises(ValueError, match="unknown"):
        task.score({"a": "yes", "extra": "no"})
    with pytest.raises(TypeError, match="mapping"):
        task.score(["yes", "no"])


def test_empty_output_valid_but_non_string_errored():
    result = make_task().score({"a": "", "b": None})
    assert not result.scores["ExactMatchScorer"][0].errored
    assert result.scores["ExactMatchScorer"][1].errored
    assert result.errors["ExactMatchScorer"]["n_valid"] == 1


@pytest.mark.parametrize(
    ("policy", "passes"), [("fail_if_all", True), ("strict", False), ("zero", False)],
)
def test_stored_bridge_error_policies(policy, passes):
    task = make_task()
    kwargs = {"outputs": {"a": "yes"}, "on_error": policy}
    if passes:
        assert task.to_test_result(**kwargs).passed
    else:
        with pytest.raises(MltkAssertionError):
            task.to_test_result(**kwargs)


@pytest.mark.parametrize("policy", ["fail_if_all", "strict", "zero"])
def test_valid_fraction_enforced_for_every_policy(policy):
    with pytest.raises(MltkAssertionError, match="valid fraction"):
        make_task().to_test_result(
            outputs={"a": "yes"}, min_accuracy=0, on_error=policy,
            min_valid_fraction=0.75,
        )


def test_all_missing_outputs_are_not_a_vacuous_pass():
    task = make_task()
    result = task.score({})
    assert result.metrics == {}
    assert not result.passed
    assert result.errors["ExactMatchScorer"]["errored"] == 2
    with pytest.raises(MltkAssertionError, match="all 2 samples errored"):
        task.to_test_result(outputs={})


def test_zero_policy_uses_zero_values_and_valid_fraction_boundary():
    result = make_task().to_test_result(
        outputs={"a": "yes"}, on_error="zero", min_accuracy=0.5,
        min_valid_fraction=0.5,
    )
    assert result.passed
    assert result.details["metrics"]["ExactMatchScorer/mean"] == 0.5


@pytest.mark.parametrize("fraction", [0, -1, 1.1, True, float("nan")])
def test_invalid_valid_fraction_rejected(fraction):
    with pytest.raises(ValueError, match="min_valid_fraction"):
        make_task().to_test_result(outputs={}, min_valid_fraction=fraction)


def test_bridge_requires_one_input_source_and_preserves_invalid_policy():
    task = make_task()
    with pytest.raises(ValueError, match="exactly one"):
        task.to_test_result()
    with pytest.raises(ValueError, match="exactly one"):
        task.to_test_result(lambda _: "yes", outputs={})
    with pytest.raises(MltkAssertionError):
        task.to_test_result(outputs={}, on_error="unknown")
