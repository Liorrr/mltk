"""Regression tests for errored eval samples and error policies."""

from __future__ import annotations

import math

import pytest

from mltk.eval._types import EvalResult, EvalSample
from mltk.eval.scorers import ExactMatchScorer, LLMJudgeScorer
from mltk.eval.solvers import GenerateSolver
from mltk.eval.task import EvalTask


def _stateful_judge(values):
    sequence = iter(values)

    def judge(prompt: str):
        value = next(sequence)
        if isinstance(value, BaseException):
            raise value
        return value

    return judge


def _task(judge, samples=3):
    return EvalTask(
        name="judge-errors",
        solver=GenerateSolver(),
        scorers=LLMJudgeScorer(judge, max_score=5),
        dataset=[EvalSample(f"q{i}", "answer") for i in range(samples)],
    )


class TestJudgeParsingAndErrors:
    """LLM judge parsing and finite error handling."""

    def test_json_score_key_is_authoritative(self):
        # SCENARIO: JSON includes an earlier reasoning number and a score.
        # WHY: The score field must win over first-number parsing.
        # EXPECTED: The normalized score uses 8, not 3.
        result = LLMJudgeScorer(
            lambda prompt: '{"reasoning_steps": 3, "score": 8}', max_score=10
        ).score(_state())
        assert result.value == pytest.approx(0.8)
        assert result.errored is False

    def test_first_number_fallback(self):
        # SCENARIO: Judge replies with a slash-formatted score.
        # WHY: Non-JSON replies retain the first-number fallback.
        # EXPECTED: "4/5" normalizes to 0.8.
        result = LLMJudgeScorer(lambda prompt: "4/5", max_score=5).score(_state())
        assert result.value == pytest.approx(0.8)
        assert result.errored is False

    @pytest.mark.parametrize(
        "reply",
        ['{"score": null}', "no number", float("nan"), float("inf"), '{"score": NaN}'],
    )
    def test_invalid_or_nonfinite_reply_is_errored(self, reply):
        # SCENARIO: Judge produces no usable finite score.
        # WHY: Invalid judge output must not become a valid zero.
        # EXPECTED: Finite zero value, errored flag, and metadata error.
        result = LLMJudgeScorer(lambda prompt: reply).score(_state())
        assert result.value == 0.0
        assert math.isfinite(result.value)
        assert result.errored is True
        assert "error" in result.metadata


class TestErrorAggregation:
    """Per-scorer error counts and valid-only aggregation."""

    def test_errors_are_excluded_from_metrics(self):
        # SCENARIO: Judge passes, raises, then passes across three samples.
        # WHY: One failed sample must not lower valid-only aggregates.
        # EXPECTED: Mean and accuracy are 1.0 with one recorded error.
        result = _task(_stateful_judge(["5", RuntimeError("down"), "5"])).run(
            lambda prompt: "answer"
        )
        assert result.metrics["LLMJudge/correctness/accuracy"] == 1.0
        assert result.metrics["LLMJudge/correctness/mean"] == 1.0
        assert result.errors["LLMJudge/correctness"] == {
            "errored": 1,
            "n_valid": 2,
            "error_rate": 0.3333,
        }

    def test_clean_scorer_has_zero_error_entry(self):
        # SCENARIO: Exact match scores every sample cleanly.
        # WHY: Clean scorers need explicit error telemetry too.
        # EXPECTED: Error count is zero and valid count is N.
        result = EvalTask(
            name="clean",
            solver=GenerateSolver(),
            scorers=ExactMatchScorer(),
            dataset=[EvalSample("q", "answer")],
        ).run(lambda prompt: "answer")
        assert result.errors["ExactMatchScorer"] == {
            "errored": 0,
            "n_valid": 1,
            "error_rate": 0.0,
        }

    def test_no_target_zero_is_not_errored(self):
        # SCENARIO: Exact match has no target.
        # WHY: Configuration zero is not a scorer failure.
        # EXPECTED: Zero remains valid and unflagged.
        result = EvalTask(
            name="no-target",
            solver=GenerateSolver(),
            scorers=ExactMatchScorer(),
            dataset=[EvalSample("q")],
        ).run(lambda prompt: "answer")
        score = result.scores["ExactMatchScorer"][0]
        assert score.value == 0.0
        assert score.errored is False
        assert result.errors["ExactMatchScorer"]["errored"] == 0


class TestErrorPolicies:
    """to_test_result error policy and valid-fraction gates."""

    def test_fail_if_all_allows_some_errors_and_reports_all_errors(self):
        # SCENARIO: Clean exact match plus an all-errored judge, then a partial judge.
        # WHY: fail_if_all protects dead scorers but tolerates isolated failures.
        # EXPECTED: All-errored run raises; one valid judge sample passes.
        samples = [EvalSample("q", "answer") for _ in range(3)]
        dead = EvalTask(
            name="dead-judge",
            solver=GenerateSolver(),
            scorers=[ExactMatchScorer(), LLMJudgeScorer(lambda prompt: 1 / 0)],
            dataset=samples,
        )
        from mltk.core.assertion import MltkAssertionError

        with pytest.raises(MltkAssertionError) as exc:
            dead.to_test_result(lambda prompt: "answer")
        assert exc.value.result.message.startswith("Eval '")
        assert "LLMJudge/correctness" in exc.value.result.message
        assert "on_error='fail_if_all'" in exc.value.result.message

        partial = _task(_stateful_judge([RuntimeError("one"), "5", RuntimeError("two")]))
        assert partial.to_test_result(lambda prompt: "answer").passed is True

    def test_strict_fails_one_error(self):
        # SCENARIO: One of three otherwise passing judge calls raises.
        # WHY: strict must fail on any scorer error.
        # EXPECTED: Critical assertion names strict policy.
        from mltk.core.assertion import MltkAssertionError

        with pytest.raises(MltkAssertionError, match="on_error='strict'"):
            _task(_stateful_judge(["5", RuntimeError("down"), "5"])).to_test_result(
                lambda prompt: "answer", on_error="strict"
            )

    @pytest.mark.parametrize("reply", [RuntimeError("down"), "n/a"])
    def test_zero_reproduces_old_gate(self, reply):
        # SCENARIO: Scores are [1.0, errored zero, 0.6] under zero policy.
        # WHY: zero is the explicit compatibility gate for old averaging.
        # EXPECTED: Mean is 0.5333 and accuracy is 0.6667, while run() stays valid-only.
        result = _task(_stateful_judge(["5", reply, "3"])).run(
            lambda prompt: "answer"
        )
        assert result.metrics["LLMJudge/correctness/mean"] == pytest.approx(0.8)
        checked = _task(_stateful_judge(["5", reply, "3"])).to_test_result(
            lambda prompt: "answer", min_accuracy=0.5, on_error="zero"
        )
        assert checked.details["metrics"]["LLMJudge/correctness/mean"] == pytest.approx(0.5333)
        assert checked.details["metrics"]["LLMJudge/correctness/accuracy"] == pytest.approx(0.6667)

    @pytest.mark.parametrize("policy", ["fail_if_all", "strict", "zero"])
    def test_min_valid_fraction_fails_below_floor(self, policy):
        # SCENARIO: One of three samples errors and floor is 0.9.
        # WHY: The floor applies consistently to every policy.
        # EXPECTED: The result fails with the floor reason.
        from mltk.core.assertion import MltkAssertionError

        with pytest.raises(MltkAssertionError, match="valid fraction 0.6667"):
            _task(_stateful_judge(["5", RuntimeError("down"), "5"])).to_test_result(
                lambda prompt: "answer", on_error=policy, min_valid_fraction=0.9
            )

    def test_floor_boundary_and_none_are_allowed(self):
        # SCENARIO: Valid fraction is exactly 2/3.
        # WHY: The floor is inclusive and None is inert.
        # EXPECTED: Both calls pass.
        task = _task(_stateful_judge(["5", RuntimeError("down"), "5"]))
        assert task.to_test_result(lambda prompt: "answer", min_valid_fraction=2 / 3).passed
        task = _task(_stateful_judge(["5", RuntimeError("down"), "5"]))
        assert task.to_test_result(lambda prompt: "answer", min_valid_fraction=None).passed

    @pytest.mark.parametrize("floor", [0, 1.5, -0.1, True])
    def test_invalid_inputs_make_zero_model_calls(self, floor):
        # SCENARIO: Invalid policy/floor is supplied.
        # WHY: Validation must precede model execution.
        # EXPECTED: Exception is raised with zero model calls.
        calls = 0

        def model(prompt: str) -> str:
            nonlocal calls
            calls += 1
            return "answer"

        task = _task(lambda prompt: "5")
        with pytest.raises(ValueError, match="min_valid_fraction"):
            task.to_test_result(model, min_valid_fraction=floor)
        assert calls == 0

    def test_invalid_on_error_makes_zero_model_calls(self):
        # SCENARIO: Unsupported on_error value is supplied.
        # WHY: Policy typos must fail before a judge/model call.
        # EXPECTED: MltkAssertionError and zero model calls.
        from mltk.core.assertion import MltkAssertionError

        calls = 0

        def model(prompt: str) -> str:
            nonlocal calls
            calls += 1
            return "answer"

        with pytest.raises(MltkAssertionError):
            _task(lambda prompt: "5").to_test_result(model, on_error="resist")
        assert calls == 0

    def test_details_include_error_policy_fields(self):
        # SCENARIO: A passing run is converted to a TestResult.
        # WHY: Consumers need policy and telemetry in assertion details.
        # EXPECTED: Details contain errors, policy, and floor.
        result = _task(lambda prompt: "5").to_test_result(
            lambda prompt: "answer", min_valid_fraction=1.0
        )
        assert "errors" in result.details
        assert result.details["on_error"] == "fail_if_all"
        assert result.details["min_valid_fraction"] == 1.0


class TestEvalResultPassed:
    """EvalResult passed behavior with and without all-errored scorers."""

    def test_all_errored_scorer_fails_even_with_clean_metrics(self):
        # SCENARIO: One scorer is all errored and another has perfect metrics.
        # WHY: A dead scorer must not be hidden by another scorer.
        # EXPECTED: passed is False.
        result = EvalResult(
            task_name="mixed",
            metrics={"Clean/accuracy": 1.0},
            errors={"Dead": {"errored": 2, "n_valid": 0, "error_rate": 1.0}},
        )
        assert result.passed is False

    @pytest.mark.parametrize(("metric", "expected"), [(1.0, True), (0.4, False)])
    def test_clean_run_preserves_metric_rule(self, metric, expected):
        # SCENARIO: Direct clean EvalResult uses an above/below threshold metric.
        # WHY: Error tracking must not change clean-run semantics.
        # EXPECTED: passed follows the existing >= 0.5 rule.
        result = EvalResult(
            task_name="clean",
            metrics={"Scorer/accuracy": metric},
            errors={"Scorer": {"errored": 0, "n_valid": 2, "error_rate": 0.0}},
        )
        assert result.passed is expected


def _state():
    from mltk.eval._types import EvalState

    return EvalState(sample=EvalSample("question", "answer"), output="answer")
