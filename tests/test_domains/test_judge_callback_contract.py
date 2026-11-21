"""Contract tests for numeric judge callbacks (``judge_fn`` returning a number).

Every assertion that calls a numeric ``judge_fn`` validates the returned value
the same way: ``float()`` coercion, then a judge **error** for ``None``, a
boolean, a non-finite value or a value outside ``[0, 1]``. Errors are governed
by a keyword-only ``on_error`` policy (``fail_if_all``, ``strict``, ``zero``).
"""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from typing import Any

import numpy as np
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.core.result import TestResult as Result
from mltk.domains.llm.behavioral.invariance import assert_paraphrase_invariance
from mltk.domains.llm.rag import (
    assert_answer_relevancy,
    assert_context_relevancy,
    assert_faithfulness,
)
from mltk.domains.llm.safety import assert_no_hallucination
from mltk.domains.llm.span import Span, SpanKind, SpanTrace
from mltk.domains.llm.span_eval import assert_span_quality
from mltk.eval import EvalSample, EvalState, LLMJudgeScorer
from mltk.integrations.trace_quality import assert_trace_quality


def _run(call: Callable[[], Result]) -> Result:
    """Return the result whether the assertion passed or raised."""
    try:
        return call()
    except MltkAssertionError as exc:
        return exc.result


def _const(value: object) -> Callable[..., object]:
    return lambda *args: value


def _raises(*args: object) -> float:
    raise RuntimeError("judge unavailable")


# ---------------------------------------------------------------------------
# Every numeric judge path, as (judge, threshold, **policy) -> TestResult
# ---------------------------------------------------------------------------


def _faithfulness(judge: Callable[..., object], threshold: float = 0.5, **kw: Any) -> Result:
    return _run(lambda: assert_faithfulness(
        "Paris is the capital.", "Paris is the capital of France.",
        method="llm", judge_fn=judge, min_score=threshold, **kw,
    ))


def _context_relevancy(
    judge: Callable[..., object], threshold: float = 0.5, **kw: Any,
) -> Result:
    return _run(lambda: assert_context_relevancy(
        "What is the capital of France?", "Paris is the capital of France.",
        method="llm", judge_fn=judge, min_score=threshold, **kw,
    ))


def _answer_relevancy(
    judge: Callable[..., object], threshold: float = 0.5, **kw: Any,
) -> Result:
    return _run(lambda: assert_answer_relevancy(
        "What is the capital of France?", "Paris is the capital.",
        method="llm", judge_fn=judge, min_score=threshold, **kw,
    ))


def _hallucination(judge: Callable[..., object], threshold: float = 0.5, **kw: Any) -> Result:
    return _run(lambda: assert_no_hallucination(
        ["Paris is the capital."], ["Paris is the capital of France."],
        method="llm", judge_fn=judge, min_coverage=threshold, **kw,
    ))


def _invariance(judge: Callable[..., object], threshold: float = 0.5, **kw: Any) -> Result:
    # Multi-word outputs, so the classifier auto-switch to label_match never fires.
    return _run(lambda: assert_paraphrase_invariance(
        lambda prompt: "The answer is forty two.",
        ["What is the answer?", "Tell me the answer."],
        equivalence_method="judge", judge_fn=judge,
        similarity_threshold=threshold, min_invariance=0.5, **kw,
    ))


def _span_quality(judge: Callable[..., object], threshold: float = 0.5, **kw: Any) -> Result:
    trace = SpanTrace(spans=[Span(name="answer", kind=SpanKind.LLM)])
    return _run(lambda: assert_span_quality(trace, judge_fn=judge, min_score=threshold, **kw))


def _trace_quality(judge: Callable[..., object], threshold: float = 0.5, **kw: Any) -> Result:
    return _run(lambda: assert_trace_quality(
        {"output": "Paris"}, min_score=threshold, judge_fn=judge, **kw,
    ))


THRESHOLD_PATHS = {
    "faithfulness": _faithfulness,
    "context_relevancy": _context_relevancy,
    "answer_relevancy": _answer_relevancy,
    "no_hallucination": _hallucination,
    "paraphrase_invariance": _invariance,
    "span_quality": _span_quality,
    "trace_quality": _trace_quality,
}


def _scorer(judge: Callable[..., object]) -> tuple[bool, int]:
    state = EvalState(sample=EvalSample(input="q", target="a"), output="a")
    score = LLMJudgeScorer(judge_fn=judge, max_score=1.0).score(state)
    return (not score.errored and score.value >= 0.5), int(score.errored)


def _outcome(path: str, judge: Callable[..., object]) -> tuple[bool, int]:
    """(passed, errored_count) for one judge call on *path*."""
    if path == "LLMJudgeScorer":
        return _scorer(judge)
    result = THRESHOLD_PATHS[path](judge)
    return result.passed, result.details["errored_count"]


ALL_PATHS = [*THRESHOLD_PATHS, "LLMJudgeScorer"]

ACCEPTED = [
    pytest.param(0.8, id="0.8"),
    pytest.param(np.float32(0.8), id="np.float32"),
    pytest.param("0.8", id="str"),
    pytest.param(Decimal("0.8"), id="Decimal"),
]
ERRORS = [
    pytest.param(True, id="True"),
    pytest.param(np.bool_(True), id="np.bool_"),
    pytest.param(None, id="None"),
    pytest.param(float("nan"), id="NaN"),
    pytest.param(float("inf"), id="inf"),
    pytest.param(-0.1, id="-0.1"),
    pytest.param(1.5, id="1.5"),
    pytest.param(7.0, id="7.0"),
]


# ---------------------------------------------------------------------------
# D2: the value matrix, on every path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ALL_PATHS)
@pytest.mark.parametrize("value", ACCEPTED)
def test_valid_score_is_accepted_on_every_path(path: str, value: object) -> None:
    assert _outcome(path, _const(value)) == (True, 0)


@pytest.mark.parametrize("path", ALL_PATHS)
@pytest.mark.parametrize("value", ERRORS)
def test_non_score_is_a_judge_error_on_every_path(path: str, value: object) -> None:
    assert _outcome(path, _const(value)) == (False, 1)


@pytest.mark.parametrize("path", ALL_PATHS)
def test_raising_judge_is_a_judge_error_on_every_path(path: str) -> None:
    assert _outcome(path, _raises) == (False, 1)


@pytest.mark.parametrize("path", list(THRESHOLD_PATHS))
def test_error_details_name_the_policy_and_the_error(path: str) -> None:
    result = THRESHOLD_PATHS[path](_const(7.0))
    assert result.details["n_valid"] == 0
    assert result.details["on_error"] in ("fail_if_all", "strict")
    [error] = result.details["errors"]
    assert "outside [0, 1.0]" in error["error"]


@pytest.mark.parametrize("path", list(THRESHOLD_PATHS))
def test_valid_score_below_threshold_fails_without_an_error(path: str) -> None:
    # Mirror: a low but valid score is a measured failure, not a judge error.
    result = THRESHOLD_PATHS[path](_const(0.1))
    assert result.passed is False
    assert result.details["errored_count"] == 0
    assert result.details["n_valid"] == 1


def test_scorer_keeps_its_own_scale() -> None:
    # LLMJudgeScorer checks [0, max_score], not [0, 1]; booleans still error.
    state = EvalState(sample=EvalSample(input="q", target="a"), output="a")
    on_five = LLMJudgeScorer(judge_fn=_const(4.0)).score(state)
    assert not on_five.errored
    assert on_five.value == pytest.approx(0.8)
    assert LLMJudgeScorer(judge_fn=_const(7.0)).score(state).errored
    assert LLMJudgeScorer(judge_fn=_const(True)).score(state).errored
    assert LLMJudgeScorer(judge_fn=_const(np.bool_(True))).score(state).errored
    one = LLMJudgeScorer(judge_fn=_const(1)).score(state)
    assert not one.errored
    assert one.value == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# D3: on_error by shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["faithfulness", "context_relevancy", "answer_relevancy",
                                  "trace_quality"])
@pytest.mark.parametrize("on_error", ["fail_if_all", "strict", "zero"])
def test_single_call_judge_error_fails_under_every_policy(path: str, on_error: str) -> None:
    result = THRESHOLD_PATHS[path](_raises, on_error=on_error)
    assert result.passed is False
    assert result.details["errored_count"] == 1
    assert result.details["on_error"] == on_error


@pytest.mark.parametrize("path", list(THRESHOLD_PATHS))
def test_unknown_on_error_fails(path: str) -> None:
    result = THRESHOLD_PATHS[path](_const(0.8), on_error="ignore")
    assert result.passed is False
    assert "Unknown on_error" in result.message


def _two_claims(judge: Callable[[str, str], object], **kw: Any) -> Result:
    return _run(lambda: assert_no_hallucination(
        ["Paris is the capital.", "France is in Europe."],
        ["Paris is the capital of France, in Europe."],
        method="llm", judge_fn=judge, min_coverage=0.5, **kw,
    ))


def _second_claim_errors(claim: str, source: str) -> float:
    if claim.startswith("France"):
        raise RuntimeError("judge unavailable")
    return 0.9


def test_unjudgeable_claim_fails_no_hallucination_by_default() -> None:
    result = _two_claims(_second_claim_errors)
    assert result.passed is False
    assert result.details["on_error"] == "strict"
    assert result.details["errored_count"] == 1
    assert result.details["n_valid"] == 1
    assert result.details["errors"] == [
        {"index": 1, "error": "RuntimeError: judge unavailable"},
    ]


def test_unjudgeable_claim_is_excluded_under_fail_if_all() -> None:
    result = _two_claims(_second_claim_errors, on_error="fail_if_all")
    assert result.passed is True
    assert result.details["errored_count"] == 1
    assert result.details["unsupported_count"] == 0


def test_unjudgeable_claim_is_unsupported_under_zero() -> None:
    result = _two_claims(_second_claim_errors, on_error="zero")
    assert result.passed is False
    assert result.details["unsupported_count"] == 1
    assert result.details["avg_coverage"] == pytest.approx(0.45)


def test_all_claims_unjudgeable_fail_under_fail_if_all() -> None:
    result = _two_claims(_raises, on_error="fail_if_all")
    assert result.passed is False
    assert result.details["n_valid"] == 0


def test_judged_claims_all_valid_pass_by_default() -> None:
    # Mirror of the strict default: no error, every claim supported -> pass.
    result = _two_claims(_const(0.9))
    assert result.passed is True
    assert result.details["errored_count"] == 0
    assert result.details["n_valid"] == 2


def _two_spans(judge: Callable[[Span], object], **kw: Any) -> Result:
    trace = SpanTrace(spans=[
        Span(name="plan", kind=SpanKind.LLM),
        Span(name="answer", kind=SpanKind.LLM),
    ])
    return _run(lambda: assert_span_quality(trace, judge_fn=judge, min_score=0.5, **kw))


def _answer_span_errors(span: Span) -> float:
    if span.name == "answer":
        raise RuntimeError("judge unavailable")
    return 0.9


def test_unjudgeable_span_fails_span_quality_by_default() -> None:
    result = _two_spans(_answer_span_errors)
    assert result.passed is False
    assert result.details["on_error"] == "strict"
    assert result.details["errored_count"] == 1
    assert result.details["error_count"] == 0  # no error *spans*


def test_unjudgeable_span_is_excluded_under_fail_if_all() -> None:
    result = _two_spans(_answer_span_errors, on_error="fail_if_all")
    assert result.passed is True
    assert result.details["errored_count"] == 1
    assert result.details["n_valid"] == 1


def test_unjudgeable_span_fails_under_zero() -> None:
    result = _two_spans(_answer_span_errors, on_error="zero")
    assert result.passed is False
    scores = {item["name"]: item["score"] for item in result.details["per_span_scores"]}
    assert scores == {"plan": 0.9, "answer": 0.0}


def test_judge_exception_no_longer_propagates_from_span_quality() -> None:
    result = _two_spans(_raises)
    assert result.passed is False
    assert result.details["errored_count"] == 2


def _three_paraphrases(judge: Callable[[str, str], object], **kw: Any) -> Result:
    outputs = {
        "q1": "The answer is one.",
        "q2": "The answer is two.",
        "q3": "The answer is three.",
    }
    return _run(lambda: assert_paraphrase_invariance(
        outputs.__getitem__, list(outputs),
        equivalence_method="judge", judge_fn=judge,
        similarity_threshold=0.5, min_invariance=0.8, **kw,
    ))


def _first_pair_errors(a: str, b: str) -> float:
    if {a, b} == {"The answer is one.", "The answer is two."}:
        raise RuntimeError("judge unavailable")
    return 0.9


def test_errored_pair_is_excluded_by_default() -> None:
    result = _three_paraphrases(_first_pair_errors)
    assert result.passed is True
    assert result.details["on_error"] == "fail_if_all"
    assert result.details["errored_count"] == 1
    assert result.details["n_valid"] == 2
    assert result.details["invariance_rate"] == 1.0


def test_errored_pair_fails_under_strict() -> None:
    result = _three_paraphrases(_first_pair_errors, on_error="strict")
    assert result.passed is False


def test_errored_pair_is_non_equivalent_under_zero() -> None:
    result = _three_paraphrases(_first_pair_errors, on_error="zero")
    assert result.passed is False
    assert result.details["invariance_rate"] == pytest.approx(2 / 3, abs=1e-4)
    errored = [p for p in result.details["pair_scores"] if p.get("errored")]
    assert errored == [{
        "pair": (0, 1), "score": 0.0, "equivalent": False, "errored": True,
        "error": "RuntimeError: judge unavailable",
    }]


def test_all_pairs_errored_fail_by_default() -> None:
    result = _three_paraphrases(_raises)
    assert result.passed is False
    assert result.details["n_valid"] == 0
    assert result.details["invariance_rate"] is None


def test_min_valid_fraction_fails_too_many_errors() -> None:
    result = _three_paraphrases(_first_pair_errors, min_valid_fraction=0.9)
    assert result.passed is False
    assert "min_valid_fraction" in result.message


def test_min_valid_fraction_passes_enough_valid_pairs() -> None:
    # Mirror: 2/3 valid meets a 0.6 floor.
    result = _three_paraphrases(_first_pair_errors, min_valid_fraction=0.6)
    assert result.passed is True
    assert result.details["min_valid_fraction"] == 0.6


@pytest.mark.parametrize("fraction", [0, -0.1, 1.5, True, "0.5"])
def test_min_valid_fraction_out_of_range_raises(fraction: object) -> None:
    with pytest.raises(ValueError, match="min_valid_fraction"):
        _three_paraphrases(_const(0.9), min_valid_fraction=fraction)


def test_label_match_switch_never_calls_the_judge() -> None:
    # Single-token outputs switch to label_match; every pair is valid.
    def judge(a: str, b: str) -> float:
        raise AssertionError("judge must not run")

    result = assert_paraphrase_invariance(
        lambda prompt: "yes", ["q1", "q2", "q3"],
        equivalence_method="judge", judge_fn=judge,
    )
    assert result.details["method"] == "label_match"
    assert result.details["errored_count"] == 0
    assert result.details["n_valid"] == 3


# ---------------------------------------------------------------------------
# D4: zero keeps its meaning; a threshold an errored 0.0 meets raises
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", list(THRESHOLD_PATHS))
@pytest.mark.parametrize("threshold", [0.0, -0.1])
def test_zero_with_non_positive_threshold_raises(path: str, threshold: float) -> None:
    calls: list[object] = []

    def judge(*args: object) -> float:
        calls.append(args)
        return 0.9

    with pytest.raises(ValueError, match="on_error='zero'"):
        THRESHOLD_PATHS[path](judge, threshold=threshold, on_error="zero")
    assert calls == []  # raised up front


@pytest.mark.parametrize("path", list(THRESHOLD_PATHS))
@pytest.mark.parametrize("on_error", ["fail_if_all", "strict"])
def test_other_policies_accept_a_zero_threshold(path: str, on_error: str) -> None:
    # Mirror: only zero is restricted; a valid score meets a 0 threshold.
    result = THRESHOLD_PATHS[path](_const(0.9), threshold=0.0, on_error=on_error)
    assert result.passed is True


def _zero_scored(path: str, result: Result) -> float:
    details = result.details
    if path == "no_hallucination":
        assert details["unsupported_count"] == 1
        return details["avg_coverage"]
    if path == "paraphrase_invariance":
        [pair] = details["pair_scores"]
        assert pair["equivalent"] is False
        return pair["score"]
    if path == "span_quality":
        [span] = details["per_span_scores"]
        return span["score"]
    return details["score"]


@pytest.mark.parametrize("path", list(THRESHOLD_PATHS))
def test_zero_with_positive_threshold_scores_error_as_zero_and_fails(path: str) -> None:
    result = THRESHOLD_PATHS[path](_raises, threshold=0.5, on_error="zero")
    assert result.passed is False
    assert result.details["errored_count"] == 1
    assert _zero_scored(path, result) == 0.0


# ---------------------------------------------------------------------------
# D5: empty input for no_hallucination and span_quality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("claims", [[], [""], ["   ", "\t\n"]], ids=["none", "blank", "ws"])
def test_no_claims_fail_by_default(claims: list[str]) -> None:
    result = _run(lambda: assert_no_hallucination(claims, ["source text"]))
    assert result.passed is False
    assert "empty input is not allowed" in result.message


@pytest.mark.parametrize("claims", [[], ["   "]], ids=["none", "ws"])
def test_no_claims_skip_and_pass(claims: list[str]) -> None:
    skipped = assert_no_hallucination(claims, ["source"], on_empty="skip")
    assert skipped.passed is True
    assert skipped.details["skipped"] is True
    legacy = assert_no_hallucination(claims, ["source"], on_empty="pass")
    assert legacy.passed is True
    assert legacy.details["total_claims"] == len(claims)


def test_one_real_claim_among_blanks_is_checked() -> None:
    # Mirror: blanks do not make a list with a real claim empty.
    result = _run(lambda: assert_no_hallucination(
        ["   ", "Berlin is the capital of Spain."], ["Paris is the capital of France."],
        min_coverage=0.9,
    ))
    assert result.passed is False
    assert result.details["unsupported_count"] == 1


@pytest.mark.parametrize("claims", [["...", "!!"], ["\U0001f600"]], ids=["punct", "emoji"])
def test_claims_without_word_tokens_are_empty_for_lexical(claims: list[str]) -> None:
    # The lexical method skips a claim with no word tokens, so nothing is
    # measured: the list is empty input, not "all claims supported".
    result = _run(lambda: assert_no_hallucination(claims, ["Paris is the capital."]))
    assert result.passed is False
    assert "empty input is not allowed" in result.message


def test_tokenless_claims_are_checked_by_the_judge() -> None:
    # Mirror: the llm method scores any non-blank claim, so it is not empty.
    result = _run(lambda: assert_no_hallucination(
        ["!!"], ["Paris is the capital."], method="llm", judge_fn=_const(0.9),
    ))
    assert result.passed is True
    assert result.details["n_valid"] == 1


def test_real_claim_among_tokenless_claims_is_checked() -> None:
    # Mirror: one claim with word tokens makes the lexical list non-empty.
    result = _run(lambda: assert_no_hallucination(
        ["!!", "Berlin is the capital of Spain."], ["Paris is the capital of France."],
        min_coverage=0.9,
    ))
    assert result.passed is False
    assert result.details["unsupported_count"] == 1


def test_no_hallucination_unknown_on_empty_fails() -> None:
    result = _run(lambda: assert_no_hallucination(["claim"], ["claim"], on_empty="ignore"))
    assert result.passed is False
    assert "Unknown on_empty" in result.message


def test_no_spans_fail_span_quality_by_default() -> None:
    result = _run(lambda: assert_span_quality(SpanTrace(spans=[])))
    assert result.passed is False
    assert "empty input is not allowed" in result.message


def test_span_kind_filter_matching_nothing_fails_by_default() -> None:
    trace = SpanTrace(spans=[Span(name="answer", kind=SpanKind.LLM)])
    result = _run(lambda: assert_span_quality(trace, span_kinds=[SpanKind.TOOL]))
    assert result.passed is False
    assert "empty input is not allowed" in result.message


def test_span_kind_filter_matching_a_span_is_checked() -> None:
    # Mirror: a filter that matches is evaluated normally.
    trace = SpanTrace(spans=[
        Span(name="answer", kind=SpanKind.LLM),
        Span(name="search", kind=SpanKind.TOOL, status="error", error="boom"),
    ])
    result = _run(lambda: assert_span_quality(trace, span_kinds=[SpanKind.TOOL]))
    assert result.passed is False
    assert result.details["error_count"] == 1
    assert result.details["span_count"] == 1


def test_no_spans_skip_and_pass() -> None:
    skipped = assert_span_quality(SpanTrace(spans=[]), on_empty="skip")
    assert skipped.passed is True
    assert skipped.details["skipped"] is True
    legacy = assert_span_quality(SpanTrace(spans=[]), on_empty="pass")
    assert legacy.passed is True
    assert legacy.details["span_count"] == 0


def test_span_quality_unknown_on_empty_fails() -> None:
    trace = SpanTrace(spans=[Span(name="answer", kind=SpanKind.LLM)])
    result = _run(lambda: assert_span_quality(trace, on_empty="ignore"))
    assert result.passed is False
    assert "Unknown on_empty" in result.message
