"""Tests for mltk.eval.scorers — scoring pipeline."""

from __future__ import annotations

import pytest

from mltk.eval._types import EvalSample, EvalState, Score
from mltk.eval.scorers import (
    ExactMatchScorer,
    IncludesScorer,
    LLMJudgeScorer,
    PatternScorer,
    Scorer,
)

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _make_state(
    output: str = "",
    target: str | None = "4",
    inp: str = "2+2?",
) -> EvalState:
    """Build an EvalState with sensible defaults."""
    return EvalState(
        sample=EvalSample(input=inp, target=target),
        output=output,
    )


# ===============================================================
# ExactMatchScorer
# ===============================================================


class TestExactMatchScorer:
    """ExactMatchScorer: binary exact match scoring."""

    def test_exact_match_returns_one(self):
        # SCENARIO: output == target
        # WHY: perfect match must score 1.0
        # EXPECTED: value == 1.0
        scorer = ExactMatchScorer()
        state = _make_state(output="4", target="4")
        result = scorer.score(state)
        assert result.value == 1.0

    def test_mismatch_returns_zero(self):
        # SCENARIO: output != target
        # WHY: mismatch must score 0.0
        # EXPECTED: value == 0.0
        scorer = ExactMatchScorer()
        state = _make_state(output="5", target="4")
        result = scorer.score(state)
        assert result.value == 0.0

    def test_case_insensitive_default(self):
        # SCENARIO: different case, ignore_case=True
        # WHY: default is case-insensitive
        # EXPECTED: value == 1.0
        scorer = ExactMatchScorer()
        state = _make_state(output="Paris", target="paris")
        result = scorer.score(state)
        assert result.value == 1.0

    def test_case_sensitive(self):
        # SCENARIO: different case, ignore_case=False
        # WHY: strict mode must distinguish case
        # EXPECTED: value == 0.0
        scorer = ExactMatchScorer(ignore_case=False)
        state = _make_state(output="Paris", target="paris")
        result = scorer.score(state)
        assert result.value == 0.0

    def test_whitespace_normalization(self):
        # SCENARIO: extra whitespace in output
        # WHY: default strips and normalizes
        # EXPECTED: value == 1.0
        scorer = ExactMatchScorer()
        state = _make_state(
            output="  hello   world  ", target="hello world"
        )
        result = scorer.score(state)
        assert result.value == 1.0

    def test_no_whitespace_normalization(self):
        # SCENARIO: extra whitespace, strip disabled
        # WHY: strict mode preserves whitespace
        # EXPECTED: value == 0.0
        scorer = ExactMatchScorer(strip_whitespace=False)
        state = _make_state(
            output="  hello  ", target="hello"
        )
        result = scorer.score(state)
        assert result.value == 0.0

    def test_no_target_returns_zero(self):
        # SCENARIO: target is None
        # WHY: cannot match without target
        # EXPECTED: value == 0.0
        scorer = ExactMatchScorer()
        state = _make_state(output="anything", target=None)
        result = scorer.score(state)
        assert result.value == 0.0

    def test_both_empty_strings(self):
        # SCENARIO: output and target are both ""
        # WHY: empty == empty is a valid match
        # EXPECTED: value == 1.0
        scorer = ExactMatchScorer()
        state = _make_state(output="", target="")
        result = scorer.score(state)
        assert result.value == 1.0

    def test_score_has_answer_field(self):
        # SCENARIO: check Score.answer is populated
        # WHY: answer field tracks model output
        # EXPECTED: answer == output
        scorer = ExactMatchScorer()
        state = _make_state(output="42", target="42")
        result = scorer.score(state)
        assert result.answer == "42"

    def test_score_has_explanation(self):
        # SCENARIO: check explanation on match
        # WHY: explanation aids debugging
        # EXPECTED: non-empty explanation string
        scorer = ExactMatchScorer()
        state = _make_state(output="4", target="4")
        result = scorer.score(state)
        assert isinstance(result.explanation, str)
        assert len(result.explanation) > 0


# ===============================================================
# IncludesScorer
# ===============================================================


class TestIncludesScorer:
    """IncludesScorer: substring / regex containment."""

    def test_substring_found(self):
        # SCENARIO: target is substring of output
        # WHY: basic includes check
        # EXPECTED: value == 1.0
        scorer = IncludesScorer()
        state = _make_state(
            output="The answer is 4.", target="4"
        )
        result = scorer.score(state)
        assert result.value == 1.0

    def test_substring_not_found(self):
        # SCENARIO: target not in output
        # WHY: must return 0.0
        # EXPECTED: value == 0.0
        scorer = IncludesScorer()
        state = _make_state(
            output="The answer is 5.", target="4"
        )
        result = scorer.score(state)
        assert result.value == 0.0

    def test_case_insensitive_default(self):
        # SCENARIO: different case, default mode
        # WHY: default is case-insensitive
        # EXPECTED: value == 1.0
        scorer = IncludesScorer()
        state = _make_state(
            output="HELLO world", target="hello"
        )
        result = scorer.score(state)
        assert result.value == 1.0

    def test_case_sensitive(self):
        # SCENARIO: different case, case-sensitive
        # WHY: strict mode distinguishes case
        # EXPECTED: value == 0.0
        scorer = IncludesScorer(ignore_case=False)
        state = _make_state(
            output="HELLO", target="hello"
        )
        result = scorer.score(state)
        assert result.value == 0.0

    def test_regex_mode(self):
        # SCENARIO: regex pattern matches output
        # WHY: regex mode uses re.search
        # EXPECTED: value == 1.0
        scorer = IncludesScorer(regex=True)
        state = _make_state(
            output="value is 42 units",
            target=r"\d+",
        )
        result = scorer.score(state)
        assert result.value == 1.0

    def test_regex_with_groups(self):
        # SCENARIO: regex with capture groups
        # WHY: groups should not break matching
        # EXPECTED: value == 1.0
        scorer = IncludesScorer(regex=True)
        state = _make_state(
            output="score: 95%",
            target=r"score:\s*(\d+)%",
        )
        result = scorer.score(state)
        assert result.value == 1.0

    def test_regex_no_match(self):
        # SCENARIO: regex does not match
        # WHY: failed regex must return 0.0
        # EXPECTED: value == 0.0
        scorer = IncludesScorer(regex=True)
        state = _make_state(
            output="no numbers here",
            target=r"\d+",
        )
        result = scorer.score(state)
        assert result.value == 0.0

    def test_no_target_returns_zero(self):
        # SCENARIO: target is None
        # WHY: cannot check inclusion without target
        # EXPECTED: value == 0.0
        scorer = IncludesScorer()
        state = _make_state(
            output="anything", target=None
        )
        result = scorer.score(state)
        assert result.value == 0.0

    def test_empty_target_always_found(self):
        # SCENARIO: target is empty string
        # WHY: "" is substring of everything
        # EXPECTED: value == 1.0
        scorer = IncludesScorer()
        state = _make_state(output="hello", target="")
        result = scorer.score(state)
        assert result.value == 1.0

    def test_score_has_explanation(self):
        # SCENARIO: check explanation on found
        # WHY: explanation aids debugging
        # EXPECTED: non-empty explanation
        scorer = IncludesScorer()
        state = _make_state(output="yes", target="yes")
        result = scorer.score(state)
        assert len(result.explanation) > 0


# ===============================================================
# LLMJudgeScorer
# ===============================================================


class TestLLMJudgeScorer:
    """LLMJudgeScorer: LLM-based evaluation."""

    def test_high_score(self):
        # SCENARIO: judge returns 4.0/5.0
        # WHY: must normalize to 0.8
        # EXPECTED: value == 0.8
        def judge(prompt: str) -> float:
            return 4.0

        scorer = LLMJudgeScorer(judge_fn=judge)
        state = _make_state(output="good answer")
        result = scorer.score(state)
        assert result.value == pytest.approx(0.8)

    def test_low_score(self):
        # SCENARIO: judge returns 1.0/5.0
        # WHY: must normalize to 0.2
        # EXPECTED: value == 0.2
        def judge(prompt: str) -> float:
            return 1.0

        scorer = LLMJudgeScorer(judge_fn=judge)
        state = _make_state(output="bad answer")
        result = scorer.score(state)
        assert result.value == pytest.approx(0.2)

    def test_parse_score_from_text(self):
        # SCENARIO: judge returns string "4/5"
        # WHY: LLMs may return text, not floats
        # EXPECTED: parsed and normalized to 0.8
        def judge(prompt: str) -> str:
            return "I give this a 4/5"

        scorer = LLMJudgeScorer(judge_fn=judge)
        state = _make_state(output="answer")
        result = scorer.score(state)
        assert result.value == pytest.approx(0.8)

    def test_normalize_to_zero_one(self):
        # SCENARIO: judge returns 5.0/5.0
        # WHY: max score normalizes to 1.0
        # EXPECTED: value == 1.0
        def judge(prompt: str) -> float:
            return 5.0

        scorer = LLMJudgeScorer(judge_fn=judge)
        state = _make_state(output="perfect")
        result = scorer.score(state)
        assert result.value == pytest.approx(1.0)

    def test_zero_score(self):
        # SCENARIO: judge returns 0.0
        # WHY: minimum score normalizes to 0.0
        # EXPECTED: value == 0.0
        def judge(prompt: str) -> float:
            return 0.0

        scorer = LLMJudgeScorer(judge_fn=judge)
        state = _make_state(output="terrible")
        result = scorer.score(state)
        assert result.value == pytest.approx(0.0)

    def test_judge_raises_returns_zero(self):
        # SCENARIO: judge function raises exception
        # WHY: errors must be caught, not propagated
        # EXPECTED: value == 0.0, error in metadata
        def bad_judge(prompt: str) -> float:
            raise RuntimeError("judge crashed")

        scorer = LLMJudgeScorer(judge_fn=bad_judge)
        state = _make_state(output="answer")
        result = scorer.score(state)
        assert result.value == 0.0
        assert "error" in result.metadata

    def test_custom_criterion(self):
        # SCENARIO: criterion="safety"
        # WHY: criterion is configurable
        # EXPECTED: criterion in explanation
        def judge(prompt: str) -> float:
            return 3.0

        scorer = LLMJudgeScorer(
            judge_fn=judge, criterion="safety"
        )
        state = _make_state(output="answer")
        result = scorer.score(state)
        assert "safety" in result.explanation

    def test_custom_rubric(self):
        # SCENARIO: rubric provided
        # WHY: rubric must be passed to judge prompt
        # EXPECTED: rubric text appears in prompt
        prompts = []

        def capture_judge(prompt: str) -> float:
            prompts.append(prompt)
            return 3.0

        rubric = "Rate accuracy of the response."
        scorer = LLMJudgeScorer(
            judge_fn=capture_judge, rubric=rubric
        )
        state = _make_state(output="answer", inp="q")
        scorer.score(state)
        assert rubric in prompts[0]

    def test_name_includes_criterion(self):
        # SCENARIO: name property
        # WHY: name should identify the criterion
        # EXPECTED: "LLMJudge/correctness"
        def judge(prompt: str) -> float:
            return 3.0

        scorer = LLMJudgeScorer(judge_fn=judge)
        assert "correctness" in scorer.name
        assert "LLMJudge" in scorer.name

    def test_custom_max_score(self):
        # SCENARIO: max_score=10.0
        # WHY: normalization uses max_score
        # EXPECTED: 7.0/10.0 = 0.7
        def judge(prompt: str) -> float:
            return 7.0

        scorer = LLMJudgeScorer(
            judge_fn=judge, max_score=10.0
        )
        state = _make_state(output="answer")
        result = scorer.score(state)
        assert result.value == pytest.approx(0.7)

    def test_over_max_clamped_to_one(self):
        # SCENARIO: judge returns above max_score
        # WHY: value must be clamped to 1.0
        # EXPECTED: value == 1.0
        def judge(prompt: str) -> float:
            return 6.0

        scorer = LLMJudgeScorer(judge_fn=judge)
        state = _make_state(output="answer")
        result = scorer.score(state)
        assert result.value == pytest.approx(1.0)

    def test_metadata_has_raw_score(self):
        # SCENARIO: metadata includes raw score
        # WHY: raw score needed for debugging
        # EXPECTED: metadata["raw_score"] == 4.0
        def judge(prompt: str) -> float:
            return 4.0

        scorer = LLMJudgeScorer(judge_fn=judge)
        state = _make_state(output="answer")
        result = scorer.score(state)
        assert result.metadata["raw_score"] == 4.0


# ===============================================================
# PatternScorer
# ===============================================================


class TestPatternScorer:
    """PatternScorer: regex-based answer extraction."""

    def test_extract_answer(self):
        # SCENARIO: "Answer: 42" with default pattern
        # WHY: must extract and compare to target
        # EXPECTED: value == 1.0, answer == "42"
        scorer = PatternScorer()
        state = _make_state(
            output="Let me think.\nAnswer: 42",
            target="42",
        )
        result = scorer.score(state)
        assert result.value == 1.0
        assert result.answer == "42"

    def test_no_match_returns_zero(self):
        # SCENARIO: no "Answer:" pattern in output
        # WHY: extraction failure = 0.0
        # EXPECTED: value == 0.0
        scorer = PatternScorer()
        state = _make_state(
            output="I think it is 42", target="42"
        )
        result = scorer.score(state)
        assert result.value == 0.0

    def test_match_with_target_comparison(self):
        # SCENARIO: extracted != target
        # WHY: extraction succeeds but wrong answer
        # EXPECTED: value == 0.0
        scorer = PatternScorer()
        state = _make_state(
            output="Answer: 99", target="42"
        )
        result = scorer.score(state)
        assert result.value == 0.0

    def test_case_insensitive_match(self):
        # SCENARIO: "ANSWER: paris" vs target "Paris"
        # WHY: default is case-insensitive
        # EXPECTED: value == 1.0
        scorer = PatternScorer()
        state = _make_state(
            output="ANSWER: paris", target="Paris"
        )
        result = scorer.score(state)
        assert result.value == 1.0

    def test_no_target_pattern_found(self):
        # SCENARIO: no target, pattern matches
        # WHY: extraction success = 1.0 when no target
        # EXPECTED: value == 1.0
        scorer = PatternScorer()
        state = _make_state(
            output="Answer: something", target=None
        )
        result = scorer.score(state)
        assert result.value == 1.0

    def test_no_target_pattern_not_found(self):
        # SCENARIO: no target, pattern missing
        # WHY: extraction failure = 0.0
        # EXPECTED: value == 0.0
        scorer = PatternScorer()
        state = _make_state(
            output="no pattern here", target=None
        )
        result = scorer.score(state)
        assert result.value == 0.0

    def test_custom_pattern(self):
        # SCENARIO: user provides custom regex
        # WHY: pattern customization is part of API
        # EXPECTED: custom pattern extracts correctly
        scorer = PatternScorer(
            pattern=r"Result:\s*(\d+)"
        )
        state = _make_state(
            output="Result: 99", target="99"
        )
        result = scorer.score(state)
        assert result.value == 1.0
        assert result.answer == "99"

    def test_final_answer_variant(self):
        # SCENARIO: "Final answer: X" (default pattern)
        # WHY: default pattern supports this variant
        # EXPECTED: value == 1.0
        scorer = PatternScorer()
        state = _make_state(
            output="Final answer: 7", target="7"
        )
        result = scorer.score(state)
        assert result.value == 1.0

    def test_equals_sign_variant(self):
        # SCENARIO: "Answer = X" (default pattern)
        # WHY: default supports = separator
        # EXPECTED: value == 1.0
        scorer = PatternScorer()
        state = _make_state(
            output="Answer = 10", target="10"
        )
        result = scorer.score(state)
        assert result.value == 1.0

    def test_score_has_explanation(self):
        # SCENARIO: check explanation on match
        # WHY: explanation aids debugging
        # EXPECTED: non-empty explanation
        scorer = PatternScorer()
        state = _make_state(
            output="Answer: yes", target="yes"
        )
        result = scorer.score(state)
        assert len(result.explanation) > 0


# ===============================================================
# Scorer ABC
# ===============================================================


class TestScorerABC:
    """Scorer base class behavior."""

    def test_name_returns_class_name(self):
        # SCENARIO: custom scorer name
        # WHY: default name is class name
        # EXPECTED: name == "MyScorer"
        class MyScorer(Scorer):
            def score(self, state):
                return Score(value=0.5)

        assert MyScorer().name == "MyScorer"

    def test_score_value_zero(self):
        # SCENARIO: Score(value=0.0)
        # WHY: lower bound must be representable
        # EXPECTED: value == 0.0
        s = Score(value=0.0)
        assert s.value == 0.0

    def test_score_value_one(self):
        # SCENARIO: Score(value=1.0)
        # WHY: upper bound must be representable
        # EXPECTED: value == 1.0
        s = Score(value=1.0)
        assert s.value == 1.0

    def test_score_default_fields(self):
        # SCENARIO: Score with only value
        # WHY: defaults must be sensible
        # EXPECTED: empty answer, explanation, metadata
        s = Score(value=0.5)
        assert s.answer == ""
        assert s.explanation == ""
        assert s.metadata == {}

    def test_score_with_all_fields(self):
        # SCENARIO: Score with all fields set
        # WHY: all fields must be settable
        # EXPECTED: all fields have correct values
        s = Score(
            value=0.9,
            answer="Paris",
            explanation="Correct city",
            metadata={"source": "geo"},
        )
        assert s.value == 0.9
        assert s.answer == "Paris"
        assert s.explanation == "Correct city"
        assert s.metadata["source"] == "geo"
