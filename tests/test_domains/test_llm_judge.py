"""Tests for LLM-as-Judge evaluation assertions."""

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.domains.llm.judge import (
    DEFAULT_CRITERIA,
    assert_llm_judge_pairwise,
    assert_llm_judge_score,
    format_judge_prompt,
)

# -------------------------------------------------------------------
# Mock judge functions
# -------------------------------------------------------------------

def _high_scorer(prompt: str) -> float:
    """Judge that always returns a high score."""
    return 4.5


def _low_scorer(prompt: str) -> float:
    """Judge that always returns a low score."""
    return 1.5


def _string_scorer(prompt: str) -> str:
    """Judge that returns a score as a string (realistic LLM output)."""
    return "4.0"


def _winner_a(prompt: str) -> str:
    """Judge that always picks Response A."""
    return "A"


def _winner_b(prompt: str) -> str:
    """Judge that always picks Response B."""
    return "B"


def _tie_judge(prompt: str) -> str:
    """Judge that always declares a tie."""
    return "TIE"


def _error_judge(prompt: str) -> float:
    """Judge that always raises an exception."""
    raise RuntimeError("Model unavailable")


def _garbage_judge(prompt: str) -> str:
    """Judge that returns unparseable garbage."""
    return "I think the response is quite good actually"


# ===================================================================
# assert_llm_judge_score tests
# ===================================================================

class TestAssertLlmJudgeScore:
    """Tests for assert_llm_judge_score."""

    def test_high_scores_pass(self) -> None:
        """PASS: High judge scores exceed min_score."""
        result = assert_llm_judge_score(
            judge_fn=_high_scorer,
            prompts=["What is Python?", "Explain REST APIs"],
            responses=["Python is a language.", "REST is an arch."],
            min_score=3.0,
        )
        assert result.passed is True
        assert result.details["avg_score"] == 4.5
        assert result.details["n_items"] == 2
        assert result.details["scores_below_min"] == 0

    def test_low_scores_fail(self) -> None:
        """FAIL: Low judge scores below min_score."""
        with pytest.raises(MltkAssertionError) as exc:
            assert_llm_judge_score(
                judge_fn=_low_scorer,
                prompts=["What is Python?"],
                responses=["Dunno."],
                min_score=3.0,
            )
        result = exc.value.result
        assert result.passed is False
        assert result.details["avg_score"] == 1.5
        assert result.details["scores_below_min"] == 1

    def test_custom_criterion(self) -> None:
        """PASS: Custom criterion name propagated to details."""
        result = assert_llm_judge_score(
            judge_fn=_high_scorer,
            prompts=["Summarize this"],
            responses=["Here is a summary."],
            criterion="coherence",
            min_score=3.0,
        )
        assert result.passed is True
        assert result.details["criterion"] == "coherence"

    def test_custom_rubric(self) -> None:
        """PASS: Custom rubric used instead of default."""
        custom_rubric = "Rate conciseness on 1-5."
        result = assert_llm_judge_score(
            judge_fn=_high_scorer,
            prompts=["Be brief"],
            responses=["OK."],
            rubric=custom_rubric,
            min_score=3.0,
        )
        assert result.passed is True

    def test_judge_fn_raises_handled(self) -> None:
        """FAIL: judge_fn that raises gets score 0.0 with error flag."""
        with pytest.raises(MltkAssertionError) as exc:
            assert_llm_judge_score(
                judge_fn=_error_judge,
                prompts=["Hello"],
                responses=["World"],
                min_score=1.0,
            )
        result = exc.value.result
        assert result.passed is False
        assert result.details["per_item_scores"][0]["score"] == 0.0
        assert "error" in result.details["per_item_scores"][0]

    def test_per_item_scores_in_details(self) -> None:
        """Verify per_item_scores list matches number of items."""
        result = assert_llm_judge_score(
            judge_fn=_high_scorer,
            prompts=["Q1", "Q2", "Q3"],
            responses=["A1", "A2", "A3"],
            min_score=3.0,
        )
        scores = result.details["per_item_scores"]
        assert len(scores) == 3
        for item in scores:
            assert item["score"] == 4.5

    def test_string_score_parsed(self) -> None:
        """PASS: Judge returning string '4.0' is parsed to float."""
        result = assert_llm_judge_score(
            judge_fn=_string_scorer,
            prompts=["Test"],
            responses=["Answer"],
            min_score=3.0,
        )
        assert result.passed is True
        assert result.details["avg_score"] == 4.0

    def test_unparseable_score_becomes_zero(self) -> None:
        """FAIL: Judge returning garbage text gets score 0.0."""
        with pytest.raises(MltkAssertionError) as exc:
            assert_llm_judge_score(
                judge_fn=_garbage_judge,
                prompts=["Test"],
                responses=["Answer"],
                min_score=1.0,
            )
        result = exc.value.result
        # "I think the response is quite good actually" has no number
        # so score is 0.0
        assert result.details["per_item_scores"][0]["score"] == 0.0

    def test_empty_prompts_pass(self) -> None:
        """EDGE: Empty prompts list passes with zero items."""
        result = assert_llm_judge_score(
            judge_fn=_high_scorer,
            prompts=[],
            responses=[],
            min_score=3.0,
        )
        assert result.passed is True
        assert result.details["n_items"] == 0

    def test_single_item(self) -> None:
        """PASS: Single prompt/response pair works correctly."""
        result = assert_llm_judge_score(
            judge_fn=_high_scorer,
            prompts=["One question"],
            responses=["One answer"],
            min_score=4.0,
        )
        assert result.passed is True
        assert result.details["n_items"] == 1

    def test_mismatched_lengths_fail(self) -> None:
        """FAIL: Different length prompts and responses."""
        with pytest.raises(MltkAssertionError) as exc:
            assert_llm_judge_score(
                judge_fn=_high_scorer,
                prompts=["Q1", "Q2"],
                responses=["A1"],
                min_score=3.0,
            )
        result = exc.value.result
        assert result.passed is False
        assert "equal length" in result.message

    def test_exact_threshold_passes(self) -> None:
        """PASS: Score exactly equal to min_score passes."""
        def exact_scorer(prompt: str) -> float:
            return 3.0

        result = assert_llm_judge_score(
            judge_fn=exact_scorer,
            prompts=["Q"],
            responses=["A"],
            min_score=3.0,
        )
        assert result.passed is True

    def test_result_name(self) -> None:
        """Verify assertion name follows package.module.name convention."""
        result = assert_llm_judge_score(
            judge_fn=_high_scorer,
            prompts=["Q"],
            responses=["A"],
            min_score=3.0,
        )
        assert result.name == "llm.judge.score"

    def test_has_duration_ms(self) -> None:
        """Verify timed_assertion decorator populates duration_ms."""
        result = assert_llm_judge_score(
            judge_fn=_high_scorer,
            prompts=["Q"],
            responses=["A"],
            min_score=3.0,
        )
        assert result.duration_ms >= 0.0


# ===================================================================
# assert_llm_judge_pairwise tests
# ===================================================================

class TestAssertLlmJudgePairwise:
    """Tests for assert_llm_judge_pairwise."""

    def test_expected_winner_a_wins(self) -> None:
        """PASS: Judge picks A and expected_winner is 'a'."""
        result = assert_llm_judge_pairwise(
            judge_fn=_winner_a,
            prompts=["What is ML?", "Explain AI"],
            responses_a=["ML is...", "AI is..."],
            responses_b=["Dunno", "No idea"],
            expected_winner="a",
            min_win_rate=0.6,
        )
        assert result.passed is True
        assert result.details["win_rate"] == 1.0
        assert result.details["wins_a"] == 2
        assert result.details["wins_b"] == 0
        assert result.details["ties"] == 0

    def test_expected_winner_loses_fails(self) -> None:
        """FAIL: Judge picks B but expected_winner is 'a'."""
        with pytest.raises(MltkAssertionError) as exc:
            assert_llm_judge_pairwise(
                judge_fn=_winner_b,
                prompts=["Q1"],
                responses_a=["Bad answer"],
                responses_b=["Good answer"],
                expected_winner="a",
                min_win_rate=0.6,
            )
        result = exc.value.result
        assert result.passed is False
        assert result.details["win_rate"] == 0.0
        assert result.details["wins_b"] == 1

    def test_expected_winner_b(self) -> None:
        """PASS: Judge picks B and expected_winner is 'b'."""
        result = assert_llm_judge_pairwise(
            judge_fn=_winner_b,
            prompts=["Q1"],
            responses_a=["Weak"],
            responses_b=["Strong"],
            expected_winner="b",
            min_win_rate=0.6,
        )
        assert result.passed is True
        assert result.details["win_rate"] == 1.0

    def test_ties_handling(self) -> None:
        """FAIL: All ties mean 0% win rate for expected winner."""
        with pytest.raises(MltkAssertionError) as exc:
            assert_llm_judge_pairwise(
                judge_fn=_tie_judge,
                prompts=["Q1", "Q2"],
                responses_a=["A1", "A2"],
                responses_b=["B1", "B2"],
                expected_winner="a",
                min_win_rate=0.5,
            )
        result = exc.value.result
        assert result.details["ties"] == 2
        assert result.details["wins_a"] == 0
        assert result.details["wins_b"] == 0
        assert result.details["win_rate"] == 0.0

    def test_unexpected_value_counted_as_tie(self) -> None:
        """EDGE: Judge returns something other than A/B/TIE."""
        def weird_judge(prompt: str) -> str:
            return "Response A is clearly better because..."

        # "Response A is..." starts with "R", not "A", so _parse_winner
        # will not match A or B -- it becomes a tie.
        with pytest.raises(MltkAssertionError) as exc:
            assert_llm_judge_pairwise(
                judge_fn=weird_judge,
                prompts=["Q1"],
                responses_a=["A1"],
                responses_b=["B1"],
                expected_winner="a",
                min_win_rate=0.5,
            )
        result = exc.value.result
        assert result.details["ties"] == 1

    def test_judge_error_counted_as_tie(self) -> None:
        """EDGE: Judge that raises exception counted as tie."""
        with pytest.raises(MltkAssertionError) as exc:
            assert_llm_judge_pairwise(
                judge_fn=_error_judge,
                prompts=["Q1"],
                responses_a=["A1"],
                responses_b=["B1"],
                expected_winner="a",
                min_win_rate=0.5,
            )
        result = exc.value.result
        assert result.details["ties"] == 1

    def test_empty_prompts_pass(self) -> None:
        """EDGE: Empty prompts list passes with zero comparisons."""
        result = assert_llm_judge_pairwise(
            judge_fn=_winner_a,
            prompts=[],
            responses_a=[],
            responses_b=[],
            expected_winner="a",
            min_win_rate=0.6,
        )
        assert result.passed is True
        assert result.details["n_comparisons"] == 0

    def test_single_comparison(self) -> None:
        """PASS: Single prompt triple works correctly."""
        result = assert_llm_judge_pairwise(
            judge_fn=_winner_a,
            prompts=["Q1"],
            responses_a=["Great answer"],
            responses_b=["Bad answer"],
            expected_winner="a",
            min_win_rate=0.5,
        )
        assert result.passed is True
        assert result.details["n_comparisons"] == 1

    def test_mismatched_lengths_fail(self) -> None:
        """FAIL: Mismatched list lengths."""
        with pytest.raises(MltkAssertionError) as exc:
            assert_llm_judge_pairwise(
                judge_fn=_winner_a,
                prompts=["Q1", "Q2"],
                responses_a=["A1"],
                responses_b=["B1", "B2"],
                expected_winner="a",
                min_win_rate=0.5,
            )
        result = exc.value.result
        assert result.passed is False
        assert "equal length" in result.message

    def test_result_name(self) -> None:
        """Verify assertion name follows package.module.name convention."""
        result = assert_llm_judge_pairwise(
            judge_fn=_winner_a,
            prompts=["Q"],
            responses_a=["A"],
            responses_b=["B"],
            expected_winner="a",
            min_win_rate=0.5,
        )
        assert result.name == "llm.judge.pairwise"

    def test_mixed_results_partial_win_rate(self) -> None:
        """PASS: Mixed wins with win rate above threshold."""
        call_count = 0

        def alternating_judge(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return "A" if call_count % 2 == 1 else "B"

        result = assert_llm_judge_pairwise(
            judge_fn=alternating_judge,
            prompts=["Q1", "Q2", "Q3", "Q4"],
            responses_a=["A1", "A2", "A3", "A4"],
            responses_b=["B1", "B2", "B3", "B4"],
            expected_winner="a",
            min_win_rate=0.5,
        )
        assert result.passed is True
        assert result.details["wins_a"] == 2
        assert result.details["wins_b"] == 2
        assert result.details["win_rate"] == 0.5


# ===================================================================
# format_judge_prompt tests
# ===================================================================

class TestFormatJudgePrompt:
    """Tests for the format_judge_prompt helper."""

    def test_contains_criterion(self) -> None:
        """Formatted prompt includes the criterion name."""
        text = format_judge_prompt(
            prompt="What is AI?",
            response="AI is artificial intelligence.",
            criterion="helpfulness",
        )
        assert "helpfulness" in text

    def test_contains_prompt_and_response(self) -> None:
        """Formatted prompt includes the original prompt and response."""
        text = format_judge_prompt(
            prompt="What is the speed of light?",
            response="About 300,000 km/s.",
            criterion="correctness",
        )
        assert "What is the speed of light?" in text
        assert "About 300,000 km/s." in text

    def test_contains_score_range(self) -> None:
        """Formatted prompt mentions the max_score."""
        text = format_judge_prompt(
            prompt="Q",
            response="A",
            criterion="helpfulness",
            max_score=10.0,
        )
        assert "10.0" in text

    def test_custom_rubric_overrides_default(self) -> None:
        """Custom rubric text appears instead of default criterion text."""
        custom = "Rate the creativity of the response."
        text = format_judge_prompt(
            prompt="Write a poem",
            response="Roses are red...",
            criterion="creativity",
            rubric=custom,
        )
        assert custom in text

    def test_unknown_criterion_uses_generic_rubric(self) -> None:
        """Unknown criterion without rubric gets a generic fallback."""
        text = format_judge_prompt(
            prompt="Q",
            response="A",
            criterion="inventiveness",
        )
        assert "inventiveness" in text


# ===================================================================
# DEFAULT_CRITERIA tests
# ===================================================================

class TestDefaultCriteria:
    """Tests for the DEFAULT_CRITERIA dictionary."""

    def test_contains_expected_keys(self) -> None:
        """DEFAULT_CRITERIA has all six standard criteria."""
        expected_keys = {
            "helpfulness",
            "correctness",
            "coherence",
            "relevance",
            "harmlessness",
            "semantic_equivalence",
        }
        assert set(DEFAULT_CRITERIA.keys()) == expected_keys

    def test_values_are_nonempty_strings(self) -> None:
        """Every criterion rubric is a non-empty string."""
        for key, value in DEFAULT_CRITERIA.items():
            assert isinstance(value, str), f"{key} is not a string"
            assert len(value) > 10, f"{key} rubric is too short"


# ===================================================================
# Hardened edge-case and parametrized tests (S62 test hardening)
# ===================================================================


class TestJudgeScoreParametrizedCriteria:
    """Parametrize across all DEFAULT_CRITERIA."""

    @pytest.mark.parametrize(
        "criterion",
        list(DEFAULT_CRITERIA.keys()),
    )
    def test_each_default_criterion(
        self, criterion: str,
    ) -> None:
        """PASS: Every default criterion works with score."""
        result = assert_llm_judge_score(
            judge_fn=_high_scorer,
            prompts=["What is ML?"],
            responses=["ML is machine learning."],
            criterion=criterion,
            min_score=3.0,
        )
        assert result.passed is True
        assert result.details["criterion"] == criterion


class TestJudgeStringScoreFormats:
    """Judge that returns float-as-string like '4.5/5'."""

    def test_fraction_string_parsed(self) -> None:
        """PASS: '4.5/5' parses first float 4.5."""
        def fraction_scorer(prompt: str) -> str:
            return "4.5/5"

        result = assert_llm_judge_score(
            judge_fn=fraction_scorer,
            prompts=["Q"],
            responses=["A"],
            min_score=3.0,
        )
        assert result.passed is True
        assert result.details["avg_score"] == 4.5


class TestPairwiseAllTies:
    """Pairwise with all ties -- 0% win rate."""

    def test_all_ties_fail(self) -> None:
        """FAIL: All ties means 0% for expected winner."""
        with pytest.raises(MltkAssertionError) as exc:
            assert_llm_judge_pairwise(
                judge_fn=_tie_judge,
                prompts=["Q1", "Q2", "Q3"],
                responses_a=["A1", "A2", "A3"],
                responses_b=["B1", "B2", "B3"],
                expected_winner="a",
                min_win_rate=0.1,
            )
        result = exc.value.result
        assert result.details["ties"] == 3
        assert result.details["wins_a"] == 0
        assert result.details["wins_b"] == 0


class TestScoreSinglePromptEdge:
    """Score with a single prompt -- smallest valid input."""

    def test_single_prompt_high_score(self) -> None:
        """PASS: Single prompt with high score passes."""
        result = assert_llm_judge_score(
            judge_fn=_high_scorer,
            prompts=["Only question"],
            responses=["Only answer"],
            min_score=4.0,
        )
        assert result.passed is True
        assert result.details["n_items"] == 1
        assert result.details["avg_score"] == 4.5


class TestPairwiseAsymmetricLengths:
    """Pairwise with very different response lengths."""

    def test_asymmetric_responses(self) -> None:
        """PASS: Short vs long responses both work."""
        result = assert_llm_judge_pairwise(
            judge_fn=_winner_a,
            prompts=["Explain AI"],
            responses_a=["x" * 1000],
            responses_b=["y"],
            expected_winner="a",
            min_win_rate=0.5,
        )
        assert result.passed is True
        assert result.details["wins_a"] == 1


class TestFormatJudgePromptLong:
    """format_judge_prompt with very long prompt."""

    def test_long_prompt_1000_chars(self) -> None:
        """PASS: 1000+ char prompt is fully included."""
        long_prompt = "Q " * 500  # 1000 chars
        text = format_judge_prompt(
            prompt=long_prompt,
            response="Short answer.",
            criterion="helpfulness",
        )
        assert long_prompt in text
        assert "helpfulness" in text
        assert len(text) > 1000
