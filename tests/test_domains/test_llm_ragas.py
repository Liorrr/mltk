"""Tests for mltk.domains.llm.ragas — RAGAS composite score."""

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.domains.llm.ragas import assert_ragas_score, compute_ragas_score

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

GOOD_QUESTION = "What is machine learning?"
GOOD_CONTEXT = (
    "Machine learning is a branch of artificial intelligence that allows systems "
    "to learn and improve from experience without being explicitly programmed."
)
GOOD_ANSWER = "Machine learning is a branch of artificial intelligence."
GOOD_RELEVANT = ["doc1", "doc2", "doc3"]
GOOD_RETRIEVED = ["doc1", "doc2", "doc3"]


# ---------------------------------------------------------------------------
# compute_ragas_score — raw dict tests
# ---------------------------------------------------------------------------


class TestComputeRagasScore:
    """Unit tests for compute_ragas_score (no assertion side-effects)."""

    def test_compute_ragas_returns_dict(self) -> None:
        # SCENARIO: Call compute_ragas_score without optional IDs.
        # WHY: Must always return a dict; callers inspect keys before asserting.
        # EXPECTED: Dict has faithfulness, answer_relevancy, composite. No precision/recall.
        result = compute_ragas_score(GOOD_ANSWER, GOOD_QUESTION, GOOD_CONTEXT)
        assert isinstance(result, dict)
        assert "faithfulness" in result
        assert "answer_relevancy" in result
        assert "composite" in result
        assert "context_precision" not in result
        assert "context_recall" not in result

    def test_compute_ragas_with_ids_has_all_keys(self) -> None:
        # SCENARIO: Pass relevant_ids + retrieved_ids.
        # WHY: All four metrics should be present when IDs are supplied.
        # EXPECTED: Dict has all 5 keys (4 metrics + composite).
        result = compute_ragas_score(
            GOOD_ANSWER, GOOD_QUESTION, GOOD_CONTEXT,
            relevant_ids=GOOD_RELEVANT,
            retrieved_ids=GOOD_RETRIEVED,
        )
        assert "faithfulness" in result
        assert "answer_relevancy" in result
        assert "context_precision" in result
        assert "context_recall" in result
        assert "composite" in result

    def test_composite_is_mean_of_components(self) -> None:
        # SCENARIO: Verify composite arithmetic manually.
        # WHY: composite must equal mean(faithfulness, answer_relevancy) when no IDs.
        # EXPECTED: composite == (faithfulness + answer_relevancy) / 2.
        result = compute_ragas_score(GOOD_ANSWER, GOOD_QUESTION, GOOD_CONTEXT)
        expected = (result["faithfulness"] + result["answer_relevancy"]) / 2
        assert abs(result["composite"] - expected) < 1e-9

    def test_score_range_zero_to_one(self) -> None:
        # SCENARIO: Feed realistic inputs.
        # WHY: All individual scores and composite must stay in [0, 1].
        # EXPECTED: Every value in the returned dict is between 0.0 and 1.0.
        result = compute_ragas_score(
            GOOD_ANSWER, GOOD_QUESTION, GOOD_CONTEXT,
            relevant_ids=GOOD_RELEVANT,
            retrieved_ids=GOOD_RETRIEVED,
        )
        for key, val in result.items():
            assert 0.0 <= val <= 1.0, f"{key}={val} out of range"


# ---------------------------------------------------------------------------
# assert_ragas_score — assertion tests
# ---------------------------------------------------------------------------


class TestAssertRagasScore:
    """Assertion-level tests for assert_ragas_score."""

    def test_ragas_score_all_good(self) -> None:
        # SCENARIO: High-quality RAG output — answer closely mirrors context,
        #           question keywords appear in answer, perfect retrieval IDs.
        # WHY: All four metrics should be high; composite should comfortably pass 0.5.
        # EXPECTED: passes; composite >= 0.5; result has timing.
        result = assert_ragas_score(
            answer=GOOD_ANSWER,
            question=GOOD_QUESTION,
            context=GOOD_CONTEXT,
            relevant_ids=GOOD_RELEVANT,
            retrieved_ids=GOOD_RETRIEVED,
            min_score=0.5,
        )
        assert result.passed is True
        assert result.details["composite"] >= 0.5
        assert result.duration_ms > 0

    def test_ragas_score_low(self) -> None:
        # SCENARIO: Completely off-topic answer and wrong documents retrieved.
        # WHY: Faithfulness and answer_relevancy both collapse to ~0; composite < 0.5.
        # EXPECTED: raises MltkAssertionError with composite < 0.5.
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_ragas_score(
                answer="The weather is sunny today with low humidity.",
                question="What is machine learning?",
                context=GOOD_CONTEXT,
                relevant_ids=["doc1", "doc2"],
                retrieved_ids=["doc9", "doc10"],
                min_score=0.5,
            )
        assert exc_info.value.result.details["composite"] < 0.5

    def test_ragas_without_ids(self) -> None:
        # SCENARIO: No relevant_ids / retrieved_ids supplied — common when ground-truth
        #           document labels are unavailable.
        # WHY: Only faithfulness + answer_relevancy computed; no KeyError expected.
        # EXPECTED: passes; result does NOT contain context_precision / context_recall.
        result = assert_ragas_score(
            answer=GOOD_ANSWER,
            question=GOOD_QUESTION,
            context=GOOD_CONTEXT,
            min_score=0.3,
        )
        assert result.passed is True
        assert "context_precision" not in result.details
        assert "context_recall" not in result.details
        assert "faithfulness" in result.details
        assert "answer_relevancy" in result.details

    def test_ragas_with_ids(self) -> None:
        # SCENARIO: Full evaluation — all four metrics computed including precision/recall.
        # WHY: Confirm that passing IDs activates the two retrieval metrics.
        # EXPECTED: passes; result.details includes context_precision and context_recall.
        result = assert_ragas_score(
            answer=GOOD_ANSWER,
            question=GOOD_QUESTION,
            context=GOOD_CONTEXT,
            relevant_ids=GOOD_RELEVANT,
            retrieved_ids=GOOD_RETRIEVED,
            min_score=0.4,
        )
        assert result.passed is True
        assert "context_precision" in result.details
        assert "context_recall" in result.details

    def test_ragas_empty_answer(self) -> None:
        # SCENARIO: LLM returns an empty string (timeout / refusal edge case).
        # WHY: Empty answer → faithfulness=1.0 (trivial), answer_relevancy=0.0.
        #      Composite depends on balance; the assertion must not crash.
        # EXPECTED: does not raise TypeError; result has a composite score in [0, 1].
        result = assert_ragas_score(
            answer="",
            question=GOOD_QUESTION,
            context=GOOD_CONTEXT,
            min_score=0.0,  # permissive so it passes regardless
        )
        assert 0.0 <= result.details["composite"] <= 1.0

    def test_ragas_result_has_metrics_used(self) -> None:
        # SCENARIO: Inspect result details for metadata about which metrics ran.
        # WHY: Callers need to know the active metric set to interpret composite.
        # EXPECTED: result.details["metrics_used"] is a list of metric names.
        result = assert_ragas_score(
            answer=GOOD_ANSWER,
            question=GOOD_QUESTION,
            context=GOOD_CONTEXT,
            min_score=0.3,
        )
        assert "metrics_used" in result.details
        assert isinstance(result.details["metrics_used"], list)
        assert "faithfulness" in result.details["metrics_used"]
        assert "answer_relevancy" in result.details["metrics_used"]
