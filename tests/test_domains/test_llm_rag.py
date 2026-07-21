"""Tests for mltk.domains.llm.rag — RAG evaluation assertions."""

from __future__ import annotations

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.core.result import Severity
from mltk.domains.llm.rag import (
    assert_answer_relevancy,
    assert_context_precision,
    assert_context_recall,
    assert_context_relevancy,
    assert_faithfulness,
)


class TestFaithfulness:
    """Faithfulness — answer grounded in context."""

    def test_faithfulness_grounded(self) -> None:
        # SCENARIO: Answer reuses words directly from the context paragraph.
        # WHY: Faithfulness score = answer tokens found in context / total answer tokens.
        # EXPECTED: Score >= 0.7 → passes.
        context = (
            "The Eiffel Tower is located in Paris France."
            " It was built in 1889 by Gustave Eiffel."
        )
        answer = "The Eiffel Tower was built in 1889 in Paris."
        result = assert_faithfulness(answer, context, min_score=0.7)
        assert result.passed is True
        assert result.details["score"] >= 0.7
        assert result.details["grounded_tokens"] > 0

    def test_faithfulness_hallucinated(self) -> None:
        # SCENARIO: Answer contains words completely absent from the context.
        # WHY: When answer tokens do not overlap with context tokens the score is 0.
        # EXPECTED: Score < 0.7 → raises MltkAssertionError.
        context = "The Eiffel Tower is in Paris."
        answer = "Jupiter has 79 retrograde moons orbiting in spiral patterns."
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_faithfulness(answer, context, min_score=0.7)
        assert exc_info.value.result.details["score"] < 0.7

    def test_faithfulness_list_context(self) -> None:
        # SCENARIO: Context is passed as a list of retrieved chunks, not a single string.
        # WHY: The RAG pipeline returns chunks; the assertion must accept list[str].
        # EXPECTED: Chunks are merged; answer grounded in merged context → passes.
        context = [
            "Machine learning is a subset of artificial intelligence.",
            "It allows systems to learn from data without explicit programming.",
        ]
        answer = "Machine learning allows systems to learn from data."
        result = assert_faithfulness(answer, context, min_score=0.6)
        assert result.passed is True

    def test_faithfulness_empty_answer_fails_by_default(self) -> None:
        # SCENARIO: Agent returns an empty string as the answer.
        # WHY: Edge case — an empty answer has no tokens to check against context.
        # EXPECTED: Default on_empty="fail" rejects empty answers.
        context = "Some relevant context about the topic."
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_faithfulness("", context, min_score=0.7)
        assert "empty answer" in exc_info.value.result.message.lower()

    def test_faithfulness_empty_answer_skip(self) -> None:
        # SCENARIO: Caller explicitly chooses to skip empty-answer checks.
        # EXPECTED: Passes as INFO with skip metadata.
        result = assert_faithfulness(
            "", "Some relevant context.", min_score=0.7, on_empty="skip"
        )
        assert result.passed is True
        assert result.severity == Severity.INFO
        assert result.message.startswith("Skipped:")
        assert result.details == {
            "skipped": True,
            "reason": "Empty answer",
        }

    def test_faithfulness_empty_answer_pass_legacy(self) -> None:
        # SCENARIO: Caller explicitly chooses legacy empty-answer behavior.
        # EXPECTED: Empty answer passes with the original score/message.
        result = assert_faithfulness(
            "", "Some relevant context.", min_score=0.7, on_empty="pass"
        )
        assert result.passed is True
        assert result.severity == Severity.CRITICAL
        assert result.details["score"] == 1.0
        assert result.message == "Empty answer -- trivially faithful (score=1.0)"

    def test_faithfulness_result_has_timing(self) -> None:
        # SCENARIO: Every assertion is wrapped with @timed_assertion.
        # WHY: duration_ms must be populated for performance tracking.
        # EXPECTED: duration_ms > 0.
        result = assert_faithfulness(
            "Paris is a city.", "Paris is a city in France.", min_score=0.5
        )
        assert result.duration_ms > 0


class TestContextRelevancy:
    """Context relevancy — retrieved context matches question."""

    def test_context_relevancy_good(self) -> None:
        # SCENARIO: Context directly addresses the question topic and keywords.
        # WHY: Overlap of question tokens with context tokens should be high.
        # EXPECTED: Score >= 0.5 → passes.
        question = "What is the capital of France?"
        context = "Paris is the capital of France and a major European city."
        result = assert_context_relevancy(question, context, min_score=0.5)
        assert result.passed is True
        assert result.details["score"] >= 0.5

    def test_context_relevancy_irrelevant(self) -> None:
        # SCENARIO: Retriever returns a document about an entirely different topic.
        # WHY: Low question-token overlap exposes retrieval failure.
        # EXPECTED: Score < 0.5 → raises MltkAssertionError.
        question = "What is the capital of France?"
        context = "Quantum computing uses qubits for parallel calculations."
        with pytest.raises(MltkAssertionError):
            assert_context_relevancy(question, context, min_score=0.5)

    def test_context_relevancy_list_chunks(self) -> None:
        # SCENARIO: Context supplied as multiple chunks (real RAG pipeline output).
        # WHY: assert_context_relevancy must accept list[str] and merge correctly.
        # EXPECTED: Combined chunks contain question keywords → passes.
        question = "How does photosynthesis work?"
        context = [
            "Photosynthesis is the process by which plants convert light to energy.",
            "Chlorophyll in leaves absorbs sunlight.",
        ]
        result = assert_context_relevancy(question, context, min_score=0.2)
        assert result.passed is True

    def test_context_relevancy_empty_question_fails_by_default(self) -> None:
        # SCENARIO: The question string is empty.
        # EXPECTED: Default on_empty="fail" rejects the degenerate input.
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_context_relevancy("", "Paris is in France.", min_score=0.5)
        assert "empty question" in exc_info.value.result.message.lower()


class TestAnswerRelevancy:
    """Answer relevancy — answer addresses the question."""

    def test_answer_relevancy_good(self) -> None:
        # SCENARIO: Answer directly addresses the question using its keywords.
        # WHY: Core question keywords appear in the answer.
        # EXPECTED: Score >= 0.5 → passes.
        question = "What is machine learning?"
        answer = "Machine learning is a method where systems learn patterns from data."
        result = assert_answer_relevancy(question, answer, min_score=0.5)
        assert result.passed is True
        assert result.details["score"] >= 0.5

    def test_answer_relevancy_off_topic(self) -> None:
        # SCENARIO: Agent answers a completely different question.
        # WHY: None of the question's keywords appear in the off-topic answer.
        # EXPECTED: Score < 0.5 → raises MltkAssertionError.
        question = "What is machine learning?"
        answer = "The weather today is sunny with a high of 25 degrees."
        with pytest.raises(MltkAssertionError):
            assert_answer_relevancy(question, answer, min_score=0.5)

    def test_answer_relevancy_empty_question_fails_by_default(self) -> None:
        # SCENARIO: The question string is empty.
        # EXPECTED: Default on_empty="fail" rejects the degenerate input.
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_answer_relevancy("", "Machine learning uses data.", min_score=0.5)
        assert "empty question" in exc_info.value.result.message.lower()


class TestContextPrecision:
    """Context precision — |relevant ∩ retrieved| / |retrieved|."""

    def test_context_precision_pass(self) -> None:
        # SCENARIO: 3 out of 5 retrieved documents are relevant.
        # WHY: Precision = 3/5 = 0.6, above the min_precision of 0.5.
        # EXPECTED: passes with precision=0.6.
        relevant = ["doc1", "doc2", "doc3"]
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        result = assert_context_precision(relevant, retrieved, min_precision=0.5)
        assert result.passed is True
        assert abs(result.details["precision"] - 0.6) < 1e-9

    def test_context_precision_fail(self) -> None:
        # SCENARIO: Only 1 out of 5 retrieved documents is relevant.
        # WHY: Precision = 1/5 = 0.2, below min_precision 0.5.
        # EXPECTED: raises MltkAssertionError.
        relevant = ["doc1"]
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_context_precision(relevant, retrieved, min_precision=0.5)
        assert exc_info.value.result.details["precision"] < 0.5

    def test_context_precision_perfect(self) -> None:
        # SCENARIO: All retrieved documents are in the relevant set.
        # WHY: Precision = 3/3 = 1.0.
        # EXPECTED: passes with precision=1.0.
        relevant = ["doc1", "doc2", "doc3", "doc4"]
        retrieved = ["doc1", "doc2", "doc3"]
        result = assert_context_precision(relevant, retrieved, min_precision=0.9)
        assert result.passed is True
        assert abs(result.details["precision"] - 1.0) < 1e-9

    def test_context_precision_no_retrieved_fails_by_default(self) -> None:
        # SCENARIO: Retriever returns no documents.
        # EXPECTED: Default on_empty="fail" rejects undefined precision.
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_context_precision(["doc1"], [], min_precision=0.5)
        assert "no documents retrieved" in exc_info.value.result.message.lower()


class TestContextRecall:
    """Context recall — |relevant ∩ retrieved| / |relevant|."""

    def test_context_recall_pass(self) -> None:
        # SCENARIO: 3 out of 4 relevant documents are retrieved.
        # WHY: Recall = 3/4 = 0.75, above the min_recall of 0.7.
        # EXPECTED: passes with recall=0.75.
        relevant = ["doc1", "doc2", "doc3", "doc4"]
        retrieved = ["doc1", "doc2", "doc3", "doc5"]
        result = assert_context_recall(relevant, retrieved, min_recall=0.7)
        assert result.passed is True
        assert abs(result.details["recall"] - 0.75) < 1e-9

    def test_context_recall_fail(self) -> None:
        # SCENARIO: Only 1 out of 4 relevant documents is retrieved.
        # WHY: Recall = 1/4 = 0.25, below min_recall 0.7.
        # EXPECTED: raises MltkAssertionError.
        relevant = ["doc1", "doc2", "doc3", "doc4"]
        retrieved = ["doc1", "doc5", "doc6"]
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_context_recall(relevant, retrieved, min_recall=0.7)
        assert exc_info.value.result.details["recall"] < 0.7

    def test_context_recall_perfect(self) -> None:
        # SCENARIO: All relevant documents are retrieved (plus extras).
        # WHY: Recall = 3/3 = 1.0; extra retrieved docs do not affect recall.
        # EXPECTED: passes with recall=1.0.
        relevant = ["doc1", "doc2", "doc3"]
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        result = assert_context_recall(relevant, retrieved, min_recall=1.0)
        assert result.passed is True
        assert abs(result.details["recall"] - 1.0) < 1e-9

    def test_context_recall_no_relevant_fails_by_default(self) -> None:
        # SCENARIO: No relevant documents are defined.
        # EXPECTED: Default on_empty="fail" rejects undefined recall.
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_context_recall([], ["doc1"], min_recall=0.5)
        assert "no relevant documents defined" in exc_info.value.result.message.lower()
