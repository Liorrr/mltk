"""Tests for mltk.domains.llm.conversation — multi-turn conversation evaluation."""

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.domains.llm.conversation import (
    assert_conversation_completeness,
    assert_knowledge_retention,
    assert_turn_relevancy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_turns(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    """Build a turn list from (user_msg, assistant_msg) pairs."""
    turns: list[dict[str, str]] = []
    for user_msg, assistant_msg in pairs:
        turns.append({"role": "user", "content": user_msg})
        turns.append({"role": "assistant", "content": assistant_msg})
    return turns


# ---------------------------------------------------------------------------
# assert_knowledge_retention
# ---------------------------------------------------------------------------


class TestKnowledgeRetention:
    """Knowledge retention — assistant keeps referencing facts it introduced."""

    def test_knowledge_retention_good(self) -> None:
        # SCENARIO: Two assistant turns that discuss the same subject with largely
        #           shared vocabulary (Python language features).
        # WHY: Retention = Jaccard overlap between consecutive assistant turns.
        #      High shared token set → overlap well above 0.3.
        # EXPECTED: score >= 0.3 → passes.
        turns = _make_turns(
            (
                "Tell me about Python.",
                "Python is a high-level programming language known for readability and simplicity.",
            ),
            (
                "More details?",
                "Python is a high-level language for programming and data science.",
            ),
        )
        result = assert_knowledge_retention(turns, min_score=0.3)
        assert result.passed is True
        assert result.details["score"] >= 0.3
        assert result.details["assistant_turns"] == 2

    def test_knowledge_retention_amnesia(self) -> None:
        # SCENARIO: Second assistant turn shares no tokens with the first.
        # WHY: Jaccard overlap of completely different vocabularies = 0.0.
        # EXPECTED: score < 0.5 → raises MltkAssertionError.
        turns = _make_turns(
            ("My name is Alice.", "Hello Alice, great to meet you."),
            ("What is my name?", "Jupiter orbits far beyond the asteroid belt."),
        )
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_knowledge_retention(turns, min_score=0.5)
        assert exc_info.value.result.details["score"] < 0.5

    def test_knowledge_retention_timing(self) -> None:
        # SCENARIO: Standard two-turn conversation.
        # WHY: @timed_assertion must populate duration_ms > 0.
        # EXPECTED: duration_ms > 0.
        turns = _make_turns(
            ("Tell me about Python.", "Python is a programming language."),
            ("More details?", "Python supports object-oriented programming and scripting."),
        )
        result = assert_knowledge_retention(turns, min_score=0.1)
        assert result.duration_ms > 0

    def test_knowledge_retention_empty_turns(self) -> None:
        # SCENARIO: Caller passes an empty turn list.
        # WHY: Edge case — no turns means nothing to forget; should be trivially OK.
        # EXPECTED: passes with score=1.0.
        result = assert_knowledge_retention([], min_score=0.7)
        assert result.passed is True
        assert result.details["score"] == 1.0

    def test_knowledge_retention_single_assistant_turn(self) -> None:
        # SCENARIO: Conversation has only one assistant response.
        # WHY: No consecutive pair exists — nothing to measure.
        # EXPECTED: trivially passes with score=1.0.
        turns = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = assert_knowledge_retention(turns, min_score=0.7)
        assert result.passed is True
        assert result.details["score"] == 1.0


# ---------------------------------------------------------------------------
# assert_turn_relevancy
# ---------------------------------------------------------------------------


class TestTurnRelevancy:
    """Turn relevancy — each assistant response addresses the preceding question."""

    def test_turn_relevancy_good(self) -> None:
        # SCENARIO: Bot answers with content that directly echoes the question keywords.
        # WHY: "python" and "language" both appear in user question and assistant answer.
        # EXPECTED: score >= 0.5 → passes.
        turns = _make_turns(
            ("What is the Python programming language?",
             "Python is a high-level programming language known for readability."),
        )
        result = assert_turn_relevancy(turns, min_score=0.4)
        assert result.passed is True
        assert result.details["score"] >= 0.4
        assert result.details["pairs_evaluated"] == 1

    def test_turn_relevancy_off_topic(self) -> None:
        # SCENARIO: Assistant answers a completely unrelated topic.
        # WHY: Zero question-token overlap with the off-topic answer → score ≈ 0.
        # EXPECTED: score < 0.5 → raises MltkAssertionError.
        turns = _make_turns(
            ("What is machine learning?",
             "The weather today is sunny with blue skies and warm temperatures."),
        )
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_turn_relevancy(turns, min_score=0.5)
        assert exc_info.value.result.details["score"] < 0.5

    def test_turn_relevancy_multiple_pairs(self) -> None:
        # SCENARIO: Three-turn conversation; all answers address their questions.
        # WHY: Mean overlap across all three (user, assistant) pairs should be high.
        # EXPECTED: passes.
        turns = _make_turns(
            ("Tell me about neural networks.",
             "Neural networks are a machine learning model inspired by the brain."),
            ("How do neural networks learn?",
             "Neural networks learn by adjusting weights through backpropagation."),
            ("What is deep learning?",
             "Deep learning uses deep neural networks with many layers."),
        )
        result = assert_turn_relevancy(turns, min_score=0.2)
        assert result.passed is True
        assert result.details["pairs_evaluated"] == 3

    def test_turn_relevancy_empty_turns(self) -> None:
        # SCENARIO: Empty conversation list.
        # WHY: Nothing to evaluate; trivially relevant.
        # EXPECTED: passes with score=1.0.
        result = assert_turn_relevancy([], min_score=0.5)
        assert result.passed is True
        assert result.details["score"] == 1.0

    def test_turn_relevancy_single_turn(self) -> None:
        # SCENARIO: Only a single user message, no assistant response follows.
        # WHY: No adjacent (user, assistant) pair to evaluate.
        # EXPECTED: trivially passes with score=1.0.
        turns = [{"role": "user", "content": "Hello world."}]
        result = assert_turn_relevancy(turns, min_score=0.5)
        assert result.passed is True
        assert result.details["pairs_evaluated"] == 0


# ---------------------------------------------------------------------------
# assert_conversation_completeness
# ---------------------------------------------------------------------------


class TestConversationCompleteness:
    """Conversation completeness — assistant covers all expected topics."""

    def test_completeness_all_topics(self) -> None:
        # SCENARIO: Assistant response mentions all required topics explicitly.
        # WHY: Each topic token set is a subset of the assistant text tokens.
        # EXPECTED: score = 1.0 → passes.
        turns = _make_turns(
            ("Tell me about Python and Django.",
             "Python is a language. Django is a web framework built on Python."),
        )
        result = assert_conversation_completeness(
            turns, expected_topics=["python", "django"], min_coverage=1.0
        )
        assert result.passed is True
        assert result.details["score"] == 1.0
        assert result.details["missing_topics"] == []

    def test_completeness_missing_topics(self) -> None:
        # SCENARIO: Assistant only covers Python but not Django.
        # WHY: "django" tokens not found in assistant text → 1/2 topics covered = 0.5.
        # EXPECTED: score 0.5 < min_coverage 0.8 → raises MltkAssertionError.
        turns = _make_turns(
            ("Tell me about Python and Django.",
             "Python is a high-level general-purpose programming language."),
        )
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_conversation_completeness(
                turns, expected_topics=["python", "django"], min_coverage=0.8
            )
        result = exc_info.value.result
        assert result.details["score"] < 0.8
        assert "django" in result.details["missing_topics"]

    def test_completeness_partial_coverage(self) -> None:
        # SCENARIO: 3 out of 4 topics are covered; threshold is 0.7.
        # WHY: 3/4 = 0.75 >= 0.7 → should pass.
        # EXPECTED: passes.
        turns = _make_turns(
            ("Describe these topics.",
             "Python is great. Machine learning uses data. Neural networks are deep models."),
        )
        result = assert_conversation_completeness(
            turns,
            expected_topics=["python", "machine learning", "neural networks", "quantum"],
            min_coverage=0.7,
        )
        assert result.passed is True
        assert result.details["topics_found"] == 3

    def test_completeness_empty_turns(self) -> None:
        # SCENARIO: No conversation turns at all, but topics are required.
        # WHY: Empty assistant text matches nothing → score = 0.0 < min_coverage.
        # EXPECTED: raises MltkAssertionError.
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_conversation_completeness(
                [], expected_topics=["python"], min_coverage=0.5
            )
        assert exc_info.value.result.details["score"] == 0.0

    def test_completeness_no_expected_topics(self) -> None:
        # SCENARIO: Caller provides an empty expected_topics list.
        # WHY: Nothing required → trivially complete.
        # EXPECTED: passes with score=1.0.
        turns = _make_turns(("Hi.", "Hello!"))
        result = assert_conversation_completeness(turns, expected_topics=[], min_coverage=1.0)
        assert result.passed is True
        assert result.details["score"] == 1.0

    def test_completeness_result_details(self) -> None:
        # SCENARIO: Normal multi-topic conversation.
        # WHY: Verify the result carries all expected detail keys.
        # EXPECTED: result.details contains score, topics_found, topics_total, missing_topics.
        turns = _make_turns(
            ("What is ML?", "Machine learning is a subset of AI."),
        )
        result = assert_conversation_completeness(
            turns, expected_topics=["machine learning", "ai"], min_coverage=0.5
        )
        assert "score" in result.details
        assert "topics_found" in result.details
        assert "topics_total" in result.details
        assert "missing_topics" in result.details
