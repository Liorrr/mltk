"""Tests for SyntheticQAGenerator v2 features.

Covers multi-hop enhanced, conversational, and distracting
question generation in both template and LLM modes.
All LLM calls are mocked; no external dependencies required.
"""

from __future__ import annotations

import json

from mltk.domains.llm.synthetic import (
    QAPair,
    QuestionType,
    SyntheticQAGenerator,
)
from mltk.domains.llm.synthetic._templates import (
    build_prompt,
    parse_conversational_response,
)

# -- Shared helpers ----------------------------------------

SEED = 42

CHUNK_A = (
    "Python is a programming language. "
    "It was created by Guido van Rossum. "
    "Python supports multiple paradigms. "
    "The first version was released in 1991. "
    "Python is widely used in data science."
)

CHUNK_B = (
    "Rust is a systems programming language. "
    "It was developed at Mozilla Research. "
    "Rust guarantees memory safety. "
    "The first stable release was in 2015. "
    "Rust is used for performance-critical code."
)

CHUNK_C = (
    "JavaScript runs in the browser. "
    "It was created by Brendan Eich. "
    "JavaScript supports event-driven programming. "
    "Node.js enables server-side JavaScript. "
    "JavaScript is the language of the web."
)

CHUNKS = [CHUNK_A, CHUNK_B, CHUNK_C]


def _mock_llm(prompt: str) -> str:
    """Return valid JSON QA pair."""
    return (
        '{"question": "What is X?", '
        '"answer": "X is Y."}'
    )


def _mock_llm_conversational(prompt: str) -> str:
    """Return valid conversational JSON."""
    return json.dumps({
        "turns": [
            {
                "question": "What is X?",
                "answer": "X is a language.",
            },
            {
                "question": "What else about X?",
                "answer": "X is popular.",
            },
        ],
    })


def _mock_llm_conversational_3(prompt: str) -> str:
    """Return 3-turn conversational JSON."""
    return json.dumps({
        "turns": [
            {
                "question": "What is X?",
                "answer": "X is a language.",
            },
            {
                "question": "What else about X?",
                "answer": "X is popular.",
            },
            {
                "question": "Why is X popular?",
                "answer": "X has a great community.",
            },
        ],
    })


# ===========================================================
# 1. QuestionType enum updates (3 tests)
# ===========================================================


class TestQuestionTypeV2:
    """Tests for new QuestionType enum members."""

    def test_conversational_exists(self) -> None:
        """CONVERSATIONAL enum member exists."""
        assert QuestionType.CONVERSATIONAL.value == (
            "conversational"
        )

    def test_distracting_exists(self) -> None:
        """DISTRACTING enum member exists."""
        assert QuestionType.DISTRACTING.value == (
            "distracting"
        )

    def test_total_enum_count(self) -> None:
        """Enum has 7 members total (5 + 2 new)."""
        assert len(QuestionType) == 7


# ===========================================================
# 2. Multi-hop enhanced (7 tests)
# ===========================================================


class TestMultiHopEnhanced:
    """Tests for generate_multi_hop method."""

    def test_multi_hop_template_returns_pairs(
        self,
    ) -> None:
        """Template mode returns QAPair list."""
        gen = SyntheticQAGenerator(seed=SEED)
        pairs = gen.generate_multi_hop(CHUNKS, n=3)
        assert isinstance(pairs, list)
        assert len(pairs) == 3
        for p in pairs:
            assert isinstance(p, QAPair)

    def test_multi_hop_requires_2_chunks(
        self,
    ) -> None:
        """Single chunk returns empty list."""
        gen = SyntheticQAGenerator(seed=SEED)
        pairs = gen.generate_multi_hop(
            [CHUNK_A], n=5,
        )
        assert pairs == []

    def test_multi_hop_context_is_list(
        self,
    ) -> None:
        """Context is a list of source chunks."""
        gen = SyntheticQAGenerator(seed=SEED)
        pairs = gen.generate_multi_hop(CHUNKS, n=1)
        assert len(pairs) >= 1
        assert isinstance(pairs[0].context, list)
        assert len(pairs[0].context) == 2

    def test_multi_hop_question_type(self) -> None:
        """Question type is MULTI_HOP."""
        gen = SyntheticQAGenerator(seed=SEED)
        pairs = gen.generate_multi_hop(CHUNKS, n=1)
        assert len(pairs) >= 1
        assert (
            pairs[0].question_type
            == QuestionType.MULTI_HOP
        )

    def test_multi_hop_metadata_indices(
        self,
    ) -> None:
        """Metadata has chunk_indices."""
        gen = SyntheticQAGenerator(seed=SEED)
        pairs = gen.generate_multi_hop(CHUNKS, n=1)
        assert len(pairs) >= 1
        assert "chunk_indices" in pairs[0].metadata
        assert len(pairs[0].metadata["chunk_indices"]) == 2

    def test_multi_hop_llm_mode(self) -> None:
        """LLM mode generates multi-hop pairs."""
        gen = SyntheticQAGenerator(
            llm_fn=_mock_llm,
            seed=SEED,
            quality_filter=False,
        )
        pairs = gen.generate_multi_hop(CHUNKS, n=2)
        assert len(pairs) == 2
        for p in pairs:
            assert p.metadata["mode"] == "llm"

    def test_multi_hop_empty_chunks(self) -> None:
        """Empty chunks list returns empty."""
        gen = SyntheticQAGenerator(seed=SEED)
        pairs = gen.generate_multi_hop([], n=5)
        assert pairs == []


# ===========================================================
# 3. Conversational (8 tests)
# ===========================================================


class TestConversational:
    """Tests for generate_conversational method."""

    def test_conversational_returns_list_of_lists(
        self,
    ) -> None:
        """Returns list of conversations."""
        gen = SyntheticQAGenerator(seed=SEED)
        convs = gen.generate_conversational(
            CHUNKS, n=2, turns=2,
        )
        assert isinstance(convs, list)
        assert len(convs) == 2
        for conv in convs:
            assert isinstance(conv, list)
            for pair in conv:
                assert isinstance(pair, QAPair)

    def test_conversational_turns_count(
        self,
    ) -> None:
        """Each conversation has requested turns."""
        gen = SyntheticQAGenerator(seed=SEED)
        convs = gen.generate_conversational(
            CHUNKS, n=1, turns=2,
        )
        assert len(convs) >= 1
        assert len(convs[0]) == 2

    def test_conversational_3_turns(self) -> None:
        """3 turns per conversation."""
        gen = SyntheticQAGenerator(seed=SEED)
        convs = gen.generate_conversational(
            CHUNKS, n=1, turns=3,
        )
        assert len(convs) >= 1
        assert len(convs[0]) == 3

    def test_conversational_question_type(
        self,
    ) -> None:
        """All pairs have CONVERSATIONAL type."""
        gen = SyntheticQAGenerator(seed=SEED)
        convs = gen.generate_conversational(
            CHUNKS, n=1, turns=2,
        )
        for conv in convs:
            for pair in conv:
                assert (
                    pair.question_type
                    == QuestionType.CONVERSATIONAL
                )

    def test_conversational_followup_references(
        self,
    ) -> None:
        """Follow-up question references prior answer."""
        gen = SyntheticQAGenerator(seed=SEED)
        convs = gen.generate_conversational(
            CHUNKS, n=1, turns=2,
        )
        assert len(convs) >= 1
        conv = convs[0]
        assert len(conv) >= 2
        # In template mode, follow-up references prev
        prev_answer = conv[0].answer
        assert (
            prev_answer in conv[1].question
            or "Following up" in conv[1].question
        )

    def test_conversational_metadata_turn(
        self,
    ) -> None:
        """Each pair has turn number in metadata."""
        gen = SyntheticQAGenerator(seed=SEED)
        convs = gen.generate_conversational(
            CHUNKS, n=1, turns=2,
        )
        assert len(convs) >= 1
        for i, pair in enumerate(convs[0]):
            assert pair.metadata["turn"] == i + 1

    def test_conversational_llm_mode(self) -> None:
        """LLM mode generates conversations."""
        gen = SyntheticQAGenerator(
            llm_fn=_mock_llm_conversational,
            seed=SEED,
            quality_filter=False,
        )
        convs = gen.generate_conversational(
            CHUNKS, n=1, turns=2,
        )
        assert len(convs) >= 1
        assert len(convs[0]) == 2
        for pair in convs[0]:
            assert pair.metadata["mode"] == "llm"

    def test_conversational_empty_chunks(
        self,
    ) -> None:
        """Empty chunks returns empty list."""
        gen = SyntheticQAGenerator(seed=SEED)
        convs = gen.generate_conversational(
            [], n=5,
        )
        assert convs == []


# ===========================================================
# 4. Distracting (7 tests)
# ===========================================================


class TestDistracting:
    """Tests for generate_distracting method."""

    def test_distracting_returns_pairs(
        self,
    ) -> None:
        """Template mode returns QAPair list."""
        gen = SyntheticQAGenerator(seed=SEED)
        pairs = gen.generate_distracting(CHUNKS, n=3)
        assert isinstance(pairs, list)
        assert len(pairs) == 3
        for p in pairs:
            assert isinstance(p, QAPair)

    def test_distracting_requires_2_chunks(
        self,
    ) -> None:
        """Single chunk returns empty list."""
        gen = SyntheticQAGenerator(seed=SEED)
        pairs = gen.generate_distracting(
            [CHUNK_A], n=5,
        )
        assert pairs == []

    def test_distracting_question_type(
        self,
    ) -> None:
        """Question type is DISTRACTING."""
        gen = SyntheticQAGenerator(seed=SEED)
        pairs = gen.generate_distracting(CHUNKS, n=1)
        assert len(pairs) >= 1
        assert (
            pairs[0].question_type
            == QuestionType.DISTRACTING
        )

    def test_distracting_metadata_has_distractor(
        self,
    ) -> None:
        """Metadata includes distractor_chunk."""
        gen = SyntheticQAGenerator(seed=SEED)
        pairs = gen.generate_distracting(CHUNKS, n=1)
        assert len(pairs) >= 1
        assert "distractor_chunk" in pairs[0].metadata

    def test_distracting_distractor_differs(
        self,
    ) -> None:
        """Distractor chunk differs from context."""
        gen = SyntheticQAGenerator(seed=SEED)
        pairs = gen.generate_distracting(CHUNKS, n=3)
        for pair in pairs:
            assert (
                pair.metadata["distractor_chunk"]
                != pair.context
            )

    def test_distracting_llm_mode(self) -> None:
        """LLM mode generates distracting pairs."""
        gen = SyntheticQAGenerator(
            llm_fn=_mock_llm,
            seed=SEED,
            quality_filter=False,
        )
        pairs = gen.generate_distracting(CHUNKS, n=2)
        assert len(pairs) == 2
        for p in pairs:
            assert p.metadata["mode"] == "llm"

    def test_distracting_empty_chunks(
        self,
    ) -> None:
        """Empty chunks returns empty list."""
        gen = SyntheticQAGenerator(seed=SEED)
        pairs = gen.generate_distracting([], n=5)
        assert pairs == []


# ===========================================================
# 5. Template building for new types (3 tests)
# ===========================================================


class TestNewTemplates:
    """Tests for build_prompt with new question types."""

    def test_build_prompt_conversational(
        self,
    ) -> None:
        """Conversational prompt has turn count."""
        prompt = build_prompt(
            "conversational",
            context="Some text here.",
            turns=3,
        )
        assert "3-turn" in prompt
        assert "Some text here." in prompt

    def test_build_prompt_distracting(self) -> None:
        """Distracting prompt has both contexts."""
        prompt = build_prompt(
            "distracting",
            context="Target text.",
            distractor="Misleading text.",
        )
        assert "Target text." in prompt
        assert "Misleading text." in prompt

    def test_build_prompt_multi_hop_enhanced(
        self,
    ) -> None:
        """Enhanced multi-hop has numbered passages."""
        prompt = build_prompt(
            "multi_hop_enhanced",
            context="Chunk A text.",
            contexts=[
                "Chunk A text.",
                "Chunk B text.",
            ],
        )
        assert "Passage 1" in prompt
        assert "Passage 2" in prompt


# ===========================================================
# 6. Conversational response parsing (3 tests)
# ===========================================================


class TestConversationalParsing:
    """Tests for parse_conversational_response."""

    def test_parse_valid_turns(self) -> None:
        """Valid JSON with turns key parses OK."""
        raw = json.dumps({
            "turns": [
                {"question": "Q1?", "answer": "A1."},
                {"question": "Q2?", "answer": "A2."},
            ],
        })
        result = parse_conversational_response(raw)
        assert result is not None
        assert len(result) == 2
        assert result[0]["question"] == "Q1?"
        assert result[1]["answer"] == "A2."

    def test_parse_empty_returns_none(self) -> None:
        """Empty input returns None."""
        assert parse_conversational_response("") is None
        assert parse_conversational_response(
            "  ",
        ) is None

    def test_parse_invalid_json_returns_none(
        self,
    ) -> None:
        """Non-JSON returns None."""
        assert parse_conversational_response(
            "not json at all",
        ) is None


# ===========================================================
# 7. LLM mode with 3-turn conversation (1 test)
# ===========================================================


class TestLLMConversational3Turn:
    """LLM mode with 3 turns."""

    def test_llm_3_turns(self) -> None:
        """LLM mode with 3 turns."""
        gen = SyntheticQAGenerator(
            llm_fn=_mock_llm_conversational_3,
            seed=SEED,
            quality_filter=False,
        )
        convs = gen.generate_conversational(
            CHUNKS, n=1, turns=3,
        )
        assert len(convs) >= 1
        assert len(convs[0]) == 3
        for i, pair in enumerate(convs[0]):
            assert pair.metadata["turn"] == i + 1
