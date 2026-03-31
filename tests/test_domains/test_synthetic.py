"""Tests for SyntheticQAGenerator and supporting modules.

Covers QAPair dataclass, QuestionType enum, text splitter,
template mode, LLM mode (mocked), quality filter, and
integration scenarios.  All LLM calls are mocked; no
external dependencies required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mltk.domains.llm.synthetic import (
    QAPair,
    QuestionType,
    SyntheticQAGenerator,
)
from mltk.domains.llm.synthetic._quality import (
    QualityFilter,
)
from mltk.domains.llm.synthetic._splitter import (
    split_text,
)

# -- Shared helpers -----------------------------------------

SEED = 42

SAMPLE_CHUNK = (
    "Python is a programming language. "
    "It was created by Guido van Rossum. "
    "Python supports multiple paradigms. "
    "The first version was released in 1991. "
    "Python is widely used in data science."
)

SAMPLE_DOC = "\n\n".join(
    [
        " ".join(f"word{j}" for j in range(60))
        for _ in range(10)
    ]
)


def _mock_llm(prompt: str) -> str:
    """Return valid JSON QA pair."""
    return (
        '{"question": "What is X?", '
        '"answer": "X is Y."}'
    )


def _mock_llm_quality(prompt: str) -> str:
    """Return high quality scores."""
    if "Rate this QA" in prompt:
        return (
            '{"self_containment": 0.9, '
            '"answerability": 0.9}'
        )
    return (
        '{"question": "What is X?", '
        '"answer": "X is Y."}'
    )


def _mock_llm_bad_quality(prompt: str) -> str:
    """Return low quality scores."""
    if "Rate this QA" in prompt:
        return (
            '{"self_containment": 0.3, '
            '"answerability": 0.2}'
        )
    return (
        '{"question": "What?", '
        '"answer": "Dunno."}'
    )


# ===========================================================
# 1. QAPair dataclass (5 tests)
# ===========================================================


class TestQAPair:
    """Tests for the QAPair dataclass."""

    def test_qapair_fields(self) -> None:
        """All expected fields are present."""
        pair = QAPair(
            question="What is X?",
            answer="X is Y.",
            context="X is Y in the doc.",
        )
        assert pair.question == "What is X?"
        assert pair.answer == "X is Y."
        assert pair.context == "X is Y in the doc."
        assert pair.question_type == QuestionType.FACTUAL
        assert pair.metadata == {}

    def test_qapair_to_dict(self) -> None:
        """Serialization produces correct dict."""
        pair = QAPair(
            question="Q?",
            answer="A.",
            context="ctx",
            question_type=QuestionType.REASONING,
        )
        d = pair.to_dict()
        assert d["question"] == "Q?"
        assert d["answer"] == "A."
        assert d["context"] == "ctx"
        assert d["question_type"] == "reasoning"

    def test_qapair_to_dict_list_context(
        self,
    ) -> None:
        """List context is joined in to_dict."""
        pair = QAPair(
            question="Q?",
            answer="A.",
            context=["chunk1", "chunk2"],
        )
        d = pair.to_dict()
        assert d["context"] == "chunk1 chunk2"

    def test_qapair_default_type(self) -> None:
        """Default question_type is FACTUAL."""
        pair = QAPair(
            question="Q?",
            answer="A.",
            context="ctx",
        )
        assert (
            pair.question_type
            == QuestionType.FACTUAL
        )

    def test_qapair_metadata(self) -> None:
        """Custom metadata preserved in to_dict."""
        pair = QAPair(
            question="Q?",
            answer="A.",
            context="ctx",
            metadata={"source": "doc.txt", "idx": 3},
        )
        d = pair.to_dict()
        assert d["source"] == "doc.txt"
        assert d["idx"] == 3


# ===========================================================
# 2. QuestionType enum (3 tests)
# ===========================================================


class TestQuestionType:
    """Tests for QuestionType enum."""

    def test_question_type_values(self) -> None:
        """All 5 types exist with correct values."""
        expected = {
            "factual",
            "reasoning",
            "multi_hop",
            "counterfactual",
            "out_of_scope",
        }
        actual = {qt.value for qt in QuestionType}
        assert actual == expected
        assert len(QuestionType) == 5

    def test_question_type_from_string(
        self,
    ) -> None:
        """String conversion works for each type."""
        assert (
            QuestionType("factual")
            == QuestionType.FACTUAL
        )
        assert (
            QuestionType("reasoning")
            == QuestionType.REASONING
        )
        assert (
            QuestionType("multi_hop")
            == QuestionType.MULTI_HOP
        )
        assert (
            QuestionType("counterfactual")
            == QuestionType.COUNTERFACTUAL
        )
        assert (
            QuestionType("out_of_scope")
            == QuestionType.OUT_OF_SCOPE
        )

    def test_question_type_all_unique(self) -> None:
        """No duplicate values."""
        values = [qt.value for qt in QuestionType]
        assert len(values) == len(set(values))


# ===========================================================
# 3. Text splitter (8 tests)
# ===========================================================


class TestSplitter:
    """Tests for the split_text function."""

    def test_split_short_text(self) -> None:
        """Text shorter than chunk_size returns one chunk."""
        text = "Hello world this is a test."
        chunks = split_text(
            text, chunk_size=100, min_chunk_words=1,
        )
        assert len(chunks) == 1
        assert "Hello" in chunks[0]

    def test_split_exact_chunk(self) -> None:
        """Text with exactly chunk_size words."""
        words = [f"w{i}" for i in range(50)]
        text = " ".join(words)
        chunks = split_text(
            text,
            chunk_size=50,
            chunk_overlap=0,
            min_chunk_words=1,
        )
        assert len(chunks) >= 1
        assert len(chunks[0].split()) == 50

    def test_split_with_overlap(self) -> None:
        """Overlap produces shared words."""
        words = [f"w{i}" for i in range(100)]
        text = " ".join(words)
        chunks = split_text(
            text,
            chunk_size=50,
            chunk_overlap=10,
            min_chunk_words=1,
        )
        assert len(chunks) >= 2
        first_words = set(chunks[0].split())
        second_words = set(chunks[1].split())
        shared = first_words & second_words
        assert len(shared) >= 10

    def test_split_empty_text(self) -> None:
        """Empty string returns empty list."""
        assert split_text("") == []
        assert split_text("   ") == []

    def test_split_min_words_filter(self) -> None:
        """Chunks below min_words are filtered."""
        text = "short"
        chunks = split_text(
            text, chunk_size=100, min_chunk_words=30,
        )
        assert len(chunks) == 0

    def test_split_respects_paragraphs(
        self,
    ) -> None:
        """Paragraph boundaries are respected."""
        para1 = " ".join(f"a{i}" for i in range(40))
        para2 = " ".join(f"b{i}" for i in range(40))
        text = f"{para1}\n\n{para2}"
        chunks = split_text(
            text,
            chunk_size=50,
            chunk_overlap=0,
            min_chunk_words=30,
        )
        assert len(chunks) >= 2

    def test_split_single_word(self) -> None:
        """Single word text handled gracefully."""
        chunks = split_text(
            "hello", chunk_size=10, min_chunk_words=1,
        )
        assert len(chunks) <= 1

    def test_split_large_text(self) -> None:
        """10000 words produce many chunks."""
        words = [f"w{i}" for i in range(10000)]
        text = " ".join(words)
        chunks = split_text(
            text,
            chunk_size=200,
            chunk_overlap=20,
            min_chunk_words=30,
        )
        assert len(chunks) >= 40


# ===========================================================
# 4. Template mode (10 tests)
# ===========================================================


class TestTemplateMode:
    """Tests for template-mode generation."""

    def test_template_generates_qapairs(
        self,
    ) -> None:
        """Returns list of QAPair objects."""
        gen = SyntheticQAGenerator(seed=SEED)
        pairs = gen.generate_from_chunks(
            [SAMPLE_CHUNK], n=3,
        )
        assert isinstance(pairs, list)
        for p in pairs:
            assert isinstance(p, QAPair)

    def test_template_correct_count(self) -> None:
        """n=5 produces 5 pairs."""
        gen = SyntheticQAGenerator(seed=SEED)
        pairs = gen.generate_from_chunks(
            [SAMPLE_CHUNK], n=5,
        )
        assert len(pairs) == 5

    def test_template_factual_type(self) -> None:
        """Default type is FACTUAL."""
        gen = SyntheticQAGenerator(
            seed=SEED,
            question_types=[QuestionType.FACTUAL],
        )
        pairs = gen.generate_from_chunks(
            [SAMPLE_CHUNK], n=1,
        )
        assert len(pairs) == 1
        assert (
            pairs[0].question_type
            == QuestionType.FACTUAL
        )

    @pytest.mark.parametrize(
        "qt",
        list(QuestionType),
        ids=[qt.value for qt in QuestionType],
    )
    def test_template_all_types(
        self, qt: QuestionType,
    ) -> None:
        """Each type produces a pair."""
        gen = SyntheticQAGenerator(
            seed=SEED,
            question_types=[qt],
        )
        pairs = gen.generate_from_chunks(
            [SAMPLE_CHUNK], n=1,
        )
        assert len(pairs) == 1
        assert pairs[0].question_type == qt

    def test_template_from_text(self) -> None:
        """generate_from_text splits and generates."""
        gen = SyntheticQAGenerator(seed=SEED)
        pairs = gen.generate_from_text(
            SAMPLE_DOC, n=3,
        )
        assert isinstance(pairs, list)
        assert len(pairs) >= 1

    def test_template_from_chunks(self) -> None:
        """generate_from_chunks works directly."""
        gen = SyntheticQAGenerator(seed=SEED)
        chunks = [SAMPLE_CHUNK, SAMPLE_CHUNK]
        pairs = gen.generate_from_chunks(
            chunks, n=2,
        )
        assert len(pairs) == 2

    def test_template_generate_one(self) -> None:
        """generate_one returns a single QAPair."""
        gen = SyntheticQAGenerator(seed=SEED)
        pair = gen.generate_one(SAMPLE_CHUNK)
        assert isinstance(pair, QAPair)
        assert pair.question
        assert pair.answer

    def test_template_deterministic(self) -> None:
        """Same seed produces same output."""
        gen1 = SyntheticQAGenerator(seed=SEED)
        gen2 = SyntheticQAGenerator(seed=SEED)
        p1 = gen1.generate_from_chunks(
            [SAMPLE_CHUNK], n=3,
        )
        p2 = gen2.generate_from_chunks(
            [SAMPLE_CHUNK], n=3,
        )
        for a, b in zip(p1, p2, strict=True):
            assert a.question == b.question
            assert a.answer == b.answer

    def test_template_empty_chunk(self) -> None:
        """Empty chunk returns None."""
        gen = SyntheticQAGenerator(seed=SEED)
        pair = gen.generate_one("")
        assert pair is None

    def test_template_short_chunk(self) -> None:
        """Very short chunk handled gracefully."""
        gen = SyntheticQAGenerator(seed=SEED)
        pair = gen.generate_one("Hi.")
        # May return None or a pair; must not crash
        assert pair is None or isinstance(
            pair, QAPair,
        )


# ===========================================================
# 5. LLM mode (12 tests)
# ===========================================================


class TestLLMMode:
    """Tests for LLM-mode generation (all mocked)."""

    def test_llm_mode_calls_fn(self) -> None:
        """LLM function is called."""
        mock = MagicMock(return_value=(
            '{"question": "Q?", "answer": "A."}'
        ))
        gen = SyntheticQAGenerator(
            llm_fn=mock,
            seed=SEED,
            quality_filter=False,
        )
        gen.generate_one(SAMPLE_CHUNK)
        mock.assert_called()

    def test_llm_mode_prompt_contains_context(
        self,
    ) -> None:
        """Chunk text appears in the prompt."""
        captured: list[str] = []

        def capture(prompt: str) -> str:
            captured.append(prompt)
            return (
                '{"question": "Q?", '
                '"answer": "A."}'
            )

        gen = SyntheticQAGenerator(
            llm_fn=capture,
            seed=SEED,
            quality_filter=False,
        )
        gen.generate_one(SAMPLE_CHUNK)
        assert len(captured) >= 1
        assert "Python" in captured[0]

    def test_llm_mode_prompt_contains_type(
        self,
    ) -> None:
        """Question type influences prompt text."""
        captured: list[str] = []

        def capture(prompt: str) -> str:
            captured.append(prompt)
            return (
                '{"question": "Q?", '
                '"answer": "A."}'
            )

        gen = SyntheticQAGenerator(
            llm_fn=capture,
            seed=SEED,
            quality_filter=False,
        )
        gen.generate_one(
            SAMPLE_CHUNK,
            question_type=QuestionType.REASONING,
        )
        assert len(captured) >= 1
        assert "reasoning" in captured[0].lower()

    def test_llm_mode_parses_response(
        self,
    ) -> None:
        """Valid JSON response is parsed."""
        gen = SyntheticQAGenerator(
            llm_fn=_mock_llm,
            seed=SEED,
            quality_filter=False,
        )
        pair = gen.generate_one(SAMPLE_CHUNK)
        assert pair is not None
        assert pair.question == "What is X?"
        assert pair.answer == "X is Y."

    def test_llm_mode_returns_qapair(
        self,
    ) -> None:
        """Result is a QAPair instance."""
        gen = SyntheticQAGenerator(
            llm_fn=_mock_llm,
            seed=SEED,
            quality_filter=False,
        )
        pair = gen.generate_one(SAMPLE_CHUNK)
        assert isinstance(pair, QAPair)

    def test_llm_mode_reasoning_type(
        self,
    ) -> None:
        """REASONING prompt differs from FACTUAL."""
        prompts: list[str] = []

        def capture(prompt: str) -> str:
            prompts.append(prompt)
            return (
                '{"question": "Why?", '
                '"answer": "Because."}'
            )

        gen = SyntheticQAGenerator(
            llm_fn=capture,
            seed=SEED,
            quality_filter=False,
        )
        gen.generate_one(
            SAMPLE_CHUNK,
            question_type=QuestionType.FACTUAL,
        )
        gen.generate_one(
            SAMPLE_CHUNK,
            question_type=QuestionType.REASONING,
        )
        assert len(prompts) == 2
        assert prompts[0] != prompts[1]

    def test_llm_mode_counterfactual(
        self,
    ) -> None:
        """COUNTERFACTUAL prompt has What if."""
        captured: list[str] = []

        def capture(prompt: str) -> str:
            captured.append(prompt)
            return (
                '{"question": "What if?", '
                '"answer": "Then."}'
            )

        gen = SyntheticQAGenerator(
            llm_fn=capture,
            seed=SEED,
            quality_filter=False,
        )
        gen.generate_one(
            SAMPLE_CHUNK,
            question_type=(
                QuestionType.COUNTERFACTUAL
            ),
        )
        assert "counterfactual" in (
            captured[0].lower()
        )

    def test_llm_mode_out_of_scope(self) -> None:
        """OOS prompt mentions out-of-scope."""
        captured: list[str] = []

        def capture(prompt: str) -> str:
            captured.append(prompt)
            return (
                '{"question": "Email?", '
                '"answer": "Not available."}'
            )

        gen = SyntheticQAGenerator(
            llm_fn=capture,
            seed=SEED,
            quality_filter=False,
        )
        gen.generate_one(
            SAMPLE_CHUNK,
            question_type=QuestionType.OUT_OF_SCOPE,
        )
        prompt_lower = captured[0].lower()
        assert (
            "out-of-scope" in prompt_lower
            or "cannot be answered" in prompt_lower
        )

    def test_llm_mode_multi_hop(self) -> None:
        """MULTI_HOP prompt has context_b."""
        captured: list[str] = []

        def capture(prompt: str) -> str:
            captured.append(prompt)
            return (
                '{"question": "Both?", '
                '"answer": "Combined."}'
            )

        gen = SyntheticQAGenerator(
            llm_fn=capture,
            seed=SEED,
            quality_filter=False,
        )
        gen.generate_one(
            SAMPLE_CHUNK,
            question_type=QuestionType.MULTI_HOP,
            context_chunks=["Second chunk text."],
        )
        assert "Context B" in captured[0]

    def test_llm_mode_fallback_on_parse_error(
        self,
    ) -> None:
        """Bad LLM output returns None gracefully."""

        def bad_llm(prompt: str) -> str:
            return "This is not JSON at all."

        gen = SyntheticQAGenerator(
            llm_fn=bad_llm,
            seed=SEED,
            quality_filter=False,
            max_retries=0,
        )
        pair = gen.generate_one(SAMPLE_CHUNK)
        assert pair is None

    def test_llm_mode_custom_types(self) -> None:
        """Subset of question types works."""
        gen = SyntheticQAGenerator(
            llm_fn=_mock_llm,
            seed=SEED,
            quality_filter=False,
            question_types=[
                QuestionType.FACTUAL,
                QuestionType.REASONING,
            ],
        )
        pairs = gen.generate_from_chunks(
            [SAMPLE_CHUNK], n=4,
        )
        types = {p.question_type for p in pairs}
        assert types <= {
            QuestionType.FACTUAL,
            QuestionType.REASONING,
        }

    def test_llm_mode_respects_n(self) -> None:
        """n=3 produces 3 pairs."""
        gen = SyntheticQAGenerator(
            llm_fn=_mock_llm,
            seed=SEED,
            quality_filter=False,
        )
        pairs = gen.generate_from_chunks(
            [SAMPLE_CHUNK], n=3,
        )
        assert len(pairs) == 3


# ===========================================================
# 6. Quality filter (7 tests)
# ===========================================================


class TestQualityFilter:
    """Tests for QualityFilter."""

    def _pair(
        self,
        q: str = "Q?",
        a: str = "A.",
        ctx: str = "ctx",
    ) -> QAPair:
        return QAPair(
            question=q, answer=a, context=ctx,
        )

    def test_quality_filter_passes_good(
        self,
    ) -> None:
        """High scores pass the filter."""
        qf = QualityFilter(
            llm_fn=lambda p: (
                '{"self_containment": 0.9, '
                '"answerability": 0.9}'
            ),
            threshold=0.6,
        )
        assert qf.passes(self._pair())

    def test_quality_filter_rejects_bad(
        self,
    ) -> None:
        """Low scores rejected (avg 0.25 < 0.6)."""
        qf = QualityFilter(
            llm_fn=lambda p: (
                '{"self_containment": 0.3, '
                '"answerability": 0.2}'
            ),
            threshold=0.6,
        )
        assert not qf.passes(self._pair())

    def test_quality_filter_template_mode(
        self,
    ) -> None:
        """No llm_fn means always passes."""
        qf = QualityFilter(llm_fn=None)
        assert qf.passes(self._pair())

    def test_quality_filter_threshold(
        self,
    ) -> None:
        """Boundary at 0.6 threshold."""
        # avg = (0.6 + 0.6) / 2 = 0.6, passes
        at_boundary = (
            '{"self_containment": 0.6, '
            '"answerability": 0.6}'
        )
        qf = QualityFilter(
            llm_fn=lambda p: at_boundary,
            threshold=0.6,
        )
        assert qf.passes(self._pair())

        # avg = (0.5 + 0.6) / 2 = 0.55, fails
        below = (
            '{"self_containment": 0.5, '
            '"answerability": 0.6}'
        )
        qf2 = QualityFilter(
            llm_fn=lambda p: below,
            threshold=0.6,
        )
        assert not qf2.passes(self._pair())

    def test_quality_filter_retries(self) -> None:
        """Retry produces good pair eventually."""
        gen_calls = 0

        def retry_llm(prompt: str) -> str:
            nonlocal gen_calls
            # Quality scoring prompts
            if "quality evaluator" in prompt:
                return (
                    '{"self_containment": 0.9, '
                    '"answerability": 0.9}'
                )
            gen_calls += 1
            if gen_calls == 1:
                return "garbage"
            return (
                '{"question": "Q?", '
                '"answer": "A."}'
            )

        gen = SyntheticQAGenerator(
            llm_fn=retry_llm,
            seed=SEED,
            quality_filter=True,
            max_retries=2,
        )
        pair = gen.generate_one(SAMPLE_CHUNK)
        assert pair is not None

    def test_quality_filter_max_retries(
        self,
    ) -> None:
        """Gives up after max retries."""

        def always_bad(prompt: str) -> str:
            return "not json at all"

        gen = SyntheticQAGenerator(
            llm_fn=always_bad,
            seed=SEED,
            quality_filter=False,
            max_retries=1,
        )
        pair = gen.generate_one(SAMPLE_CHUNK)
        assert pair is None

    def test_quality_filter_disabled(self) -> None:
        """quality_filter=False skips scoring."""
        call_count = 0

        def counting_llm(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return (
                '{"question": "Q?", '
                '"answer": "A."}'
            )

        gen = SyntheticQAGenerator(
            llm_fn=counting_llm,
            seed=SEED,
            quality_filter=False,
        )
        pair = gen.generate_one(SAMPLE_CHUNK)
        assert pair is not None
        # Only 1 call (generation), no scoring call
        assert call_count == 1


# ===========================================================
# 7. Integration (5 tests)
# ===========================================================


class TestIntegration:
    """Integration tests across modules."""

    def test_generate_then_assert(self) -> None:
        """QAPair feeds into mock assertion."""
        gen = SyntheticQAGenerator(seed=SEED)
        pair = gen.generate_one(SAMPLE_CHUNK)
        assert pair is not None

        # Simulate assert_faithfulness check
        def mock_faithfulness(
            answer: str,
            context: str | list[str],
        ) -> bool:
            ctx = (
                context
                if isinstance(context, str)
                else " ".join(context)
            )
            return len(answer) > 0 and len(ctx) > 0

        assert mock_faithfulness(
            pair.answer, pair.context,
        )

    def test_to_dict_to_dataframe(self) -> None:
        """QAPair.to_dict works with pandas."""
        import pandas as pd

        pairs = [
            QAPair(
                question="Q1?",
                answer="A1.",
                context="c1",
            ),
            QAPair(
                question="Q2?",
                answer="A2.",
                context="c2",
                question_type=(
                    QuestionType.REASONING
                ),
            ),
        ]
        dicts = [p.to_dict() for p in pairs]
        df = pd.DataFrame(dicts)
        assert len(df) == 2
        assert "question" in df.columns
        assert "answer" in df.columns
        assert "context" in df.columns
        assert "question_type" in df.columns
        assert df.iloc[0]["question"] == "Q1?"

    def test_generate_multiple_types(
        self,
    ) -> None:
        """Mixed types in one call."""
        gen = SyntheticQAGenerator(
            llm_fn=_mock_llm,
            seed=SEED,
            quality_filter=False,
            question_types=[
                QuestionType.FACTUAL,
                QuestionType.REASONING,
                QuestionType.COUNTERFACTUAL,
            ],
        )
        pairs = gen.generate_from_chunks(
            [SAMPLE_CHUNK], n=6,
        )
        assert len(pairs) == 6
        types = {p.question_type for p in pairs}
        assert len(types) >= 2

    def test_seed_reproducibility(self) -> None:
        """Same seed, same results across runs."""
        gen1 = SyntheticQAGenerator(seed=99)
        gen2 = SyntheticQAGenerator(seed=99)
        p1 = gen1.generate_from_chunks(
            [SAMPLE_CHUNK], n=5,
        )
        p2 = gen2.generate_from_chunks(
            [SAMPLE_CHUNK], n=5,
        )
        assert len(p1) == len(p2)
        for a, b in zip(p1, p2, strict=True):
            assert a.question == b.question
            assert a.answer == b.answer
            assert a.context == b.context

    def test_large_document(self) -> None:
        """5000 word document, n=20."""
        words = [f"w{i}" for i in range(5000)]
        doc = " ".join(words)
        gen = SyntheticQAGenerator(
            seed=SEED,
            chunk_size=200,
            min_chunk_words=30,
        )
        pairs = gen.generate_from_text(doc, n=20)
        assert isinstance(pairs, list)
        assert len(pairs) >= 1
        for p in pairs:
            assert isinstance(p, QAPair)
            assert p.question
            assert p.answer
