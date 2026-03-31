"""Synthetic QA pair generation for ML test data.

Provides ``SyntheticQAGenerator`` -- the core class for
generating question-answer pairs from documents or text chunks.
Pairs feed directly into mltk's RAG assertions
(``assert_faithfulness``, ``assert_answer_relevancy``, etc.).

Two modes:

- **Template mode** (default): deterministic, zero-dependency,
  CI-safe.  Extracts key sentences and converts them to
  interrogative form.  Good for smoke tests.
- **LLM mode** (provide ``llm_fn``): delegates generation to
  any ``str -> str`` callable.  Higher quality and diversity.
  Works with OpenAI, Anthropic, Ollama, or any other backend.

Research context:  RAGAS ``TestsetGenerator`` requires a
knowledge graph (2+ LLM calls per chunk).  DeepEval
``Synthesizer`` couples to a ``DeepEvalBaseLLM`` subclass.
mltk uses a simple ``Callable[[str], str]`` -- the same
pattern as ``ParaphraseGenerator`` -- for zero-framework
coupling and easy mocking in tests.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from mltk.domains.llm.synthetic._quality import (
    QualityFilter,
)
from mltk.domains.llm.synthetic._splitter import split_text
from mltk.domains.llm.synthetic._templates import (
    build_prompt,
    declarative_to_interrogative,
    extract_key_sentences,
    parse_conversational_response,
    parse_response,
)

# ---------------------------------------------------------------
# QuestionType enum
# ---------------------------------------------------------------


class QuestionType(str, Enum):
    """Categories of synthetic questions.

    Each type targets a different evaluation dimension:

    - **FACTUAL**: Answer is explicitly stated in the context.
      Tests basic retrieval accuracy.
    - **REASONING**: Answer requires a logical inference step.
      Tests comprehension beyond literal matching.
    - **MULTI_HOP**: Answer requires combining information
      from multiple context chunks.  Tests cross-document
      reasoning.
    - **COUNTERFACTUAL**: Hypothetical "what if" questions.
      Tests whether models handle altered premises correctly.
    - **OUT_OF_SCOPE**: Question is related to the topic but
      unanswerable from the context.  Tests refusal and
      "I don't know" behavior.
    - **CONVERSATIONAL**: Multi-turn dialogue where follow-up
      questions build on previous answers.
    - **DISTRACTING**: Questions with misleading elements
      injected from a different context chunk.

    Example::

        from mltk.domains.llm.synthetic import QuestionType

        # Use as filter:
        gen.generate_from_text(
            text, question_types=[QuestionType.FACTUAL],
        )
    """

    FACTUAL = "factual"
    REASONING = "reasoning"
    MULTI_HOP = "multi_hop"
    COUNTERFACTUAL = "counterfactual"
    OUT_OF_SCOPE = "out_of_scope"
    CONVERSATIONAL = "conversational"
    DISTRACTING = "distracting"


# ---------------------------------------------------------------
# QAPair dataclass
# ---------------------------------------------------------------


@dataclass
class QAPair:
    """A single synthetic question-answer pair.

    Designed for direct integration with mltk assertions::

        pair = generator.generate_one(chunk)
        assert_faithfulness(pair.answer, pair.context)
        assert_answer_relevancy(
            pair.question, pair.answer, pair.context,
        )

    Attributes:
        question: The generated question string.
        answer: The reference (ground truth) answer.
        context: The source chunk(s) used to generate
            the pair.  A string for single-chunk, or a
            list of strings for multi-hop.
        question_type: The question category.
        metadata: Optional dict for source document info,
            chunk index, quality scores, etc.
    """

    question: str
    answer: str
    context: str | list[str]
    question_type: QuestionType = QuestionType.FACTUAL
    metadata: dict[str, Any] = field(
        default_factory=dict,
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON or pandas.

        Returns:
            Dict with ``question``, ``answer``, ``context``,
            ``question_type``, and any metadata keys merged
            at the top level.

        Example::

            pair.to_dict()
            # {"question": "...", "answer": "...",
            #  "context": "...", "question_type": "factual"}
        """
        ctx = (
            self.context
            if isinstance(self.context, str)
            else " ".join(self.context)
        )
        return {
            "question": self.question,
            "answer": self.answer,
            "context": ctx,
            "question_type": self.question_type.value,
            **self.metadata,
        }


# ---------------------------------------------------------------
# SyntheticQAGenerator
# ---------------------------------------------------------------


class SyntheticQAGenerator:
    """Generate synthetic QA pairs from documents or text chunks.

    Works in two modes:

    **Template mode** (default, zero-dependency):
    Generates deterministic questions from structural patterns.
    Suitable for smoke tests and CI -- no API calls, no costs.

    **LLM mode** (provide ``llm_fn``):
    Delegates question generation to a user-supplied LLM
    callable.  Higher quality and diversity; any LLM backend
    works.

    Args:
        llm_fn: ``str -> str`` callable that accepts a prompt
            and returns the LLM response.  If ``None``,
            template mode is used.
        question_types: List of ``QuestionType`` to generate.
            Defaults to all five types.
        chunk_size: Word count per chunk when splitting raw
            text.  Default 512.
        chunk_overlap: Overlapping words between consecutive
            chunks.  Default 50.
        min_chunk_words: Minimum words a chunk must have to
            be considered.  Default 30.
        quality_filter: If ``True`` and ``llm_fn`` is
            provided, score each generated QA pair and
            discard below-threshold pairs.
        quality_threshold: Minimum score (0-1) for both
            self-containment and answerability.  Default 0.6.
        max_retries: Max LLM retries on quality failure.
            Default 1.
        seed: Random seed for reproducible chunk selection.

    Example -- template mode::

        from mltk.domains.llm.synthetic import (
            SyntheticQAGenerator,
        )

        gen = SyntheticQAGenerator()
        pairs = gen.generate_from_text(
            my_document_text, n=10,
        )

    Example -- LLM mode::

        import openai

        def my_llm(prompt: str) -> str:
            return openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt},
                ],
            ).choices[0].message.content

        gen = SyntheticQAGenerator(llm_fn=my_llm)
        pairs = gen.generate_from_text(document, n=20)

    Example -- integration with mltk assertions::

        gen = SyntheticQAGenerator(llm_fn=my_llm)
        pairs = gen.generate_from_chunks(my_chunks, n=15)

        for pair in pairs:
            assert_faithfulness(
                pair.answer, pair.context, min_score=0.7,
            )
    """

    def __init__(
        self,
        llm_fn: Callable[[str], str] | None = None,
        question_types: (
            list[QuestionType] | None
        ) = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_words: int = 30,
        quality_filter: bool = True,
        quality_threshold: float = 0.6,
        max_retries: int = 1,
        seed: int | None = None,
    ) -> None:
        self._llm_fn = llm_fn
        self._question_types = (
            question_types
            if question_types is not None
            else list(QuestionType)
        )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._min_chunk_words = min_chunk_words
        self._quality_filter_enabled = quality_filter
        self._quality_threshold = quality_threshold
        self._max_retries = max_retries
        self._rng = random.Random(seed)

        # Quality filter only active in LLM mode
        filter_llm = (
            llm_fn if quality_filter else None
        )
        self._quality = QualityFilter(
            llm_fn=filter_llm,
            threshold=quality_threshold,
            max_retries=max_retries,
        )

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def generate_from_text(
        self,
        text: str,
        n: int = 10,
        question_types: (
            list[QuestionType] | None
        ) = None,
    ) -> list[QAPair]:
        """Split text into chunks and generate *n* QA pairs.

        The text is split using a word-count chunker that
        respects paragraph boundaries.  Chunks are then
        distributed across the requested question types.

        Args:
            text: Raw document text to generate from.
            n: Number of QA pairs to generate.  The actual
                count may be lower if the text is too short
                or quality filtering removes pairs.
            question_types: Override the instance-level
                question types for this call.

        Returns:
            List of ``QAPair`` instances.

        Example::

            pairs = gen.generate_from_text(
                open("doc.txt").read(), n=20,
            )
        """
        chunks = split_text(
            text,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            min_chunk_words=self._min_chunk_words,
        )
        if not chunks:
            return []

        return self.generate_from_chunks(
            chunks, n=n, question_types=question_types,
        )

    def generate_from_chunks(
        self,
        chunks: list[str],
        n: int = 10,
        question_types: (
            list[QuestionType] | None
        ) = None,
    ) -> list[QAPair]:
        """Generate QA pairs from pre-split chunks.

        Distributes generation across chunks and question
        types as evenly as possible.  For multi-hop questions,
        adjacent chunk pairs are used automatically.

        Args:
            chunks: List of text chunks to generate from.
            n: Target number of QA pairs.
            question_types: Override question types for this
                call.

        Returns:
            List of ``QAPair`` instances.

        Example::

            chunks = ["First chunk...", "Second chunk..."]
            pairs = gen.generate_from_chunks(chunks, n=5)
        """
        if not chunks:
            return []

        types = question_types or self._question_types
        pairs: list[QAPair] = []
        assignments = self._distribute(
            chunks, types, n,
        )

        for chunk_idx, qtype in assignments:
            if len(pairs) >= n:
                break

            context_chunks = None
            if (
                qtype == QuestionType.MULTI_HOP
                and len(chunks) > 1
            ):
                other_idx = (
                    chunk_idx + 1
                    if chunk_idx + 1 < len(chunks)
                    else chunk_idx - 1
                )
                context_chunks = [chunks[other_idx]]

            pair = self.generate_one(
                chunks[chunk_idx],
                question_type=qtype,
                context_chunks=context_chunks,
            )
            if pair is not None:
                pair.metadata["chunk_index"] = chunk_idx
                pairs.append(pair)

        return pairs

    def generate_one(
        self,
        chunk: str,
        question_type: QuestionType = (
            QuestionType.FACTUAL
        ),
        context_chunks: list[str] | None = None,
    ) -> QAPair | None:
        """Generate a single QA pair from one chunk.

        Dispatches to template mode or LLM mode based on
        whether ``llm_fn`` was provided at init time.

        Args:
            chunk: The primary context chunk.
            question_type: Category of question to generate.
            context_chunks: Additional chunks for multi-hop
                questions.  Ignored for other types.

        Returns:
            A ``QAPair``, or ``None`` if generation fails
            (e.g., chunk too short, quality filter rejects).

        Example::

            pair = gen.generate_one(
                "Python is an interpreted language.",
                question_type=QuestionType.FACTUAL,
            )
            if pair:
                print(pair.question, pair.answer)
        """
        if not chunk or not chunk.strip():
            return None

        if self._llm_fn is not None:
            return self._generate_llm(
                chunk, question_type, context_chunks,
            )
        return self._generate_template(
            chunk, question_type, context_chunks,
        )

    def to_dataframe(self, pairs: list[QAPair]) -> Any:
        """Convert QAPairs to a pandas DataFrame.

        Requires ``pandas`` to be installed.  Each QAPair
        is serialized via ``to_dict()`` and stacked into a
        DataFrame.

        Args:
            pairs: List of QAPair instances.

        Returns:
            A ``pandas.DataFrame``.

        Raises:
            ImportError: If pandas is not installed.
        """
        import pandas as pd  # noqa: PLC0415
        return pd.DataFrame(
            [p.to_dict() for p in pairs],
        )

    def to_jsonl(
        self,
        pairs: list[QAPair],
        path: str,
    ) -> None:
        """Write QAPairs to a JSONL file.

        Each line is a JSON object produced by
        ``QAPair.to_dict()``.

        Args:
            pairs: List of QAPair instances to write.
            path: Filesystem path for the output file.

        Example::

            gen.to_jsonl(pairs, "output/qa_pairs.jsonl")
        """
        with open(path, "w", encoding="utf-8") as f:
            for pair in pairs:
                line = json.dumps(
                    pair.to_dict(), ensure_ascii=False,
                )
                f.write(line + "\n")

    # ---------------------------------------------------------
    # v2: Multi-hop, Conversational, Distracting
    # ---------------------------------------------------------

    def generate_multi_hop(
        self,
        chunks: list[str],
        n: int = 5,
    ) -> list[QAPair]:
        """Generate multi-hop questions from chunk pairs.

        Selects pairs (or triples) of chunks and generates
        questions that require cross-chunk reasoning.

        Args:
            chunks: List of text chunks.  Must contain at
                least 2 chunks.
            n: Target number of QA pairs to generate.

        Returns:
            List of ``QAPair`` with
            ``question_type=MULTI_HOP`` and context set
            to the list of source chunks.
        """
        if len(chunks) < 2:
            return []

        pairs: list[QAPair] = []
        indices = list(range(len(chunks)))

        for i in range(n):
            if len(pairs) >= n:
                break

            idx_a = indices[i % len(indices)]
            idx_b = indices[
                (i + 1) % len(indices)
            ]
            if idx_a == idx_b:
                idx_b = indices[
                    (i + 2) % len(indices)
                ]

            selected = [chunks[idx_a], chunks[idx_b]]

            pair = self._generate_multi_hop_pair(
                selected,
            )
            if pair is not None:
                pair.metadata["chunk_indices"] = [
                    idx_a, idx_b,
                ]
                pairs.append(pair)

        return pairs

    def _generate_multi_hop_pair(
        self,
        selected_chunks: list[str],
    ) -> QAPair | None:
        """Generate one multi-hop pair from chunks."""
        if self._llm_fn is not None:
            return self._llm_multi_hop_enhanced(
                selected_chunks,
            )
        return self._template_multi_hop_enhanced(
            selected_chunks,
        )

    def _template_multi_hop_enhanced(
        self,
        selected_chunks: list[str],
    ) -> QAPair | None:
        """Template multi-hop from N chunks."""
        sentences: list[str] = []
        for chunk in selected_chunks:
            sents = extract_key_sentences(chunk, 1)
            if sents:
                sentences.append(sents[0])

        if len(sentences) < 2:
            return None

        question = (
            "Considering that "
            f"{sentences[0].strip().rstrip('.')} "
            f"and {sentences[1].strip().rstrip('.')}"
            ", what can be concluded?"
        )
        answer = " ".join(
            s.strip() for s in sentences
        )

        return QAPair(
            question=question,
            answer=answer,
            context=list(selected_chunks),
            question_type=QuestionType.MULTI_HOP,
            metadata={"mode": "template"},
        )

    def _llm_multi_hop_enhanced(
        self,
        selected_chunks: list[str],
    ) -> QAPair | None:
        """LLM multi-hop from N chunks."""
        assert self._llm_fn is not None

        prompt = build_prompt(
            "multi_hop_enhanced",
            context=selected_chunks[0],
            contexts=selected_chunks,
        )

        for attempt in range(1 + self._max_retries):
            try:
                raw = self._llm_fn(prompt)
                parsed = parse_response(raw)
                if parsed is None:
                    continue

                pair = QAPair(
                    question=parsed["question"],
                    answer=parsed["answer"],
                    context=list(selected_chunks),
                    question_type=(
                        QuestionType.MULTI_HOP
                    ),
                    metadata={
                        "mode": "llm",
                        "attempt": attempt + 1,
                    },
                )

                if self._quality_filter_enabled:
                    score = self._quality.score(pair)
                    pair.metadata[
                        "quality_score"
                    ] = score
                    if score < self._quality_threshold:
                        continue

                return pair

            except Exception:  # noqa: BLE001
                continue

        return None

    def generate_conversational(
        self,
        chunks: list[str],
        n: int = 5,
        turns: int = 2,
    ) -> list[list[QAPair]]:
        """Generate multi-turn conversations from chunks.

        Each item in the returned list is a conversation
        (a list of ``QAPair`` objects forming a dialogue).
        Follow-up questions build on previous answers.

        Args:
            chunks: List of text chunks to generate from.
            n: Number of conversations to generate.
            turns: Number of question-answer turns per
                conversation.  Default 2.

        Returns:
            List of conversations.  Each conversation is
            a list of ``QAPair`` with
            ``question_type=CONVERSATIONAL``.
        """
        if not chunks:
            return []

        conversations: list[list[QAPair]] = []
        indices = list(range(len(chunks)))

        for i in range(n):
            if len(conversations) >= n:
                break

            idx = indices[i % len(indices)]
            conv = self._generate_conversation(
                chunks[idx], turns=turns,
            )
            if conv:
                for pair in conv:
                    pair.metadata["chunk_index"] = idx
                conversations.append(conv)

        return conversations

    def _generate_conversation(
        self,
        chunk: str,
        turns: int = 2,
    ) -> list[QAPair] | None:
        """Generate one multi-turn conversation."""
        if self._llm_fn is not None:
            return self._llm_conversation(
                chunk, turns,
            )
        return self._template_conversation(
            chunk, turns,
        )

    def _template_conversation(
        self,
        chunk: str,
        turns: int = 2,
    ) -> list[QAPair] | None:
        """Template conversation from a chunk."""
        sentences = extract_key_sentences(
            chunk, turns * 2,
        )
        if len(sentences) < turns:
            return None

        conv: list[QAPair] = []
        for i in range(turns):
            sent = sentences[i]
            if i == 0:
                question = declarative_to_interrogative(
                    sent,
                )
                if question is None:
                    question = (
                        "What is described here: "
                        f"{sent}?"
                    )
            else:
                prev_answer = conv[i - 1].answer
                question = (
                    f"Following up on "
                    f'"{prev_answer}", '
                    f"what else can be said?"
                )

            conv.append(QAPair(
                question=question,
                answer=sent.strip().rstrip("."),
                context=chunk,
                question_type=(
                    QuestionType.CONVERSATIONAL
                ),
                metadata={
                    "mode": "template",
                    "turn": i + 1,
                },
            ))

        return conv

    def _llm_conversation(
        self,
        chunk: str,
        turns: int = 2,
    ) -> list[QAPair] | None:
        """LLM conversation from a chunk."""
        assert self._llm_fn is not None

        prompt = build_prompt(
            "conversational",
            context=chunk,
            turns=turns,
        )

        for attempt in range(1 + self._max_retries):
            try:
                raw = self._llm_fn(prompt)
                parsed = parse_conversational_response(
                    raw,
                )
                if parsed is None:
                    continue
                if len(parsed) < turns:
                    continue

                conv: list[QAPair] = []
                for i, turn_data in enumerate(
                    parsed[:turns],
                ):
                    conv.append(QAPair(
                        question=turn_data["question"],
                        answer=turn_data["answer"],
                        context=chunk,
                        question_type=(
                            QuestionType.CONVERSATIONAL
                        ),
                        metadata={
                            "mode": "llm",
                            "turn": i + 1,
                            "attempt": attempt + 1,
                        },
                    ))
                return conv

            except Exception:  # noqa: BLE001
                continue

        return None

    def generate_distracting(
        self,
        chunks: list[str],
        n: int = 5,
    ) -> list[QAPair]:
        """Generate questions with misleading elements.

        Pairs each target chunk with a random distractor
        from a different chunk.  The generated question
        includes a misleading detail from the distractor.

        Args:
            chunks: List of text chunks.  Must contain at
                least 2 chunks.
            n: Target number of QA pairs to generate.

        Returns:
            List of ``QAPair`` with
            ``question_type=DISTRACTING``.  Each pair's
            metadata includes ``distractor_chunk``.
        """
        if len(chunks) < 2:
            return []

        pairs: list[QAPair] = []
        indices = list(range(len(chunks)))

        for i in range(n):
            if len(pairs) >= n:
                break

            target_idx = indices[i % len(indices)]
            distractor_idx = indices[
                (i + 1) % len(indices)
            ]
            if target_idx == distractor_idx:
                distractor_idx = indices[
                    (i + 2) % len(indices)
                ]

            pair = self._generate_distracting_pair(
                chunks[target_idx],
                chunks[distractor_idx],
            )
            if pair is not None:
                pair.metadata[
                    "chunk_index"
                ] = target_idx
                pair.metadata[
                    "distractor_chunk"
                ] = chunks[distractor_idx]
                pairs.append(pair)

        return pairs

    def _generate_distracting_pair(
        self,
        target: str,
        distractor: str,
    ) -> QAPair | None:
        """Generate one distracting pair."""
        if self._llm_fn is not None:
            return self._llm_distracting(
                target, distractor,
            )
        return self._template_distracting(
            target, distractor,
        )

    def _template_distracting(
        self,
        target: str,
        distractor: str,
    ) -> QAPair | None:
        """Template distracting question."""
        target_sents = extract_key_sentences(
            target, 1,
        )
        distractor_sents = extract_key_sentences(
            distractor, 1,
        )
        if not target_sents or not distractor_sents:
            return None

        t_sent = target_sents[0].strip().rstrip(".")
        d_sent = distractor_sents[0].strip().rstrip(".")

        question = (
            f"While {d_sent.lower()}, {t_sent.lower()}?"
        )
        answer = t_sent

        return QAPair(
            question=question,
            answer=answer,
            context=target,
            question_type=QuestionType.DISTRACTING,
            metadata={"mode": "template"},
        )

    def _llm_distracting(
        self,
        target: str,
        distractor: str,
    ) -> QAPair | None:
        """LLM distracting question."""
        assert self._llm_fn is not None

        prompt = build_prompt(
            "distracting",
            context=target,
            distractor=distractor,
        )

        for attempt in range(1 + self._max_retries):
            try:
                raw = self._llm_fn(prompt)
                parsed = parse_response(raw)
                if parsed is None:
                    continue

                pair = QAPair(
                    question=parsed["question"],
                    answer=parsed["answer"],
                    context=target,
                    question_type=(
                        QuestionType.DISTRACTING
                    ),
                    metadata={
                        "mode": "llm",
                        "attempt": attempt + 1,
                    },
                )

                if self._quality_filter_enabled:
                    score = self._quality.score(pair)
                    pair.metadata[
                        "quality_score"
                    ] = score
                    if score < self._quality_threshold:
                        continue

                return pair

            except Exception:  # noqa: BLE001
                continue

        return None

    # ---------------------------------------------------------
    # Template mode (deterministic, zero-dep)
    # ---------------------------------------------------------

    def _generate_template(
        self,
        chunk: str,
        question_type: QuestionType,
        context_chunks: list[str] | None = None,
    ) -> QAPair | None:
        """Generate a QA pair using template heuristics.

        Extracts key sentences, converts to interrogative
        form, and uses the original sentence as the answer.
        Deterministic and CI-safe.
        """
        if question_type == QuestionType.MULTI_HOP:
            return self._template_multi_hop(
                chunk, context_chunks,
            )
        if question_type == QuestionType.OUT_OF_SCOPE:
            return self._template_out_of_scope(chunk)
        if question_type == QuestionType.COUNTERFACTUAL:
            return self._template_counterfactual(chunk)

        # FACTUAL and REASONING share the same template
        # logic (REASONING gets a metadata tag)
        sentences = extract_key_sentences(chunk, 3)
        if not sentences:
            return None

        for sent in sentences:
            question = declarative_to_interrogative(sent)
            if question is not None:
                return QAPair(
                    question=question,
                    answer=sent.strip().rstrip("."),
                    context=chunk,
                    question_type=question_type,
                    metadata={
                        "mode": "template",
                        "source_sentence": sent,
                    },
                )

        # Fallback: wrap the first sentence as a question
        fallback = (
            "What is described here: "
            f"{sentences[0]}?"
        )
        return QAPair(
            question=fallback,
            answer=sentences[0],
            context=chunk,
            question_type=question_type,
            metadata={
                "mode": "template",
                "fallback": True,
            },
        )

    def _template_multi_hop(
        self,
        chunk: str,
        context_chunks: list[str] | None,
    ) -> QAPair | None:
        """Template multi-hop: combine two chunk sentences."""
        sent_a = extract_key_sentences(chunk, 1)
        if not sent_a:
            return None

        if context_chunks:
            sent_b = extract_key_sentences(
                context_chunks[0], 1,
            )
            context: str | list[str] = [
                chunk, context_chunks[0],
            ]
        else:
            all_sents = extract_key_sentences(chunk, 2)
            if len(all_sents) < 2:
                return None
            sent_b = [all_sents[1]]
            context = chunk

        if not sent_b:
            return None

        question = (
            "Considering that "
            f"{sent_a[0].strip().rstrip('.')} "
            f"and {sent_b[0].strip().rstrip('.')}, "
            "what can be concluded?"
        )
        answer = (
            f"{sent_a[0].strip()} "
            f"{sent_b[0].strip()}"
        )

        return QAPair(
            question=question,
            answer=answer,
            context=context,
            question_type=QuestionType.MULTI_HOP,
            metadata={"mode": "template"},
        )

    def _template_out_of_scope(
        self,
        chunk: str,
    ) -> QAPair | None:
        """Template out-of-scope: ask about related topic."""
        sentences = extract_key_sentences(chunk, 1)
        if not sentences:
            return None

        words = sentences[0].split()
        topic_words = [
            w for w in words
            if len(w) > 3 and w[0].isupper()
        ]
        topic = (
            topic_words[0] if topic_words else words[0]
        )

        question = (
            "What are the long-term implications "
            f"of {topic} that are not discussed here?"
        )
        answer = (
            "This information is not available "
            "in the provided context."
        )

        return QAPair(
            question=question,
            answer=answer,
            context=chunk,
            question_type=QuestionType.OUT_OF_SCOPE,
            metadata={"mode": "template"},
        )

    def _template_counterfactual(
        self,
        chunk: str,
    ) -> QAPair | None:
        """Template counterfactual: invert a key statement."""
        sentences = extract_key_sentences(chunk, 1)
        if not sentences:
            return None

        sent = sentences[0].strip().rstrip(".")
        question = (
            f"What if {sent.lower()} were not true?"
        )
        answer = (
            "If that were not the case, the information "
            "in the context would be contradicted."
        )

        return QAPair(
            question=question,
            answer=answer,
            context=chunk,
            question_type=QuestionType.COUNTERFACTUAL,
            metadata={"mode": "template"},
        )

    # ---------------------------------------------------------
    # LLM mode
    # ---------------------------------------------------------

    def _generate_llm(
        self,
        chunk: str,
        question_type: QuestionType,
        context_chunks: list[str] | None = None,
    ) -> QAPair | None:
        """Generate a QA pair using the LLM callable.

        Builds a type-specific prompt, calls the LLM, parses
        the response, and optionally quality-filters.
        """
        assert self._llm_fn is not None

        context_b = (
            context_chunks[0] if context_chunks else None
        )
        prompt = build_prompt(
            question_type.value, chunk, context_b,
        )

        for attempt in range(1 + self._max_retries):
            try:
                raw = self._llm_fn(prompt)
                parsed = parse_response(raw)
                if parsed is None:
                    continue

                ctx: str | list[str] = chunk
                if (
                    question_type
                    == QuestionType.MULTI_HOP
                    and context_chunks
                ):
                    ctx = [chunk] + list(context_chunks)

                pair = QAPair(
                    question=parsed["question"],
                    answer=parsed["answer"],
                    context=ctx,
                    question_type=question_type,
                    metadata={
                        "mode": "llm",
                        "attempt": attempt + 1,
                    },
                )

                # Quality filter (LLM mode only)
                if self._quality_filter_enabled:
                    score = self._quality.score(pair)
                    pair.metadata[
                        "quality_score"
                    ] = score
                    if score < self._quality_threshold:
                        continue

                return pair

            except Exception:  # noqa: BLE001
                continue

        return None

    # ---------------------------------------------------------
    # Distribution helper
    # ---------------------------------------------------------

    def _distribute(
        self,
        chunks: list[str],
        types: list[QuestionType],
        n: int,
    ) -> list[tuple[int, QuestionType]]:
        """Distribute *n* tasks across chunks and types.

        Produces (chunk_index, question_type) tuples,
        cycling through types and shuffled chunk indices
        for even coverage.

        Args:
            chunks: Available text chunks.
            types: Question types to cycle through.
            n: Target number of assignments.

        Returns:
            List of (chunk_index, QuestionType) tuples.
        """
        assignments: list[tuple[int, QuestionType]] = []
        chunk_indices = list(range(len(chunks)))
        self._rng.shuffle(chunk_indices)

        type_cycle = types * ((n // len(types)) + 1)

        for i in range(n):
            chunk_idx = chunk_indices[
                i % len(chunk_indices)
            ]
            qtype = type_cycle[i % len(type_cycle)]
            assignments.append((chunk_idx, qtype))

        return assignments
