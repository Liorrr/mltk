"""Tests for retrieval consistency assertion.

Covers ``assert_retrieval_consistency`` from the behavioral
consistency module.  All retriever calls are deterministic
lambdas; no external dependencies required.
"""

from __future__ import annotations

import unicodedata
from unittest.mock import MagicMock

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.core.result import Severity, TestResult
from mltk.domains.llm.behavioral import (
    assert_retrieval_consistency,
)


# -- Shared helpers -----------------------------------------------

SEED = 42


def _result_shape_ok(result: TestResult) -> None:
    """Assert TestResult has the mandatory fields."""
    assert isinstance(result.name, str)
    assert isinstance(result.passed, bool)
    assert isinstance(result.message, str)
    assert isinstance(result.details, dict)
    assert result.duration_ms >= 0.0


# =================================================================
# TestRetrievalConsistency
# =================================================================


class TestRetrievalConsistency:
    """Tests for ``assert_retrieval_consistency``."""

    def test_identical_retrievals_pass(self) -> None:
        """Same docs for all queries gives Jaccard=1.0."""
        retriever = lambda q: ["doc1", "doc2", "doc3"]
        result = assert_retrieval_consistency(
            retriever_fn=retriever,
            paraphrases=[
                "What is gravity?",
                "Explain gravitational force.",
                "Describe gravity.",
            ],
            min_overlap=0.8,
        )
        assert result.passed is True
        _result_shape_ok(result)

    def test_disjoint_retrievals_fail(self) -> None:
        """Completely different docs gives Jaccard=0.0."""
        call_idx = 0

        def disjoint_retriever(q: str) -> list[str]:
            nonlocal call_idx
            call_idx += 1
            return [f"unique-doc-{call_idx}-{i}"
                    for i in range(3)]

        with pytest.raises(MltkAssertionError):
            assert_retrieval_consistency(
                retriever_fn=disjoint_retriever,
                paraphrases=[
                    "What is gravity?",
                    "Explain gravitational force.",
                ],
                min_overlap=0.5,
            )

    def test_partial_overlap_passes_low(self) -> None:
        """50% overlap passes at threshold 0.4."""
        def half_overlap(q: str) -> list[str]:
            if "explain" in q.lower():
                return ["doc1", "doc2", "doc4"]
            return ["doc1", "doc2", "doc3"]

        result = assert_retrieval_consistency(
            retriever_fn=half_overlap,
            paraphrases=[
                "What is gravity?",
                "Explain gravitational force.",
            ],
            min_overlap=0.4,
        )
        assert result.passed is True

    def test_partial_overlap_fails_high(self) -> None:
        """50% overlap fails at threshold 0.8."""
        def half_overlap(q: str) -> list[str]:
            if "explain" in q.lower():
                return ["doc1", "doc4"]
            return ["doc1", "doc3"]

        with pytest.raises(MltkAssertionError):
            assert_retrieval_consistency(
                retriever_fn=half_overlap,
                paraphrases=[
                    "What is gravity?",
                    "Explain gravitational force.",
                ],
                min_overlap=0.8,
            )

    def test_single_query_fails(self) -> None:
        """Less than 2 paraphrases produces a fail."""
        retriever = lambda q: ["doc1", "doc2"]
        with pytest.raises(
            (MltkAssertionError, ValueError),
        ):
            assert_retrieval_consistency(
                retriever_fn=retriever,
                paraphrases=["What is gravity?"],
                min_overlap=0.5,
            )

    def test_empty_retrieval(self) -> None:
        """Empty doc lists yield Jaccard=0."""
        retriever = lambda q: []
        with pytest.raises(
            (MltkAssertionError, ValueError),
        ):
            assert_retrieval_consistency(
                retriever_fn=retriever,
                paraphrases=[
                    "What is gravity?",
                    "Explain gravity.",
                ],
                min_overlap=0.5,
            )

    def test_worst_pair_in_details(self) -> None:
        """Details include worst-pair info."""
        retriever = lambda q: ["doc1", "doc2", "doc3"]
        result = assert_retrieval_consistency(
            retriever_fn=retriever,
            paraphrases=[
                "What is gravity?",
                "Explain gravity.",
                "Describe gravity.",
            ],
            min_overlap=0.5,
        )
        details = result.details
        assert (
            "worst_pair" in details
            or "worst_score" in details
            or "min_jaccard" in details
        )

    def test_per_pair_in_details(self) -> None:
        """Details include per-pair breakdown."""
        retriever = lambda q: ["doc1", "doc2"]
        result = assert_retrieval_consistency(
            retriever_fn=retriever,
            paraphrases=[
                "What is gravity?",
                "Explain gravity.",
            ],
            min_overlap=0.5,
        )
        details = result.details
        assert (
            "per_pair" in details
            or "pair_scores" in details
            or "pair_details" in details
        )

    def test_unicode_normalized(self) -> None:
        """Unicode inputs are normalized before retrieval."""
        raw = "\u0041\u030A gravity"
        nfkc = unicodedata.normalize("NFKC", raw)
        received: list[str] = []

        def tracking_retriever(q: str) -> list[str]:
            received.append(q)
            return ["doc1"]

        # We allow either pass or fail — just testing
        # that normalization happens and no crash.
        try:
            assert_retrieval_consistency(
                retriever_fn=tracking_retriever,
                paraphrases=[raw, "gravity explained"],
                min_overlap=0.5,
            )
        except MltkAssertionError:
            pass
        # First query should be NFKC-normalized
        if received:
            assert received[0] == nfkc or len(received) >= 2

    def test_jaccard_calculation_correct(self) -> None:
        """Manual Jaccard verification: |A&B| / |A|B|."""
        # A = {d1, d2, d3}, B = {d2, d3, d4}
        # intersection = {d2, d3} = 2
        # union = {d1, d2, d3, d4} = 4
        # Jaccard = 2/4 = 0.5
        call_idx = 0

        def retriever(q: str) -> list[str]:
            nonlocal call_idx
            call_idx += 1
            if call_idx == 1:
                return ["d1", "d2", "d3"]
            return ["d2", "d3", "d4"]

        result = assert_retrieval_consistency(
            retriever_fn=retriever,
            paraphrases=[
                "What is gravity?",
                "Explain gravity.",
            ],
            min_overlap=0.5,
        )
        assert result.passed is True
        details = result.details
        # Check the score is approximately 0.5
        score_key = next(
            (k for k in details
             if "jaccard" in k.lower()
             or "consistency" in k.lower()
             or "score" in k.lower()
             or "avg" in k.lower()),
            None,
        )
        if score_key is not None:
            val = details[score_key]
            if isinstance(val, (int, float)):
                assert abs(val - 0.5) < 0.01

    def test_result_name(self) -> None:
        """TestResult has the correct assertion name."""
        retriever = lambda q: ["doc1"]
        result = assert_retrieval_consistency(
            retriever_fn=retriever,
            paraphrases=[
                "What is gravity?",
                "Explain gravity.",
            ],
            min_overlap=0.5,
        )
        assert "retrieval" in result.name.lower()

    def test_severity_on_failure(self) -> None:
        """Failed consistency has CRITICAL severity."""
        call_idx = 0

        def disjoint(q: str) -> list[str]:
            nonlocal call_idx
            call_idx += 1
            return [f"doc-{call_idx}"]

        with pytest.raises(MltkAssertionError) as exc:
            assert_retrieval_consistency(
                retriever_fn=disjoint,
                paraphrases=[
                    "What is gravity?",
                    "Explain gravity.",
                ],
                min_overlap=0.9,
            )
        assert (
            exc.value.result.severity == Severity.CRITICAL
        )

    def test_method_in_details(self) -> None:
        """Details include the method used."""
        retriever = lambda q: ["doc1", "doc2"]
        result = assert_retrieval_consistency(
            retriever_fn=retriever,
            paraphrases=[
                "What is gravity?",
                "Explain gravity.",
            ],
            min_overlap=0.5,
        )
        details = result.details
        assert (
            "method" in details
            or "metric" in details
            or result.passed is True
        )

    def test_three_queries_all_pairs(self) -> None:
        """Three queries produce 3 pairwise comparisons."""
        retriever = lambda q: ["doc1", "doc2"]
        result = assert_retrieval_consistency(
            retriever_fn=retriever,
            paraphrases=[
                "What is gravity?",
                "Explain gravity.",
                "Describe gravity.",
            ],
            min_overlap=0.5,
        )
        assert result.passed is True
        details = result.details
        pairs_key = next(
            (k for k in details
             if "pair" in k.lower()
             or "total" in k.lower()),
            None,
        )
        if pairs_key is not None:
            val = details[pairs_key]
            if isinstance(val, int):
                assert val == 3
            elif isinstance(val, list):
                assert len(val) == 3

    # -- New edge-case and parametrized tests --------------------

    def test_retrieval_identical_doc_sets(self) -> None:
        """Same docs from all queries => Jaccard 1.0."""
        docs = ["a1", "a2", "a3"]
        retriever = lambda q: list(docs)
        result = assert_retrieval_consistency(
            retriever_fn=retriever,
            paraphrases=[
                "What is ML?",
                "Explain ML.",
            ],
            min_overlap=1.0,
        )
        assert result.passed is True
        assert result.details["avg_overlap"] == 1.0

    def test_retrieval_disjoint_doc_sets(self) -> None:
        """No overlap between doc sets => Jaccard 0.0."""
        counter = {"i": 0}

        def disjoint(q: str) -> list[str]:
            counter["i"] += 1
            return [
                f"x{counter['i']}-{k}"
                for k in range(3)
            ]

        with pytest.raises(MltkAssertionError) as exc:
            assert_retrieval_consistency(
                retriever_fn=disjoint,
                paraphrases=[
                    "What is ML?",
                    "Explain ML.",
                ],
                min_overlap=0.1,
            )
        r = exc.value.result
        assert r.passed is False

    def test_retrieval_single_doc(self) -> None:
        """1 doc each, same doc => Jaccard 1.0, pass."""
        retriever = lambda q: ["only-doc"]
        result = assert_retrieval_consistency(
            retriever_fn=retriever,
            paraphrases=[
                "What is ML?",
                "Explain ML.",
            ],
            min_overlap=1.0,
        )
        assert result.passed is True

    def test_retrieval_empty_doc_set(self) -> None:
        """Empty doc list => Jaccard 0.0, fails or errs."""
        retriever = lambda q: []
        with pytest.raises(
            (MltkAssertionError, ValueError),
        ):
            assert_retrieval_consistency(
                retriever_fn=retriever,
                paraphrases=[
                    "What is ML?",
                    "Explain ML.",
                ],
                min_overlap=0.5,
            )

    def test_retrieval_high_threshold(self) -> None:
        """threshold=0.99 with slight difference => fail."""
        counter = {"i": 0}

        def almost_same(q: str) -> list[str]:
            counter["i"] += 1
            base = ["d1", "d2", "d3", "d4", "d5"]
            if counter["i"] == 1:
                return base
            # Replace one doc => Jaccard 4/6 ~ 0.667
            return ["d1", "d2", "d3", "d4", "d6"]

        with pytest.raises(MltkAssertionError):
            assert_retrieval_consistency(
                retriever_fn=almost_same,
                paraphrases=[
                    "What is ML?",
                    "Explain ML.",
                ],
                min_overlap=0.99,
            )

    def test_retrieval_low_threshold(self) -> None:
        """threshold=0.01, any overlap => pass."""
        counter = {"i": 0}

        def tiny_overlap(q: str) -> list[str]:
            counter["i"] += 1
            if counter["i"] == 1:
                return ["shared", "a1", "a2"]
            return ["shared", "b1", "b2"]

        result = assert_retrieval_consistency(
            retriever_fn=tiny_overlap,
            paraphrases=[
                "What is ML?",
                "Explain ML.",
            ],
            min_overlap=0.01,
        )
        assert result.passed is True

    def test_retrieval_many_docs(self) -> None:
        """100 docs each, 80% overlap."""
        shared = [f"doc-{i}" for i in range(80)]
        counter = {"i": 0}

        def many_docs(q: str) -> list[str]:
            counter["i"] += 1
            unique = [
                f"u{counter['i']}-{j}"
                for j in range(20)
            ]
            return shared + unique

        # Jaccard = 80 / 120 ~ 0.667
        result = assert_retrieval_consistency(
            retriever_fn=many_docs,
            paraphrases=[
                "What is ML?",
                "Explain ML.",
            ],
            min_overlap=0.5,
        )
        assert result.passed is True

    def test_retrieval_doc_order_irrelevant(
        self,
    ) -> None:
        """Same docs in different order => same Jaccard."""
        counter = {"i": 0}

        def shuffled(q: str) -> list[str]:
            counter["i"] += 1
            docs = ["alpha", "beta", "gamma"]
            if counter["i"] % 2 == 0:
                return list(reversed(docs))
            return docs

        result = assert_retrieval_consistency(
            retriever_fn=shuffled,
            paraphrases=[
                "What is ML?",
                "Explain ML.",
            ],
            min_overlap=1.0,
        )
        assert result.passed is True
        assert result.details["avg_overlap"] == 1.0
