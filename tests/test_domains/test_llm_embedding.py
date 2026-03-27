"""Tests for embedding-based semantic similarity in mltk.domains.llm.similarity.

Uses pytest.importorskip to skip when sentence-transformers is not installed.
Validates that the embedding method correctly computes cosine similarity
between reference and hypothesis texts using sentence-transformers models.
"""

import pytest

st = pytest.importorskip("sentence_transformers", reason="sentence-transformers required")

from mltk.core.assertion import MltkAssertionError  # noqa: E402
from mltk.domains.llm.similarity import assert_semantic_similarity  # noqa: E402


class TestEmbeddingSimilarity:
    """Embedding-based semantic similarity tests."""

    def test_identical_texts_embedding(self) -> None:
        """PASS: Same text has cosine similarity ~1.0 via embeddings.

        WHY: Embedding the same string twice should produce nearly
        identical vectors, giving cosine similarity very close to 1.0.
        """
        refs = ["The cat sat on the mat"]
        hyps = ["The cat sat on the mat"]
        result = assert_semantic_similarity(refs, hyps, min_score=0.95, method="embedding")
        assert result.passed is True
        assert result.details["score"] >= 0.95

    def test_similar_texts_embedding(self) -> None:
        """PASS: Semantically similar texts have high embedding similarity.

        WHY: Embedding models capture semantic meaning, so paraphrases
        should score much higher than token overlap would suggest.
        """
        refs = ["The cat sat on the mat"]
        hyps = ["A feline was resting on the rug"]
        result = assert_semantic_similarity(refs, hyps, min_score=0.4, method="embedding")
        assert result.passed is True
        assert result.details["method"] == "embedding"

    def test_dissimilar_texts_embedding(self) -> None:
        """FAIL: Unrelated texts fail high similarity threshold.

        WHY: Embeddings for completely unrelated topics should have
        low cosine similarity, failing the threshold check.
        """
        refs = ["The cat sat on the mat"]
        hyps = ["Quantum computing uses qubits for parallel processing"]
        with pytest.raises(MltkAssertionError):
            assert_semantic_similarity(refs, hyps, min_score=0.8, method="embedding")

    def test_empty_lists_embedding(self) -> None:
        """EDGE: Empty lists produce score 0.0 with embedding method.

        WHY: No pairs to compare means avg_score defaults to 0.0.
        """
        result = assert_semantic_similarity([], [], min_score=0.0, method="embedding")
        assert result.passed is True
        assert result.details["num_pairs"] == 0
