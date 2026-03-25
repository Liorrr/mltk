"""Tests for embedding drift detection — cosine, euclidean, MMD."""

import numpy as np
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.data.embedding_drift import assert_no_embedding_drift


class TestEmbeddingDrift:
    """Embedding drift detection tests."""

    def test_identical_embeddings_cosine(self) -> None:
        """PASS: Same embeddings have zero cosine distance."""
        rng = np.random.default_rng(42)
        emb = rng.normal(0, 1, (100, 64))
        result = assert_no_embedding_drift(emb, emb, method="cosine", threshold=0.01)
        assert result.passed is True

    def test_shifted_embeddings_cosine(self) -> None:
        """FAIL: Embeddings shifted in different direction."""
        ref = np.ones((100, 64))
        cur = -np.ones((100, 64))
        with pytest.raises(MltkAssertionError):
            assert_no_embedding_drift(ref, cur, method="cosine", threshold=0.5)

    def test_euclidean_method(self) -> None:
        """PASS: Similar embeddings have small euclidean distance."""
        rng = np.random.default_rng(42)
        emb = rng.normal(0, 1, (100, 32))
        result = assert_no_embedding_drift(emb, emb, method="euclidean", threshold=1.0)
        assert result.passed is True

    def test_mmd_method(self) -> None:
        """PASS: Identical embeddings have MMD near 0."""
        rng = np.random.default_rng(42)
        emb = rng.normal(0, 1, (50, 16))
        result = assert_no_embedding_drift(emb, emb, method="mmd", threshold=0.1)
        assert result.passed is True

    def test_unknown_method(self) -> None:
        """FAIL: Invalid method raises error."""
        with pytest.raises(MltkAssertionError):
            assert_no_embedding_drift(np.array([[1]]), np.array([[1]]), method="invalid")
