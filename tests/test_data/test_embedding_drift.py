"""Tests for embedding drift detection — cosine, euclidean, MMD."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.data.embedding_drift import _mmd, assert_no_embedding_drift


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

    def test_mmd_uses_seeded_random_subsample_instead_of_first_n(self) -> None:
        """MMD must sample from the whole array, not only the leading rows."""
        ref = np.zeros((300, 4))
        cur = np.zeros((300, 4))
        cur[200:] = 10.0

        distance = _mmd(ref, cur, gamma=0.5, seed=0)

        assert distance > 0.01

    def test_mmd_seed_makes_random_subsample_deterministic(self) -> None:
        """The same seed must produce stable MMD estimates."""
        rng = np.random.default_rng(7)
        ref = rng.normal(0, 1, (300, 4))
        cur = rng.normal(0.5, 1, (300, 4))

        first = _mmd(ref, cur, seed=123)
        second = _mmd(ref, cur, seed=123)

        assert first == second

    def test_cosine_rust_backend_runtime_errors_propagate(self, monkeypatch) -> None:
        """Only Rust import failures should fall back to numpy."""
        rust_module = types.ModuleType("mltk._rust")

        def broken_centroid(*_args: object) -> float:
            raise RuntimeError("rust computation failed")

        rust_module.centroid_cosine_distance = broken_centroid
        monkeypatch.setitem(sys.modules, "mltk._rust", rust_module)

        with pytest.raises(RuntimeError, match="rust computation failed"):
            assert_no_embedding_drift(
                np.ones((4, 2)),
                np.ones((4, 2)),
                method="cosine",
            )

    def test_unknown_method(self) -> None:
        """FAIL: Invalid method raises error."""
        with pytest.raises(MltkAssertionError):
            assert_no_embedding_drift(np.array([[1]]), np.array([[1]]), method="invalid")
