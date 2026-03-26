"""Tests for mltk.domains.llm.bertscore — BERTScore assertion."""

from __future__ import annotations

import numpy as np
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.domains.llm.bertscore import assert_bertscore


class TestBERTScoreIdentical:
    # SCENARIO: Reference and hypothesis have the exact same token embeddings.
    # WHY: Identical embeddings should yield perfect precision, recall, and F1.
    # EXPECTED: passed=True, f1 == 1.0, p == 1.0, r == 1.0.

    def test_bertscore_identical(self) -> None:
        """Identical token embeddings produce F1 = 1.0."""
        embs = np.eye(4)  # 4 orthonormal basis vectors (4, 4)
        result = assert_bertscore(embs, embs, min_f1=0.99)

        assert result.passed is True
        assert abs(result.details["f1"] - 1.0) < 1e-6
        assert abs(result.details["precision"] - 1.0) < 1e-6
        assert abs(result.details["recall"] - 1.0) < 1e-6


class TestBERTScoreOrthogonal:
    # SCENARIO: Reference and hypothesis token embeddings are perfectly orthogonal —
    #           each pair has cosine similarity 0.
    # WHY: Zero similarity → zero precision and recall → F1 = 0.
    # EXPECTED: passed=False (MltkAssertionError raised), f1 ~= 0.0.

    def test_bertscore_orthogonal(self) -> None:
        """Orthogonal token embeddings produce F1 ~= 0.0 and fail the assertion."""
        ref = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        hyp = np.array([[0.0, 0.0, 1.0]])  # orthogonal to all ref tokens

        with pytest.raises(MltkAssertionError) as exc_info:
            assert_bertscore(ref, hyp, min_f1=0.5)

        result = exc_info.value.result
        assert result.passed is False
        assert result.details["f1"] < 0.1


class TestBERTScoreAboveThreshold:
    # SCENARIO: Hypothesis embeddings are close (but not identical) to reference.
    # WHY: Verifies that a good-but-imperfect match passes when min_f1 is reasonable.
    # EXPECTED: passed=True, F1 above min_f1.

    def test_bertscore_above_threshold(self) -> None:
        """Near-identical embeddings pass a reasonable min_f1 threshold."""
        rng = np.random.default_rng(0)
        base = rng.standard_normal((5, 16))
        # Hypothesis = reference + tiny noise → very similar
        noise = rng.standard_normal((5, 16)) * 0.01
        ref = base / np.linalg.norm(base, axis=1, keepdims=True)
        hyp = (base + noise)
        hyp = hyp / np.linalg.norm(hyp, axis=1, keepdims=True)

        result = assert_bertscore(ref, hyp, min_f1=0.9)
        assert result.passed is True
        assert result.details["f1"] >= 0.9


class TestBERTScoreBelowThreshold:
    # SCENARIO: Hypothesis embeddings are very different from reference.
    # WHY: A poor semantic match should fail when min_f1 is set to a high bar.
    # EXPECTED: MltkAssertionError raised, result.passed=False.

    def test_bertscore_below_threshold(self) -> None:
        """Dissimilar embeddings fail a high min_f1 threshold."""
        rng = np.random.default_rng(42)
        ref = rng.standard_normal((4, 8))
        # Hypothesis is the negative → low cosine similarity
        hyp = -ref + rng.standard_normal((4, 8)) * 2.0

        with pytest.raises(MltkAssertionError) as exc_info:
            assert_bertscore(ref, hyp, min_f1=0.9)

        result = exc_info.value.result
        assert result.passed is False
        assert result.details["f1"] < 0.9


class TestBERTScoreEmpty:
    # SCENARIO: Empty embeddings are passed to assert_bertscore.
    # WHY: Empty input is an edge case that should fail gracefully with a clear error,
    #      not crash with an unhandled exception.
    # EXPECTED: MltkAssertionError raised with a descriptive message.

    def test_bertscore_empty_ref(self) -> None:
        """Empty reference embeddings produce a clear failure, not a crash."""
        hyp = np.array([[1.0, 0.0]])
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_bertscore(np.array([]), hyp, min_f1=0.5)
        assert "empty" in exc_info.value.result.message.lower()

    def test_bertscore_empty_hyp(self) -> None:
        """Empty hypothesis embeddings produce a clear failure, not a crash."""
        ref = np.array([[1.0, 0.0]])
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_bertscore(ref, np.array([]), min_f1=0.5)
        assert "empty" in exc_info.value.result.message.lower()


class TestBERTScoreMetadata:
    # SCENARIO: Verify TestResult carries all expected metadata fields.
    # WHY: Downstream tooling (reports, dashboards) depends on structured metadata.
    # EXPECTED: result.details has precision, recall, f1, ref_tokens, hyp_tokens,
    #           embedding_dim, min_f1 keys.

    def test_bertscore_result_has_metadata(self) -> None:
        """TestResult contains all required metadata fields."""
        ref = np.eye(3)
        hyp = np.eye(3)
        result = assert_bertscore(ref, hyp, min_f1=0.5)

        expected_keys = (
            "precision", "recall", "f1", "min_f1",
            "ref_tokens", "hyp_tokens", "embedding_dim",
        )
        for key in expected_keys:
            assert key in result.details, f"Missing key '{key}' in result.details"

        assert result.details["ref_tokens"] == 3
        assert result.details["hyp_tokens"] == 3
        assert result.details["embedding_dim"] == 3
        assert result.duration_ms >= 0
