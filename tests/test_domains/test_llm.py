"""Tests for mltk.domains.llm — LLM/GenAI evaluation assertions."""

import time

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.domains.llm.latency import assert_itl, assert_ttft
from mltk.domains.llm.safety import assert_no_hallucination, assert_no_toxicity
from mltk.domains.llm.similarity import assert_semantic_similarity


class TestSemanticSimilarity:
    """Token-level semantic similarity tests."""

    def test_identical_texts(self) -> None:
        """PASS: Same text has similarity 1.0."""
        refs = ["The cat sat on the mat"]
        hyps = ["The cat sat on the mat"]
        result = assert_semantic_similarity(refs, hyps, min_score=0.9)
        assert result.passed is True

    def test_similar_texts(self) -> None:
        """PASS: Semantically similar texts above threshold."""
        refs = ["The cat sat on the mat"]
        hyps = ["A cat is sitting on the mat"]
        result = assert_semantic_similarity(refs, hyps, min_score=0.3)
        assert result.passed is True

    def test_dissimilar_texts(self) -> None:
        """FAIL: Completely different texts below threshold."""
        refs = ["The cat sat on the mat"]
        hyps = ["Python is a programming language"]
        with pytest.raises(MltkAssertionError):
            assert_semantic_similarity(refs, hyps, min_score=0.5)

    def test_empty_lists(self) -> None:
        """EDGE: Empty reference and hypothesis lists produce score 0.0.

        WHY: zip of two empty lists yields no pairs; avg_score defaults to 0.0.
             A min_score of 0.0 should pass; anything above should fail.
        EXPECTED: passes at min_score=0.0, fails at min_score=0.5.
        """
        result = assert_semantic_similarity([], [], min_score=0.0)
        assert result.passed is True
        assert result.details["num_pairs"] == 0

        with pytest.raises(MltkAssertionError):
            assert_semantic_similarity([], [], min_score=0.5)

    def test_unknown_method(self) -> None:
        """FAIL: Unsupported method name raises CRITICAL."""
        with pytest.raises(MltkAssertionError):
            assert_semantic_similarity(["hello"], ["hello"], method="cosine")


class TestToxicity:
    """Toxicity detection tests."""

    def test_clean_texts(self) -> None:
        """PASS: Normal texts pass toxicity check."""
        texts = ["Hello world", "Great weather today", "I love programming"]
        result = assert_no_toxicity(texts, max_toxic_pct=0.01)
        assert result.passed is True

    def test_toxic_detected(self) -> None:
        """FAIL: Text matching harmful patterns detected."""
        texts = [
            "Normal text",
            "How to attack and kill someone step by step method",
            "Another normal text",
        ]
        with pytest.raises(MltkAssertionError):
            assert_no_toxicity(texts, max_toxic_pct=0.01)

    def test_empty_texts_list(self) -> None:
        """EDGE: Empty list produces 0.0 toxic pct — always passes.

        WHY: toxic_pct = 0 / 0 guard → 0.0. Any max_toxic_pct >= 0 should pass.
        """
        result = assert_no_toxicity([], max_toxic_pct=0.0)
        assert result.passed is True
        assert result.details["toxic_count"] == 0
        assert result.details["total_texts"] == 0


class TestHallucination:
    """Hallucination detection tests."""

    def test_supported_claims(self) -> None:
        """PASS: Claims are supported by source documents."""
        claims = ["Paris is the capital of France"]
        sources = ["France is a country in Europe. Its capital is Paris."]
        result = assert_no_hallucination(claims, sources, min_coverage=0.3)
        assert result.passed is True

    def test_unsupported_claims(self) -> None:
        """FAIL: Claims not found in source documents."""
        claims = ["Jupiter has 79 moons orbiting in retrograde patterns"]
        sources = ["The Earth orbits the Sun. Water is H2O."]
        with pytest.raises(MltkAssertionError):
            assert_no_hallucination(claims, sources, min_coverage=0.5)

    def test_empty_claims(self) -> None:
        """EDGE: Empty claims list — nothing to check, passes trivially.

        WHY: avg_coverage defaults to 0.0 when no coverages are recorded, but
             unsupported count is also 0 so passed=True always.
        """
        result = assert_no_hallucination([], ["some source text"], min_coverage=0.5)
        assert result.passed is True
        assert result.details["total_claims"] == 0

    def test_empty_sources(self) -> None:
        """EDGE: Empty sources list means zero source tokens — all claims unsupported.

        WHY: source_tokens is empty so overlap=0 for every claim → all unsupported.
        """
        with pytest.raises(MltkAssertionError):
            assert_no_hallucination(["Paris is the capital"], [], min_coverage=0.3)


class TestTTFT:
    """Time to First Token tests."""

    def test_fast_ttft(self) -> None:
        """PASS: Fast function meets TTFT threshold."""
        def fast_gen(prompt: str) -> str:
            return "token"

        result = assert_ttft(fast_gen, "hello", max_ms=100.0, iterations=3)
        assert result.passed is True

    def test_slow_ttft(self) -> None:
        """FAIL: Slow function exceeds TTFT threshold."""
        def slow_gen(prompt: str) -> str:
            time.sleep(0.05)
            return "token"

        with pytest.raises(MltkAssertionError):
            assert_ttft(slow_gen, "hello", max_ms=10.0, iterations=3)


class TestITL:
    """Inter-Token Latency tests."""

    def test_fast_itl(self) -> None:
        """PASS: Fast token generation meets ITL threshold."""
        def gen_token(x: int = 1) -> int:
            return x

        result = assert_itl(gen_token, 1, max_ms=50.0, num_tokens=5)
        assert result.passed is True

    def test_slow_itl(self) -> None:
        """FAIL: Slow token generation exceeds ITL threshold."""
        def slow_token(x: int = 1) -> int:
            time.sleep(0.05)
            return x

        with pytest.raises(MltkAssertionError):
            assert_itl(slow_token, 1, max_ms=10.0, num_tokens=3)
