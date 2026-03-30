"""End-to-end behavioral pipeline tests.

Verifies the FULL workflow:
  ParaphraseGenerator -> model_fn -> assertion
so interface mismatches between components are caught
even when unit tests pass individually.

All model and backend calls are mocked.  No external
dependencies, no network, no time.sleep.
"""

from __future__ import annotations

import hashlib
from unittest.mock import MagicMock

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.core.result import TestResult
from mltk.domains.llm.behavioral import (
    ParaphraseGenerator,
    assert_directional_expectation,
    assert_format_invariance,
    assert_output_stability,
    assert_paraphrase_invariance,
    assert_retrieval_consistency,
    assert_semantic_equivalence,
)

# -- Helpers / constants -----------------------------------------

SEED = 42


def _result_ok(result: TestResult) -> None:
    """Verify TestResult structural contract."""
    assert isinstance(result.name, str)
    assert isinstance(result.passed, bool)
    assert isinstance(result.message, str)
    assert isinstance(result.details, dict)
    assert result.duration_ms >= 0.0


# -- Mock models -------------------------------------------------


def _consistent(text: str) -> str:
    """Always returns the same answer."""
    return "ML is a subset of AI"


_COUNTER = 0


def _inconsistent(text: str) -> str:
    """Different answer every call."""
    global _COUNTER  # noqa: PLW0603
    _COUNTER += 1
    return f"answer-{_COUNTER}"


def _hash_model(text: str) -> str:
    """Deterministic but input-dependent."""
    h = hashlib.md5(
        text.encode(), usedforsecurity=False,
    ).hexdigest()[:8]
    return f"result-{h}"


def _classifier(text: str) -> str:
    """Deterministic short label."""
    return "positive"


def _length_model(text: str) -> str:
    """Longer input -> longer output."""
    return "x" * max(1, len(text))


# ================================================================
# E2E pipeline tests
# ================================================================


class TestBehavioralE2EPipeline:
    """End-to-end: ParaphraseGenerator -> model -> assert."""

    def test_consistent_model_full_pipeline(
        self,
    ) -> None:
        """Generate paraphrases -> consistent model -> pass."""
        gen = ParaphraseGenerator()
        paraphrases = gen.generate_template(
            "What is machine learning?", n=4,
        )
        assert len(paraphrases) >= 2
        result = assert_paraphrase_invariance(
            model_fn=_consistent,
            paraphrases=paraphrases,
            equivalence_method="token_f1",
        )
        _result_ok(result)
        assert result.passed

    def test_inconsistent_model_full_pipeline(
        self,
    ) -> None:
        """Generate paraphrases -> bad model -> fail."""
        global _COUNTER  # noqa: PLW0603
        _COUNTER = 0
        gen = ParaphraseGenerator()
        paraphrases = gen.generate_template(
            "What is machine learning?", n=4,
        )
        with pytest.raises(MltkAssertionError) as exc:
            assert_paraphrase_invariance(
                model_fn=_inconsistent,
                paraphrases=paraphrases,
                equivalence_method="token_f1",
            )
        result = exc.value.result
        _result_ok(result)
        assert not result.passed

    def test_generator_to_format_invariance(
        self,
    ) -> None:
        """Template transforms + format invariance."""
        result = assert_format_invariance(
            model_fn=_consistent,
            input_text="What is machine learning?",
            equivalence_method="token_f1",
        )
        _result_ok(result)
        assert result.passed
        rate = result.details["invariance_rate"]
        assert rate == 1.0

    def test_generator_to_stability(self) -> None:
        """Generate inputs -> stability with n_runs."""
        gen = ParaphraseGenerator()
        paraphrases = gen.generate_template(
            "What is machine learning?", n=3,
        )
        result = assert_output_stability(
            model_fn=_consistent,
            inputs=paraphrases,
            n_runs=3,
            equivalence_method="label_match",
        )
        _result_ok(result)
        assert result.passed

    def test_generator_to_retrieval(self) -> None:
        """Generate paraphrases -> retrieval consistency."""
        gen = ParaphraseGenerator()
        paraphrases = gen.generate_template(
            "What is machine learning?", n=3,
        )
        def retriever(q):
            return ["doc1", "doc2", "doc3"]
        result = assert_retrieval_consistency(
            retriever_fn=retriever,
            paraphrases=paraphrases,
            min_overlap=0.7,
        )
        _result_ok(result)
        assert result.passed

    def test_pipeline_with_llm_generator_mock(
        self,
    ) -> None:
        """LLM-based generator -> invariance (mocked)."""
        mock_llm = MagicMock(
            return_value=(
                "1. What does ML mean?\n"
                "2. Define machine learning\n"
                "3. Explain ML\n"
            ),
        )
        gen = ParaphraseGenerator()
        paraphrases = gen.generate_llm(
            "What is machine learning?",
            llm_fn=mock_llm,
            n=3,
        )
        assert len(paraphrases) == 3
        result = assert_paraphrase_invariance(
            model_fn=_consistent,
            paraphrases=paraphrases,
            equivalence_method="token_f1",
        )
        _result_ok(result)
        assert result.passed

    def test_full_behavioral_suite(self) -> None:
        """Run ALL 7 assertions on one mock model."""
        paraphrases = [
            "What is ML?",
            "Explain ML",
            "Describe ML",
        ]

        # 1. Paraphrase invariance
        r1 = assert_paraphrase_invariance(
            model_fn=_consistent,
            paraphrases=paraphrases,
            equivalence_method="token_f1",
        )
        _result_ok(r1)
        assert r1.passed

        # 2. Format invariance
        r2 = assert_format_invariance(
            model_fn=_consistent,
            input_text="What is ML?",
            equivalence_method="token_f1",
        )
        _result_ok(r2)
        assert r2.passed

        # 3. Output stability
        r3 = assert_output_stability(
            model_fn=_consistent,
            inputs=paraphrases,
            n_runs=3,
            equivalence_method="label_match",
        )
        _result_ok(r3)
        assert r3.passed

        # 4. Semantic equivalence (token_f1)
        r4 = assert_semantic_equivalence(
            "ML is a subset of AI",
            "ML is a subset of AI",
            method="token_f1",
            min_score=0.5,
        )
        _result_ok(r4)
        assert r4.passed

        # 5. Directional expectation
        r5 = assert_directional_expectation(
            model_fn=_length_model,
            input_text="short",
            perturbation=lambda t: t + " extra words",
            direction_fn=lambda o, p: len(p) > len(o),
            perturbation_name="add_words",
        )
        _result_ok(r5)
        assert r5.passed

        # 6. Retrieval consistency
        def ret_fn(q):
            return ["doc1", "doc2"]
        r6 = assert_retrieval_consistency(
            retriever_fn=ret_fn,
            paraphrases=paraphrases,
            min_overlap=0.5,
        )
        _result_ok(r6)
        assert r6.passed

        # 7. Paraphrase invariance (label_match)
        r7 = assert_paraphrase_invariance(
            model_fn=_classifier,
            paraphrases=paraphrases,
            equivalence_method="label_match",
        )
        _result_ok(r7)
        assert r7.passed

    def test_unstable_model_detected(self) -> None:
        """Unstable model fails output stability."""
        global _COUNTER  # noqa: PLW0603
        _COUNTER = 100
        with pytest.raises(MltkAssertionError) as exc:
            assert_output_stability(
                model_fn=_inconsistent,
                inputs=["What is ML?"],
                n_runs=4,
                equivalence_method="label_match",
            )
        result = exc.value.result
        _result_ok(result)
        assert not result.passed

    def test_retrieval_divergent_detected(
        self,
    ) -> None:
        """Retriever returning different docs -> fail."""
        call_count = 0

        def divergent_retriever(q: str) -> list[str]:
            nonlocal call_count
            call_count += 1
            return [f"doc-{call_count}"]

        gen = ParaphraseGenerator()
        paraphrases = gen.generate_template(
            "What is machine learning?", n=3,
        )
        with pytest.raises(MltkAssertionError) as exc:
            assert_retrieval_consistency(
                retriever_fn=divergent_retriever,
                paraphrases=paraphrases,
                min_overlap=0.5,
            )
        result = exc.value.result
        _result_ok(result)
        assert not result.passed

    def test_directional_expectation_pipeline(
        self,
    ) -> None:
        """Directional: add negative -> changes output."""
        def sentiment_model(text: str) -> str:
            if "negative" in text.lower():
                return "bad product"
            return "great product"

        result = assert_directional_expectation(
            model_fn=sentiment_model,
            input_text="Review this product",
            perturbation=lambda t: "negative " + t,
            direction_fn=lambda o, p: (
                "bad" in p.lower()
            ),
            perturbation_name="negative_shift",
        )
        _result_ok(result)
        assert result.passed

    def test_semantic_equiv_in_pipeline(
        self,
    ) -> None:
        """Semantic equivalence with token_f1 method."""
        gen = ParaphraseGenerator()
        paraphrases = gen.generate_template(
            "What is deep learning?", n=2,
        )
        out_a = _consistent(paraphrases[0])
        out_b = _consistent(paraphrases[1])
        result = assert_semantic_equivalence(
            out_a,
            out_b,
            method="token_f1",
            min_score=0.5,
        )
        _result_ok(result)
        assert result.passed

    def test_format_invariance_custom_transforms(
        self,
    ) -> None:
        """Custom transforms piped through pipeline."""
        transforms = [
            lambda t: t.upper(),
            lambda t: t.lower(),
            lambda t: t.strip(),
        ]
        result = assert_format_invariance(
            model_fn=_consistent,
            input_text="What is AI?",
            transforms=transforms,
            equivalence_method="token_f1",
        )
        _result_ok(result)
        assert result.passed

    def test_classifier_auto_detects_label_match(
        self,
    ) -> None:
        """Short label outputs auto-switch to label_match."""
        paraphrases = [
            "Is this positive?",
            "Would you call this positive?",
            "Positive or negative?",
        ]
        result = assert_paraphrase_invariance(
            model_fn=_classifier,
            paraphrases=paraphrases,
            equivalence_method="token_f1",
        )
        _result_ok(result)
        assert result.passed
        assert result.details["method"] == "label_match"

    def test_llm_generator_unified_api(self) -> None:
        """ParaphraseGenerator.generate dispatches correctly."""
        gen = ParaphraseGenerator()

        # Template method
        tmpl = gen.generate(
            "What is ML?", n=3, method="template",
        )
        assert len(tmpl) >= 1

        # LLM method
        mock_fn = MagicMock(
            return_value="1. Explain ML\n2. Define ML\n",
        )
        llm = gen.generate(
            "What is ML?",
            n=2,
            method="llm",
            llm_fn=mock_fn,
        )
        assert len(llm) == 2
        mock_fn.assert_called_once()
