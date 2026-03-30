"""Tests for ParaphraseGenerator.

Covers the ``ParaphraseGenerator`` class from the behavioral
consistency module.  All LLM calls are mocked; no external
dependencies required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mltk.domains.llm.behavioral import (
    ParaphraseGenerator,
)


# -- Shared helpers -----------------------------------------------

SEED = 42


# =================================================================
# TestParaphraseGenerator
# =================================================================


class TestParaphraseGenerator:
    """Tests for ``ParaphraseGenerator``."""

    def test_template_produces_n_results(self) -> None:
        """Template method returns the requested count."""
        gen = ParaphraseGenerator()
        results = gen.generate(
            "What is gravity?", n=5,
            method="template",
        )
        assert isinstance(results, list)
        assert len(results) == 5

    def test_template_deterministic(self) -> None:
        """Same input produces same output every time."""
        gen = ParaphraseGenerator()
        run1 = gen.generate(
            "What is gravity?", n=3,
            method="template",
        )
        run2 = gen.generate(
            "What is gravity?", n=3,
            method="template",
        )
        assert run1 == run2

    def test_template_question_reformulation(
        self,
    ) -> None:
        """Question inputs get rephrased variants."""
        gen = ParaphraseGenerator()
        results = gen.generate(
            "What is X?", n=3, method="template",
        )
        assert len(results) == 3
        # Each result should be a non-empty string
        for r in results:
            assert isinstance(r, str)
            assert len(r) > 0

    def test_template_filler_insertion(self) -> None:
        """Template can insert filler words."""
        gen = ParaphraseGenerator()
        results = gen.generate(
            "Explain quantum mechanics", n=3,
            method="template",
        )
        assert len(results) == 3
        for r in results:
            assert isinstance(r, str)
            assert len(r) > 0

    def test_template_contraction(self) -> None:
        """Template applies contractions or shortening."""
        gen = ParaphraseGenerator()
        results = gen.generate(
            "What is the capital of France?", n=3,
            method="template",
        )
        assert len(results) == 3
        for r in results:
            assert isinstance(r, str)

    def test_llm_method_calls_llm_fn(self) -> None:
        """LLM method delegates to provided llm_fn."""
        mock_llm = MagicMock(
            return_value=(
                "Rephrase 1\nRephrase 2\nRephrase 3"
            ),
        )
        gen = ParaphraseGenerator()
        results = gen.generate(
            "What is gravity?", n=3,
            method="llm", llm_fn=mock_llm,
        )
        mock_llm.assert_called_once()
        assert len(results) == 3

    def test_llm_method_without_fn_fails(self) -> None:
        """LLM method without llm_fn raises ValueError."""
        gen = ParaphraseGenerator()
        with pytest.raises(
            (ValueError, TypeError),
        ):
            gen.generate(
                "What is gravity?", n=3,
                method="llm",
            )

    def test_generate_dispatches_correctly(
        self,
    ) -> None:
        """generate() dispatches to the right method."""
        gen = ParaphraseGenerator()
        res_t = gen.generate(
            "test input", n=2, method="template",
        )
        assert len(res_t) == 2

        mock_llm = MagicMock(
            return_value="p1\np2",
        )
        res_l = gen.generate(
            "test input", n=2,
            method="llm", llm_fn=mock_llm,
        )
        assert len(res_l) == 2
        mock_llm.assert_called_once()

    def test_all_paraphrases_are_unique(self) -> None:
        """All returned paraphrases are distinct."""
        gen = ParaphraseGenerator()
        results = gen.generate(
            "What is gravity?", n=5,
            method="template",
        )
        assert len(set(results)) == len(results)

    def test_original_not_in_results(self) -> None:
        """Original text is not among the paraphrases."""
        original = "What is gravity?"
        gen = ParaphraseGenerator()
        results = gen.generate(
            original, n=5, method="template",
        )
        assert original not in results

    def test_n_equals_one(self) -> None:
        """Requesting a single paraphrase works."""
        gen = ParaphraseGenerator()
        results = gen.generate(
            "Explain photosynthesis", n=1,
            method="template",
        )
        assert len(results) == 1
        assert isinstance(results[0], str)

    def test_empty_input_handled(self) -> None:
        """Empty input does not crash the generator."""
        gen = ParaphraseGenerator()
        results = gen.generate(
            "", n=3, method="template",
        )
        assert isinstance(results, list)
        assert len(results) == 3
