"""Length-contract tests for NLP generation assertions."""

from __future__ import annotations

import pytest

from mltk.domains.nlp.generation import assert_rouge


def test_rouge_mismatched_lengths_raise_before_optional_dependency_import() -> None:
    references = ["the cat sat on the mat", "the dog played in the park"]
    hypotheses = ["the cat is on the mat"]

    with pytest.raises(ValueError, match="assert_rouge: length mismatch"):
        assert_rouge(references, hypotheses, variant="rougeL", min_score=0.3)
