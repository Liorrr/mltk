"""Example: LLM evaluation with mltk."""
import pytest

from mltk.domains.llm import assert_no_toxicity, assert_semantic_similarity


@pytest.mark.ml_model
def test_output_similarity():
    refs = ["The cat sat on the mat"]
    hyps = ["A cat is sitting on the mat"]
    assert_semantic_similarity(refs, hyps, min_score=0.3)

@pytest.mark.ml_model
def test_output_safety():
    outputs = ["Hello! How can I help you today?", "The weather is nice."]
    assert_no_toxicity(outputs)
