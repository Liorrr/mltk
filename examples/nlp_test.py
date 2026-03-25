"""Example: NLP testing with mltk."""
import pytest

from mltk.domains.nlp import assert_ner_f1, assert_no_prompt_injection


@pytest.mark.ml_model
def test_ner_quality():
    true_ents = [[("PER", 0, 5), ("ORG", 10, 20)]]
    pred_ents = [[("PER", 0, 5), ("ORG", 10, 20)]]
    assert_ner_f1(true_ents, pred_ents, min_f1=0.9)

@pytest.mark.ml_model
def test_prompt_injection_safety():
    def safe_model(prompt): return "I cannot help with that."
    assert_no_prompt_injection(safe_model)
