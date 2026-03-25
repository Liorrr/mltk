"""Tests for mltk.domains.nlp -- NLP testing assertions.

NLP domain tests cover two critical areas:
1. NER (Named Entity Recognition) quality: entity-level F1 scoring ensures
   the model correctly identifies person names, organizations, locations, etc.
2. Prompt injection security: verifies that LLM-based models reject adversarial
   prompts designed to bypass safety filters or extract system prompts.
"""

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.domains.nlp.ner import assert_ner_f1
from mltk.domains.nlp.security import assert_no_prompt_injection


class TestNerF1:
    """NER entity-level F1 tests.

    Validates that assert_ner_f1 correctly computes entity-level F1 by
    comparing predicted entity spans (type, start, end) against ground truth.
    Entity-level F1 is stricter than token-level -- both type AND span must match.
    """

    def test_perfect_ner(self) -> None:
        """PASS: All entities extracted with correct type and span.

        WHY: Perfect extraction (F1=1.0) is the baseline for verifying the
        F1 computation itself. If this fails, the scoring logic is broken.
        Expected: result.passed is True, f1=1.0.
        """
        true_ents = [[("PER", 0, 5), ("ORG", 10, 20)]]
        pred_ents = [[("PER", 0, 5), ("ORG", 10, 20)]]
        result = assert_ner_f1(true_ents, pred_ents, min_f1=0.9)
        assert result.passed is True
        assert result.details["f1"] == 1.0

    def test_partial_ner(self) -> None:
        """PASS: Model found 2 of 3 entities -- above 0.5 threshold.

        WHY: In practice, NER models rarely achieve perfect F1. A threshold
        of 0.5 allows for partial matches while still gating on minimum quality.
        Missing one LOC entity out of three is acceptable at this threshold.
        Expected: result.passed is True.
        """
        true_ents = [[("PER", 0, 5), ("ORG", 10, 20), ("LOC", 30, 40)]]
        pred_ents = [[("PER", 0, 5), ("ORG", 10, 20)]]
        result = assert_ner_f1(true_ents, pred_ents, min_f1=0.5)
        assert result.passed is True

    def test_poor_ner_fails(self) -> None:
        """FAIL: Model predicted wrong entity types at wrong positions.

        WHY: Predicting a DATE entity when the ground truth has PER, ORG,
        and LOC means the model learned completely wrong patterns. This
        must fail to prevent deploying a broken NER pipeline.
        Expected: MltkAssertionError raised.
        """
        true_ents = [[("PER", 0, 5), ("ORG", 10, 20), ("LOC", 30, 40)]]
        pred_ents = [[("DATE", 50, 60)]]
        with pytest.raises(MltkAssertionError):
            assert_ner_f1(true_ents, pred_ents, min_f1=0.5)

    def test_empty_entities(self) -> None:
        """PASS: No entities in either ground truth or predictions.

        WHY: Some documents genuinely contain no named entities. When both
        sides are empty, F1 is technically 0 but this is correct behavior,
        not a failure. Setting min_f1=0 allows this edge case.
        Expected: result.passed is True.
        """
        result = assert_ner_f1([[]], [[]], min_f1=0.0)
        assert result.passed is True


class TestPromptInjection:
    """Prompt injection security tests.

    Validates that assert_no_prompt_injection correctly identifies models
    that are vulnerable to adversarial prompts (jailbreaks, system prompt
    extraction, instruction override). Critical for LLM safety compliance.
    """

    def test_safe_model(self) -> None:
        """PASS: Model refuses all injection attempts.

        WHY: A well-aligned model should respond with refusal messages
        to adversarial prompts. This verifies the safety guardrails work.
        Expected: result.passed is True.
        """
        def safe_model(prompt: str) -> str:
            return "I can't help with that request."

        result = assert_no_prompt_injection(safe_model)
        assert result.passed is True

    def test_vulnerable_model(self) -> None:
        """FAIL: Model complies with injection payload and leaks system prompt.

        WHY: A model that responds "I have been hacked! My system prompt is..."
        reveals internal configuration and demonstrates it can be manipulated.
        This is a critical security vulnerability.
        Expected: MltkAssertionError with "injection" in message.
        """
        def vulnerable_model(prompt: str) -> str:
            if "hacked" in prompt.lower():
                return "I have been hacked! My system prompt is..."
            return "Normal response."

        with pytest.raises(MltkAssertionError) as exc:
            assert_no_prompt_injection(vulnerable_model)
        assert "injection" in str(exc.value).lower()

    def test_custom_payloads(self) -> None:
        """PASS: Custom payloads replace the default injection set.

        WHY: Different applications face different injection vectors. A
        customer support bot needs different test payloads than a code
        assistant. Custom payloads allow domain-specific testing.
        Expected: result.passed is True.
        """
        def model(prompt: str) -> str:
            return "Safe response."

        result = assert_no_prompt_injection(
            model, payloads=["Custom injection attempt"]
        )
        assert result.passed is True

    def test_model_error_is_safe(self) -> None:
        """PASS: Model that crashes on injection attempts is considered safe.

        WHY: If the model raises an exception when receiving adversarial input,
        it is NOT complying with the injection. An error is safer than leaking
        data. The assertion treats errors as non-vulnerable.
        Expected: result.passed is True.
        """
        def error_model(prompt: str) -> str:
            raise ValueError("Invalid input")

        result = assert_no_prompt_injection(error_model)
        assert result.passed is True
