"""Tests for semantic equivalence and directional expectation.

Covers ``assert_semantic_equivalence`` and
``assert_directional_expectation`` from the behavioral
consistency module.  All NLI / embedding calls are mocked;
no external dependencies required.
"""

from __future__ import annotations

import unicodedata
from unittest.mock import MagicMock, patch

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.core.result import Severity, TestResult
from mltk.domains.llm.behavioral import (
    assert_directional_expectation,
    assert_semantic_equivalence,
)

# -- Shared helpers -----------------------------------------------

SEED = 42


def _result_shape_ok(result: TestResult) -> None:
    """Assert TestResult has the mandatory fields."""
    assert isinstance(result.name, str)
    assert isinstance(result.passed, bool)
    assert isinstance(result.message, str)
    assert isinstance(result.details, dict)
    assert result.duration_ms >= 0.0


# =================================================================
# TestSemanticEquivalence
# =================================================================


class TestSemanticEquivalence:
    """Tests for ``assert_semantic_equivalence``."""

    def test_identical_texts_pass(self) -> None:
        """Identical texts are always equivalent."""
        result = assert_semantic_equivalence(
            text_a="The sky is blue.",
            text_b="The sky is blue.",
            method="token_f1",
        )
        assert result.passed is True
        _result_shape_ok(result)

    def test_contradictory_texts_fail(self) -> None:
        """Contradictory texts detected via NLI mock."""
        with patch(
            "mltk.domains.llm._backends"
            ".nli_bidirectional",
        ) as mock_nli:
            mock_nli.return_value = {
                "forward": {
                    "entailment": 0.05,
                    "contradiction": 0.90,
                    "neutral": 0.05,
                    "label": "contradiction",
                },
                "backward": {
                    "entailment": 0.04,
                    "contradiction": 0.91,
                    "neutral": 0.05,
                    "label": "contradiction",
                },
                "equivalent": False,
                "contradiction": True,
            }
            with pytest.raises(MltkAssertionError):
                assert_semantic_equivalence(
                    text_a="This is the best.",
                    text_b="This is the worst.",
                    method="nli",
                )

    @patch(
        "mltk.domains.llm._backends"
        ".nli_bidirectional",
    )
    def test_method_nli_default(
        self, mock_nli: MagicMock,
    ) -> None:
        """Default method uses NLI bidirectional."""
        mock_nli.return_value = {
            "forward": {
                "entailment": 0.92,
                "contradiction": 0.03,
                "neutral": 0.05,
                "label": "entailment",
            },
            "backward": {
                "entailment": 0.89,
                "contradiction": 0.04,
                "neutral": 0.07,
                "label": "entailment",
            },
            "equivalent": True,
            "contradiction": False,
        }
        result = assert_semantic_equivalence(
            text_a="The cat sat on a mat.",
            text_b="A cat was sitting on the mat.",
            method="nli",
        )
        assert result.passed is True
        mock_nli.assert_called_once()
        assert result.details.get("method") == "nli"

    @patch(
        "mltk.domains.llm._backends"
        ".embedding_cosine_single",
    )
    def test_method_embedding(
        self, mock_emb: MagicMock,
    ) -> None:
        """Embedding method uses cosine similarity."""
        mock_emb.return_value = 0.95
        result = assert_semantic_equivalence(
            text_a="The cat sat on a mat.",
            text_b="A cat was sitting on the mat.",
            method="embedding",
        )
        assert result.passed is True
        mock_emb.assert_called_once()
        assert result.details.get("method") == "embedding"

    def test_method_token_f1(self) -> None:
        """Token F1 zero-dep fallback works."""
        result = assert_semantic_equivalence(
            text_a="the quick brown fox",
            text_b="the quick brown fox",
            method="token_f1",
        )
        assert result.passed is True
        assert result.details.get("method") == "token_f1"

    def test_unknown_method_fails(self) -> None:
        """Unknown method produces a failing TestResult."""
        with pytest.raises(
            (MltkAssertionError, ValueError),
        ):
            assert_semantic_equivalence(
                text_a="hello",
                text_b="hello",
                method="bogus_method",
            )

    @patch(
        "mltk.domains.llm._backends"
        ".nli_bidirectional",
    )
    def test_contradiction_flag_in_details(
        self, mock_nli: MagicMock,
    ) -> None:
        """NLI method includes contradiction flag."""
        mock_nli.return_value = {
            "forward": {
                "entailment": 0.10,
                "contradiction": 0.85,
                "neutral": 0.05,
                "label": "contradiction",
            },
            "backward": {
                "entailment": 0.08,
                "contradiction": 0.87,
                "neutral": 0.05,
                "label": "contradiction",
            },
            "equivalent": False,
            "contradiction": True,
        }
        with pytest.raises(MltkAssertionError) as exc:
            assert_semantic_equivalence(
                text_a="It is raining.",
                text_b="It is sunny.",
                method="nli",
            )
        details = exc.value.result.details
        assert "contradiction" in details
        assert details["contradiction"] is True

    @patch(
        "mltk.domains.llm._backends"
        ".nli_bidirectional",
    )
    def test_forward_backward_scores_in_details(
        self, mock_nli: MagicMock,
    ) -> None:
        """NLI result stores forward/backward scores."""
        mock_nli.return_value = {
            "forward": {
                "entailment": 0.91,
                "contradiction": 0.04,
                "neutral": 0.05,
                "label": "entailment",
            },
            "backward": {
                "entailment": 0.88,
                "contradiction": 0.05,
                "neutral": 0.07,
                "label": "entailment",
            },
            "equivalent": True,
            "contradiction": False,
        }
        result = assert_semantic_equivalence(
            text_a="Dogs are mammals.",
            text_b="Mammals include dogs.",
            method="nli",
        )
        details = result.details
        assert "forward_entailment" in details
        assert "backward_entailment" in details

    @patch(
        "mltk.domains.llm._backends"
        ".embedding_cosine_single",
    )
    def test_min_score_threshold(
        self, mock_emb: MagicMock,
    ) -> None:
        """Score below threshold fails."""
        mock_emb.return_value = 0.55
        with pytest.raises(MltkAssertionError):
            assert_semantic_equivalence(
                text_a="apples and oranges",
                text_b="cars and trucks",
                method="embedding",
                min_score=0.8,
            )

    def test_unicode_normalized(self) -> None:
        """Inputs are unicode-normalized before comparison."""
        # \u0041\u030A = "A with ring" (decomposed)
        raw_a = "\u0041\u030A sentence"
        nfkc_a = unicodedata.normalize("NFKC", raw_a)
        result = assert_semantic_equivalence(
            text_a=raw_a,
            text_b=nfkc_a,
            method="token_f1",
        )
        assert result.passed is True

    def test_empty_text_handled(self) -> None:
        """Empty strings do not crash; result is returned."""
        result = assert_semantic_equivalence(
            text_a="",
            text_b="",
            method="token_f1",
        )
        _result_shape_ok(result)

    @patch(
        "mltk.domains.llm._backends"
        ".embedding_cosine_single",
    )
    def test_score_in_details(
        self, mock_emb: MagicMock,
    ) -> None:
        """Score is included in TestResult details."""
        mock_emb.return_value = 0.92
        result = assert_semantic_equivalence(
            text_a="hello world",
            text_b="hello earth",
            method="embedding",
        )
        assert "score" in result.details

    def test_result_name(self) -> None:
        """TestResult has the correct assertion name."""
        result = assert_semantic_equivalence(
            text_a="same text",
            text_b="same text",
            method="token_f1",
        )
        assert "semantic" in result.name.lower()

    def test_severity_on_failure(self) -> None:
        """Failed equivalence has CRITICAL severity."""
        with pytest.raises(MltkAssertionError) as exc:
            assert_semantic_equivalence(
                text_a="alpha bravo charlie",
                text_b="x y z",
                method="token_f1",
                min_score=0.99,
            )
        assert (
            exc.value.result.severity == Severity.CRITICAL
        )

    def test_high_threshold_identical_passes(
        self,
    ) -> None:
        """Identical texts pass even at threshold=1.0."""
        result = assert_semantic_equivalence(
            text_a="exact match",
            text_b="exact match",
            method="token_f1",
            min_score=1.0,
        )
        assert result.passed is True


# =================================================================
# TestDirectionalExpectation
# =================================================================


class TestDirectionalExpectation:
    """Tests for ``assert_directional_expectation``."""

    def test_length_reduction_passes(self) -> None:
        """Adding 'be brief' produces shorter output."""
        def model(text: str) -> str:
            if "brief" in text.lower():
                return "short"
            return "a much longer response here"

        result = assert_directional_expectation(
            model_fn=model,
            input_text="Explain quantum computing",
            perturbation=lambda t: t + ". Be brief.",
            direction_fn=lambda o, p: len(p) < len(o),
        )
        assert result.passed is True
        _result_shape_ok(result)

    def test_wrong_direction_fails(self) -> None:
        """Output going wrong direction fails."""
        def model(text: str) -> str:
            return "same length answer"

        with pytest.raises(MltkAssertionError):
            assert_directional_expectation(
                model_fn=model,
                input_text="Explain gravity",
                perturbation=lambda t: t + ". Be brief.",
                direction_fn=lambda o, p: len(p) < len(o),
            )

    def test_perturbation_name_in_details(self) -> None:
        """Perturbation description in details if provided."""
        def model(text: str) -> str:
            if "brief" in text.lower():
                return "short"
            return "a much longer response here"

        result = assert_directional_expectation(
            model_fn=model,
            input_text="Explain gravity",
            perturbation=lambda t: t + ". Be brief.",
            direction_fn=lambda o, p: len(p) < len(o),
            perturbation_name="add_brevity",
        )
        details = result.details
        assert (
            "perturbation_name" in details
            or "perturbation" in details
        )

    def test_original_and_perturbed_in_details(
        self,
    ) -> None:
        """Details include original and perturbed outputs."""
        def model(text: str) -> str:
            if "brief" in text.lower():
                return "short"
            return "a much longer response here"

        result = assert_directional_expectation(
            model_fn=model,
            input_text="Explain gravity",
            perturbation=lambda t: t + ". Be brief.",
            direction_fn=lambda o, p: len(p) < len(o),
        )
        details = result.details
        assert (
            "original_output" in details
            or "original" in details
        )
        assert (
            "perturbed_output" in details
            or "perturbed" in details
        )

    def test_model_exception_handled(self) -> None:
        """Model that raises is handled gracefully."""
        def exploding_model(text: str) -> str:
            raise RuntimeError("model crashed")

        with pytest.raises(
            (MltkAssertionError, RuntimeError),
        ):
            assert_directional_expectation(
                model_fn=exploding_model,
                input_text="Explain gravity",
                perturbation=lambda t: t + " briefly",
                direction_fn=lambda o, p: len(p) < len(o),
            )

    def test_custom_direction_fn(self) -> None:
        """Custom direction_fn evaluates correctly."""
        call_count = 0

        def model(text: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "answer without numbers"
            return "answer with 42 numbers 7"

        result = assert_directional_expectation(
            model_fn=model,
            input_text="Give facts",
            perturbation=lambda t: t + " with numbers",
            direction_fn=lambda o, p: any(
                c.isdigit() for c in p
            ),
        )
        assert result.passed is True

    def test_identity_perturbation_same_output(
        self,
    ) -> None:
        """Identity perturbation with always-true fn passes."""
        def model(text: str) -> str:
            return "fixed output"

        result = assert_directional_expectation(
            model_fn=model,
            input_text="Test input",
            perturbation=lambda t: t,
            direction_fn=lambda o, p: True,
        )
        assert result.passed is True

    def test_direction_fn_false_always_fails(
        self,
    ) -> None:
        """direction_fn returning False always fails."""
        def model(text: str) -> str:
            return "some output"

        with pytest.raises(MltkAssertionError):
            assert_directional_expectation(
                model_fn=model,
                input_text="Test input",
                perturbation=lambda t: t + " more",
                direction_fn=lambda o, p: False,
            )

    def test_result_name(self) -> None:
        """TestResult has the correct assertion name."""
        def model(text: str) -> str:
            if "brief" in text:
                return "short"
            return "longer answer"

        result = assert_directional_expectation(
            model_fn=model,
            input_text="Explain gravity",
            perturbation=lambda t: t + ". Be brief.",
            direction_fn=lambda o, p: len(p) < len(o),
        )
        assert "directional" in result.name.lower()

    def test_severity_on_failure(self) -> None:
        """Failed directional check has CRITICAL severity."""
        def model(text: str) -> str:
            return "same"

        with pytest.raises(MltkAssertionError) as exc:
            assert_directional_expectation(
                model_fn=model,
                input_text="Explain gravity",
                perturbation=lambda t: t + ". Be brief.",
                direction_fn=lambda o, p: len(p) < len(o),
            )
        assert (
            exc.value.result.severity == Severity.CRITICAL
        )

    def test_input_text_passed_to_model(self) -> None:
        """Both original and perturbed text pass through."""
        received: list[str] = []

        def tracking_model(text: str) -> str:
            received.append(text)
            if len(received) == 1:
                return "long original answer"
            return "short"

        assert_directional_expectation(
            model_fn=tracking_model,
            input_text="Explain gravity",
            perturbation=lambda t: t + ". Be brief.",
            direction_fn=lambda o, p: len(p) < len(o),
        )
        assert len(received) == 2
        assert received[0] == "Explain gravity"
        assert received[1] == "Explain gravity. Be brief."


# =================================================================
# Hardening: parametrized + edge-case tests (appended)
# =================================================================


class TestSemanticEquivalenceHardening:
    """Extra edge-case tests for semantic equivalence."""

    def test_semantic_equivalence_identical(
        self,
    ) -> None:
        """Identical text -> always passes."""
        result = assert_semantic_equivalence(
            text_a="exact same text here",
            text_b="exact same text here",
            method="token_f1",
        )
        assert result.passed is True

    @patch(
        "mltk.domains.llm._backends"
        ".nli_bidirectional",
    )
    def test_semantic_equivalence_contradiction(
        self, mock_nli: MagicMock,
    ) -> None:
        """Opposite meanings -> fails."""
        mock_nli.return_value = {
            "forward": {
                "entailment": 0.03,
                "contradiction": 0.92,
                "neutral": 0.05,
                "label": "contradiction",
            },
            "backward": {
                "entailment": 0.04,
                "contradiction": 0.90,
                "neutral": 0.06,
                "label": "contradiction",
            },
            "equivalent": False,
            "contradiction": True,
        }
        with pytest.raises(MltkAssertionError):
            assert_semantic_equivalence(
                text_a="The door is open.",
                text_b="The door is closed.",
                method="nli",
            )

    @patch(
        "mltk.domains.llm._backends"
        ".embedding_cosine_single",
    )
    def test_semantic_equivalence_threshold(
        self, mock_emb: MagicMock,
    ) -> None:
        """Score at boundary threshold = 0.5."""
        mock_emb.return_value = 0.5
        result = assert_semantic_equivalence(
            text_a="apples",
            text_b="oranges",
            method="embedding",
            min_score=0.5,
        )
        assert result.passed is True

    def test_semantic_equivalence_single_pair(
        self,
    ) -> None:
        """Single pair of short texts."""
        result = assert_semantic_equivalence(
            text_a="hi",
            text_b="hi",
            method="token_f1",
        )
        assert result.passed is True

    def test_semantic_equivalence_unicode(
        self,
    ) -> None:
        """Unicode inputs do not crash."""
        result = assert_semantic_equivalence(
            text_a="\u6771\u4eac\u306f\u65e5\u672c",
            text_b="\u6771\u4eac\u306f\u65e5\u672c",
            method="token_f1",
        )
        _result_shape_ok(result)


class TestDirectionalExpectationHardening:
    """Extra edge-case tests for directional expectation."""

    def test_directional_all_positive(self) -> None:
        """All perturbations go in expected direction."""
        call_count = 0

        def model(text: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "original long response here"
            return "short"

        result = assert_directional_expectation(
            model_fn=model,
            input_text="Explain gravity",
            perturbation=lambda t: t + ". Be brief.",
            direction_fn=lambda o, p: len(p) < len(o),
        )
        assert result.passed is True

    def test_directional_mixed_fails(self) -> None:
        """Wrong direction -> fails."""
        def model(text: str) -> str:
            return "same length output always"

        with pytest.raises(MltkAssertionError):
            assert_directional_expectation(
                model_fn=model,
                input_text="Explain gravity",
                perturbation=lambda t: t + " briefly",
                direction_fn=(
                    lambda o, p: len(p) < len(o)
                ),
            )

    def test_directional_empty_perturbation_fn(
        self,
    ) -> None:
        """Perturbation that returns empty string."""
        def model(text: str) -> str:
            if text:
                return "some output"
            return ""

        result = assert_directional_expectation(
            model_fn=model,
            input_text="test",
            perturbation=lambda t: "",
            direction_fn=(
                lambda o, p: len(p) <= len(o)
            ),
        )
        _result_shape_ok(result)

    def test_directional_magnitude_large(self) -> None:
        """Large expected change direction."""
        counter = [0]

        def model(text: str) -> str:
            counter[0] += 1
            if counter[0] == 1:
                return "x" * 10
            return "x" * 10000

        result = assert_directional_expectation(
            model_fn=model,
            input_text="Write a lot",
            perturbation=lambda t: t + " (verbose)",
            direction_fn=lambda o, p: len(p) > len(o),
        )
        assert result.passed is True

    def test_directional_custom_metric(self) -> None:
        """User-provided metric_fn via direction_fn."""
        counter = [0]

        def model(text: str) -> str:
            counter[0] += 1
            if counter[0] == 1:
                return "no numbers here"
            return "has 42 numbers and 7 digits"

        def has_more_digits(
            orig: str, perturbed: str,
        ) -> bool:
            o_count = sum(
                1 for c in orig if c.isdigit()
            )
            p_count = sum(
                1 for c in perturbed if c.isdigit()
            )
            return p_count > o_count

        result = assert_directional_expectation(
            model_fn=model,
            input_text="Give facts",
            perturbation=lambda t: t + " with numbers",
            direction_fn=has_more_digits,
        )
        assert result.passed is True
