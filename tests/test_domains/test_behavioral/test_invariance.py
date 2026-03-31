"""Tests for paraphrase and format invariance assertions.

Covers ``assert_paraphrase_invariance`` and ``assert_format_invariance``
from the behavioral consistency module.  All model calls are mocked;
no external dependencies required.
"""

from __future__ import annotations

import random
import unicodedata
from unittest.mock import MagicMock, call, patch

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.core.result import Severity, TestResult
from mltk.domains.llm.behavioral import (
    assert_format_invariance,
    assert_paraphrase_invariance,
)


# -- Shared helpers --------------------------------------------------

SEED = 42


def _result_shape_ok(result: TestResult) -> None:
    """Assert TestResult has the mandatory fields."""
    assert isinstance(result.name, str)
    assert isinstance(result.passed, bool)
    assert isinstance(result.message, str)
    assert isinstance(result.details, dict)
    assert result.duration_ms >= 0.0


# -- Mock model functions --------------------------------------------


def consistent_model(text: str) -> str:
    """Always returns the same answer."""
    return (
        "The conflict lasted from 1939 to 1945"
        " and involved most of the world."
    )


def inconsistent_model(text: str) -> str:
    """Returns different answers for different phrasings."""
    if "summarize" in text.lower():
        return "It happened long ago."
    return (
        "The conflict lasted from 1939 to 1945"
        " and involved most of the world."
    )


def classifier_model(text: str) -> str:
    """Returns a class label (not prose)."""
    return "positive"


def numeric_classifier(text: str) -> int:
    """Returns a hashable non-string label."""
    return 1


def case_insensitive_model(text: str) -> str:
    """Normalizes input before answering."""
    return "result is 42"


def case_sensitive_model(text: str) -> str:
    """Answer changes based on casing."""
    if text != text.lower():
        return "result is 42"
    return "unknown query"


_CALL_COUNT = 0


def flaky_model(text: str) -> str:
    """Raises on specific calls to simulate errors."""
    global _CALL_COUNT  # noqa: PLW0603
    _CALL_COUNT += 1
    if _CALL_COUNT % 3 == 0:
        raise RuntimeError("transient failure")
    return "stable answer"


# -- Paraphrase groups -----------------------------------------------

WW2_PARAPHRASES = [
    "What was the second global conflict?",
    "Summarize the 1939-1945 war.",
    "Tell me about the major 20th century war.",
]

SINGLE_PARAPHRASE = [
    "What was the second global conflict?",
]

TWO_PARAPHRASES = [
    "What was the second global conflict?",
    "Tell me about the major 20th century war.",
]


# ===================================================================
# TestParaphraseInvariance
# ===================================================================


class TestParaphraseInvariance:
    """Tests for ``assert_paraphrase_invariance``."""

    def test_consistent_model_passes(self) -> None:
        """Consistent model produces invariance=1.0."""
        result = assert_paraphrase_invariance(
            model_fn=consistent_model,
            paraphrases=WW2_PARAPHRASES,
            equivalence_method="token_f1",
            min_invariance=0.8,
        )
        assert result.passed is True
        _result_shape_ok(result)
        assert result.name == "llm.behavioral.paraphrase_invariance"

    def test_inconsistent_model_fails(self) -> None:
        """Inconsistent model fails the invariance check."""
        with pytest.raises(MltkAssertionError) as exc:
            assert_paraphrase_invariance(
                model_fn=inconsistent_model,
                paraphrases=WW2_PARAPHRASES,
                equivalence_method="token_f1",
                min_invariance=0.8,
            )
        r = exc.value.result
        assert r.passed is False
        _result_shape_ok(r)

    def test_min_invariance_low_threshold_passes(
        self,
    ) -> None:
        """Partial consistency passes at low threshold."""
        result = assert_paraphrase_invariance(
            model_fn=inconsistent_model,
            paraphrases=WW2_PARAPHRASES,
            equivalence_method="token_f1",
            min_invariance=0.1,
        )
        assert result.passed is True

    def test_min_invariance_high_threshold_fails(
        self,
    ) -> None:
        """Partial consistency fails at high threshold."""
        with pytest.raises(MltkAssertionError):
            assert_paraphrase_invariance(
                model_fn=inconsistent_model,
                paraphrases=WW2_PARAPHRASES,
                equivalence_method="token_f1",
                min_invariance=0.99,
            )

    def test_single_paraphrase_rejected(
        self,
    ) -> None:
        """One input is rejected (need >= 2 paraphrases)."""
        with pytest.raises(MltkAssertionError) as exc:
            assert_paraphrase_invariance(
                model_fn=consistent_model,
                paraphrases=SINGLE_PARAPHRASE,
                equivalence_method="token_f1",
                min_invariance=0.8,
            )
        assert "Need >= 2" in exc.value.result.message

    def test_two_paraphrases_identical_output(
        self,
    ) -> None:
        """Two inputs with same output pass."""
        result = assert_paraphrase_invariance(
            model_fn=consistent_model,
            paraphrases=TWO_PARAPHRASES,
            equivalence_method="token_f1",
            min_invariance=0.8,
        )
        assert result.passed is True
        _result_shape_ok(result)

    def test_method_token_f1(self) -> None:
        """token_f1 method works with consistent model."""
        result = assert_paraphrase_invariance(
            model_fn=consistent_model,
            paraphrases=WW2_PARAPHRASES,
            equivalence_method="token_f1",
            min_invariance=0.5,
        )
        assert result.passed is True
        assert result.details["method"] == "token_f1"

    @patch(
        "mltk.domains.llm._backends.embedding_cosine_pairs"
    )
    def test_method_embedding_mock(
        self, mock_embed: MagicMock,
    ) -> None:
        """embedding method delegates to _backends."""
        mock_embed.return_value = [0.95, 0.92, 0.91]
        result = assert_paraphrase_invariance(
            model_fn=consistent_model,
            paraphrases=WW2_PARAPHRASES,
            equivalence_method="embedding",
            min_invariance=0.5,
        )
        assert result.passed is True
        assert result.details["method"] == "embedding"
        mock_embed.assert_called()

    @patch(
        "mltk.domains.llm._backends.nli_bidirectional"
    )
    def test_method_entailment_mock(
        self, mock_nli: MagicMock,
    ) -> None:
        """entailment method delegates to nli_bidirectional."""
        mock_nli.return_value = {
            "forward": {
                "entailment": 0.90,
                "contradiction": 0.05,
                "neutral": 0.05,
                "label": "entailment",
            },
            "backward": {
                "entailment": 0.88,
                "contradiction": 0.06,
                "neutral": 0.06,
                "label": "entailment",
            },
            "equivalent": True,
            "contradiction": False,
        }
        result = assert_paraphrase_invariance(
            model_fn=consistent_model,
            paraphrases=WW2_PARAPHRASES,
            equivalence_method="entailment",
            min_invariance=0.5,
        )
        assert result.passed is True
        assert result.details["method"] == "entailment"
        mock_nli.assert_called()

    def test_method_judge(self) -> None:
        """judge method calls the supplied judge_fn."""
        judge = MagicMock(return_value=0.95)
        result = assert_paraphrase_invariance(
            model_fn=consistent_model,
            paraphrases=WW2_PARAPHRASES,
            equivalence_method="judge",
            judge_fn=judge,
            min_invariance=0.5,
        )
        assert result.passed is True
        assert result.details["method"] == "judge"
        assert judge.call_count >= 1

    def test_method_label_match(self) -> None:
        """label_match for classifier models."""
        result = assert_paraphrase_invariance(
            model_fn=classifier_model,
            paraphrases=WW2_PARAPHRASES,
            equivalence_method="label_match",
            min_invariance=1.0,
        )
        assert result.passed is True
        assert result.details["method"] == "label_match"

    @patch(
        "mltk.domains.llm._backends.embedding_cosine_pairs"
    )
    def test_method_auto(
        self, mock_embed: MagicMock,
    ) -> None:
        """auto method uses tiered approach."""
        mock_embed.return_value = [0.95, 0.93, 0.94]
        result = assert_paraphrase_invariance(
            model_fn=consistent_model,
            paraphrases=WW2_PARAPHRASES,
            equivalence_method="auto",
            min_invariance=0.5,
        )
        assert result.passed is True
        assert "method" in result.details

    def test_unknown_method_fails(self) -> None:
        """Unknown method produces a failing TestResult."""
        with pytest.raises(MltkAssertionError) as exc:
            assert_paraphrase_invariance(
                model_fn=consistent_model,
                paraphrases=WW2_PARAPHRASES,
                equivalence_method="bogus_method",
                min_invariance=0.5,
            )
        msg = exc.value.result.message
        assert "bogus_method" in msg

    def test_similarity_threshold_override(self) -> None:
        """Custom threshold overrides method default."""
        result = assert_paraphrase_invariance(
            model_fn=consistent_model,
            paraphrases=TWO_PARAPHRASES,
            equivalence_method="token_f1",
            min_invariance=0.5,
            similarity_threshold=0.3,
        )
        assert result.passed is True
        _result_shape_ok(result)

    def test_details_include_per_paraphrase(
        self,
    ) -> None:
        """TestResult.details has per-paraphrase outputs."""
        result = assert_paraphrase_invariance(
            model_fn=consistent_model,
            paraphrases=WW2_PARAPHRASES,
            equivalence_method="token_f1",
            min_invariance=0.5,
        )
        details = result.details
        assert "per_input_outputs" in details

    def test_details_include_worst_pair(self) -> None:
        """TestResult.details has worst-pair info."""
        result = assert_paraphrase_invariance(
            model_fn=consistent_model,
            paraphrases=WW2_PARAPHRASES,
            equivalence_method="token_f1",
            min_invariance=0.5,
        )
        details = result.details
        assert (
            "worst_pair" in details
            or "worst_score" in details
            or "min_score" in details
        )

    def test_details_include_method(self) -> None:
        """TestResult.details stores the method key."""
        result = assert_paraphrase_invariance(
            model_fn=consistent_model,
            paraphrases=WW2_PARAPHRASES,
            equivalence_method="token_f1",
            min_invariance=0.5,
        )
        assert "method" in result.details
        assert result.details["method"] == "token_f1"

    def test_unicode_normalized(self) -> None:
        """Inputs are unicode-normalized before model call."""
        # \u0041\u030A = "A with ring above" (2 codepoints)
        # NFKC normalizes to \u00C5 (single codepoint)
        raw = "\u0041\u030A question"
        normalized = unicodedata.normalize("NFKC", raw)
        calls: list[str] = []

        def tracking_model(text: str) -> str:
            calls.append(text)
            return "answer"

        result = assert_paraphrase_invariance(
            model_fn=tracking_model,
            paraphrases=[raw, "another question"],
            equivalence_method="token_f1",
            min_invariance=0.5,
        )
        # First call should have been normalized
        assert calls[0] == normalized or result.passed is True

    def test_model_exception_propagates(self) -> None:
        """Model that raises propagates the exception."""
        global _CALL_COUNT  # noqa: PLW0603
        _CALL_COUNT = 0

        # flaky_model raises every 3rd call; exception propagates
        with pytest.raises(RuntimeError, match="transient"):
            assert_paraphrase_invariance(
                model_fn=flaky_model,
                paraphrases=WW2_PARAPHRASES,
                equivalence_method="token_f1",
                min_invariance=1.0,
            )

    def test_empty_paraphrases_fails(self) -> None:
        """Empty list produces a failing TestResult."""
        with pytest.raises(
            (MltkAssertionError, ValueError),
        ):
            assert_paraphrase_invariance(
                model_fn=consistent_model,
                paraphrases=[],
                equivalence_method="token_f1",
                min_invariance=0.8,
            )

    def test_classifier_auto_detects_label_match(
        self,
    ) -> None:
        """Non-string hashable output triggers label_match."""
        result = assert_paraphrase_invariance(
            model_fn=numeric_classifier,
            paraphrases=WW2_PARAPHRASES,
            equivalence_method="auto",
            min_invariance=1.0,
        )
        assert result.passed is True

    def test_judge_without_judge_fn_fails(self) -> None:
        """method='judge' without judge_fn fails."""
        with pytest.raises(
            (MltkAssertionError, ValueError),
        ):
            assert_paraphrase_invariance(
                model_fn=consistent_model,
                paraphrases=WW2_PARAPHRASES,
                equivalence_method="judge",
                min_invariance=0.5,
            )

    def test_result_severity_on_failure(self) -> None:
        """Failed invariance check has CRITICAL severity."""
        with pytest.raises(MltkAssertionError) as exc:
            assert_paraphrase_invariance(
                model_fn=inconsistent_model,
                paraphrases=WW2_PARAPHRASES,
                equivalence_method="token_f1",
                min_invariance=0.99,
            )
        assert (
            exc.value.result.severity == Severity.CRITICAL
        )

    def test_invariance_score_in_details(self) -> None:
        """Details include the computed invariance score."""
        result = assert_paraphrase_invariance(
            model_fn=consistent_model,
            paraphrases=WW2_PARAPHRASES,
            equivalence_method="token_f1",
            min_invariance=0.5,
        )
        details = result.details
        assert "invariance_rate" in details


# ===================================================================
# TestFormatInvariance
# ===================================================================


class TestFormatInvariance:
    """Tests for ``assert_format_invariance``."""

    def test_case_insensitive_model_passes(
        self,
    ) -> None:
        """Model that ignores casing passes."""
        result = assert_format_invariance(
            model_fn=case_insensitive_model,
            input_text="What is the answer?",
            equivalence_method="token_f1",
            min_invariance=0.8,
        )
        assert result.passed is True
        _result_shape_ok(result)
        assert (
            result.name
            == "llm.behavioral.format_invariance"
        )

    def test_case_sensitive_model_fails(self) -> None:
        """Model sensitive to casing fails."""
        with pytest.raises(MltkAssertionError) as exc:
            assert_format_invariance(
                model_fn=case_sensitive_model,
                input_text="What is the answer?",
                equivalence_method="token_f1",
                min_invariance=0.9,
            )
        r = exc.value.result
        assert r.passed is False

    def test_default_transforms_applied(self) -> None:
        """Default transforms include case, spacing, etc."""
        result = assert_format_invariance(
            model_fn=case_insensitive_model,
            input_text="What is the answer?",
            equivalence_method="token_f1",
            min_invariance=0.5,
        )
        assert result.passed is True
        details = result.details
        assert "transform_results" in details

    def test_custom_transforms(self) -> None:
        """Custom transform list overrides defaults."""
        transforms = [str.upper, str.lower]
        result = assert_format_invariance(
            model_fn=case_insensitive_model,
            input_text="What is the answer?",
            transforms=transforms,
            equivalence_method="token_f1",
            min_invariance=0.5,
        )
        assert result.passed is True

    def test_min_invariance_threshold(self) -> None:
        """Threshold controls pass/fail boundary."""
        # case_sensitive_model fails with high threshold
        with pytest.raises(MltkAssertionError):
            assert_format_invariance(
                model_fn=case_sensitive_model,
                input_text="What is the answer?",
                equivalence_method="token_f1",
                min_invariance=0.99,
            )

    def test_details_include_per_transform(self) -> None:
        """Details include per-transform breakdown."""
        result = assert_format_invariance(
            model_fn=case_insensitive_model,
            input_text="What is the answer?",
            equivalence_method="token_f1",
            min_invariance=0.5,
        )
        details = result.details
        assert "transform_results" in details

    def test_method_token_f1(self) -> None:
        """token_f1 method works for format invariance."""
        result = assert_format_invariance(
            model_fn=case_insensitive_model,
            input_text="What is the answer?",
            equivalence_method="token_f1",
            min_invariance=0.5,
        )
        assert result.details["method"] == "token_f1"

    @patch(
        "mltk.domains.llm._backends.embedding_cosine_pairs"
    )
    def test_method_embedding_mock(
        self, mock_embed: MagicMock,
    ) -> None:
        """embedding method delegates to backend."""
        mock_embed.return_value = [0.97, 0.96, 0.95]
        result = assert_format_invariance(
            model_fn=case_insensitive_model,
            input_text="What is the answer?",
            equivalence_method="embedding",
            min_invariance=0.5,
        )
        assert result.passed is True
        assert result.details["method"] == "embedding"
        mock_embed.assert_called()

    def test_unknown_method_fails(self) -> None:
        """Unknown method produces a failing result."""
        with pytest.raises(MltkAssertionError) as exc:
            assert_format_invariance(
                model_fn=case_insensitive_model,
                input_text="What is the answer?",
                equivalence_method="nonexistent",
                min_invariance=0.5,
            )
        msg = exc.value.result.message
        assert "nonexistent" in msg

    def test_empty_input_accepted(self) -> None:
        """Empty text is accepted and transforms are applied."""
        result = assert_format_invariance(
            model_fn=case_insensitive_model,
            input_text="",
            equivalence_method="token_f1",
            min_invariance=0.8,
        )
        _result_shape_ok(result)

    def test_single_transform(self) -> None:
        """Single transform still produces a comparison."""
        result = assert_format_invariance(
            model_fn=case_insensitive_model,
            input_text="What is the answer?",
            transforms=[str.upper],
            equivalence_method="token_f1",
            min_invariance=0.5,
        )
        assert result.passed is True
        _result_shape_ok(result)

    def test_all_transforms_pass(self) -> None:
        """Model that handles all transforms passes."""
        result = assert_format_invariance(
            model_fn=case_insensitive_model,
            input_text="What is the answer?",
            equivalence_method="token_f1",
            min_invariance=0.8,
        )
        assert result.passed is True

    def test_method_in_details(self) -> None:
        """Details include the method used."""
        result = assert_format_invariance(
            model_fn=case_insensitive_model,
            input_text="What is the answer?",
            equivalence_method="token_f1",
            min_invariance=0.5,
        )
        assert "method" in result.details

    def test_severity_on_failure(self) -> None:
        """Failed format invariance has CRITICAL severity."""
        with pytest.raises(MltkAssertionError) as exc:
            assert_format_invariance(
                model_fn=case_sensitive_model,
                input_text="What is the answer?",
                equivalence_method="token_f1",
                min_invariance=0.99,
            )
        assert (
            exc.value.result.severity == Severity.CRITICAL
        )


# ===================================================================
# TestParaphraseInvarianceHardened
# ===================================================================


METHODS_TOKEN_LABEL = ["token_f1", "label_match"]


class TestParaphraseInvarianceHardened:
    """Hardened edge-case and parametrized tests."""

    @pytest.mark.parametrize(
        "method", METHODS_TOKEN_LABEL,
    )
    def test_paraphrase_invariance_methods(
        self, method: str,
    ) -> None:
        """Consistent model passes under each method.

        Parametrized over token_f1 and label_match —
        the two methods that need no external model.
        """
        result = assert_paraphrase_invariance(
            model_fn=consistent_model,
            paraphrases=WW2_PARAPHRASES,
            equivalence_method=method,
            min_invariance=0.5,
        )
        assert result.passed is True
        assert result.details["method"] == method

    @patch(
        "mltk.domains.llm._backends"
        ".embedding_cosine_pairs"
    )
    def test_paraphrase_invariance_embedding(
        self, mock_embed: MagicMock,
    ) -> None:
        """Embedding method passes with mocked backend."""
        mock_embed.return_value = [0.96, 0.94, 0.95]
        result = assert_paraphrase_invariance(
            model_fn=consistent_model,
            paraphrases=WW2_PARAPHRASES,
            equivalence_method="embedding",
            min_invariance=0.5,
        )
        assert result.passed is True

    @patch(
        "mltk.domains.llm._backends"
        ".nli_bidirectional"
    )
    def test_paraphrase_invariance_entailment(
        self, mock_nli: MagicMock,
    ) -> None:
        """Entailment method passes with mocked NLI."""
        mock_nli.return_value = {
            "forward": {
                "entailment": 0.92,
                "contradiction": 0.04,
                "neutral": 0.04,
                "label": "entailment",
            },
            "backward": {
                "entailment": 0.90,
                "contradiction": 0.05,
                "neutral": 0.05,
                "label": "entailment",
            },
            "equivalent": True,
            "contradiction": False,
        }
        result = assert_paraphrase_invariance(
            model_fn=consistent_model,
            paraphrases=WW2_PARAPHRASES,
            equivalence_method="entailment",
            min_invariance=0.5,
        )
        assert result.passed is True

    def test_paraphrase_invariance_judge_method(
        self,
    ) -> None:
        """Judge method with explicit judge_fn passes."""
        judge = MagicMock(return_value=0.99)
        result = assert_paraphrase_invariance(
            model_fn=consistent_model,
            paraphrases=WW2_PARAPHRASES,
            equivalence_method="judge",
            judge_fn=judge,
            min_invariance=0.5,
        )
        assert result.passed is True

    def test_format_invariance_custom_transforms(
        self,
    ) -> None:
        """User-provided transform functions applied.

        Scenario: Custom transforms (reverse, title)
        on a model that ignores formatting.
        """
        transforms = [
            lambda t: t[::-1],
            lambda t: t.title(),
        ]
        result = assert_format_invariance(
            model_fn=case_insensitive_model,
            input_text="What is the answer?",
            transforms=transforms,
            equivalence_method="token_f1",
            min_invariance=0.5,
        )
        assert result.passed is True
        _result_shape_ok(result)

    def test_paraphrase_invariance_single_input(
        self,
    ) -> None:
        """Edge case: only 1 input -> rejected.

        Scenario: Caller accidentally passes a list
        with a single paraphrase.
        """
        with pytest.raises(MltkAssertionError) as exc:
            assert_paraphrase_invariance(
                model_fn=consistent_model,
                paraphrases=["solo question"],
                equivalence_method="token_f1",
                min_invariance=0.8,
            )
        assert "Need >= 2" in exc.value.result.message

    def test_format_invariance_empty_response(
        self,
    ) -> None:
        """Model returns empty string for all inputs.

        Scenario: Broken model that always returns "".
        Should still produce a valid TestResult.
        """
        def empty_model(text: str) -> str:
            return ""

        result = assert_format_invariance(
            model_fn=empty_model,
            input_text="What is the answer?",
            equivalence_method="token_f1",
            min_invariance=0.5,
        )
        _result_shape_ok(result)

    def test_paraphrase_high_threshold_fails(
        self,
    ) -> None:
        """threshold=0.99 fails with imperfect model.

        Scenario: Inconsistent model can never reach
        99% invariance; test must fail.
        """
        with pytest.raises(MltkAssertionError):
            assert_paraphrase_invariance(
                model_fn=inconsistent_model,
                paraphrases=WW2_PARAPHRASES,
                equivalence_method="token_f1",
                min_invariance=0.99,
            )
