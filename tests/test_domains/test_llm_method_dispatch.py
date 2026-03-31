"""Tests for S67 method dispatch — hallucination + RAG assertions.

Backward compatibility freeze tests ensure the refactored assertions
produce identical results when called without the new ``method`` param.
Method variant tests verify each dispatch path (embedding, nli, llm)
produces valid TestResult objects with the expected structure.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.core.result import TestResult
from mltk.domains.llm.rag import (
    assert_answer_relevancy,
    assert_context_relevancy,
    assert_faithfulness,
)
from mltk.domains.llm.safety import (
    _HALLUCINATION_METHODS,
    assert_no_hallucination,
)

# -- Shared fixtures ------------------------------------------------

CLAIMS = ["Paris is the capital of France"]
SOURCES = [
    "France is a country in Europe. Its capital is Paris."
]
CTX = (
    "The Eiffel Tower is in Paris France."
    " It was built in 1889 by Gustave Eiffel."
)
QUESTION = "Where is the Eiffel Tower located?"
ANSWER = "The Eiffel Tower is located in Paris France."


def _result_shape_ok(result: TestResult) -> None:
    """Assert TestResult has the mandatory fields."""
    assert isinstance(result.name, str)
    assert isinstance(result.passed, bool)
    assert isinstance(result.message, str)
    assert isinstance(result.details, dict)
    assert result.duration_ms >= 0.0


# ===================================================================
# Backward compatibility freeze tests
# ===================================================================


class TestBackwardCompatibility:
    """Freeze tests: refactored assertions must match old behavior."""

    # -- assert_no_hallucination -----------------------------------

    def test_hallucination_no_method_param(self) -> None:
        """Calling without method= works exactly as before."""
        result = assert_no_hallucination(
            CLAIMS, SOURCES, min_coverage=0.3,
        )
        assert result.passed is True
        _result_shape_ok(result)
        assert result.name == "llm.hallucination"
        assert result.details["total_claims"] == 1
        assert result.details["avg_coverage"] >= 0.3

    def test_hallucination_lexical_matches_default(
        self,
    ) -> None:
        """method='lexical' produces same result as default."""
        default = assert_no_hallucination(
            CLAIMS, SOURCES, min_coverage=0.3,
        )
        explicit = assert_no_hallucination(
            CLAIMS, SOURCES, method="lexical",
            min_coverage=0.3,
        )
        assert default.passed == explicit.passed
        assert (
            default.details["avg_coverage"]
            == explicit.details["avg_coverage"]
        )
        assert (
            default.details["unsupported_count"]
            == explicit.details["unsupported_count"]
        )

    def test_hallucination_overlap_alias(self) -> None:
        """method='overlap' is accepted as backward-compat alias."""
        result = assert_no_hallucination(
            CLAIMS, SOURCES, method="overlap",
            min_coverage=0.3,
        )
        assert result.passed is True
        # After alias resolution, method stored as "lexical"
        assert result.details["method"] == "lexical"

    def test_hallucination_default_method_is_lexical(
        self,
    ) -> None:
        """Default call stores method='lexical' in details."""
        result = assert_no_hallucination(
            CLAIMS, SOURCES, min_coverage=0.3,
        )
        assert result.details["method"] == "lexical"

    def test_hallucination_result_structure(self) -> None:
        """TestResult has all expected detail keys."""
        result = assert_no_hallucination(
            CLAIMS, SOURCES, min_coverage=0.3,
        )
        expected_keys = {
            "unsupported_count",
            "total_claims",
            "avg_coverage",
            "min_coverage",
            "method",
        }
        assert expected_keys.issubset(result.details.keys())

    # -- assert_faithfulness ---------------------------------------

    def test_faithfulness_no_method_param(self) -> None:
        """Calling without method= works exactly as before."""
        result = assert_faithfulness(
            ANSWER, CTX, min_score=0.5,
        )
        assert result.passed is True
        _result_shape_ok(result)
        assert result.name == "llm.rag.faithfulness"
        assert result.details["score"] >= 0.5

    def test_faithfulness_result_structure(self) -> None:
        """TestResult has all expected detail keys."""
        result = assert_faithfulness(
            ANSWER, CTX, min_score=0.5,
        )
        expected_keys = {
            "score",
            "min_score",
            "answer_tokens",
            "context_tokens",
            "grounded_tokens",
        }
        assert expected_keys.issubset(result.details.keys())

    # -- assert_context_relevancy ----------------------------------

    def test_context_relevancy_no_method_param(self) -> None:
        """Calling without method= works exactly as before."""
        result = assert_context_relevancy(
            QUESTION, CTX, min_score=0.3,
        )
        assert result.passed is True
        _result_shape_ok(result)
        assert result.name == "llm.rag.context_relevancy"

    def test_context_relevancy_result_structure(
        self,
    ) -> None:
        """TestResult has all expected detail keys."""
        result = assert_context_relevancy(
            QUESTION, CTX, min_score=0.3,
        )
        expected_keys = {
            "score",
            "min_score",
            "question_tokens",
            "context_tokens",
            "matched_tokens",
        }
        assert expected_keys.issubset(result.details.keys())

    # -- assert_answer_relevancy -----------------------------------

    def test_answer_relevancy_no_method_param(self) -> None:
        """Calling without method= works exactly as before."""
        result = assert_answer_relevancy(
            QUESTION, ANSWER, min_score=0.3,
        )
        assert result.passed is True
        _result_shape_ok(result)
        assert result.name == "llm.rag.answer_relevancy"

    def test_answer_relevancy_result_structure(
        self,
    ) -> None:
        """TestResult has all expected detail keys."""
        result = assert_answer_relevancy(
            QUESTION, ANSWER, min_score=0.3,
        )
        expected_keys = {
            "score",
            "min_score",
            "question_tokens",
            "answer_tokens",
            "matched_tokens",
        }
        assert expected_keys.issubset(result.details.keys())


# ===================================================================
# Method dispatch tests
# ===================================================================


class TestMethodDispatch:
    """Test each method variant produces valid TestResult."""

    # -- Unknown method --------------------------------------------

    def test_hallucination_unknown_method(self) -> None:
        """Unknown method returns failing result with hint."""
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_hallucination(
                CLAIMS, SOURCES, method="bogus",
            )
        msg = exc.value.result.message
        assert "bogus" in msg
        # Message should list supported methods
        for m in _HALLUCINATION_METHODS:
            assert m in msg

    def test_hallucination_unknown_preserves_severity(
        self,
    ) -> None:
        """Unknown method result is CRITICAL severity."""
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_hallucination(
                CLAIMS, SOURCES, method="invalid",
            )
        from mltk.core.result import Severity
        assert (
            exc.value.result.severity == Severity.CRITICAL
        )

    # -- Embedding method ------------------------------------------

    @patch(
        "mltk.domains.llm._backends.embedding_cosine_single"
    )
    def test_hallucination_embedding_method(
        self, mock_embed: MagicMock,
    ) -> None:
        """method='embedding' dispatches to cosine backend."""
        mock_embed.return_value = 0.85
        result = assert_no_hallucination(
            CLAIMS, SOURCES, method="embedding",
            min_coverage=0.5,
        )
        assert result.passed is True
        assert result.details["method"] == "embedding"
        assert result.details["avg_coverage"] == pytest.approx(
            0.85, abs=0.01,
        )
        mock_embed.assert_called_once()

    @patch(
        "mltk.domains.llm._backends.embedding_cosine_single"
    )
    def test_hallucination_embedding_fail(
        self, mock_embed: MagicMock,
    ) -> None:
        """method='embedding' fails when score < min_coverage."""
        mock_embed.return_value = 0.1
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_hallucination(
                CLAIMS, SOURCES, method="embedding",
                min_coverage=0.5,
            )
        r = exc.value.result
        assert r.details["unsupported_count"] == 1

    # -- NLI method ------------------------------------------------

    @patch(
        "mltk.domains.llm._backends.nli_entailment_score"
    )
    def test_hallucination_nli_method(
        self, mock_nli: MagicMock,
    ) -> None:
        """method='nli' dispatches to entailment backend."""
        mock_nli.return_value = {
            "contradiction": 0.05,
            "entailment": 0.90,
            "neutral": 0.05,
            "label": "entailment",
        }
        result = assert_no_hallucination(
            CLAIMS, SOURCES, method="nli",
            min_coverage=0.5,
        )
        assert result.passed is True
        assert result.details["method"] == "nli"
        assert result.details["avg_coverage"] == pytest.approx(
            0.90, abs=0.01,
        )
        mock_nli.assert_called_once()

    @patch(
        "mltk.domains.llm._backends.nli_entailment_score"
    )
    def test_hallucination_nli_fail(
        self, mock_nli: MagicMock,
    ) -> None:
        """method='nli' fails on low entailment score."""
        mock_nli.return_value = {
            "contradiction": 0.70,
            "entailment": 0.10,
            "neutral": 0.20,
            "label": "contradiction",
        }
        with pytest.raises(MltkAssertionError):
            assert_no_hallucination(
                CLAIMS, SOURCES, method="nli",
                min_coverage=0.5,
            )

    # -- LLM (judge_fn) method ------------------------------------

    def test_hallucination_llm_method(self) -> None:
        """method='llm' uses supplied judge_fn."""
        judge = MagicMock(return_value=0.95)
        result = assert_no_hallucination(
            CLAIMS, SOURCES, method="llm",
            judge_fn=judge, min_coverage=0.5,
        )
        assert result.passed is True
        assert result.details["method"] == "llm"
        judge.assert_called_once()

    def test_hallucination_llm_without_judge_fn(
        self,
    ) -> None:
        """method='llm' without judge_fn fails immediately."""
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_hallucination(
                CLAIMS, SOURCES, method="llm",
            )
        msg = exc.value.result.message
        assert "judge_fn" in msg

    def test_hallucination_llm_low_score(self) -> None:
        """method='llm' fails when judge gives low score."""
        judge = MagicMock(return_value=0.1)
        with pytest.raises(MltkAssertionError):
            assert_no_hallucination(
                CLAIMS, SOURCES, method="llm",
                judge_fn=judge, min_coverage=0.5,
            )

    # -- Lexical (explicit) ----------------------------------------

    def test_hallucination_lexical_explicit(self) -> None:
        """method='lexical' is explicitly accepted."""
        result = assert_no_hallucination(
            CLAIMS, SOURCES, method="lexical",
            min_coverage=0.3,
        )
        assert result.passed is True
        assert result.details["method"] == "lexical"


# ===================================================================
# Method key in details
# ===================================================================


class TestMethodInDetails:
    """Every hallucination method stores 'method' in details."""

    def test_lexical_has_method_key(self) -> None:
        """Lexical path stores method in details."""
        result = assert_no_hallucination(
            CLAIMS, SOURCES, method="lexical",
            min_coverage=0.3,
        )
        assert "method" in result.details
        assert result.details["method"] == "lexical"

    def test_overlap_resolves_to_lexical(self) -> None:
        """Overlap alias resolves to 'lexical' in details."""
        result = assert_no_hallucination(
            CLAIMS, SOURCES, method="overlap",
            min_coverage=0.3,
        )
        assert result.details["method"] == "lexical"

    @patch(
        "mltk.domains.llm._backends.embedding_cosine_single"
    )
    def test_embedding_has_method_key(
        self, mock_embed: MagicMock,
    ) -> None:
        """Embedding path stores method in details."""
        mock_embed.return_value = 0.9
        result = assert_no_hallucination(
            CLAIMS, SOURCES, method="embedding",
            min_coverage=0.3,
        )
        assert result.details["method"] == "embedding"

    @patch(
        "mltk.domains.llm._backends.nli_entailment_score"
    )
    def test_nli_has_method_key(
        self, mock_nli: MagicMock,
    ) -> None:
        """NLI path stores method in details."""
        mock_nli.return_value = {
            "contradiction": 0.05,
            "entailment": 0.90,
            "neutral": 0.05,
            "label": "entailment",
        }
        result = assert_no_hallucination(
            CLAIMS, SOURCES, method="nli",
            min_coverage=0.3,
        )
        assert result.details["method"] == "nli"

    def test_llm_has_method_key(self) -> None:
        """LLM path stores method in details."""
        judge = MagicMock(return_value=0.9)
        result = assert_no_hallucination(
            CLAIMS, SOURCES, method="llm",
            judge_fn=judge, min_coverage=0.3,
        )
        assert result.details["method"] == "llm"


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    """Edge cases for method dispatch paths."""

    def test_empty_claims_all_methods(self) -> None:
        """Empty claims list passes for lexical method."""
        result = assert_no_hallucination(
            [], SOURCES, method="lexical",
            min_coverage=0.5,
        )
        assert result.passed is True
        assert result.details["total_claims"] == 0

    @patch(
        "mltk.domains.llm._backends.embedding_cosine_single"
    )
    def test_whitespace_claim_skipped_embedding(
        self, mock_embed: MagicMock,
    ) -> None:
        """Whitespace-only claims are skipped in embedding."""
        mock_embed.return_value = 0.9
        result = assert_no_hallucination(
            ["  ", "Paris is in France"], SOURCES,
            method="embedding", min_coverage=0.3,
        )
        # Only non-blank claim processed
        mock_embed.assert_called_once()
        assert result.passed is True

    @patch(
        "mltk.domains.llm._backends.nli_entailment_score"
    )
    def test_whitespace_claim_skipped_nli(
        self, mock_nli: MagicMock,
    ) -> None:
        """Whitespace-only claims are skipped in nli."""
        mock_nli.return_value = {
            "contradiction": 0.0,
            "entailment": 0.95,
            "neutral": 0.05,
            "label": "entailment",
        }
        result = assert_no_hallucination(
            ["  ", "Paris is in France"], SOURCES,
            method="nli", min_coverage=0.3,
        )
        mock_nli.assert_called_once()
        assert result.passed is True

    def test_whitespace_claim_skipped_llm(self) -> None:
        """Whitespace-only claims are skipped in llm."""
        judge = MagicMock(return_value=0.9)
        result = assert_no_hallucination(
            ["  ", "Paris is in France"], SOURCES,
            method="llm", judge_fn=judge,
            min_coverage=0.3,
        )
        judge.assert_called_once()
        assert result.passed is True

    def test_multiple_claims_lexical(self) -> None:
        """Multiple claims: one supported, one not."""
        claims = [
            "Paris is the capital of France",
            "Jupiter has 79 moons orbiting retrograde",
        ]
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_hallucination(
                claims, SOURCES, method="lexical",
                min_coverage=0.5,
            )
        r = exc.value.result
        assert r.details["unsupported_count"] >= 1
        assert r.details["total_claims"] == 2

    def test_message_includes_method_name(self) -> None:
        """Result message mentions which method was used."""
        result = assert_no_hallucination(
            CLAIMS, SOURCES, method="lexical",
            min_coverage=0.3,
        )
        assert "method=lexical" in result.message


# ===================================================================
# Hardening: parametrized + edge-case tests (appended)
# ===================================================================


class TestHardeningMethodDispatch:
    """Extra edge-case and parametrized tests."""

    def test_hallucination_lexical_exact_match(
        self,
    ) -> None:
        """Claim equals source exactly -> passes."""
        text = "Paris is the capital of France"
        result = assert_no_hallucination(
            [text], [text], method="lexical",
            min_coverage=0.3,
        )
        assert result.passed is True
        assert result.details["avg_coverage"] >= 0.99

    def test_hallucination_lexical_no_overlap(
        self,
    ) -> None:
        """Zero token overlap -> fails."""
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_hallucination(
                ["alpha bravo charlie"],
                ["delta echo foxtrot"],
                method="lexical",
                min_coverage=0.3,
            )
        r = exc.value.result
        assert r.details["unsupported_count"] >= 1
        assert r.details["avg_coverage"] < 0.3

    @pytest.mark.parametrize(
        "method",
        ["lexical", "embedding", "nli", "llm"],
    )
    def test_faithfulness_all_methods_parametrized(
        self, method: str,
    ) -> None:
        """Each faithfulness method accepts valid input."""
        if method == "embedding":
            with patch(
                "mltk.domains.llm._backends"
                ".embedding_cosine_single",
                return_value=0.9,
            ):
                result = assert_faithfulness(
                    ANSWER, CTX, min_score=0.3,
                    method=method,
                )
        elif method == "nli":
            mock_ret = {
                "contradiction": 0.02,
                "entailment": 0.93,
                "neutral": 0.05,
                "label": "entailment",
            }
            with patch(
                "mltk.domains.llm._backends"
                ".nli_entailment_score",
                return_value=mock_ret,
            ):
                result = assert_faithfulness(
                    ANSWER, CTX, min_score=0.3,
                    method=method,
                )
        elif method == "llm":
            judge = MagicMock(return_value=0.9)
            result = assert_faithfulness(
                ANSWER, CTX, min_score=0.3,
                method=method, judge_fn=judge,
            )
        else:
            result = assert_faithfulness(
                ANSWER, CTX, min_score=0.3,
                method=method,
            )
        assert result.passed is True
        _result_shape_ok(result)

    def test_context_relevancy_empty_context(
        self,
    ) -> None:
        """Empty context string -> fails gracefully."""
        with pytest.raises(MltkAssertionError) as exc:
            assert_context_relevancy(
                QUESTION, "", min_score=0.0,
            )
        r = exc.value.result
        _result_shape_ok(r)
        assert "empty" in r.message.lower()

    def test_answer_relevancy_long_text(self) -> None:
        """1000+ word response still works."""
        rng = __import__("random").Random(42)
        words = [
            "the", "model", "explains", "quantum",
            "physics", "thoroughly", "in", "detail",
            "with", "examples",
        ]
        long_answer = " ".join(
            rng.choice(words) for _ in range(1100)
        )
        result = assert_answer_relevancy(
            QUESTION, long_answer, min_score=0.0,
        )
        _result_shape_ok(result)

    def test_hallucination_unicode_input(self) -> None:
        """Japanese/Arabic text does not crash."""
        claims_uni = [
            "\u6771\u4eac\u306f\u65e5\u672c\u306e"
            "\u9996\u90fd\u3067\u3059",
        ]
        sources_uni = [
            "\u0627\u0644\u0642\u0627\u0647\u0631\u0629 "
            "\u0647\u064a \u0639\u0627\u0635\u0645\u0629 "
            "\u0645\u0635\u0631",
        ]
        with pytest.raises(MltkAssertionError):
            assert_no_hallucination(
                claims_uni, sources_uni,
                method="lexical", min_coverage=0.5,
            )

    def test_method_invalid_raises(self) -> None:
        """method='invalid' raises MltkAssertionError."""
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_hallucination(
                CLAIMS, SOURCES, method="invalid",
            )
        msg = exc.value.result.message
        assert "invalid" in msg

    def test_faithfulness_multiple_contexts(
        self,
    ) -> None:
        """List of context strings joined properly."""
        contexts = [
            "The Eiffel Tower is in Paris.",
            "It was built in 1889.",
        ]
        result = assert_faithfulness(
            "The Eiffel Tower is in Paris.",
            contexts, min_score=0.3,
        )
        assert result.passed is True

    @patch(
        "mltk.domains.llm._backends"
        ".embedding_cosine_single",
    )
    def test_hallucination_embedding_mock(
        self, mock_embed: MagicMock,
    ) -> None:
        """Embedding mock returns controlled score."""
        mock_embed.return_value = 0.72
        result = assert_no_hallucination(
            CLAIMS, SOURCES, method="embedding",
            min_coverage=0.5,
        )
        assert result.passed is True
        assert result.details["avg_coverage"] == (
            pytest.approx(0.72, abs=0.01)
        )

    @patch(
        "mltk.domains.llm._backends"
        ".nli_entailment_score",
    )
    def test_answer_relevancy_nli_mock(
        self, mock_nli: MagicMock,
    ) -> None:
        """NLI mock for answer relevancy dispatch."""
        mock_nli.return_value = {
            "contradiction": 0.02,
            "entailment": 0.92,
            "neutral": 0.06,
            "label": "entailment",
        }
        result = assert_answer_relevancy(
            QUESTION, ANSWER, min_score=0.3,
            method="nli",
        )
        assert result.passed is True
        assert result.details["method"] == "nli"
