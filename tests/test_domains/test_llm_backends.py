"""Tests for mltk.domains.llm._backends — model-backed helpers.

Unit tests mock all external model calls. Integration tests that
actually load sentence-transformers are guarded with importorskip
and marked slow so they only run when the dependency is present.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from mltk.domains.llm._backends import (
    _NLI_LABELS,
    _softmax,
    normalize_unicode,
)


# ===================================================================
# normalize_unicode
# ===================================================================


class TestNormalizeUnicode:
    """NFKC normalization + invisible character stripping."""

    def test_nfkc_normalization(self) -> None:
        """Fullwidth chars are normalized to ASCII equiv."""
        # U+FF21 = fullwidth 'A', U+FF22 = fullwidth 'B'
        text = "\uff21\uff22\uff23"
        result = normalize_unicode(text)
        assert result == "ABC"

    def test_zero_width_stripped(self) -> None:
        """Zero-width chars are removed from text."""
        # U+200B = zero-width space
        # U+200C = zero-width non-joiner
        # U+200D = zero-width joiner
        text = "hel\u200blo\u200cwor\u200dld"
        result = normalize_unicode(text)
        assert result == "helloworld"

    def test_normal_text_unchanged(self) -> None:
        """Regular ASCII text passes through unchanged."""
        text = "Hello world 123"
        assert normalize_unicode(text) == text

    def test_preserves_newlines(self) -> None:
        """Newlines and tabs survive the Cf/Cc filter."""
        text = "line1\nline2\ttab"
        assert normalize_unicode(text) == text

    def test_bidi_override_stripped(self) -> None:
        """Bidirectional override chars (Cf) are removed."""
        # U+202E = right-to-left override
        # U+202C = pop directional formatting
        text = "abc\u202edef\u202cghi"
        result = normalize_unicode(text)
        assert result == "abcdefghi"

    def test_empty_string(self) -> None:
        """Empty string returns empty string."""
        assert normalize_unicode("") == ""

    def test_combining_chars_composed(self) -> None:
        """NFKC composes decomposed sequences."""
        # e + combining acute = e-acute
        decomposed = "e\u0301"
        result = normalize_unicode(decomposed)
        assert result == "\u00e9"  # e-acute composed


# ===================================================================
# _softmax
# ===================================================================


class TestSoftmax:
    """Softmax utility used by NLI scoring."""

    def test_softmax_sums_to_one(self) -> None:
        """Softmax output sums to approximately 1.0."""
        import numpy as np
        logits = [2.0, 1.0, 0.1]
        probs = _softmax(logits)
        assert sum(probs) == pytest.approx(1.0, abs=1e-6)

    def test_softmax_highest_input_wins(self) -> None:
        """Largest logit maps to largest probability."""
        import numpy as np
        logits = [10.0, 1.0, 0.1]
        probs = _softmax(logits)
        assert probs[0] > probs[1] > probs[2]

    def test_softmax_equal_inputs(self) -> None:
        """Equal logits produce uniform distribution."""
        probs = _softmax([1.0, 1.0, 1.0])
        assert probs[0] == pytest.approx(1.0 / 3, abs=1e-6)
        assert probs[1] == pytest.approx(1.0 / 3, abs=1e-6)

    def test_softmax_negative_logits(self) -> None:
        """Negative logits still produce valid probabilities."""
        probs = _softmax([-1.0, -2.0, -3.0])
        assert all(p > 0 for p in probs)
        assert sum(probs) == pytest.approx(1.0, abs=1e-6)


# ===================================================================
# NLI labels constant
# ===================================================================


class TestNLILabels:
    """Verify label ordering matches model expectations."""

    def test_label_count(self) -> None:
        """Exactly 3 NLI labels defined."""
        assert len(_NLI_LABELS) == 3

    def test_label_names(self) -> None:
        """Labels are contradiction, entailment, neutral."""
        assert _NLI_LABELS == [
            "contradiction",
            "entailment",
            "neutral",
        ]


# ===================================================================
# Embedding backend — import error path
# ===================================================================


class TestEmbeddingBackend:
    """Tests for embedding_cosine_single / _pairs."""

    @patch.dict(
        "sys.modules",
        {"sentence_transformers": None},
    )
    def test_import_error_message(self) -> None:
        """Clear error when sentence-transformers missing."""
        # Force-clear the lru_cache so the mock takes effect
        from mltk.domains.llm._backends import (
            _load_sentence_model,
        )
        _load_sentence_model.cache_clear()
        with pytest.raises(ImportError, match="mltk"):
            _load_sentence_model("all-MiniLM-L6-v2")

    @patch(
        "mltk.domains.llm._backends._load_sentence_model"
    )
    def test_cosine_identical_texts_mock(
        self, mock_load: MagicMock,
    ) -> None:
        """Identical texts get score ~1.0 via mock model."""
        import numpy as np
        rng = np.random.default_rng(42)
        vec = rng.standard_normal(384).astype(np.float32)

        mock_model = MagicMock()
        mock_model.encode.return_value = np.array(
            [vec],
        )
        mock_load.return_value = mock_model

        from mltk.domains.llm._backends import (
            embedding_cosine_single,
        )
        score = embedding_cosine_single(
            "hello world", "hello world",
        )
        assert score == pytest.approx(1.0, abs=0.01)

    @patch(
        "mltk.domains.llm._backends._load_sentence_model"
    )
    def test_cosine_orthogonal_texts_mock(
        self, mock_load: MagicMock,
    ) -> None:
        """Orthogonal vectors produce score ~0.0."""
        import numpy as np
        rng = np.random.default_rng(42)
        vec_a = np.zeros(384, dtype=np.float32)
        vec_a[0] = 1.0
        vec_b = np.zeros(384, dtype=np.float32)
        vec_b[1] = 1.0

        mock_model = MagicMock()
        mock_model.encode.side_effect = [
            np.array([vec_a]),
            np.array([vec_b]),
        ]
        mock_load.return_value = mock_model

        from mltk.domains.llm._backends import (
            embedding_cosine_pairs,
        )
        scores = embedding_cosine_pairs(
            ["text a"], ["text b"],
        )
        assert scores[0] == pytest.approx(0.0, abs=0.01)

    @patch(
        "mltk.domains.llm._backends._load_sentence_model"
    )
    def test_cosine_zero_norm_returns_zero(
        self, mock_load: MagicMock,
    ) -> None:
        """Zero-norm vector produces score 0.0."""
        import numpy as np
        zero_vec = np.zeros(384, dtype=np.float32)
        real_vec = np.ones(384, dtype=np.float32)

        mock_model = MagicMock()
        mock_model.encode.side_effect = [
            np.array([zero_vec]),
            np.array([real_vec]),
        ]
        mock_load.return_value = mock_model

        from mltk.domains.llm._backends import (
            embedding_cosine_pairs,
        )
        scores = embedding_cosine_pairs(
            ["empty"], ["real"],
        )
        assert scores[0] == 0.0


# ===================================================================
# NLI backend — import error path
# ===================================================================


class TestNLIBackend:
    """Tests for nli_entailment_score / nli_bidirectional."""

    @patch.dict(
        "sys.modules",
        {"sentence_transformers": None},
    )
    def test_import_error_message(self) -> None:
        """Clear error when sentence-transformers missing."""
        from mltk.domains.llm._backends import (
            _load_nli_model,
        )
        _load_nli_model.cache_clear()
        with pytest.raises(ImportError, match="mltk"):
            _load_nli_model(
                "cross-encoder/nli-deberta-v3-base"
            )

    @patch(
        "mltk.domains.llm._backends._load_nli_model"
    )
    def test_entailment_high_score_mock(
        self, mock_load: MagicMock,
    ) -> None:
        """High entailment logit yields entailment label."""
        import numpy as np
        # logits: [contradiction, entailment, neutral]
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array(
            [[-2.0, 5.0, -1.0]],
        )
        mock_load.return_value = mock_model

        from mltk.domains.llm._backends import (
            nli_entailment_score,
        )
        result = nli_entailment_score(
            "Paris is in France",
            "Paris is a city in France",
        )
        assert result["label"] == "entailment"
        assert result["entailment"] > 0.9

    @patch(
        "mltk.domains.llm._backends._load_nli_model"
    )
    def test_contradiction_score_mock(
        self, mock_load: MagicMock,
    ) -> None:
        """High contradiction logit yields that label."""
        import numpy as np
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array(
            [[5.0, -2.0, -1.0]],
        )
        mock_load.return_value = mock_model

        from mltk.domains.llm._backends import (
            nli_entailment_score,
        )
        result = nli_entailment_score(
            "The sky is blue", "The sky is green",
        )
        assert result["label"] == "contradiction"
        assert result["contradiction"] > 0.9

    @patch(
        "mltk.domains.llm._backends._load_nli_model"
    )
    def test_bidirectional_equivalent_mock(
        self, mock_load: MagicMock,
    ) -> None:
        """Bidirectional entailment detects equivalence."""
        import numpy as np
        mock_model = MagicMock()
        # Both directions: high entailment
        mock_model.predict.return_value = np.array(
            [[-2.0, 5.0, -1.0]],
        )
        mock_load.return_value = mock_model

        from mltk.domains.llm._backends import (
            nli_bidirectional,
        )
        result = nli_bidirectional(
            "Cats are animals", "Cats are animals",
        )
        assert result["equivalent"] is True
        assert result["contradiction"] is False
