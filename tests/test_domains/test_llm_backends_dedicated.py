"""Dedicated tests for mltk.domains.llm._backends.

This module is the FOUNDATION for all method dispatch:
model loading, embedding cosine, NLI scoring, and caching.
If it breaks, every behavioral assertion breaks.

All external model calls are mocked.  No network, no
time.sleep, fixed seeds throughout.
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mltk.domains.llm._backends import (
    _MODEL_REVISIONS,
    _NLI_LABELS,
    _softmax,
    embedding_cosine_pairs,
    embedding_cosine_single,
    nli_bidirectional,
    nli_entailment_score,
    normalize_unicode,
)

SEED = 42
RNG = np.random.default_rng(SEED)


# ================================================================
# Model revisions (SEC-2 supply-chain defense)
# ================================================================


class TestModelRevisions:
    """Pinned revisions for known models."""

    def test_revision_dict_has_minilm(self) -> None:
        """MiniLM revision is present."""
        assert "all-MiniLM-L6-v2" in _MODEL_REVISIONS

    def test_revision_dict_has_deberta(self) -> None:
        """DeBERTa NLI revision is present."""
        key = "cross-encoder/nli-deberta-v3-base"
        assert key in _MODEL_REVISIONS

    def test_revision_format_is_hex(self) -> None:
        """All revisions are lowercase hex strings."""
        hex_re = re.compile(r"^[0-9a-f]+$")
        for model, rev in _MODEL_REVISIONS.items():
            assert hex_re.match(rev), (
                f"{model}: '{rev}' is not hex"
            )

    def test_revision_length_reasonable(self) -> None:
        """Revisions are >= 8 hex chars (short SHA)."""
        for model, rev in _MODEL_REVISIONS.items():
            assert len(rev) >= 8, (
                f"{model}: revision too short"
            )


# ================================================================
# LRU caching behavior
# ================================================================


class TestLruCaching:
    """Model caching avoids redundant loads."""

    @patch(
        "mltk.domains.llm._backends.SentenceTransformer",
        create=True,
    )
    def test_sentence_model_cached(
        self, mock_st: MagicMock,
    ) -> None:
        """Second call reuses cache (1 init only)."""
        from mltk.domains.llm._backends import (
            _load_sentence_model,
        )
        _load_sentence_model.cache_clear()
        mock_st.return_value = MagicMock()

        with patch(
            "mltk.domains.llm._backends."
            "SentenceTransformer",
            mock_st,
            create=True,
        ):
            import mltk.domains.llm._backends as be
            be._load_sentence_model.cache_clear()

            # Patch the import inside the function
            with patch.dict("sys.modules", {
                "sentence_transformers": MagicMock(
                    SentenceTransformer=mock_st,
                ),
            }):
                be._load_sentence_model.cache_clear()
                be._load_sentence_model("test-model")
                be._load_sentence_model("test-model")
                assert mock_st.call_count == 1
                be._load_sentence_model.cache_clear()

    @patch(
        "mltk.domains.llm._backends.CrossEncoder",
        create=True,
    )
    def test_nli_model_cached(
        self, mock_ce: MagicMock,
    ) -> None:
        """NLI model: second call reuses cache."""
        from mltk.domains.llm._backends import (
            _load_nli_model,
        )
        _load_nli_model.cache_clear()

        with patch.dict("sys.modules", {
            "sentence_transformers": MagicMock(
                CrossEncoder=mock_ce,
            ),
        }):
            _load_nli_model.cache_clear()
            _load_nli_model("test-nli")
            _load_nli_model("test-nli")
            assert mock_ce.call_count == 1
            _load_nli_model.cache_clear()


# ================================================================
# Embedding cosine similarity
# ================================================================


class TestEmbeddingCosine:
    """Embedding similarity computation."""

    @patch(
        "mltk.domains.llm._backends"
        "._load_sentence_model",
    )
    def test_identical_texts_high_score(
        self, mock_load: MagicMock,
    ) -> None:
        """Same text -> score near 1.0."""
        rng = np.random.default_rng(SEED)
        vec = rng.standard_normal(384).astype(
            np.float32,
        )
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array(
            [vec],
        )
        mock_load.return_value = mock_model

        score = embedding_cosine_single(
            "hello world", "hello world",
        )
        assert score == pytest.approx(1.0, abs=0.01)

    @patch(
        "mltk.domains.llm._backends"
        "._load_sentence_model",
    )
    def test_orthogonal_vectors_zero(
        self, mock_load: MagicMock,
    ) -> None:
        """Orthogonal embeddings -> 0.0."""
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

        scores = embedding_cosine_pairs(
            ["a"], ["b"],
        )
        assert scores[0] == pytest.approx(
            0.0, abs=0.01,
        )

    @patch(
        "mltk.domains.llm._backends"
        "._load_sentence_model",
    )
    def test_zero_vector_handled(
        self, mock_load: MagicMock,
    ) -> None:
        """Zero-norm vector -> 0.0, not NaN."""
        zero = np.zeros(384, dtype=np.float32)
        real = np.ones(384, dtype=np.float32)

        mock_model = MagicMock()
        mock_model.encode.side_effect = [
            np.array([zero]),
            np.array([real]),
        ]
        mock_load.return_value = mock_model

        scores = embedding_cosine_pairs(
            ["empty"], ["real"],
        )
        assert scores[0] == 0.0
        assert not np.isnan(scores[0])

    @patch(
        "mltk.domains.llm._backends"
        "._load_sentence_model",
    )
    def test_strict_zip_mismatch_raises(
        self, mock_load: MagicMock,
    ) -> None:
        """Mismatched list lengths raises ValueError."""
        rng = np.random.default_rng(SEED)
        v1 = rng.standard_normal(384).astype(
            np.float32,
        )
        v2 = rng.standard_normal(384).astype(
            np.float32,
        )

        mock_model = MagicMock()
        mock_model.encode.side_effect = [
            np.array([v1]),
            np.array([v2, v2]),
        ]
        mock_load.return_value = mock_model

        with pytest.raises(ValueError, match="longer"):
            embedding_cosine_pairs(
                ["one"], ["two", "three"],
            )

    @patch(
        "mltk.domains.llm._backends"
        "._load_sentence_model",
    )
    def test_multiple_pairs_batch(
        self, mock_load: MagicMock,
    ) -> None:
        """Multiple pairs scored in one call."""
        rng = np.random.default_rng(SEED)
        va = rng.standard_normal(
            (3, 384),
        ).astype(np.float32)
        vb = rng.standard_normal(
            (3, 384),
        ).astype(np.float32)

        mock_model = MagicMock()
        mock_model.encode.side_effect = [va, vb]
        mock_load.return_value = mock_model

        scores = embedding_cosine_pairs(
            ["a", "b", "c"],
            ["d", "e", "f"],
        )
        assert len(scores) == 3
        for s in scores:
            assert -1.01 <= s <= 1.01

    @patch(
        "mltk.domains.llm._backends"
        "._load_sentence_model",
    )
    def test_negative_cosine_possible(
        self, mock_load: MagicMock,
    ) -> None:
        """Opposing vectors can produce negative score."""
        vec_a = np.ones(384, dtype=np.float32)
        vec_b = -np.ones(384, dtype=np.float32)

        mock_model = MagicMock()
        mock_model.encode.side_effect = [
            np.array([vec_a]),
            np.array([vec_b]),
        ]
        mock_load.return_value = mock_model

        scores = embedding_cosine_pairs(
            ["pos"], ["neg"],
        )
        assert scores[0] == pytest.approx(
            -1.0, abs=0.01,
        )


# ================================================================
# NLI (Natural Language Inference)
# ================================================================


class TestNLI:
    """NLI entailment computation."""

    @patch(
        "mltk.domains.llm._backends._load_nli_model",
    )
    def test_entailment_scores_sum_to_one(
        self, mock_load: MagicMock,
    ) -> None:
        """Softmax outputs sum to ~1.0."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array(
            [[-1.0, 3.0, 0.5]],
        )
        mock_load.return_value = mock_model

        result = nli_entailment_score("a", "b")
        total = (
            result["contradiction"]
            + result["entailment"]
            + result["neutral"]
        )
        assert total == pytest.approx(1.0, abs=1e-5)

    @patch(
        "mltk.domains.llm._backends._load_nli_model",
    )
    def test_bidirectional_symmetric_identical(
        self, mock_load: MagicMock,
    ) -> None:
        """Identical texts -> equivalent=True."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array(
            [[-3.0, 5.0, -1.0]],
        )
        mock_load.return_value = mock_model

        result = nli_bidirectional("same", "same")
        assert result["equivalent"] is True
        assert result["contradiction"] is False

    @patch(
        "mltk.domains.llm._backends._load_nli_model",
    )
    def test_contradiction_detected(
        self, mock_load: MagicMock,
    ) -> None:
        """Contradictory mock -> contradiction=True."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array(
            [[5.0, -3.0, -1.0]],
        )
        mock_load.return_value = mock_model

        result = nli_bidirectional(
            "The sky is blue",
            "The sky is green",
        )
        assert result["contradiction"] is True
        assert result["equivalent"] is False

    def test_label_order_correct(self) -> None:
        """Labels match nli-deberta-v3-base convention."""
        assert _NLI_LABELS[0] == "contradiction"
        assert _NLI_LABELS[1] == "entailment"
        assert _NLI_LABELS[2] == "neutral"

    @patch(
        "mltk.domains.llm._backends._load_nli_model",
    )
    def test_entailment_label_selected(
        self, mock_load: MagicMock,
    ) -> None:
        """Highest logit selects the label."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array(
            [[-2.0, 6.0, -1.0]],
        )
        mock_load.return_value = mock_model

        result = nli_entailment_score("p", "h")
        assert result["label"] == "entailment"

    @patch(
        "mltk.domains.llm._backends._load_nli_model",
    )
    def test_neutral_label_selected(
        self, mock_load: MagicMock,
    ) -> None:
        """Neutral wins when it has highest logit."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array(
            [[-2.0, -1.0, 6.0]],
        )
        mock_load.return_value = mock_model

        result = nli_entailment_score("p", "h")
        assert result["label"] == "neutral"

    @patch(
        "mltk.domains.llm._backends._load_nli_model",
    )
    def test_bidirectional_asymmetric(
        self, mock_load: MagicMock,
    ) -> None:
        """Forward entails but backward neutral -> not equiv."""
        mock_model = MagicMock()
        # Forward: entailment wins
        # Backward: neutral wins
        mock_model.predict.side_effect = [
            np.array([[-2.0, 5.0, -1.0]]),
            np.array([[-2.0, -1.0, 5.0]]),
        ]
        mock_load.return_value = mock_model

        result = nli_bidirectional("broad", "narrow")
        assert result["equivalent"] is False
        assert result["contradiction"] is False


# ================================================================
# Softmax edge cases
# ================================================================


class TestSoftmaxEdgeCases:
    """Softmax stability and correctness."""

    def test_large_logits_stable(self) -> None:
        """Large logits do not overflow."""
        probs = _softmax([1000.0, 1.0, 0.1])
        assert all(np.isfinite(p) for p in probs)
        assert sum(probs) == pytest.approx(
            1.0, abs=1e-5,
        )

    def test_all_zeros(self) -> None:
        """All-zero logits -> uniform distribution."""
        probs = _softmax([0.0, 0.0, 0.0])
        for p in probs:
            assert p == pytest.approx(
                1.0 / 3, abs=1e-6,
            )

    def test_single_element(self) -> None:
        """Single logit -> probability 1.0."""
        probs = _softmax([42.0])
        assert probs[0] == pytest.approx(
            1.0, abs=1e-6,
        )


# ================================================================
# normalize_unicode re-export
# ================================================================


class TestNormalizeUnicode:
    """Verify re-export from _backends."""

    def test_normalize_accessible_from_backends(
        self,
    ) -> None:
        """normalize_unicode importable from _backends."""
        assert callable(normalize_unicode)
        result = normalize_unicode("hello")
        assert result == "hello"

    def test_strips_zero_width_chars(self) -> None:
        """Zero-width space removed via _backends path."""
        text = "he\u200bllo"
        assert normalize_unicode(text) == "hello"

    def test_nfkc_via_backends(self) -> None:
        """Fullwidth chars normalized via _backends."""
        text = "\uff21\uff22"
        assert normalize_unicode(text) == "AB"
