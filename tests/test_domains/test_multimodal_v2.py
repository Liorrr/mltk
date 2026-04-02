"""Tests for Multimodal v2 assertions -- metrics, hallucination, OCR.

This module tests four new assertions added in the Multimodal v2 sprint:

1. **CLIPScore** (``assert_clip_score``): Image-text alignment via cosine
   similarity.  Tests both the zero-dep embedding path (pure numpy) and
   the model path (mocked open-clip).

2. **Edit preservation** (``assert_edit_preservation``): SSIM and
   pixel_diff methods for measuring how much an edit preserves the
   original image.

3. **Object hallucination** (``assert_object_hallucination``): POPE-style
   probing for VLM hallucination detection.

4. **OCR accuracy** (``assert_ocr_accuracy``): Character and word error
   rates for OCR output evaluation.

All external dependencies (open-clip, scikit-image, Pillow) are mocked.
No GPU, no model downloads, no network access in tests.
"""

from __future__ import annotations

import io
import sys
import types
from unittest.mock import patch

import numpy as np
import pytest

from mltk.core.assertion import MltkAssertionError

# Also verify __init__ re-exports
from mltk.domains.multimodal import (
    assert_clip_score as clip_from_init,
)
from mltk.domains.multimodal import (
    assert_edit_preservation as edit_from_init,
)
from mltk.domains.multimodal import (
    assert_object_hallucination as halluc_from_init,
)
from mltk.domains.multimodal import (
    assert_ocr_accuracy as ocr_from_init,
)
from mltk.domains.multimodal.hallucination import (
    _parse_yes_no,
    assert_object_hallucination,
)
from mltk.domains.multimodal.metrics import (
    _cosine_similarity,
    assert_clip_score,
    assert_edit_preservation,
)
from mltk.domains.multimodal.vlm import (
    _levenshtein_distance,
    assert_ocr_accuracy,
)

# ===============================================================
# Helpers for creating fake images
# ===============================================================


def _make_png_bytes(
    width: int = 4,
    height: int = 4,
    color: tuple = (128, 128, 128),
) -> bytes:
    """Create minimal PNG bytes using Pillow for test images."""
    from PIL import Image

    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_png_bytes_from_array(arr: np.ndarray) -> bytes:
    """Convert a numpy array to PNG bytes."""
    from PIL import Image

    img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ===============================================================
# SHARED: _cosine_similarity tests
# ===============================================================


class TestCosineSimilarity:
    """Tests for the shared _cosine_similarity helper."""

    def test_identical_vectors(self) -> None:
        """Identical vectors should have similarity 1.0."""
        v = np.array([1.0, 2.0, 3.0])
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        """Orthogonal vectors should have similarity 0.0."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        """Opposite vectors should have similarity -1.0."""
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self) -> None:
        """Zero-norm vectors should return 0.0 (degenerate)."""
        zero = np.array([0.0, 0.0, 0.0])
        nonzero = np.array([1.0, 2.0, 3.0])
        assert _cosine_similarity(zero, nonzero) == 0.0
        assert _cosine_similarity(nonzero, zero) == 0.0

    def test_magnitude_invariant(self) -> None:
        """Cosine similarity ignores vector magnitude."""
        a = np.array([1.0, 1.0])
        b = np.array([100.0, 100.0])
        assert _cosine_similarity(a, b) == pytest.approx(1.0)


# ===============================================================
# CLIPScore tests
# ===============================================================


class TestClipScore:
    """Tests for assert_clip_score -- embedding and model paths."""

    def test_embedding_identical_vectors(self) -> None:
        """Identical embeddings should yield score ~1.0, passing."""
        emb = np.array([1.0, 0.0, 0.0, 0.0])
        result = assert_clip_score(
            image_embedding=emb,
            text_embedding=emb,
            min_score=0.9,
        )
        assert result.passed
        assert result.details["score"] == pytest.approx(1.0)
        assert result.details["method"] == "embedding"

    def test_embedding_orthogonal_vectors(self) -> None:
        """Orthogonal embeddings should yield score ~0.0, failing."""
        img = np.array([1.0, 0.0, 0.0])
        txt = np.array([0.0, 1.0, 0.0])
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_clip_score(
                image_embedding=img,
                text_embedding=txt,
                min_score=0.25,
            )
        result = exc_info.value.result
        assert result.details["score"] == pytest.approx(0.0)

    def test_embedding_partial_similarity(self) -> None:
        """Partially similar embeddings produce mid-range score."""
        img = np.array([1.0, 1.0, 0.0])
        txt = np.array([1.0, 0.0, 0.0])
        result = assert_clip_score(
            image_embedding=img,
            text_embedding=txt,
            min_score=0.5,
        )
        assert result.passed
        expected = 1.0 / np.sqrt(2.0)
        assert result.details["score"] == pytest.approx(
            expected, abs=0.001
        )

    def test_embedding_threshold_exact_boundary(self) -> None:
        """Score exactly at threshold should pass."""
        # cos(a, b) where a=[3,4] b=[4,3]
        a = np.array([3.0, 4.0])
        b = np.array([4.0, 3.0])
        cos = float(
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        )
        result = assert_clip_score(
            image_embedding=a,
            text_embedding=b,
            min_score=cos,
        )
        assert result.passed

    def test_embedding_threshold_fail(self) -> None:
        """Score below threshold should raise."""
        a = np.array([1.0, 0.0])
        b = np.array([0.7, 0.7])
        with pytest.raises(MltkAssertionError):
            assert_clip_score(
                image_embedding=a,
                text_embedding=b,
                min_score=0.99,
            )

    def test_missing_both_paths_raises_value_error(self) -> None:
        """Neither embeddings nor image+text should raise ValueError."""
        with pytest.raises(ValueError, match="neither complete"):
            assert_clip_score(min_score=0.25)

    def test_partial_embedding_raises_value_error(self) -> None:
        """Only image_embedding without text_embedding raises."""
        with pytest.raises(ValueError, match="neither complete"):
            assert_clip_score(
                image_embedding=np.array([1.0]),
                min_score=0.25,
            )

    def test_partial_model_raises_value_error(self) -> None:
        """Only image without text raises ValueError."""
        with pytest.raises(ValueError, match="neither complete"):
            assert_clip_score(
                image=b"fake_image",
                min_score=0.25,
            )

    def test_model_path_with_mocked_encode(self) -> None:
        """Model path should encode and compute cosine similarity."""
        fake_img_emb = np.array([1.0, 0.0, 0.0])
        fake_txt_emb = np.array([0.9, 0.1, 0.0])

        with patch(
            "mltk.domains.multimodal.metrics._encode_clip",
            return_value=(fake_img_emb, fake_txt_emb),
        ):
            result = assert_clip_score(
                image=b"fake_png_data",
                text="a test image",
                min_score=0.1,
            )

        assert result.passed
        assert result.details["method"] == "model"
        expected_cos = _cosine_similarity(
            fake_img_emb, fake_txt_emb
        )
        assert result.details["score"] == pytest.approx(
            expected_cos, abs=0.001
        )

    def test_model_path_import_error(self) -> None:
        """Model path gracefully handles missing open-clip."""
        with patch(
            "mltk.domains.multimodal.metrics._encode_clip",
            side_effect=ImportError("no open-clip"),
        ):
            with pytest.raises(MltkAssertionError) as exc_info:
                assert_clip_score(
                    image=b"fake",
                    text="test",
                    min_score=0.25,
                )
            result = exc_info.value.result
            assert result.details["score"] == 0.0
            assert "open-clip" in result.details["error"]

    def test_assertion_name(self) -> None:
        """Assertion name follows multimodal.metrics.* pattern."""
        emb = np.array([1.0])
        result = assert_clip_score(
            image_embedding=emb,
            text_embedding=emb,
            min_score=0.5,
        )
        assert result.name == "multimodal.metrics.clip_score"

    def test_high_dimensional_embeddings(self) -> None:
        """Works with high-dimensional embeddings (CLIP=512)."""
        rng = np.random.RandomState(42)
        a = rng.randn(512)
        result = assert_clip_score(
            image_embedding=a,
            text_embedding=a,
            min_score=0.99,
        )
        assert result.passed
        assert result.details["score"] == pytest.approx(1.0)


# ===============================================================
# Edit Preservation tests (SSIM + pixel_diff)
# ===============================================================


class TestEditPreservation:
    """Tests for assert_edit_preservation -- SSIM and pixel_diff."""

    def test_identical_images_pixel_diff(self) -> None:
        """Identical images should score 1.0 with pixel_diff."""
        png = _make_png_bytes(8, 8, (100, 100, 100))
        result = assert_edit_preservation(
            original=png,
            edited=png,
            method="pixel_diff",
            threshold=0.99,
        )
        assert result.passed
        assert result.details["score"] == pytest.approx(1.0)

    def test_different_images_pixel_diff(self) -> None:
        """Very different images should score low."""
        white = _make_png_bytes(8, 8, (255, 255, 255))
        black = _make_png_bytes(8, 8, (0, 0, 0))
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_edit_preservation(
                original=white,
                edited=black,
                method="pixel_diff",
                threshold=0.5,
            )
        result = exc_info.value.result
        assert result.details["score"] < 0.1

    def test_partially_different_pixel_diff(self) -> None:
        """Partially different images produce mid-range score."""
        gray128 = _make_png_bytes(8, 8, (128, 128, 128))
        gray192 = _make_png_bytes(8, 8, (192, 192, 192))
        result = assert_edit_preservation(
            original=gray128,
            edited=gray192,
            method="pixel_diff",
            threshold=0.5,
        )
        assert result.passed
        assert 0.5 < result.details["score"] < 1.0

    def test_identical_images_ssim(self) -> None:
        """Identical images should score 1.0 with SSIM."""
        arr = np.full((32, 32, 3), 128, dtype=np.uint8)
        png = _make_png_bytes_from_array(arr)
        try:
            result = assert_edit_preservation(
                original=png,
                edited=png,
                method="ssim",
                threshold=0.99,
            )
            assert result.passed
            assert result.details["score"] == pytest.approx(
                1.0, abs=0.001
            )
        except ImportError:
            pytest.skip("scikit-image not installed")

    def test_different_images_ssim(self) -> None:
        """Different images should score low with SSIM."""
        white = np.full((32, 32, 3), 255, dtype=np.uint8)
        black = np.zeros((32, 32, 3), dtype=np.uint8)
        png_w = _make_png_bytes_from_array(white)
        png_b = _make_png_bytes_from_array(black)
        try:
            with pytest.raises(MltkAssertionError) as exc_info:
                assert_edit_preservation(
                    original=png_w,
                    edited=png_b,
                    method="ssim",
                    threshold=0.5,
                )
            result = exc_info.value.result
            assert result.details["score"] < 0.5
        except ImportError:
            pytest.skip("scikit-image not installed")

    def test_ssim_import_error(self) -> None:
        """Missing scikit-image raises ImportError for SSIM."""
        png = _make_png_bytes(8, 8)
        with patch.dict(sys.modules, {"skimage": None}):
            # Force the import to fail by removing the module
            with patch(
                "mltk.domains.multimodal.metrics"
                ".structural_similarity",
                side_effect=ImportError,
                create=True,
            ):
                with pytest.raises(
                    ImportError, match="scikit-image"
                ):
                    assert_edit_preservation(
                        original=png,
                        edited=png,
                        method="ssim",
                    )

    def test_pixel_diff_threshold_boundary(self) -> None:
        """Score exactly at threshold should pass."""
        png = _make_png_bytes(8, 8, (128, 128, 128))
        result = assert_edit_preservation(
            original=png,
            edited=png,
            method="pixel_diff",
            threshold=1.0,
        )
        assert result.passed

    def test_invalid_method_raises(self) -> None:
        """Invalid method string raises ValueError."""
        png = _make_png_bytes(4, 4)
        with pytest.raises(ValueError, match="Unknown method"):
            assert_edit_preservation(
                original=png,
                edited=png,
                method="invalid",
            )

    def test_max_image_size_resize(self) -> None:
        """Large images are resized before comparison."""
        large = _make_png_bytes(1024, 1024, (100, 100, 100))
        result = assert_edit_preservation(
            original=large,
            edited=large,
            method="pixel_diff",
            threshold=0.99,
            max_image_size=64,
        )
        assert result.passed
        assert result.details["max_image_size"] == 64

    def test_assertion_name(self) -> None:
        """Assertion name follows multimodal.metrics.* pattern."""
        png = _make_png_bytes(4, 4)
        result = assert_edit_preservation(
            original=png,
            edited=png,
            method="pixel_diff",
        )
        assert result.name == (
            "multimodal.metrics.edit_preservation"
        )

    def test_pillow_import_error(self) -> None:
        """Missing Pillow produces a failing TestResult."""
        with patch(
            "mltk.domains.multimodal.metrics._load_and_resize",
            side_effect=ImportError(
                "Pillow is required for image comparison. "
                "Install with: pip install mltk[multimodal]"
            ),
        ):
            with pytest.raises(MltkAssertionError) as exc_info:
                assert_edit_preservation(
                    original=b"fake",
                    edited=b"fake",
                    method="pixel_diff",
                )
            result = exc_info.value.result
            assert not result.passed
            assert "Pillow" in result.message


# ===============================================================
# Object Hallucination (POPE) tests
# ===============================================================


class TestObjectHallucination:
    """Tests for assert_object_hallucination -- POPE-style probing."""

    @staticmethod
    def _make_vqa(
        present: set[str] | None = None,
    ) -> types.FunctionType:
        """Create a mock VQA function that knows which objects exist.

        Returns "Yes, ..." for present objects and "No, ..." for
        absent ones.
        """
        present = present or set()

        def vqa_fn(
            question: str,
            image: object | None,
            description: str | None,
        ) -> str:
            for obj in present:
                if obj.lower() in question.lower():
                    return f"Yes, there is a {obj} in the image."
            return "No, I don't see that in the image."

        return vqa_fn

    def test_all_present_correct(self) -> None:
        """All present objects answered correctly."""
        vqa = self._make_vqa(present={"sofa", "tv"})
        result = assert_object_hallucination(
            vqa_fn=vqa,
            image=None,
            objects_present=["sofa", "tv"],
            objects_absent=[],
            threshold=1.0,
            image_description="A living room.",
        )
        assert result.passed
        assert result.details["score"] == pytest.approx(1.0)

    def test_all_absent_correct(self) -> None:
        """All absent objects correctly denied."""
        vqa = self._make_vqa(present=set())
        result = assert_object_hallucination(
            vqa_fn=vqa,
            image=None,
            objects_present=[],
            objects_absent=["elephant", "car"],
            threshold=1.0,
            image_description="Empty room.",
        )
        assert result.passed
        assert result.details["hallucination_rate"] == 0.0

    def test_hallucination_detected(self) -> None:
        """VLM that hallucinates absent objects should fail."""
        # VLM says "yes" to everything -- always hallucinates
        def hallucinating_vqa(q, img, desc):
            return "Yes, I see it."

        with pytest.raises(MltkAssertionError) as exc_info:
            assert_object_hallucination(
                vqa_fn=hallucinating_vqa,
                image=None,
                objects_present=["sofa"],
                objects_absent=["elephant", "car", "rocket"],
                threshold=0.9,
                image_description="A room with a sofa.",
            )
        result = exc_info.value.result
        assert result.details["hallucination_rate"] == pytest.approx(
            1.0
        )
        assert result.details["false_positives"] == 3

    def test_threshold_boundary(self) -> None:
        """Accuracy exactly at threshold should pass."""
        vqa = self._make_vqa(present={"cat"})
        result = assert_object_hallucination(
            vqa_fn=vqa,
            image=None,
            objects_present=["cat"],
            objects_absent=["dog"],
            threshold=1.0,
            image_description="A cat on a mat.",
        )
        assert result.passed

    def test_empty_objects_raises_value_error(self) -> None:
        """Both empty lists should raise ValueError."""
        vqa = self._make_vqa()
        with pytest.raises(ValueError, match="non-empty"):
            assert_object_hallucination(
                vqa_fn=vqa,
                image=None,
                objects_present=[],
                objects_absent=[],
                image_description="Nothing.",
            )

    def test_single_present_object(self) -> None:
        """Single present object works."""
        vqa = self._make_vqa(present={"book"})
        result = assert_object_hallucination(
            vqa_fn=vqa,
            image=None,
            objects_present=["book"],
            objects_absent=[],
            image_description="A book on a table.",
        )
        assert result.passed

    def test_single_absent_object(self) -> None:
        """Single absent object works."""
        vqa = self._make_vqa(present=set())
        result = assert_object_hallucination(
            vqa_fn=vqa,
            image=None,
            objects_present=[],
            objects_absent=["unicorn"],
            image_description="An empty field.",
        )
        assert result.passed

    def test_vqa_fn_error_handling(self) -> None:
        """Errors from vqa_fn are caught and reported."""
        def failing_vqa(q, img, desc):
            raise RuntimeError("VLM crashed")

        with pytest.raises(MltkAssertionError) as exc_info:
            assert_object_hallucination(
                vqa_fn=failing_vqa,
                image=None,
                objects_present=["cat"],
                objects_absent=["dog"],
                threshold=0.9,
                image_description="A scene.",
            )
        result = exc_info.value.result
        assert result.details["errors"] is not None
        assert any(
            "RuntimeError" in e for e in result.details["errors"]
        )

    def test_image_description_passed_to_vqa(self) -> None:
        """image_description is forwarded to vqa_fn."""
        received_desc = []

        def capture_vqa(q, img, desc):
            received_desc.append(desc)
            return "No"

        assert_object_hallucination(
            vqa_fn=capture_vqa,
            image=None,
            objects_present=[],
            objects_absent=["car"],
            image_description="A park with trees.",
        )
        assert received_desc[0] == "A park with trees."

    def test_per_object_details(self) -> None:
        """per_object list contains detailed per-probe results."""
        vqa = self._make_vqa(present={"lamp"})
        result = assert_object_hallucination(
            vqa_fn=vqa,
            image=None,
            objects_present=["lamp"],
            objects_absent=["piano"],
            image_description="A desk with a lamp.",
        )
        per_obj = result.details["per_object"]
        assert len(per_obj) == 2
        assert per_obj[0]["object"] == "lamp"
        assert per_obj[0]["correct"] is True
        assert per_obj[1]["object"] == "piano"
        assert per_obj[1]["correct"] is True

    def test_ambiguous_answer_treated_as_incorrect(self) -> None:
        """Ambiguous answers (no yes/no) count as incorrect."""
        def ambiguous_vqa(q, img, desc):
            return "I'm not sure about that."

        with pytest.raises(MltkAssertionError):
            assert_object_hallucination(
                vqa_fn=ambiguous_vqa,
                image=None,
                objects_present=["cat"],
                objects_absent=[],
                threshold=0.5,
                image_description="A scene.",
            )

    def test_assertion_name(self) -> None:
        """Assertion name follows multimodal.hallucination.*."""
        vqa = self._make_vqa(present=set())
        result = assert_object_hallucination(
            vqa_fn=vqa,
            image=None,
            objects_present=[],
            objects_absent=["x"],
            image_description="Empty.",
        )
        assert result.name == (
            "multimodal.hallucination.object_hallucination"
        )

    def test_mixed_correct_and_incorrect(self) -> None:
        """Mixed results produce correct accuracy score."""
        # VLM knows about cat but hallucinates car
        def mixed_vqa(q, img, desc):
            if "cat" in q.lower():
                return "Yes"
            if "dog" in q.lower():
                return "Yes"  # hallucination!
            return "No"

        with pytest.raises(MltkAssertionError) as exc_info:
            assert_object_hallucination(
                vqa_fn=mixed_vqa,
                image=None,
                objects_present=["cat"],
                objects_absent=["dog", "fish"],
                threshold=1.0,
                image_description="A scene with a cat.",
            )
        result = exc_info.value.result
        # 2 correct (cat=yes, fish=no) out of 3
        assert result.details["score"] == pytest.approx(
            2.0 / 3.0, abs=0.01
        )
        assert result.details["false_positives"] == 1


# ===============================================================
# Parse yes/no helper tests
# ===============================================================


class TestParseYesNo:
    """Tests for the _parse_yes_no helper in hallucination.py."""

    def test_plain_yes(self) -> None:
        assert _parse_yes_no("yes") == "yes"

    def test_plain_no(self) -> None:
        assert _parse_yes_no("no") == "no"

    def test_sentence_with_yes(self) -> None:
        assert _parse_yes_no(
            "Yes, there is a cat in the image."
        ) == "yes"

    def test_sentence_with_no(self) -> None:
        assert _parse_yes_no(
            "No, I don't see a dog."
        ) == "no"

    def test_ambiguous_returns_none(self) -> None:
        assert _parse_yes_no("I'm not sure.") is None

    def test_both_yes_and_no_first_wins(self) -> None:
        """When both appear, the first one wins."""
        assert _parse_yes_no("No, yes maybe") == "no"
        assert _parse_yes_no("Yes, no wait") == "yes"

    def test_case_insensitive(self) -> None:
        assert _parse_yes_no("YES") == "yes"
        assert _parse_yes_no("NO") == "no"
        assert _parse_yes_no("yEs") == "yes"


# ===============================================================
# OCR accuracy tests (CER / WER)
# ===============================================================


class TestOcrAccuracy:
    """Tests for assert_ocr_accuracy -- CER and WER."""

    def test_exact_match_cer(self) -> None:
        """Exact match should have CER 0.0."""
        result = assert_ocr_accuracy(
            expected_text="Invoice #1234",
            actual_text="Invoice #1234",
            method="cer",
            threshold=0.05,
        )
        assert result.passed
        assert result.details["error_rate"] == 0.0
        assert result.details["score"] == pytest.approx(1.0)

    def test_exact_match_wer(self) -> None:
        """Exact match should have WER 0.0."""
        result = assert_ocr_accuracy(
            expected_text="hello world",
            actual_text="hello world",
            method="wer",
            threshold=0.05,
        )
        assert result.passed
        assert result.details["error_rate"] == 0.0

    def test_completely_different_cer(self) -> None:
        """Completely different strings produce high CER."""
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_ocr_accuracy(
                expected_text="abc",
                actual_text="xyz",
                method="cer",
                threshold=0.1,
            )
        result = exc_info.value.result
        assert result.details["error_rate"] == pytest.approx(1.0)

    def test_single_char_difference_cer(self) -> None:
        """One character difference: CER = 1/N."""
        result = assert_ocr_accuracy(
            expected_text="Invoice",
            actual_text="Iuvoice",
            method="cer",
            threshold=0.2,
        )
        # edit distance = 2 (u->n, swap positions)
        # CER = edit_dist / 7
        assert result.details["error_rate"] <= 0.3

    def test_wer_single_word_wrong(self) -> None:
        """One wrong word out of four: WER = 0.25."""
        result = assert_ocr_accuracy(
            expected_text="the quick brown fox",
            actual_text="the quick brown dog",
            method="wer",
            threshold=0.3,
        )
        assert result.passed
        assert result.details["error_rate"] == pytest.approx(0.25)

    def test_empty_strings_both(self) -> None:
        """Both empty strings: error_rate 0.0."""
        result = assert_ocr_accuracy(
            expected_text="",
            actual_text="",
            method="cer",
            threshold=0.1,
        )
        assert result.passed
        assert result.details["error_rate"] == 0.0

    def test_empty_reference_nonempty_hypothesis(self) -> None:
        """Empty reference with non-empty hypothesis: error 1.0."""
        with pytest.raises(MltkAssertionError):
            assert_ocr_accuracy(
                expected_text="",
                actual_text="extra text",
                method="cer",
                threshold=0.5,
            )

    def test_unicode_text_cer(self) -> None:
        """Unicode characters are handled correctly."""
        result = assert_ocr_accuracy(
            expected_text="cafe\u0301",
            actual_text="cafe\u0301",
            method="cer",
            threshold=0.1,
        )
        assert result.passed

    def test_threshold_pass_boundary(self) -> None:
        """Error rate exactly at threshold should pass."""
        # "ab" vs "ac" = 1 edit, CER = 1/2 = 0.5
        result = assert_ocr_accuracy(
            expected_text="ab",
            actual_text="ac",
            method="cer",
            threshold=0.5,
        )
        assert result.passed

    def test_threshold_fail(self) -> None:
        """Error rate above threshold should fail."""
        with pytest.raises(MltkAssertionError):
            assert_ocr_accuracy(
                expected_text="ab",
                actual_text="ac",
                method="cer",
                threshold=0.1,
            )

    def test_invalid_method_raises(self) -> None:
        """Invalid method string raises ValueError."""
        with pytest.raises(ValueError, match="method must be"):
            assert_ocr_accuracy(
                expected_text="a",
                actual_text="b",
                method="invalid",
            )

    def test_assertion_name(self) -> None:
        """Assertion name follows multimodal.vlm.* pattern."""
        result = assert_ocr_accuracy(
            expected_text="test",
            actual_text="test",
        )
        assert result.name == "multimodal.vlm.ocr_accuracy"

    def test_error_rate_can_exceed_one(self) -> None:
        """CER can exceed 1.0 when hypothesis is much longer."""
        result = assert_ocr_accuracy(
            expected_text="a",
            actual_text="abcdef",
            method="cer",
            threshold=10.0,
        )
        assert result.details["error_rate"] > 1.0


# ===============================================================
# Levenshtein distance tests
# ===============================================================


class TestLevenshteinDistance:
    """Tests for the _levenshtein_distance helper."""

    def test_identical_sequences(self) -> None:
        assert _levenshtein_distance(
            list("abc"), list("abc")
        ) == 0

    def test_empty_sequences(self) -> None:
        assert _levenshtein_distance([], []) == 0

    def test_one_empty(self) -> None:
        assert _levenshtein_distance(
            list("abc"), []
        ) == 3
        assert _levenshtein_distance(
            [], list("abc")
        ) == 3

    def test_single_substitution(self) -> None:
        assert _levenshtein_distance(
            list("cat"), list("car")
        ) == 1

    def test_single_insertion(self) -> None:
        assert _levenshtein_distance(
            list("cat"), list("cats")
        ) == 1

    def test_single_deletion(self) -> None:
        assert _levenshtein_distance(
            list("cats"), list("cat")
        ) == 1

    def test_classic_kitten_sitting(self) -> None:
        """Classic example: kitten -> sitting = 3 edits."""
        assert _levenshtein_distance(
            list("kitten"), list("sitting")
        ) == 3

    def test_word_level(self) -> None:
        """Works with word lists too (for WER)."""
        assert _levenshtein_distance(
            ["the", "cat", "sat"],
            ["the", "dog", "sat"],
        ) == 1


# ===============================================================
# Module exports tests
# ===============================================================


class TestModuleExports:
    """Verify that __init__.py re-exports all v2 assertions."""

    def test_clip_score_exported(self) -> None:
        assert clip_from_init is assert_clip_score

    def test_edit_preservation_exported(self) -> None:
        assert edit_from_init is assert_edit_preservation

    def test_object_hallucination_exported(self) -> None:
        assert halluc_from_init is assert_object_hallucination

    def test_ocr_accuracy_exported(self) -> None:
        assert ocr_from_init is assert_ocr_accuracy

    def test_all_in_package_all(self) -> None:
        """All v2 names appear in __all__."""
        from mltk.domains import multimodal

        all_names = multimodal.__all__
        assert "assert_clip_score" in all_names
        assert "assert_edit_preservation" in all_names
        assert "assert_object_hallucination" in all_names
        assert "assert_ocr_accuracy" in all_names


# ===============================================================
# Edge-case / hardening tests (appended)
# ===============================================================


class TestClipScoreEdgeCases:
    """Edge-case tests for CLIPScore."""

    def test_empty_text_description(self) -> None:
        """Empty text with model path still produces result."""
        fake_img_emb = np.array([1.0, 0.0])
        fake_txt_emb = np.array([0.5, 0.5])

        with patch(
            "mltk.domains.multimodal.metrics._encode_clip",
            return_value=(fake_img_emb, fake_txt_emb),
        ):
            result = assert_clip_score(
                image=b"fake_png",
                text="",
                min_score=0.1,
            )
        assert result.passed
        assert result.details["method"] == "model"

    def test_very_long_text(self) -> None:
        """1000+ char text with embedding path works."""
        rng = np.random.RandomState(99)
        img_emb = rng.randn(128)
        txt_emb = img_emb.copy()
        result = assert_clip_score(
            image_embedding=img_emb,
            text_embedding=txt_emb,
            min_score=0.9,
        )
        assert result.passed
        assert result.details["score"] == pytest.approx(1.0)


class TestEditPreservationEdgeCases:
    """Edge-case tests for edit preservation."""

    def test_ssim_identical_returns_one(self) -> None:
        """SSIM of identical images should be ~1.0."""
        rng = np.random.RandomState(42)
        arr = rng.randint(
            0, 256, (32, 32, 3), dtype=np.uint8,
        )
        png = _make_png_bytes_from_array(arr)
        try:
            result = assert_edit_preservation(
                original=png,
                edited=png,
                method="ssim",
                threshold=0.99,
            )
            assert result.passed
            assert result.details["score"] >= 0.99
        except ImportError:
            pytest.skip("scikit-image not installed")


class TestOcrAccuracyEdgeCases:
    """Edge-case tests for OCR accuracy."""

    def test_both_empty_strings_cer(self) -> None:
        """Both empty strings produce 0.0 error rate."""
        result = assert_ocr_accuracy(
            expected_text="",
            actual_text="",
            method="cer",
            threshold=0.0,
        )
        assert result.passed
        assert result.details["error_rate"] == 0.0

    def test_both_empty_strings_wer(self) -> None:
        """Both empty strings with WER produce 0.0 error."""
        result = assert_ocr_accuracy(
            expected_text="",
            actual_text="",
            method="wer",
            threshold=0.0,
        )
        assert result.passed
        assert result.details["error_rate"] == 0.0

    def test_unicode_cjk_chars(self) -> None:
        """CJK characters handled by CER correctly."""
        result = assert_ocr_accuracy(
            expected_text="\u4f60\u597d\u4e16\u754c",
            actual_text="\u4f60\u597d\u4e16\u754c",
            method="cer",
            threshold=0.0,
        )
        assert result.passed
        assert result.details["error_rate"] == 0.0

    def test_unicode_emoji_chars(self) -> None:
        """Emoji characters handled by CER correctly."""
        result = assert_ocr_accuracy(
            expected_text="\U0001f600\U0001f680\U0001f30d",
            actual_text="\U0001f600\U0001f680\U0001f30e",
            method="cer",
            threshold=0.5,
        )
        assert result.passed
        rate = result.details["error_rate"]
        assert rate == pytest.approx(1.0 / 3.0, abs=0.01)


class TestPOPEEdgeCases:
    """Edge-case tests for POPE hallucination detection."""

    @staticmethod
    def _make_vqa_all_yes():
        """VQA that always answers yes."""
        def vqa_fn(q, img, desc):
            return "Yes, I see it."
        return vqa_fn

    @staticmethod
    def _make_vqa_all_no():
        """VQA that always answers no."""
        def vqa_fn(q, img, desc):
            return "No, I don't see that."
        return vqa_fn

    def test_all_yes_with_all_present(self) -> None:
        """All-yes VQA with only present objects passes."""
        vqa = self._make_vqa_all_yes()
        result = assert_object_hallucination(
            vqa_fn=vqa,
            image=None,
            objects_present=["cat", "dog", "bird"],
            objects_absent=[],
            threshold=1.0,
            image_description="Animals.",
        )
        assert result.passed
        assert result.details["score"] == 1.0

    def test_all_yes_with_absent_objects_fails(self) -> None:
        """All-yes VQA hallucinates all absent objects."""
        vqa = self._make_vqa_all_yes()
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_object_hallucination(
                vqa_fn=vqa,
                image=None,
                objects_present=[],
                objects_absent=["car", "plane"],
                threshold=0.5,
                image_description="Empty.",
            )
        result = exc_info.value.result
        assert result.details["hallucination_rate"] == 1.0
        assert result.details["false_positives"] == 2

    def test_all_no_with_all_absent(self) -> None:
        """All-no VQA with only absent objects passes."""
        vqa = self._make_vqa_all_no()
        result = assert_object_hallucination(
            vqa_fn=vqa,
            image=None,
            objects_present=[],
            objects_absent=["x", "y", "z"],
            threshold=1.0,
            image_description="Nothing.",
        )
        assert result.passed
        assert result.details["hallucination_rate"] == 0.0

    def test_all_no_with_present_objects_fails(self) -> None:
        """All-no VQA misses all present objects."""
        vqa = self._make_vqa_all_no()
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_object_hallucination(
                vqa_fn=vqa,
                image=None,
                objects_present=["cat", "dog"],
                objects_absent=[],
                threshold=0.5,
                image_description="Pets.",
            )
        result = exc_info.value.result
        assert result.details["false_negatives"] == 2
        assert result.details["score"] == 0.0
