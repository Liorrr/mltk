"""Tests for mltk.domains.multimodal -- multimodal evaluation suite.

This test module covers the full multimodal subpackage:

1. **Embedding-based alignment** (existing): Image-text alignment via
   cosine similarity and cross-modal consistency via agreement rate.

2. **Image utilities** (new): ImageInput loading from paths and bytes,
   base64 encoding, prompt construction with image context.

3. **LLM-judge alignment** (new): Prompt faithfulness (does image match
   the prompt?) and image coherence (does image fit the text context?).

4. **VLM evaluation** (new): Image helpfulness (does image help answer
   the question?) and VQA accuracy (is the visual answer correct?).

All LLM-judge assertions use mock judge functions returning JSON with
a score field.  No real LLM calls are made in tests.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.domains.multimodal import (
    assert_cross_modal_consistency,
    assert_image_coherence,
    assert_image_helpfulness,
    assert_image_text_alignment,
    assert_prompt_faithfulness,
    assert_vqa_accuracy,
    image_to_base64,
    load_image,
)
from mltk.domains.multimodal._image import (
    _build_image_prompt,
    _validate_image_pillow,
)

# ===============================================================
# Fixtures
# ===============================================================


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Minimal valid PNG bytes (1x1 transparent pixel).

    WHY: A real PNG header lets us test image loading without
    requiring external files.  This is the smallest valid PNG.
    """
    # 1x1 transparent PNG (67 bytes)
    return (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x06\x00\x00\x00"
        b"\x1f\x15\xc4\x89\x00\x00\x00\nIDATx"
        b"\x9cc\x00\x01\x00\x00\x05\x00\x01"
        b"\r\n\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )


@pytest.fixture
def sample_image_file(sample_image_bytes: bytes) -> Path:
    """Write sample PNG to a temp file and return the path.

    WHY: Tests that accept file paths need an actual file on disk.
    Uses tempfile for cross-platform compatibility.
    """
    with tempfile.NamedTemporaryFile(
        suffix=".png", delete=False
    ) as f:
        f.write(sample_image_bytes)
        return Path(f.name)


def _mock_judge_pass(prompt: str) -> str:
    """Mock judge that always returns a passing score."""
    return json.dumps(
        {"score": 0.85, "reasoning": "Good alignment"}
    )


def _mock_judge_fail(prompt: str) -> str:
    """Mock judge that always returns a failing score."""
    return json.dumps(
        {"score": 0.2, "reasoning": "Poor alignment"}
    )


def _mock_judge_high(prompt: str) -> str:
    """Mock judge that returns a perfect score."""
    return json.dumps(
        {"score": 1.0, "reasoning": "Perfect match"}
    )


def _mock_judge_boundary(prompt: str) -> str:
    """Mock judge that returns exactly 0.7 (boundary)."""
    return json.dumps(
        {"score": 0.7, "reasoning": "Borderline"}
    )


def _mock_judge_plain_number(prompt: str) -> str:
    """Mock judge that returns a plain number string."""
    return "0.85"


def _mock_judge_error(prompt: str) -> str:
    """Mock judge that raises an exception."""
    raise RuntimeError("LLM service unavailable")


def _mock_judge_garbage(prompt: str) -> str:
    """Mock judge that returns unparseable garbage."""
    return "I cannot evaluate this image."


# ===============================================================
# Part 1: Image Utilities (_image.py)
# ===============================================================


class TestLoadImage:
    """Tests for load_image() -- resolving ImageInput to bytes."""

    def test_load_from_bytes(
        self, sample_image_bytes: bytes
    ) -> None:
        """PASS: bytes input returned as-is.

        WHY: API responses often provide images as raw bytes.
        load_image should pass them through without copying.
        """
        result = load_image(sample_image_bytes)
        assert result == sample_image_bytes
        assert isinstance(result, bytes)

    def test_load_from_path_str(
        self, sample_image_file: Path
    ) -> None:
        """PASS: String path loads file bytes.

        WHY: Most users pass file paths as strings.  The loader
        should read the file and return its contents.
        """
        result = load_image(str(sample_image_file))
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_load_from_path_object(
        self, sample_image_file: Path
    ) -> None:
        """PASS: pathlib.Path loads file bytes.

        WHY: pathlib is the modern Python path API.  Supporting
        Path objects avoids users having to call str() manually.
        """
        result = load_image(sample_image_file)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_load_invalid_type_raises(self) -> None:
        """FAIL: Invalid type raises TypeError.

        WHY: Passing an integer or list is a programming error.
        A clear TypeError helps the user fix the call site.
        """
        with pytest.raises(TypeError, match="Expected str"):
            load_image(12345)  # type: ignore[arg-type]

    def test_load_missing_file_raises(self) -> None:
        """FAIL: Non-existent path raises FileNotFoundError.

        WHY: A typo in the path should fail fast with the OS
        error, not silently return empty bytes.
        """
        with pytest.raises(
            (FileNotFoundError, OSError)
        ):
            load_image("/nonexistent/path/image.png")

    def test_load_empty_bytes(self) -> None:
        """PASS: Empty bytes are returned as-is.

        WHY: Empty bytes may be valid in certain contexts (e.g.,
        placeholder). load_image is a transport function, not a
        validator.  Validation happens downstream.
        """
        result = load_image(b"")
        assert result == b""


class TestImageToBase64:
    """Tests for image_to_base64() -- encoding for LLM prompts."""

    def test_bytes_to_base64(
        self, sample_image_bytes: bytes
    ) -> None:
        """PASS: Bytes are base64-encoded correctly.

        WHY: LLM APIs accept images as base64 strings in prompts.
        The encoding must be reversible.
        """
        import base64

        b64 = image_to_base64(sample_image_bytes)
        assert isinstance(b64, str)
        # Verify it decodes back to the original
        decoded = base64.b64decode(b64)
        assert decoded == sample_image_bytes

    def test_file_to_base64(
        self, sample_image_file: Path
    ) -> None:
        """PASS: File path is loaded and base64-encoded.

        WHY: End-to-end path: file -> bytes -> base64.  The user
        should be able to pass a path directly.
        """
        b64 = image_to_base64(sample_image_file)
        assert isinstance(b64, str)
        assert len(b64) > 0

    def test_empty_bytes_base64(self) -> None:
        """PASS: Empty bytes produce empty base64 string.

        WHY: base64 of empty input is empty string.  This is
        mathematically correct and should not raise.
        """
        b64 = image_to_base64(b"")
        assert b64 == ""


class TestBuildImagePrompt:
    """Tests for _build_image_prompt() -- prompt construction."""

    def test_with_description(self) -> None:
        """PASS: Description path uses text directly.

        WHY: When image_description is provided, the prompt should
        include the description text without loading any image.
        This is the Pillow-free path.
        """
        prompt = _build_image_prompt(
            instruction="Evaluate this.",
            image_description="A red car on a highway.",
        )
        assert "A red car on a highway." in prompt
        assert "Image Description" in prompt

    def test_with_image_bytes(
        self, sample_image_bytes: bytes
    ) -> None:
        """PASS: Image path base64-encodes the image.

        WHY: When only image bytes are provided (no description),
        the prompt should contain the base64 encoding.
        """
        prompt = _build_image_prompt(
            instruction="Evaluate this.",
            image=sample_image_bytes,
        )
        assert "Image (base64)" in prompt
        assert len(prompt) > 50

    def test_description_takes_precedence(
        self, sample_image_bytes: bytes
    ) -> None:
        """PASS: Description is used even when image is provided.

        WHY: image_description is the escape hatch for users who
        pre-describe images.  It should take precedence over raw
        image data to avoid unnecessary base64 encoding.
        """
        prompt = _build_image_prompt(
            instruction="Evaluate this.",
            image=sample_image_bytes,
            image_description="Description wins.",
        )
        assert "Description wins." in prompt
        assert "base64" not in prompt.lower()

    def test_neither_provided_raises(self) -> None:
        """FAIL: No image and no description raises ValueError.

        WHY: The assertion cannot evaluate without any image
        context.  A clear error prevents silent failures.
        """
        with pytest.raises(
            ValueError, match="Either 'image' or"
        ):
            _build_image_prompt(
                instruction="Evaluate this."
            )


class TestValidateImagePillow:
    """Tests for _validate_image_pillow() -- optional Pillow path."""

    def test_pillow_import_error(self) -> None:
        """FAIL: Missing Pillow raises ImportError with hint.

        WHY: Users without Pillow should get a clear message
        telling them how to install it, not a raw ImportError.
        """
        with patch.dict(
            "sys.modules", {"PIL": None, "PIL.Image": None}
        ):
            with pytest.raises(
                ImportError, match="mltk\\[multimodal\\]"
            ):
                _validate_image_pillow(b"fake image data")


# ===============================================================
# Part 2: Existing Embedding-Based Assertions
# ===============================================================


class TestImageTextAlignment:
    """Image-text alignment tests using cosine similarity.

    These are the original tests from the pre-subpackage era,
    preserved to verify backward compatibility after migration.
    """

    def test_aligned_embeddings_pass(self) -> None:
        """PASS: Nearly identical embeddings yield cosine ~1.0."""
        img = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        txt = np.array(
            [[0.95, 0.05, 0.0], [0.05, 0.95, 0.0]]
        )
        result = assert_image_text_alignment(
            img, txt, min_cosine=0.5
        )
        assert result.passed is True
        assert result.details["avg_cosine"] > 0.9
        assert result.details["n_pairs"] == 2

    def test_misaligned_embeddings_fail(self) -> None:
        """FAIL: Orthogonal embeddings yield cosine ~0.0."""
        img = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        txt = np.array(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
        )
        with pytest.raises(MltkAssertionError):
            assert_image_text_alignment(
                img, txt, min_cosine=0.5
            )

    def test_single_pair(self) -> None:
        """PASS: Single image-text pair."""
        img = np.array([[0.6, 0.8, 0.0]])
        txt = np.array([[0.6, 0.8, 0.0]])
        result = assert_image_text_alignment(
            img, txt, min_cosine=0.9
        )
        assert result.passed is True
        assert result.details["n_pairs"] == 1

    def test_empty_embeddings_fail(self) -> None:
        """FAIL: Empty arrays cannot be evaluated."""
        img = np.array([]).reshape(0, 3)
        txt = np.array([]).reshape(0, 3)
        with pytest.raises(MltkAssertionError):
            assert_image_text_alignment(
                img, txt, min_cosine=0.5
            )

    def test_1d_input_treated_as_single_pair(self) -> None:
        """PASS: 1D vectors are reshaped to single pair."""
        img = np.array([1.0, 0.0, 0.0])
        txt = np.array([1.0, 0.0, 0.0])
        result = assert_image_text_alignment(
            img, txt, min_cosine=0.9
        )
        assert result.passed is True
        assert result.details["n_pairs"] == 1


class TestCrossModalConsistency:
    """Cross-modal consistency tests."""

    def test_perfect_agreement(self) -> None:
        """PASS: All predictions match (100% agreement)."""
        a = ["cat", "dog", "bird"]
        b = ["cat", "dog", "bird"]
        result = assert_cross_modal_consistency(
            a, b, min_agreement=0.9
        )
        assert result.passed is True
        assert result.details["agreement_rate"] == 1.0

    def test_zero_agreement_fails(self) -> None:
        """FAIL: No predictions match (0% agreement)."""
        a = [0, 1, 2, 3]
        b = [3, 2, 1, 0]
        with pytest.raises(MltkAssertionError):
            assert_cross_modal_consistency(
                a, b, min_agreement=0.5
            )

    def test_partial_agreement(self) -> None:
        """PASS: 75% agreement with disagreement indices."""
        a = np.array([1, 1, 0, 1])
        b = np.array([1, 1, 1, 1])
        result = assert_cross_modal_consistency(
            a, b, min_agreement=0.7
        )
        assert result.passed is True
        assert result.details["agreement_rate"] == 0.75
        assert result.details["disagreements"] == [2]

    def test_empty_predictions_fail(self) -> None:
        """FAIL: Empty predictions cannot be compared."""
        with pytest.raises(MltkAssertionError):
            assert_cross_modal_consistency(
                [], [], min_agreement=0.5
            )


# ===============================================================
# Part 3: Prompt Faithfulness (alignment.py - new)
# ===============================================================


class TestPromptFaithfulness:
    """Prompt faithfulness -- does image match the text prompt?

    Uses mock judge functions to test scoring logic without
    real LLM calls.
    """

    def test_pass_with_description(self) -> None:
        """PASS: High judge score with image_description.

        WHY: The most common path -- user pre-describes the image
        with a VLM, then a text-only judge evaluates alignment.
        No Pillow needed.
        """
        result = assert_prompt_faithfulness(
            prompt="A red car on a highway",
            image=None,
            judge_fn=_mock_judge_pass,
            image_description="Photo of a red sedan on a road.",
        )
        assert result.passed is True
        assert result.details["score"] == 0.85
        assert result.name == (
            "multimodal.alignment.prompt_faithfulness"
        )

    def test_fail_low_score(self) -> None:
        """FAIL: Judge returns score below threshold.

        WHY: When the image does not match the prompt (e.g.,
        asked for red car, got blue truck), the judge scores
        low and the assertion raises.
        """
        with pytest.raises(MltkAssertionError):
            assert_prompt_faithfulness(
                prompt="A sunset over mountains",
                image=None,
                judge_fn=_mock_judge_fail,
                image_description="A photo of a cat.",
            )

    def test_pass_with_image_bytes(
        self, sample_image_bytes: bytes
    ) -> None:
        """PASS: Raw image bytes are base64-encoded for judge.

        WHY: Users who have image bytes (from API response)
        should be able to pass them directly.
        """
        result = assert_prompt_faithfulness(
            prompt="A test pattern",
            image=sample_image_bytes,
            judge_fn=_mock_judge_pass,
        )
        assert result.passed is True

    def test_pass_with_image_file(
        self, sample_image_file: Path
    ) -> None:
        """PASS: File path loaded and encoded for judge.

        WHY: Most users have images on disk.  Passing a path
        should work end-to-end.
        """
        result = assert_prompt_faithfulness(
            prompt="A test image",
            image=sample_image_file,
            judge_fn=_mock_judge_pass,
        )
        assert result.passed is True

    def test_custom_min_score(self) -> None:
        """PASS: Custom min_score threshold works.

        WHY: Different use cases need different thresholds.
        A strict evaluation (0.9) should fail with a 0.85 score.
        """
        result = assert_prompt_faithfulness(
            prompt="A test",
            image=None,
            judge_fn=_mock_judge_pass,  # returns 0.85
            min_score=0.8,
            image_description="test",
        )
        assert result.passed is True

    def test_strict_threshold_fails(self) -> None:
        """FAIL: Score 0.85 fails with min_score=0.9.

        WHY: When the threshold is set higher than the score,
        the assertion should fail.
        """
        with pytest.raises(MltkAssertionError):
            assert_prompt_faithfulness(
                prompt="A test",
                image=None,
                judge_fn=_mock_judge_pass,  # 0.85
                min_score=0.9,
                image_description="test",
            )

    def test_boundary_score_passes(self) -> None:
        """PASS: Score exactly equals min_score.

        WHY: The boundary condition (score == min_score) should
        pass, not fail.  This is >= not >.
        """
        result = assert_prompt_faithfulness(
            prompt="Boundary test",
            image=None,
            judge_fn=_mock_judge_boundary,  # 0.7
            min_score=0.7,
            image_description="test",
        )
        assert result.passed is True

    def test_judge_error_fails(self) -> None:
        """FAIL: Judge exception results in score 0.0.

        WHY: If the judge LLM is unavailable, the score defaults
        to 0.0.  The error is captured in details for debugging.
        """
        with pytest.raises(MltkAssertionError):
            assert_prompt_faithfulness(
                prompt="Error test",
                image=None,
                judge_fn=_mock_judge_error,
                image_description="test",
            )

    def test_judge_garbage_fails(self) -> None:
        """FAIL: Unparseable judge response results in score 0.0.

        WHY: If the judge returns text without a number, the
        parser cannot extract a score.  Default to 0.0.
        """
        with pytest.raises(MltkAssertionError):
            assert_prompt_faithfulness(
                prompt="Garbage test",
                image=None,
                judge_fn=_mock_judge_garbage,
                image_description="test",
            )

    def test_plain_number_response(self) -> None:
        """PASS: Judge returns plain number (no JSON).

        WHY: Some judges return just "0.85" without JSON wrapper.
        The regex fallback should parse this correctly.
        """
        result = assert_prompt_faithfulness(
            prompt="Plain number",
            image=None,
            judge_fn=_mock_judge_plain_number,
            min_score=0.8,
            image_description="test",
        )
        assert result.passed is True
        assert result.details["score"] == 0.85

    def test_details_contain_prompt_text(self) -> None:
        """PASS: Details include the original prompt text.

        WHY: For debugging, the TestResult should carry the
        prompt that was evaluated so users can trace failures.
        """
        result = assert_prompt_faithfulness(
            prompt="Specific prompt text here",
            image=None,
            judge_fn=_mock_judge_pass,
            image_description="test",
        )
        assert (
            result.details["prompt_text"]
            == "Specific prompt text here"
        )


# ===============================================================
# Part 4: Image Coherence (alignment.py - new)
# ===============================================================


class TestImageCoherence:
    """Image coherence -- does image fit the text context?"""

    def test_pass_coherent_image(self) -> None:
        """PASS: Image matches the text context.

        WHY: An X-ray image in a medical report about fractures
        is coherent.  The judge scores high.
        """
        result = assert_image_coherence(
            text="The patient's X-ray shows a fracture.",
            image=None,
            judge_fn=_mock_judge_pass,
            image_description="An X-ray of a broken bone.",
        )
        assert result.passed is True
        assert result.details["score"] == 0.85
        assert result.name == (
            "multimodal.alignment.image_coherence"
        )

    def test_fail_incoherent_image(self) -> None:
        """FAIL: Image contradicts the text context.

        WHY: A sunset photo in a medical report is incoherent.
        The judge scores low and the assertion raises.
        """
        with pytest.raises(MltkAssertionError):
            assert_image_coherence(
                text="Bone fracture analysis results.",
                image=None,
                judge_fn=_mock_judge_fail,
                image_description="A sunset at the beach.",
            )

    def test_with_image_bytes(
        self, sample_image_bytes: bytes
    ) -> None:
        """PASS: Raw bytes path works."""
        result = assert_image_coherence(
            text="Test context",
            image=sample_image_bytes,
            judge_fn=_mock_judge_pass,
        )
        assert result.passed is True

    def test_boundary_score(self) -> None:
        """PASS: Boundary score (exactly min_score) passes."""
        result = assert_image_coherence(
            text="Boundary",
            image=None,
            judge_fn=_mock_judge_boundary,
            min_score=0.7,
            image_description="test",
        )
        assert result.passed is True

    def test_judge_error_fails(self) -> None:
        """FAIL: Judge error results in score 0.0."""
        with pytest.raises(MltkAssertionError):
            assert_image_coherence(
                text="Error test",
                image=None,
                judge_fn=_mock_judge_error,
                image_description="test",
            )

    def test_details_contain_text_context(self) -> None:
        """PASS: Details include the text context."""
        result = assert_image_coherence(
            text="Important context here",
            image=None,
            judge_fn=_mock_judge_pass,
            image_description="test",
        )
        assert (
            result.details["text_context"]
            == "Important context here"
        )


# ===============================================================
# Part 5: Image Helpfulness (vlm.py - new)
# ===============================================================


class TestImageHelpfulness:
    """Image helpfulness -- does image help answer the question?"""

    def test_pass_helpful_image(self) -> None:
        """PASS: Image is helpful for the question.

        WHY: A photo of a red car helps answer "What color is
        the car?"  The judge scores high.
        """
        result = assert_image_helpfulness(
            question="What color is the car?",
            image=None,
            answer="The car is red.",
            judge_fn=_mock_judge_pass,
            image_description="A red sedan in a parking lot.",
        )
        assert result.passed is True
        assert result.details["score"] == 0.85
        assert result.name == (
            "multimodal.vlm.image_helpfulness"
        )

    def test_fail_unhelpful_image(self) -> None:
        """FAIL: Image is irrelevant to the question.

        WHY: A photo of food does not help answer a question
        about car colors.  The judge scores low.
        """
        with pytest.raises(MltkAssertionError):
            assert_image_helpfulness(
                question="What color is the car?",
                image=None,
                answer="The car is red.",
                judge_fn=_mock_judge_fail,
                image_description="A plate of spaghetti.",
            )

    def test_with_image_bytes(
        self, sample_image_bytes: bytes
    ) -> None:
        """PASS: Raw bytes path works for helpfulness."""
        result = assert_image_helpfulness(
            question="Describe this image.",
            image=sample_image_bytes,
            answer="A small test image.",
            judge_fn=_mock_judge_pass,
        )
        assert result.passed is True

    def test_perfect_score(self) -> None:
        """PASS: Perfect helpfulness score."""
        result = assert_image_helpfulness(
            question="What is shown?",
            image=None,
            answer="A diagram of the system.",
            judge_fn=_mock_judge_high,
            image_description="System architecture diagram.",
        )
        assert result.passed is True
        assert result.details["score"] == 1.0

    def test_custom_min_score(self) -> None:
        """PASS: Custom min_score works correctly."""
        result = assert_image_helpfulness(
            question="Test",
            image=None,
            answer="Answer",
            judge_fn=_mock_judge_pass,  # 0.85
            min_score=0.5,
            image_description="test",
        )
        assert result.passed is True

    def test_details_contain_question_and_answer(self) -> None:
        """PASS: Details include question and answer."""
        result = assert_image_helpfulness(
            question="My question",
            image=None,
            answer="My answer",
            judge_fn=_mock_judge_pass,
            image_description="test",
        )
        assert result.details["question"] == "My question"
        assert result.details["answer"] == "My answer"

    def test_judge_error_fails(self) -> None:
        """FAIL: Judge exception results in score 0.0."""
        with pytest.raises(MltkAssertionError):
            assert_image_helpfulness(
                question="Error",
                image=None,
                answer="Error",
                judge_fn=_mock_judge_error,
                image_description="test",
            )

    def test_judge_garbage_fails(self) -> None:
        """FAIL: Unparseable judge response fails."""
        with pytest.raises(MltkAssertionError):
            assert_image_helpfulness(
                question="Garbage",
                image=None,
                answer="Garbage",
                judge_fn=_mock_judge_garbage,
                image_description="test",
            )


# ===============================================================
# Part 6: VQA Accuracy (vlm.py - new)
# ===============================================================


class TestVqaAccuracyExactMatch:
    """VQA accuracy with exact string matching (no judge)."""

    def test_exact_match_pass(self) -> None:
        """PASS: Expected and actual answers match exactly.

        WHY: For unambiguous answers (colors, counts, yes/no),
        exact match is fast and reliable.
        """
        result = assert_vqa_accuracy(
            question="How many dogs?",
            image=None,
            expected_answer="2",
            actual_answer="2",
        )
        assert result.passed is True
        assert result.details["score"] == 1.0
        assert result.details["method"] == "exact_match"
        assert result.name == "multimodal.vlm.vqa_accuracy"

    def test_exact_match_case_insensitive(self) -> None:
        """PASS: Case-insensitive comparison.

        WHY: "Red" and "red" are the same answer.  The
        normalization should handle this.
        """
        result = assert_vqa_accuracy(
            question="What color?",
            image=None,
            expected_answer="Red",
            actual_answer="red",
        )
        assert result.passed is True
        assert result.details["score"] == 1.0

    def test_exact_match_whitespace(self) -> None:
        """PASS: Leading/trailing whitespace is stripped.

        WHY: Model outputs often have extra whitespace.
        """
        result = assert_vqa_accuracy(
            question="Color?",
            image=None,
            expected_answer="blue",
            actual_answer="  blue  ",
        )
        assert result.passed is True

    def test_exact_match_fail(self) -> None:
        """FAIL: Different answers fail exact match.

        WHY: "red" != "blue" -- wrong VQA answer.
        """
        with pytest.raises(MltkAssertionError):
            assert_vqa_accuracy(
                question="What color?",
                image=None,
                expected_answer="red",
                actual_answer="blue",
            )

    def test_exact_match_partial_no_match(self) -> None:
        """FAIL: Partial match is not exact match.

        WHY: "a red car" != "red" in exact mode.  This is
        strict by design -- use judge mode for semantic match.
        """
        with pytest.raises(MltkAssertionError):
            assert_vqa_accuracy(
                question="What color?",
                image=None,
                expected_answer="red",
                actual_answer="a red car",
            )

    def test_exact_match_empty_strings(self) -> None:
        """PASS: Both empty strings match."""
        result = assert_vqa_accuracy(
            question="Anything?",
            image=None,
            expected_answer="",
            actual_answer="",
        )
        assert result.passed is True

    def test_exact_match_details(self) -> None:
        """PASS: Details contain all fields."""
        result = assert_vqa_accuracy(
            question="Q",
            image=None,
            expected_answer="A",
            actual_answer="a",
        )
        assert result.details["question"] == "Q"
        assert result.details["expected_answer"] == "A"
        assert result.details["actual_answer"] == "a"


class TestVqaAccuracyWithJudge:
    """VQA accuracy with LLM judge (semantic comparison)."""

    def test_judge_pass(self) -> None:
        """PASS: Judge scores high semantic equivalence.

        WHY: "two dogs" and "2 dogs" are semantically the same
        even though they differ as strings.  The judge catches this.
        """
        result = assert_vqa_accuracy(
            question="How many dogs?",
            image=None,
            expected_answer="two dogs",
            actual_answer="2 dogs",
            judge_fn=_mock_judge_pass,
            image_description="Photo with two dogs.",
        )
        assert result.passed is True
        assert result.details["method"] == "llm_judge"
        assert result.details["score"] == 0.85

    def test_judge_fail(self) -> None:
        """FAIL: Judge scores low -- semantically different."""
        with pytest.raises(MltkAssertionError):
            assert_vqa_accuracy(
                question="What color?",
                image=None,
                expected_answer="red",
                actual_answer="I see a blue sky",
                judge_fn=_mock_judge_fail,
                image_description="test",
            )

    def test_judge_with_image_bytes(
        self, sample_image_bytes: bytes
    ) -> None:
        """PASS: Judge mode with raw image bytes."""
        result = assert_vqa_accuracy(
            question="Describe this.",
            image=sample_image_bytes,
            expected_answer="test pattern",
            actual_answer="test pattern",
            judge_fn=_mock_judge_pass,
        )
        assert result.passed is True

    def test_judge_error_fails(self) -> None:
        """FAIL: Judge error results in score 0.0."""
        with pytest.raises(MltkAssertionError):
            assert_vqa_accuracy(
                question="Error",
                image=None,
                expected_answer="a",
                actual_answer="a",
                judge_fn=_mock_judge_error,
                image_description="test",
            )

    def test_judge_without_image_context(self) -> None:
        """PASS: Judge mode works without image context.

        WHY: VQA accuracy can be evaluated on the text answers
        alone -- the judge compares expected vs actual without
        needing the original image.
        """
        result = assert_vqa_accuracy(
            question="Capital of France?",
            image=None,
            expected_answer="Paris",
            actual_answer="Paris, France",
            judge_fn=_mock_judge_pass,
        )
        assert result.passed is True

    def test_judge_custom_min_score(self) -> None:
        """FAIL: Strict min_score with moderate judge score."""
        with pytest.raises(MltkAssertionError):
            assert_vqa_accuracy(
                question="Test",
                image=None,
                expected_answer="a",
                actual_answer="b",
                judge_fn=_mock_judge_pass,  # 0.85
                min_score=0.9,
                image_description="test",
            )


# ===============================================================
# Part 7: Edge Cases and Cross-Cutting Concerns
# ===============================================================


class TestImageDescriptionEscapeHatch:
    """image_description parameter -- the Pillow-free path.

    All four LLM-judge assertions accept image_description as an
    escape hatch.  When provided, the assertion never loads the
    image and Pillow is never imported.
    """

    def test_faithfulness_no_pillow_needed(self) -> None:
        """PASS: Prompt faithfulness with description only."""
        result = assert_prompt_faithfulness(
            prompt="Red car",
            image=None,
            judge_fn=_mock_judge_pass,
            image_description="A red car parked outside.",
        )
        assert result.passed is True

    def test_coherence_no_pillow_needed(self) -> None:
        """PASS: Image coherence with description only."""
        result = assert_image_coherence(
            text="Medical report",
            image=None,
            judge_fn=_mock_judge_pass,
            image_description="X-ray image.",
        )
        assert result.passed is True

    def test_helpfulness_no_pillow_needed(self) -> None:
        """PASS: Image helpfulness with description only."""
        result = assert_image_helpfulness(
            question="What is shown?",
            image=None,
            answer="A diagram.",
            judge_fn=_mock_judge_pass,
            image_description="Architecture diagram.",
        )
        assert result.passed is True

    def test_vqa_no_pillow_needed(self) -> None:
        """PASS: VQA accuracy with description only."""
        result = assert_vqa_accuracy(
            question="Color?",
            image=None,
            expected_answer="red",
            actual_answer="red",
            judge_fn=_mock_judge_pass,
            image_description="Red object.",
        )
        assert result.passed is True


class TestTimingDecorator:
    """Verify @timed_assertion populates duration_ms."""

    def test_faithfulness_has_timing(self) -> None:
        """PASS: duration_ms is populated."""
        result = assert_prompt_faithfulness(
            prompt="Timing test",
            image=None,
            judge_fn=_mock_judge_pass,
            image_description="test",
        )
        assert result.duration_ms >= 0.0

    def test_coherence_has_timing(self) -> None:
        """PASS: duration_ms is populated."""
        result = assert_image_coherence(
            text="Timing test",
            image=None,
            judge_fn=_mock_judge_pass,
            image_description="test",
        )
        assert result.duration_ms >= 0.0

    def test_helpfulness_has_timing(self) -> None:
        """PASS: duration_ms is populated."""
        result = assert_image_helpfulness(
            question="Timing",
            image=None,
            answer="test",
            judge_fn=_mock_judge_pass,
            image_description="test",
        )
        assert result.duration_ms >= 0.0

    def test_vqa_has_timing(self) -> None:
        """PASS: duration_ms is populated."""
        result = assert_vqa_accuracy(
            question="Timing",
            image=None,
            expected_answer="a",
            actual_answer="a",
        )
        assert result.duration_ms >= 0.0


class TestAssertionNames:
    """Verify assertion name conventions.

    All multimodal assertions must follow the naming pattern
    multimodal.{submodule}.{assertion_name}.
    """

    def test_prompt_faithfulness_name(self) -> None:
        result = assert_prompt_faithfulness(
            prompt="t",
            image=None,
            judge_fn=_mock_judge_pass,
            image_description="d",
        )
        assert result.name == (
            "multimodal.alignment.prompt_faithfulness"
        )

    def test_image_coherence_name(self) -> None:
        result = assert_image_coherence(
            text="t",
            image=None,
            judge_fn=_mock_judge_pass,
            image_description="d",
        )
        assert result.name == (
            "multimodal.alignment.image_coherence"
        )

    def test_image_helpfulness_name(self) -> None:
        result = assert_image_helpfulness(
            question="q",
            image=None,
            answer="a",
            judge_fn=_mock_judge_pass,
            image_description="d",
        )
        assert result.name == (
            "multimodal.vlm.image_helpfulness"
        )

    def test_vqa_accuracy_name(self) -> None:
        result = assert_vqa_accuracy(
            question="q",
            image=None,
            expected_answer="a",
            actual_answer="a",
        )
        assert result.name == "multimodal.vlm.vqa_accuracy"

    def test_image_text_alignment_name(self) -> None:
        """Existing assertion preserves its name."""
        img = np.array([[1.0, 0.0]])
        txt = np.array([[1.0, 0.0]])
        result = assert_image_text_alignment(
            img, txt, min_cosine=0.5
        )
        assert result.name == (
            "multimodal.image_text_alignment"
        )

    def test_cross_modal_consistency_name(self) -> None:
        """Existing assertion preserves its name."""
        result = assert_cross_modal_consistency(
            ["a"], ["a"], min_agreement=0.5
        )
        assert result.name == (
            "multimodal.cross_modal_consistency"
        )
