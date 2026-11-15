"""Tests for face recognition assertions."""

import numpy as np
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.domains.cv.face import assert_face_far


class TestFaceFAR:
    """False Accept Rate at the operating threshold."""

    def test_low_far(self) -> None:
        """PASS: Non-mate similarities well below the threshold."""
        sims = np.array([0.95, 0.90, 0.85, 0.10, 0.05, 0.02])
        labels = np.array([1, 1, 1, 0, 0, 0])
        result = assert_face_far(sims, labels, threshold=0.5, max_far=0.1)
        assert result.passed is True
        assert result.details["far"] == 0.0
        assert result.details["tar"] == 1.0

    def test_high_far(self) -> None:
        """FAIL: Non-mate similarities above the threshold."""
        sims = np.array([0.9, 0.85, 0.88, 0.87])
        labels = np.array([1, 1, 0, 0])
        with pytest.raises(MltkAssertionError):
            assert_face_far(sims, labels, threshold=0.8, max_far=0.01)

    def test_hand_computed_far_and_tar(self) -> None:
        """FAR = 1/4 non-mates accepted, TAR = 2/3 mates accepted at 0.5."""
        sims = [0.9, 0.8, 0.4, 0.7, 0.3, 0.2, 0.1]
        labels = [1, 1, 1, 0, 0, 0, 0]
        result = assert_face_far(sims, labels, threshold=0.5, max_far=0.25)
        d = result.details
        assert result.passed is True
        assert d["far"] == pytest.approx(0.25)
        assert d["false_accepts"] == 1
        assert d["total_non_mates"] == 4
        assert d["tar"] == pytest.approx(2 / 3)
        assert d["true_accepts"] == 2
        assert d["total_mates"] == 3
        assert d["threshold"] == 0.5

    def test_boundary_equality_is_an_accept(self) -> None:
        """A similarity exactly at the threshold is accepted (>=)."""
        sims = [0.6, 0.5, 0.6, 0.5, 0.1]
        labels = [1, 1, 0, 0, 0]
        with pytest.raises(MltkAssertionError) as exc:
            assert_face_far(sims, labels, threshold=0.5, max_far=0.5)
        d = exc.value.result.details
        assert d["false_accepts"] == 2  # 0.6 and 0.5 (== threshold)
        assert d["far"] == pytest.approx(2 / 3)
        assert d["true_accepts"] == 2  # the mate at exactly 0.5 counts
        result = assert_face_far(sims, labels, threshold=0.5000001, max_far=0.5)
        assert result.details["false_accepts"] == 1

    def test_operating_threshold_not_median_heuristic(self) -> None:
        """Mirror of the removed heuristic: the median of mate scores (0.8)
        gives FAR 0 and would pass; at the deployed threshold (0.5) two of
        four non-mates are accepted, so the result must fail and report 0.5.
        """
        sims = [0.9, 0.8, 0.7, 0.6, 0.65, 0.1, 0.2]
        labels = [1, 1, 1, 0, 0, 0, 0]
        median_threshold = float(np.median([0.9, 0.8, 0.7]))
        with pytest.raises(MltkAssertionError) as exc:
            assert_face_far(sims, labels, threshold=0.5, max_far=0.1)
        d = exc.value.result.details
        assert d["threshold"] == 0.5
        assert d["threshold"] != median_threshold
        assert d["far"] == pytest.approx(0.5)
        # At the median the same data would pass -- that was the old report.
        old = assert_face_far(sims, labels, threshold=median_threshold, max_far=0.1)
        assert old.details["far"] == 0.0

    def test_threshold_is_required(self) -> None:
        with pytest.raises(TypeError):
            assert_face_far([0.9, 0.1], [1, 0], max_far=0.5)  # type: ignore[call-arg]

    def test_threshold_is_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            assert_face_far([0.9, 0.1], [1, 0], 0.5)  # type: ignore[misc]


class TestFaceFAREmpty:
    """Empty mate or non-mate sets follow on_empty and fail by default."""

    def test_no_non_mates_fails_by_default(self) -> None:
        with pytest.raises(MltkAssertionError, match="FAR is undefined"):
            assert_face_far([0.95, 0.90], [1, 1], threshold=0.5)

    def test_no_mates_fails_by_default(self) -> None:
        with pytest.raises(MltkAssertionError, match="TAR at the threshold is undefined"):
            assert_face_far([0.2, 0.1], [0, 0], threshold=0.5)

    def test_empty_input_fails_by_default(self) -> None:
        with pytest.raises(MltkAssertionError):
            assert_face_far([], [], threshold=0.5)

    def test_no_non_mates_skip_and_pass(self) -> None:
        skipped = assert_face_far([0.9], [1], threshold=0.5, on_empty="skip")
        assert skipped.passed is True
        assert skipped.details["skipped"] is True
        passed = assert_face_far([0.9], [1], threshold=0.5, on_empty="pass")
        assert passed.passed is True
        assert passed.details.get("skipped") is not True

    def test_no_mates_skip(self) -> None:
        result = assert_face_far([0.2, 0.1], [0, 0], threshold=0.5, on_empty="skip")
        assert result.details["skipped"] is True

    def test_no_mates_pass_still_checks_far(self) -> None:
        """on_empty='pass' waives the empty mate set but FAR is still checked."""
        result = assert_face_far([0.2, 0.1], [0, 0], threshold=0.5, on_empty="pass")
        assert result.passed is True
        assert result.details["tar"] is None
        assert result.details["far"] == 0.0
        with pytest.raises(MltkAssertionError, match="FAR too high"):
            assert_face_far([0.9, 0.1], [0, 0], threshold=0.5, on_empty="pass")

    def test_unknown_on_empty_fails(self) -> None:
        with pytest.raises(MltkAssertionError, match="Unknown on_empty"):
            assert_face_far([0.9, 0.1], [1, 0], threshold=0.5, on_empty="zero")  # type: ignore[arg-type]


class TestFaceFARValidation:
    """Malformed inputs raise ValueError instead of producing a number."""

    def test_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="equal lengths"):
            assert_face_far([0.9, 0.1, 0.2], [1, 0], threshold=0.5)

    @pytest.mark.parametrize("labels", [[1, 2], [1, -1], [1, 0.5], ["a", "b"]])
    def test_labels_outside_zero_one(self, labels: list) -> None:
        with pytest.raises(ValueError, match="labels must be 0"):
            assert_face_far([0.9, 0.1], labels, threshold=0.5)

    def test_bool_labels_accepted(self) -> None:
        result = assert_face_far([0.9, 0.1], [True, False], threshold=0.5, max_far=0.5)
        assert result.details["total_mates"] == 1

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_similarity(self, bad: float) -> None:
        with pytest.raises(ValueError, match="finite"):
            assert_face_far([0.9, bad], [1, 0], threshold=0.5)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_non_finite_threshold(self, bad: float) -> None:
        # A NaN threshold would make every comparison False and FAR 0.
        with pytest.raises(ValueError, match="threshold must be a finite"):
            assert_face_far([0.9, 0.8], [1, 0], threshold=bad)

    @pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan")])
    def test_max_far_out_of_range(self, bad: float) -> None:
        with pytest.raises(ValueError, match="max_far"):
            assert_face_far([0.9, 0.1], [1, 0], threshold=0.5, max_far=bad)

    def test_two_dimensional_rejected(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            assert_face_far([[0.9, 0.1]], [[1, 0]], threshold=0.5)
