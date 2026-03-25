"""Tests for face recognition assertions."""

import numpy as np
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.domains.cv.face import assert_face_far


class TestFaceFAR:
    """False Accept Rate tests."""

    def test_low_far(self) -> None:
        """PASS: Non-mate similarities well below mate similarities."""
        sims = np.array([0.95, 0.90, 0.85, 0.10, 0.05, 0.02])
        labels = np.array([1, 1, 1, 0, 0, 0])
        result = assert_face_far(sims, labels, max_far=0.1)
        assert result.passed is True

    def test_high_far(self) -> None:
        """FAIL: Non-mate similarities too close to mates."""
        sims = np.array([0.9, 0.85, 0.88, 0.87])
        labels = np.array([1, 1, 0, 0])
        with pytest.raises(MltkAssertionError):
            assert_face_far(sims, labels, max_far=0.01)

    def test_no_non_mates(self) -> None:
        """PASS: No non-mate pairs — nothing to check."""
        sims = np.array([0.95, 0.90])
        labels = np.array([1, 1])
        result = assert_face_far(sims, labels, max_far=0.001)
        assert result.passed is True

    def test_far_details(self) -> None:
        """Result includes FAR and threshold in details."""
        sims = np.array([0.9, 0.1])
        labels = np.array([1, 0])
        result = assert_face_far(sims, labels, max_far=0.5)
        assert "far" in result.details
        assert "threshold" in result.details
