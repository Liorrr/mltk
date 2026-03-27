"""Tests for mltk.model.ab_test -- A/B test statistical significance.

Validates that assert_ab_significance correctly identifies statistically
significant improvements between two model versions using bootstrap
confidence intervals.
"""

import numpy as np
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.model.ab_test import assert_ab_significance


class TestABSignificance:
    """A/B test significance assertion tests."""

    def test_significant_improvement(self) -> None:
        """PASS: Model B is clearly better than A.

        WHY: When B consistently outperforms A by a large margin,
        the bootstrap CI should be entirely above 0, confirming
        statistical significance.
        """
        scores_a = [0.70, 0.72, 0.71, 0.69, 0.73, 0.70, 0.71, 0.72]
        scores_b = [0.90, 0.92, 0.91, 0.89, 0.93, 0.90, 0.91, 0.92]
        result = assert_ab_significance(scores_a, scores_b)
        assert result.passed is True
        assert result.details["mean_diff"] > 0
        assert result.details["ci_lower"] > 0
        assert result.details["p_value"] < 0.05

    def test_no_significant_difference(self) -> None:
        """FAIL: Models A and B are essentially the same.

        WHY: When scores overlap heavily, the CI should include 0,
        meaning we cannot reject the null hypothesis of no difference.
        """
        rng = np.random.default_rng(42)
        scores_a = rng.normal(0.80, 0.05, size=20).tolist()
        scores_b = rng.normal(0.80, 0.05, size=20).tolist()
        with pytest.raises(MltkAssertionError):
            assert_ab_significance(scores_a, scores_b)

    def test_empty_scores(self) -> None:
        """FAIL: Empty score arrays produce clear error.

        WHY: Guard against pipeline bugs that produce zero predictions.
        """
        with pytest.raises(MltkAssertionError) as exc:
            assert_ab_significance([], [])
        assert "empty" in str(exc.value).lower()

    def test_unequal_lengths(self) -> None:
        """FAIL: Mismatched array lengths produce clear error.

        WHY: A/B tests require paired samples. Different lengths mean
        the evaluation sets were not properly aligned.
        """
        with pytest.raises(MltkAssertionError) as exc:
            assert_ab_significance([0.8, 0.9], [0.85])
        assert "equal length" in str(exc.value)
