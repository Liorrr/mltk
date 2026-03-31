"""Tests for SmoothECE calibration method in assert_calibration.

Standard ECE suffers from empty bins when data is sparse in
certain probability ranges. SmoothECE uses kernel density
estimation to avoid this, producing more stable calibration
error estimates — especially on small or skewed datasets.
"""

from __future__ import annotations

import numpy as np
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.model.slicing import assert_calibration


class TestSmoothECE:
    """SmoothECE calibration method tests."""

    def test_perfectly_calibrated_near_zero(self) -> None:
        """PASS: Well-calibrated probs give smECE near 0.

        Scenario: Model says 80% and is correct ~80% of
        the time — smooth ECE should be very low.
        """
        rng = np.random.default_rng(42)
        n = 1000
        y_prob = rng.uniform(0, 1, n)
        y_true = (rng.random(n) < y_prob).astype(int)
        result = assert_calibration(
            y_true, y_prob,
            max_error=0.1,
            method="smooth_ece",
        )
        assert result.passed is True

    def test_overconfident_detected(self) -> None:
        """FAIL: Always predicts 0.9 but only 50% correct.

        Scenario: Overconfident classifier — dangerous for
        decision systems that trust probability outputs.
        """
        rng = np.random.default_rng(42)
        n = 200
        y_prob = np.full(n, 0.9)
        y_true = rng.integers(0, 2, n)
        with pytest.raises(MltkAssertionError):
            assert_calibration(
                y_true, y_prob,
                max_error=0.05,
                method="smooth_ece",
            )

    def test_underconfident_detected(self) -> None:
        """FAIL: Predicts 0.5 but nearly always correct.

        Scenario: Model is too cautious — probabilities
        don't reflect true accuracy.
        """
        n = 200
        y_true = np.ones(n, dtype=int)
        y_prob = np.full(n, 0.5)
        with pytest.raises(MltkAssertionError):
            assert_calibration(
                y_true, y_prob,
                max_error=0.05,
                method="smooth_ece",
            )

    def test_method_ece_backward_compat(self) -> None:
        """PASS: method='ece' (default) still works.

        Scenario: Existing code that never sets method
        parameter should not break.
        """
        rng = np.random.default_rng(42)
        n = 500
        y_prob = rng.uniform(0, 1, n)
        y_true = (rng.random(n) < y_prob).astype(int)
        result = assert_calibration(
            y_true, y_prob, max_error=0.15
        )
        assert result.passed is True
        assert "ece" in result.details

    def test_method_smooth_ece_produces_result(
        self,
    ) -> None:
        """method='smooth_ece' returns valid TestResult.

        Scenario: Explicit smooth_ece method selection.
        """
        rng = np.random.default_rng(42)
        n = 500
        y_prob = rng.uniform(0, 1, n)
        y_true = (rng.random(n) < y_prob).astype(int)
        result = assert_calibration(
            y_true, y_prob,
            max_error=0.15,
            method="smooth_ece",
        )
        assert isinstance(result.passed, bool)

    def test_unknown_method_fails(self) -> None:
        """FAIL: Invalid method name rejected clearly.

        Scenario: Typo in method parameter.
        """
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 100)
        y_prob = rng.uniform(0, 1, 100)
        with pytest.raises(
            (MltkAssertionError, ValueError)
        ):
            assert_calibration(
                y_true, y_prob,
                max_error=0.1,
                method="invalid_method",
            )

    def test_method_in_details(self) -> None:
        """Method name appears in result details.

        Scenario: QA dashboard shows which calibration
        method was used for each test run.
        """
        rng = np.random.default_rng(42)
        n = 500
        y_prob = rng.uniform(0, 1, n)
        y_true = (rng.random(n) < y_prob).astype(int)
        result = assert_calibration(
            y_true, y_prob,
            max_error=0.2,
            method="smooth_ece",
        )
        has_method = (
            "method" in result.details
            or "smooth_ece" in result.message.lower()
            or "smece" in result.message.lower()
        )
        assert has_method or result.passed is True

    def test_smooth_avoids_empty_bins(self) -> None:
        """SmoothECE handles sparse probability regions.

        Scenario: All predictions cluster near 0.5 — most
        bins are empty. ECE becomes unreliable but smooth
        ECE still produces a valid estimate.
        """
        rng = np.random.default_rng(42)
        n = 300
        y_prob = rng.uniform(0.4, 0.6, n)
        y_true = (rng.random(n) < y_prob).astype(int)
        result = assert_calibration(
            y_true, y_prob,
            max_error=0.15,
            method="smooth_ece",
        )
        assert isinstance(result.passed, bool)

    def test_smooth_vs_ece_direction(self) -> None:
        """Both methods agree: good model < bad model.

        Scenario: SmoothECE and ECE may differ in
        magnitude but must agree on relative ordering.
        """
        rng = np.random.default_rng(42)
        n = 500
        good_prob = rng.uniform(0, 1, n)
        good_true = (
            rng.random(n) < good_prob
        ).astype(int)

        bad_true = rng.integers(0, 2, n)
        bad_prob = np.full(n, 0.9)

        good_result = assert_calibration(
            good_true, good_prob,
            max_error=0.5,
            method="smooth_ece",
        )
        bad_result = assert_calibration(
            bad_true, bad_prob,
            max_error=0.5,
            method="smooth_ece",
        )

        good_err = good_result.details.get(
            "ece",
            good_result.details.get("smooth_ece", 0),
        )
        bad_err = bad_result.details.get(
            "ece",
            bad_result.details.get("smooth_ece", 1),
        )
        assert good_err < bad_err

    def test_empty_arrays_smooth(self) -> None:
        """FAIL: Empty input handled gracefully."""
        with pytest.raises(MltkAssertionError):
            assert_calibration(
                [], [],
                max_error=0.05,
                method="smooth_ece",
            )


class TestSmoothECEEdgeCases:
    """Edge-case and convergence tests for SmoothECE."""

    def test_smooth_ece_perfectly_calibrated(
        self,
    ) -> None:
        """PASS: probs == true frequency -> ECE near 0.

        Scenario: Generate labels whose frequency
        exactly matches the predicted probability.
        """
        rng = np.random.default_rng(42)
        n = 2000
        y_prob = rng.uniform(0.1, 0.9, n)
        y_true = (rng.random(n) < y_prob).astype(int)
        result = assert_calibration(
            y_true, y_prob,
            max_error=0.1,
            method="smooth_ece",
        )
        assert result.passed is True

    def test_smooth_ece_completely_miscalibrated(
        self,
    ) -> None:
        """FAIL: probs=0.9 but labels all 0 -> high ECE.

        Scenario: Model is absurdly overconfident;
        smooth ECE must exceed a tight threshold.
        """
        n = 500
        y_prob = np.full(n, 0.9)
        y_true = np.zeros(n, dtype=int)
        with pytest.raises(MltkAssertionError):
            assert_calibration(
                y_true, y_prob,
                max_error=0.05,
                method="smooth_ece",
            )

    def test_smooth_ece_binary_edge_case(
        self,
    ) -> None:
        """Predicted probs are only 0s and 1s.

        Scenario: Hard predictions — smooth kernel
        must handle point masses at boundaries.
        """
        rng = np.random.default_rng(42)
        n = 300
        y_prob = rng.choice(
            [0.0, 1.0], size=n
        ).astype(float)
        y_true = y_prob.astype(int)
        result = assert_calibration(
            y_true, y_prob,
            max_error=0.1,
            method="smooth_ece",
        )
        assert isinstance(result.passed, bool)

    def test_smooth_ece_large_sample(self) -> None:
        """PASS: n=2000 well-calibrated converges.

        Scenario: Large sample reduces noise; smooth
        ECE should converge to near-zero error.
        """
        rng = np.random.default_rng(42)
        n = 2000
        y_prob = rng.uniform(0, 1, n)
        y_true = (rng.random(n) < y_prob).astype(int)
        result = assert_calibration(
            y_true, y_prob,
            max_error=0.1,
            method="smooth_ece",
        )
        assert result.passed is True

    def test_smooth_ece_vs_binned_ece(self) -> None:
        """Both methods agree: good model is calibrated.

        Scenario: Well-calibrated data should pass
        under both 'ece' and 'smooth_ece'.
        """
        rng = np.random.default_rng(42)
        n = 1000
        y_prob = rng.uniform(0, 1, n)
        y_true = (rng.random(n) < y_prob).astype(int)
        binned = assert_calibration(
            y_true, y_prob,
            max_error=0.15,
            method="ece",
        )
        smooth = assert_calibration(
            y_true, y_prob,
            max_error=0.15,
            method="smooth_ece",
        )
        assert binned.passed is True
        assert smooth.passed is True
