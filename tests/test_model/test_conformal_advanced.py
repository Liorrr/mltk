"""Tests for advanced conformal prediction assertions.

Tests for ``assert_conformal_calibration`` and ``assert_conditional_coverage``
which verify calibration fidelity and per-group coverage fairness for
conformal prediction intervals.
"""

import math

import numpy as np
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.core.result import Severity
from mltk.model.conformal import (
    assert_conditional_coverage,
    assert_conformal_calibration,
)

# ---------------------------------------------------------------------------
# assert_conformal_calibration
# ---------------------------------------------------------------------------


class TestConformalCalibrationPassing:
    """Cases where empirical coverage matches the nominal promise."""

    def test_perfect_calibration_at_90_passes(self) -> None:
        """PASS: Empirical coverage exactly matches 90% nominal.

        Scenario: 90 of 100 points are covered -- exactly what was
        promised.  Deviation is 0.0, well within any tolerance.
        """
        rng = np.random.default_rng(42)
        n = 100
        y_true = rng.normal(0, 1, n)
        y_lower = y_true - 1.0
        y_upper = y_true + 1.0
        # Ensure exactly 90 are covered, 10 are not
        y_lower[:10] = y_true[:10] + 10.0
        y_upper[:10] = y_true[:10] + 11.0

        result = assert_conformal_calibration(
            y_true, y_lower, y_upper, nominal_coverage=0.9, tolerance=0.02,
        )
        assert result.passed is True
        assert result.details["empirical_coverage"] == pytest.approx(0.9)
        assert result.details["deviation"] == pytest.approx(0.0)
        assert result.details["direction"] == "calibrated"
        assert result.details["n_total"] == 100

    def test_wide_tolerance_makes_imperfect_pass(self) -> None:
        """PASS: 85% coverage with tolerance=0.1 passes for 90% nominal.

        Scenario: Tolerance is wide enough to accommodate a 5-point
        shortfall.  |0.85 - 0.9| = 0.05 <= 0.1.
        """
        n = 20
        y_true = np.arange(n, dtype=float)
        y_lower = y_true - 0.5
        y_upper = y_true + 0.5
        # Miss 3 of 20 => 85% coverage
        y_lower[:3] = y_true[:3] + 100.0
        y_upper[:3] = y_true[:3] + 101.0

        result = assert_conformal_calibration(
            y_true, y_lower, y_upper,
            nominal_coverage=0.9,
            tolerance=0.1,
        )
        assert result.passed is True
        assert result.details["empirical_coverage"] == pytest.approx(0.85)


class TestConformalCalibrationFailing:
    """Cases where empirical coverage deviates too far from nominal."""

    def test_under_coverage_fails(self) -> None:
        """FAIL: 82% actual when 90% was promised -- under-covering.

        Scenario: Model under-covers by 8 percentage points.  With
        default tolerance=0.02 this is a clear miscalibration.  The
        direction should be "under".
        """
        n = 100
        y_true = np.arange(n, dtype=float)
        y_lower = y_true - 0.5
        y_upper = y_true + 0.5
        # Miss 18 of 100 => 82% coverage
        y_lower[:18] = y_true[:18] + 100.0
        y_upper[:18] = y_true[:18] + 101.0

        with pytest.raises(MltkAssertionError) as exc:
            assert_conformal_calibration(
                y_true, y_lower, y_upper,
                nominal_coverage=0.9,
                tolerance=0.02,
            )
        result = exc.value.result
        assert result.details["direction"] == "under"
        assert result.details["empirical_coverage"] == pytest.approx(0.82)
        assert "miscalibrated" in result.message

    def test_over_coverage_fails(self) -> None:
        """FAIL: 99% actual when 90% was promised -- over-covering.

        Scenario: Model is overly conservative, producing unnecessarily
        wide intervals.  The two-sided check catches this: 99% is far
        from the 90% promise.  Direction should be "over".
        """
        n = 100
        y_true = np.arange(n, dtype=float)
        y_lower = y_true - 0.5
        y_upper = y_true + 0.5
        # Miss only 1 of 100 => 99% coverage
        y_lower[0] = y_true[0] + 100.0
        y_upper[0] = y_true[0] + 101.0

        with pytest.raises(MltkAssertionError) as exc:
            assert_conformal_calibration(
                y_true, y_lower, y_upper,
                nominal_coverage=0.9,
                tolerance=0.02,
            )
        result = exc.value.result
        assert result.details["direction"] == "over"
        assert result.details["empirical_coverage"] == pytest.approx(0.99)

    def test_tight_tolerance_makes_near_perfect_fail(self) -> None:
        """FAIL: 91% coverage with tolerance=0.01 fails for 90% nominal.

        Scenario: Very tight tolerance rejects even small deviations.
        |0.91 - 0.9| = 0.01 is NOT <= 0.01 due to the strict inequality?
        Actually 0.01 <= 0.01 is True, so we need 92% to fail.
        Actually let's use 92%: |0.92 - 0.9| = 0.02 > 0.01.
        """
        n = 100
        y_true = np.arange(n, dtype=float)
        y_lower = y_true - 0.5
        y_upper = y_true + 0.5
        # Miss 8 of 100 => 92% coverage
        y_lower[:8] = y_true[:8] + 100.0
        y_upper[:8] = y_true[:8] + 101.0

        with pytest.raises(MltkAssertionError) as exc:
            assert_conformal_calibration(
                y_true, y_lower, y_upper,
                nominal_coverage=0.9,
                tolerance=0.01,
            )
        result = exc.value.result
        assert result.details["deviation"] == pytest.approx(0.02)
        assert result.details["direction"] == "over"


class TestConformalCalibrationEdgeCases:
    """Edge cases for conformal calibration."""

    def test_single_sample_covered(self) -> None:
        """Edge: Single covered sample gives 100% empirical coverage.

        Scenario: n=1, covered.  With nominal_coverage=0.9,
        deviation = |1.0 - 0.9| = 0.1, which exceeds default tolerance
        of 0.02.  Should fail (over-coverage).
        """
        y_true = np.array([5.0])
        y_lower = np.array([4.0])
        y_upper = np.array([6.0])

        with pytest.raises(MltkAssertionError) as exc:
            assert_conformal_calibration(
                y_true, y_lower, y_upper,
                nominal_coverage=0.9,
                tolerance=0.02,
            )
        result = exc.value.result
        assert result.details["n_total"] == 1
        assert result.details["empirical_coverage"] == 1.0
        assert result.details["direction"] == "over"

    def test_deviation_value_is_correct(self) -> None:
        """Verify the deviation field matches the actual difference.

        Scenario: 85 of 100 covered = 0.85 empirical, nominal 0.9.
        Deviation should be |0.85 - 0.9| = 0.05 exactly.
        """
        n = 100
        y_true = np.arange(n, dtype=float)
        y_lower = y_true - 0.5
        y_upper = y_true + 0.5
        # Miss 15 of 100 => 85% coverage
        y_lower[:15] = y_true[:15] + 100.0
        y_upper[:15] = y_true[:15] + 101.0

        result = assert_conformal_calibration(
            y_true, y_lower, y_upper,
            nominal_coverage=0.9,
            tolerance=0.1,  # wide enough to pass
            severity=Severity.WARNING,
        )
        assert result.details["deviation"] == pytest.approx(0.05)
        assert result.details["direction"] == "under"
        assert result.details["empirical_coverage"] == pytest.approx(0.85)
        assert result.details["nominal_coverage"] == 0.9

    def test_empty_arrays_fail(self) -> None:
        """FAIL: Empty arrays cannot produce calibration estimate."""
        with pytest.raises(MltkAssertionError):
            assert_conformal_calibration(
                np.array([]), np.array([]), np.array([]),
                nominal_coverage=0.9,
            )

    def test_has_duration_ms(self) -> None:
        """PASS: timed_assertion decorator populates duration_ms."""
        n = 50
        y_true = np.arange(n, dtype=float)
        y_lower = y_true - 0.5
        y_upper = y_true + 0.5
        # Miss 5 => 90%
        y_lower[:5] = y_true[:5] + 100.0
        y_upper[:5] = y_true[:5] + 101.0

        result = assert_conformal_calibration(
            y_true, y_lower, y_upper,
            nominal_coverage=0.9,
            tolerance=0.02,
        )
        assert result.duration_ms >= 0.0


# ---------------------------------------------------------------------------
# assert_conditional_coverage
# ---------------------------------------------------------------------------


class TestConditionalCoveragePassing:
    """Cases where all groups meet per-group coverage requirements."""

    def test_all_groups_above_threshold_passes(self) -> None:
        """PASS: Every group has coverage above min_group_coverage.

        Scenario: Two groups (A, B) each with perfect coverage.
        Both exceed the 80% minimum.
        """
        n_per_group = 20
        y_a = np.arange(n_per_group, dtype=float)
        y_b = np.arange(n_per_group, dtype=float) + 100.0
        y_true = np.concatenate([y_a, y_b])
        y_lower = y_true - 0.5
        y_upper = y_true + 0.5
        groups = np.array(["A"] * n_per_group + ["B"] * n_per_group)

        result = assert_conditional_coverage(
            y_true, y_lower, y_upper, groups,
            min_group_coverage=0.8,
            min_group_size=10,
        )
        assert result.passed is True
        assert result.details["per_group"]["A"]["coverage"] == pytest.approx(1.0)
        assert result.details["per_group"]["B"]["coverage"] == pytest.approx(1.0)
        assert result.details["groups_below_threshold"] == []
        assert result.details["groups_skipped"] == []

    def test_single_group_works(self) -> None:
        """PASS: Single group with adequate coverage.

        Scenario: All data belongs to one group. Conditional coverage
        reduces to marginal coverage.
        """
        n = 20
        y_true = np.arange(n, dtype=float)
        y_lower = y_true - 0.5
        y_upper = y_true + 0.5
        groups = np.array(["only_group"] * n)

        result = assert_conditional_coverage(
            y_true, y_lower, y_upper, groups,
            min_group_coverage=0.8,
            min_group_size=5,
        )
        assert result.passed is True
        assert "only_group" in result.details["per_group"]
        assert result.details["worst_group"] == "only_group"
        assert result.details["worst_coverage"] == pytest.approx(1.0)

    def test_groups_below_min_size_skipped(self) -> None:
        """PASS: Small groups are skipped, not used for pass/fail.

        Scenario: Group "tiny" has only 3 samples (below min_group_size
        of 10) with 0% coverage.  It should be skipped and not cause
        failure.  Group "big" has 20 samples with 100% coverage.
        """
        y_big = np.arange(20, dtype=float)
        y_tiny = np.array([100.0, 101.0, 102.0])
        y_true = np.concatenate([y_big, y_tiny])
        y_lower = y_true - 0.5
        y_upper = y_true + 0.5
        # Make tiny group have 0% coverage
        y_lower[-3:] = y_true[-3:] + 100.0
        y_upper[-3:] = y_true[-3:] + 101.0
        groups = np.array(["big"] * 20 + ["tiny"] * 3)

        result = assert_conditional_coverage(
            y_true, y_lower, y_upper, groups,
            min_group_coverage=0.8,
            min_group_size=10,
        )
        assert result.passed is True
        assert "tiny" in result.details["groups_skipped"]
        assert result.details["groups_below_threshold"] == []
        # tiny should still appear in per_group for transparency
        assert "tiny" in result.details["per_group"]


class TestConditionalCoverageFailing:
    """Cases where at least one group fails coverage requirements."""

    def test_one_weak_group_fails(self) -> None:
        """FAIL: One group has coverage below min_group_coverage.

        Scenario: Group A has 100% coverage, Group B has 50% coverage
        (only 10 of 20 covered).  The overall average might look fine,
        but per-group analysis reveals the problem.  worst_group should
        identify group B.
        """
        n_per = 20
        y_a = np.arange(n_per, dtype=float)
        y_b = np.arange(n_per, dtype=float) + 100.0
        y_true = np.concatenate([y_a, y_b])
        y_lower = y_true - 0.5
        y_upper = y_true + 0.5
        # Make group B have 50% coverage (miss 10 of 20)
        for i in range(10):
            idx = n_per + i
            y_lower[idx] = y_true[idx] + 100.0
            y_upper[idx] = y_true[idx] + 101.0
        groups = np.array(["A"] * n_per + ["B"] * n_per)

        with pytest.raises(MltkAssertionError) as exc:
            assert_conditional_coverage(
                y_true, y_lower, y_upper, groups,
                min_group_coverage=0.8,
                min_group_size=10,
            )
        result = exc.value.result
        assert result.details["worst_group"] == "B"
        assert result.details["worst_coverage"] == pytest.approx(0.5)
        assert "B" in result.details["groups_below_threshold"]
        assert "A" not in result.details["groups_below_threshold"]

    def test_mixed_groups_with_varying_coverage(self) -> None:
        """FAIL: Multiple groups with varying coverage, one below threshold.

        Scenario: Three groups (X, Y, Z) with coverages 100%, 90%, and
        60% respectively.  Only Z fails the 80% threshold.
        """
        n = 20
        y_x = np.arange(n, dtype=float)
        y_y = np.arange(n, dtype=float) + 50.0
        y_z = np.arange(n, dtype=float) + 100.0
        y_true = np.concatenate([y_x, y_y, y_z])
        y_lower = y_true - 0.5
        y_upper = y_true + 0.5

        # Miss 2 in Y (90%), miss 8 in Z (60%)
        for i in range(2):
            idx = n + i
            y_lower[idx] = y_true[idx] + 100.0
            y_upper[idx] = y_true[idx] + 101.0
        for i in range(8):
            idx = 2 * n + i
            y_lower[idx] = y_true[idx] + 100.0
            y_upper[idx] = y_true[idx] + 101.0

        groups = np.array(["X"] * n + ["Y"] * n + ["Z"] * n)

        with pytest.raises(MltkAssertionError) as exc:
            assert_conditional_coverage(
                y_true, y_lower, y_upper, groups,
                min_group_coverage=0.8,
                min_group_size=10,
            )
        result = exc.value.result
        assert result.details["worst_group"] == "Z"
        assert result.details["worst_coverage"] == pytest.approx(0.6)
        assert "Z" in result.details["groups_below_threshold"]
        # Y should NOT be below threshold (90% >= 80%)
        assert "Y" not in result.details["groups_below_threshold"]


class TestConditionalCoverageEdgeCases:
    """Edge cases for conditional coverage."""

    def test_empty_arrays_fail(self) -> None:
        """FAIL: Empty arrays cannot produce conditional coverage."""
        with pytest.raises(MltkAssertionError):
            assert_conditional_coverage(
                np.array([]), np.array([]), np.array([]),
                groups=[],
                min_group_coverage=0.8,
            )

    def test_all_groups_too_small_passes(self) -> None:
        """PASS: When all groups are below min_group_size, nothing to evaluate.

        Scenario: Every group has fewer samples than min_group_size.
        Since no group can be reliably evaluated, the assertion passes
        (no evidence of failure) and all groups appear in groups_skipped.
        """
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        y_lower = y_true + 100.0  # 0% coverage for all
        y_upper = y_true + 101.0
        groups = np.array(["A", "A", "B", "B", "C", "C"])

        result = assert_conditional_coverage(
            y_true, y_lower, y_upper, groups,
            min_group_coverage=0.8,
            min_group_size=10,
            severity=Severity.WARNING,
        )
        assert result.passed is True
        assert set(result.details["groups_skipped"]) == {"A", "B", "C"}
        assert result.details["groups_below_threshold"] == []
        assert result.details["worst_group"] is None
        assert math.isnan(result.details["worst_coverage"])

    def test_per_group_details_structure(self) -> None:
        """Verify per_group dict has the expected structure.

        Scenario: Two groups, verify that each group entry has
        'coverage', 'n_covered', and 'n_total' keys.
        """
        n = 15
        y_true = np.arange(2 * n, dtype=float)
        y_lower = y_true - 0.5
        y_upper = y_true + 0.5
        groups = np.array(["alpha"] * n + ["beta"] * n)

        result = assert_conditional_coverage(
            y_true, y_lower, y_upper, groups,
            min_group_coverage=0.8,
            min_group_size=5,
        )
        assert result.passed is True
        for g_label in ["alpha", "beta"]:
            g_info = result.details["per_group"][g_label]
            assert "coverage" in g_info
            assert "n_covered" in g_info
            assert "n_total" in g_info
            assert g_info["n_total"] == n
            assert g_info["n_covered"] == n
            assert g_info["coverage"] == pytest.approx(1.0)

    def test_has_duration_ms(self) -> None:
        """PASS: timed_assertion decorator populates duration_ms."""
        n = 20
        y_true = np.arange(n, dtype=float)
        y_lower = y_true - 0.5
        y_upper = y_true + 0.5
        groups = np.array(["G"] * n)

        result = assert_conditional_coverage(
            y_true, y_lower, y_upper, groups,
            min_group_coverage=0.8,
            min_group_size=5,
        )
        assert result.duration_ms >= 0.0
