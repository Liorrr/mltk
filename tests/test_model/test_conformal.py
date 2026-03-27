"""Tests for mltk.model.conformal -- prediction interval and set validation.

These tests verify that assert_interval_coverage and assert_prediction_set_size
correctly gate on coverage thresholds and set size limits.
"""

import numpy as np
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.model.conformal import assert_interval_coverage, assert_prediction_set_size

# ---------------------------------------------------------------------------
# assert_interval_coverage
# ---------------------------------------------------------------------------


class TestIntervalCoveragePassing:
    """Cases where intervals meet coverage requirements."""

    def test_perfect_coverage_passes(self) -> None:
        """PASS: All true values fall inside their intervals.

        Scenario: Every prediction interval contains the ground truth,
        achieving 100% empirical coverage -- easily exceeds 90% target.
        """
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_lower = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        y_upper = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        result = assert_interval_coverage(y_true, y_lower, y_upper, target_coverage=0.9)
        assert result.passed is True
        assert result.details["empirical_coverage"] == 1.0
        assert result.details["n_covered"] == 5
        assert result.details["n_total"] == 5

    def test_partial_coverage_at_threshold_passes(self) -> None:
        """PASS: Coverage exactly at (target - tolerance) boundary.

        Scenario: 17 out of 20 points covered = 0.85, which equals
        the threshold of 0.9 - 0.05 = 0.85. Boundary passes.
        """
        rng = np.random.default_rng(42)
        n = 20
        y_true = rng.normal(0, 1, n)
        # Cover 17 of 20 points (85%)
        y_lower = y_true - 1.0
        y_upper = y_true + 1.0
        # Move 3 intervals so they miss
        y_lower[:3] = y_true[:3] + 10.0
        y_upper[:3] = y_true[:3] + 11.0
        result = assert_interval_coverage(
            y_true, y_lower, y_upper, target_coverage=0.9, tolerance=0.05
        )
        assert result.passed is True
        assert result.details["empirical_coverage"] == pytest.approx(0.85)

    def test_wide_intervals_larger_avg_width(self) -> None:
        """PASS: Wide intervals have larger reported avg_width than narrow ones.

        Scenario: Two models predicting the same targets -- one with tight
        intervals, one with wide intervals. Both cover 100%, but width differs.
        """
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Narrow intervals
        narrow_lo = y_true - 0.1
        narrow_hi = y_true + 0.1
        result_narrow = assert_interval_coverage(y_true, narrow_lo, narrow_hi)
        assert result_narrow.passed is True

        # Wide intervals
        wide_lo = y_true - 10.0
        wide_hi = y_true + 10.0
        result_wide = assert_interval_coverage(y_true, wide_lo, wide_hi)
        assert result_wide.passed is True

        assert result_wide.details["avg_width"] > result_narrow.details["avg_width"]
        assert result_narrow.details["avg_width"] == pytest.approx(0.2)
        assert result_wide.details["avg_width"] == pytest.approx(20.0)


class TestIntervalCoverageFailing:
    """Cases where intervals fail to meet coverage requirements."""

    def test_below_tolerance_fails(self) -> None:
        """FAIL: Coverage well below target - tolerance.

        Scenario: Only 2 of 10 points covered = 20% coverage.
        Target 90% - tolerance 5% = 85% required. Should fail hard.
        """
        y_true = np.arange(10, dtype=float)
        y_lower = np.full(10, 100.0)  # Intervals far away
        y_upper = np.full(10, 200.0)
        # Cover only 2 points
        y_lower[0] = -1.0
        y_upper[0] = 1.0
        y_lower[1] = 0.0
        y_upper[1] = 2.0
        with pytest.raises(MltkAssertionError) as exc:
            assert_interval_coverage(
                y_true, y_lower, y_upper, target_coverage=0.9, tolerance=0.05
            )
        assert "coverage=" in str(exc.value)

    def test_zero_coverage_fails(self) -> None:
        """FAIL: No points covered -- all intervals miss completely.

        Scenario: Intervals are entirely disjoint from true values.
        """
        y_true = np.array([1.0, 2.0, 3.0])
        y_lower = np.array([100.0, 200.0, 300.0])
        y_upper = np.array([101.0, 201.0, 301.0])
        with pytest.raises(MltkAssertionError):
            assert_interval_coverage(y_true, y_lower, y_upper, target_coverage=0.9)


class TestIntervalCoverageEdgeCases:
    """Edge cases for interval coverage."""

    def test_single_point_covered(self) -> None:
        """PASS: Single observation covered -- 100% coverage.

        Scenario: Edge case with n=1 sample. Coverage is either 0% or 100%.
        """
        y_true = np.array([5.0])
        y_lower = np.array([4.0])
        y_upper = np.array([6.0])
        result = assert_interval_coverage(y_true, y_lower, y_upper, target_coverage=0.9)
        assert result.passed is True
        assert result.details["empirical_coverage"] == 1.0
        assert result.details["n_total"] == 1
        assert result.details["median_width"] == pytest.approx(2.0)

    def test_single_point_not_covered(self) -> None:
        """FAIL: Single observation outside interval -- 0% coverage.

        Scenario: n=1 and the point is missed entirely.
        """
        y_true = np.array([5.0])
        y_lower = np.array([10.0])
        y_upper = np.array([20.0])
        with pytest.raises(MltkAssertionError):
            assert_interval_coverage(y_true, y_lower, y_upper, target_coverage=0.9)

    def test_empty_arrays_fail(self) -> None:
        """FAIL: Empty arrays cannot produce coverage.

        Scenario: Pipeline bug produces empty predictions.
        """
        with pytest.raises(MltkAssertionError):
            assert_interval_coverage(
                np.array([]), np.array([]), np.array([]), target_coverage=0.9
            )

    def test_details_contain_all_fields(self) -> None:
        """PASS: Result details include all documented fields.

        Scenario: Verify the full contract of returned metadata.
        """
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_lower = y_true - 0.5
        y_upper = y_true + 0.5
        result = assert_interval_coverage(y_true, y_lower, y_upper, target_coverage=0.9)
        assert result.passed is True
        expected_keys = {
            "empirical_coverage",
            "target_coverage",
            "tolerance",
            "n_covered",
            "n_total",
            "avg_width",
            "median_width",
        }
        assert expected_keys.issubset(result.details.keys())
        assert result.details["avg_width"] == pytest.approx(1.0)
        assert result.details["median_width"] == pytest.approx(1.0)

    def test_length_mismatch_fails(self) -> None:
        """FAIL: Mismatched array lengths are caught.

        Scenario: y_true has 3 elements but y_lower has 2.
        """
        with pytest.raises(MltkAssertionError) as exc:
            assert_interval_coverage(
                np.array([1.0, 2.0, 3.0]),
                np.array([0.0, 1.0]),
                np.array([2.0, 3.0, 4.0]),
            )
        assert "mismatch" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# assert_prediction_set_size
# ---------------------------------------------------------------------------


class TestPredictionSetSizePassing:
    """Cases where prediction sets meet size constraints."""

    def test_small_sets_pass(self) -> None:
        """PASS: Average set size is well below maximum.

        Scenario: Conformal classifier produces tight prediction sets
        with 1-2 classes each, well under max_avg_size=3.
        """
        sets = [{"cat"}, {"dog", "cat"}, {"bird"}, {"cat", "bird"}]
        result = assert_prediction_set_size(sets, max_avg_size=3.0)
        assert result.passed is True
        assert result.details["avg_size"] == pytest.approx(1.5)
        assert result.details["n_sets"] == 4

    def test_all_singletons(self) -> None:
        """PASS: Every set contains exactly one class.

        Scenario: High-confidence conformal classifier outputs singletons.
        Average size = 1.0, no empties.
        """
        sets = [["a"], ["b"], ["c"], ["d"], ["e"]]
        result = assert_prediction_set_size(sets, max_avg_size=2.0)
        assert result.passed is True
        assert result.details["avg_size"] == pytest.approx(1.0)
        assert result.details["min_size"] == pytest.approx(1.0)
        assert result.details["max_size"] == pytest.approx(1.0)
        assert result.details["empty_count"] == 0

    def test_regression_float_widths_pass(self) -> None:
        """PASS: Regression intervals with small average width.

        Scenario: Conformal regression produces interval widths as floats.
        Average width 2.0, well under max_avg_size=5.0.
        """
        widths = np.array([1.5, 2.0, 2.5, 1.8, 2.2])
        result = assert_prediction_set_size(widths, max_avg_size=5.0)
        assert result.passed is True
        assert result.details["avg_size"] == pytest.approx(2.0)
        assert result.details["n_sets"] == 5
        assert result.details["empty_count"] == 0


class TestPredictionSetSizeFailing:
    """Cases where prediction sets fail constraints."""

    def test_large_sets_fail(self) -> None:
        """FAIL: Average set size exceeds maximum.

        Scenario: Under-confident model outputs too many classes per set.
        """
        sets = [{"a", "b", "c", "d"}, {"a", "b", "c"}, {"a", "b", "c", "d", "e"}]
        with pytest.raises(MltkAssertionError) as exc:
            assert_prediction_set_size(sets, max_avg_size=2.0)
        assert "avg_size=" in str(exc.value)

    def test_empty_set_fraction_fails(self) -> None:
        """FAIL: Too many empty prediction sets.

        Scenario: Model produces empty prediction sets for uncertain inputs.
        3 of 5 empty = 60%, far above max_empty_frac=0.1.
        """
        sets: list[list[str]] = [[], [], [], ["a"], ["b"]]
        with pytest.raises(MltkAssertionError) as exc:
            assert_prediction_set_size(sets, max_avg_size=10.0, max_empty_frac=0.1)
        assert "empty_frac=" in str(exc.value)

    def test_both_conditions_fail(self) -> None:
        """FAIL: Both avg size and empty fraction exceed limits.

        Scenario: Model is both under-confident (large sets) and has
        missing predictions (empty sets).
        """
        sets: list[list[str]] = [
            [],
            [],
            ["a", "b", "c", "d", "e"],
            ["a", "b", "c", "d"],
        ]
        with pytest.raises(MltkAssertionError) as exc:
            assert_prediction_set_size(
                sets, max_avg_size=1.0, max_empty_frac=0.1
            )
        msg = str(exc.value)
        assert "avg_size=" in msg
        assert "empty_frac=" in msg


class TestPredictionSetSizeEdgeCases:
    """Edge cases for prediction set size."""

    def test_all_empty_sets_fail(self) -> None:
        """FAIL: Every prediction set is empty.

        Scenario: Broken model producing no predictions at all.
        """
        sets: list[list[str]] = [[], [], []]
        with pytest.raises(MltkAssertionError) as exc:
            assert_prediction_set_size(sets, max_avg_size=5.0, max_empty_frac=0.1)
        assert "empty_frac=" in str(exc.value)

    def test_all_empty_sets_pass_with_high_tolerance(self) -> None:
        """PASS: All empty sets tolerated when max_empty_frac=1.0.

        Scenario: User explicitly allows 100% empty sets, and avg_size=0
        is below max_avg_size.
        """
        from mltk.core.result import Severity

        sets: list[list[str]] = [[], [], []]
        result = assert_prediction_set_size(
            sets,
            max_avg_size=5.0,
            max_empty_frac=1.0,
            severity=Severity.WARNING,
        )
        assert result.passed is True
        assert result.details["empty_frac"] == pytest.approx(1.0)
        assert result.details["avg_size"] == pytest.approx(0.0)

    def test_regression_zero_widths_are_empty(self) -> None:
        """Regression mode: zero-width intervals count as empty.

        Scenario: Some intervals collapse to zero width, counted as empty.
        """
        widths = np.array([0.0, 0.0, 3.0, 4.0, 5.0])
        result = assert_prediction_set_size(
            widths, max_avg_size=10.0, max_empty_frac=0.5
        )
        assert result.passed is True
        assert result.details["empty_count"] == 2
        assert result.details["empty_frac"] == pytest.approx(0.4)

    def test_empty_input_fails(self) -> None:
        """FAIL: Empty input list produces error.

        Scenario: Pipeline bug produces no prediction sets at all.
        """
        with pytest.raises(MltkAssertionError):
            assert_prediction_set_size([], max_avg_size=5.0)

    def test_details_contain_all_fields(self) -> None:
        """PASS: Result details include all documented fields.

        Scenario: Verify the full contract of returned metadata.
        """
        sets = [{"a", "b"}, {"c"}]
        result = assert_prediction_set_size(sets, max_avg_size=5.0)
        assert result.passed is True
        expected_keys = {
            "avg_size",
            "max_size",
            "min_size",
            "empty_count",
            "empty_frac",
            "n_sets",
        }
        assert expected_keys.issubset(result.details.keys())
        assert result.details["max_size"] == pytest.approx(2.0)
        assert result.details["min_size"] == pytest.approx(1.0)

    def test_has_duration_ms(self) -> None:
        """PASS: timed_assertion decorator populates duration_ms.

        Scenario: Verify the decorator wiring is correct.
        """
        sets = [{"a"}, {"b"}]
        result = assert_prediction_set_size(sets, max_avg_size=5.0)
        assert result.duration_ms >= 0.0


# ---------------------------------------------------------------------------
# Hardening: parametrized, edge-case, and integration tests (S-hardening)
# ---------------------------------------------------------------------------


class TestIntervalCoverageParametrized:
    """Parametrized interval coverage tests across target levels."""

    @pytest.mark.parametrize(
        "target_coverage",
        [0.5, 0.7, 0.8, 0.9, 0.95],
        ids=["50pct", "70pct", "80pct", "90pct", "95pct"],
    )
    def test_perfect_coverage_at_various_targets(
        self, target_coverage: float
    ) -> None:
        """PASS: 100% empirical coverage passes at every target level.

        WHY: If every true value is inside its interval, empirical coverage
        is 1.0 which always exceeds any target <= 1.0.
        """
        rng = np.random.default_rng(42)
        n = 100
        y_true = rng.normal(0, 1, n)
        y_lower = y_true - 2.0
        y_upper = y_true + 2.0
        result = assert_interval_coverage(
            y_true, y_lower, y_upper, target_coverage=target_coverage,
        )
        assert result.passed is True
        assert result.details["empirical_coverage"] == pytest.approx(1.0)
        assert result.details["target_coverage"] == target_coverage


class TestPredictionSetSizeHardening:
    """Additional edge cases for prediction set size."""

    def test_nested_lists_different_lengths(self) -> None:
        """PASS: Sets of varying sizes are handled correctly.

        WHY: Real conformal classifiers produce sets of different sizes
        depending on input confidence. The assertion must compute the
        average over heterogeneous set sizes.
        """
        sets = [
            ["a"],
            ["a", "b", "c", "d", "e"],
            ["a", "b"],
            [],
            ["a", "b", "c"],
        ]
        result = assert_prediction_set_size(
            sets, max_avg_size=5.0, max_empty_frac=0.5,
        )
        assert result.passed is True
        # avg = (1 + 5 + 2 + 0 + 3) / 5 = 2.2
        assert result.details["avg_size"] == pytest.approx(2.2)
        assert result.details["max_size"] == pytest.approx(5.0)
        assert result.details["min_size"] == pytest.approx(0.0)
        assert result.details["empty_count"] == 1


class TestConformalPerformance:
    """Performance tests for conformal prediction assertions."""

    def test_large_arrays_interval_coverage(self) -> None:
        """PERF: 10,000 element arrays complete without error.

        WHY: Production conformal prediction often produces thousands of
        intervals. The assertion must handle this without performance issues.
        """
        rng = np.random.default_rng(42)
        n = 10_000
        y_true = rng.normal(0, 1, n)
        y_lower = y_true - 1.5
        y_upper = y_true + 1.5
        result = assert_interval_coverage(
            y_true, y_lower, y_upper, target_coverage=0.9,
        )
        assert result.passed is True
        assert result.details["n_total"] == 10_000
        assert result.details["empirical_coverage"] == pytest.approx(1.0)

    def test_large_arrays_prediction_set_size(self) -> None:
        """PERF: 10,000 regression widths complete without error."""
        rng = np.random.default_rng(42)
        widths = rng.uniform(0.5, 3.0, 10_000)
        result = assert_prediction_set_size(widths, max_avg_size=5.0)
        assert result.passed is True
        assert result.details["n_sets"] == 10_000


class TestConformalEdgeCasesHardening:
    """Additional edge cases for conformal assertions."""

    def test_nan_in_y_true(self) -> None:
        """EDGE: NaN in y_true should produce coverage < 1.0.

        WHY: NaN comparisons return False in numpy, so NaN values are never
        covered by any interval. The assertion should not crash; it should
        count NaN entries as uncovered.
        """
        y_true = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        y_lower = np.array([0.5, 0.5, 2.5, 3.5, 4.5])
        y_upper = np.array([1.5, 1.5, 3.5, 4.5, 5.5])
        result = assert_interval_coverage(
            y_true, y_lower, y_upper, target_coverage=0.5, tolerance=0.3,
        )
        # 4 out of 5 covered (NaN is not covered)
        assert result.details["empirical_coverage"] == pytest.approx(0.8)
        assert result.details["n_covered"] == 4

    def test_negative_interval_widths(self) -> None:
        """EDGE: When y_lower > y_upper, no point can be covered.

        WHY: Inverted intervals (lower bound above upper bound) are
        mathematically empty sets. Coverage should be 0% and the assertion
        should fail at any non-zero target.
        """
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_lower = np.array([10.0, 20.0, 30.0, 40.0, 50.0])  # higher than y_upper
        y_upper = np.array([0.0, 0.0, 0.0, 0.0, 0.0])        # lower than y_lower
        with pytest.raises(MltkAssertionError):
            assert_interval_coverage(
                y_true, y_lower, y_upper, target_coverage=0.9,
            )

    def test_integer_arrays(self) -> None:
        """PASS: Integer arrays are accepted and produce correct coverage.

        WHY: Users may pass integer labels or counts. The assertion must
        handle int dtype via np.asarray conversion without error.
        """
        y_true = np.array([1, 2, 3, 4, 5])
        y_lower = np.array([0, 1, 2, 3, 4])
        y_upper = np.array([2, 3, 4, 5, 6])
        result = assert_interval_coverage(
            y_true, y_lower, y_upper, target_coverage=0.9,
        )
        assert result.passed is True
        assert result.details["empirical_coverage"] == pytest.approx(1.0)
        assert result.details["avg_width"] == pytest.approx(2.0)

    def test_coverage_boundary_with_tolerance(self) -> None:
        """PASS: Coverage exactly at (target - tolerance) passes.

        WHY: The threshold is inclusive (>=), so empirical coverage that exactly
        equals target_coverage - tolerance should pass.
        """
        # 8 out of 10 covered = 0.80 == 0.90 - 0.10
        y_true = np.arange(10, dtype=float)
        y_lower = y_true - 0.5
        y_upper = y_true + 0.5
        # Move 2 intervals far away
        y_lower[0] = 100.0
        y_upper[0] = 101.0
        y_lower[1] = 100.0
        y_upper[1] = 101.0
        result = assert_interval_coverage(
            y_true, y_lower, y_upper, target_coverage=0.9, tolerance=0.1,
        )
        assert result.passed is True
        assert result.details["empirical_coverage"] == pytest.approx(0.8)
