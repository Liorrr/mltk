"""Tests for mltk.monitor.anomaly -- anomaly detection on test metrics.

Anomaly detection catches *relative* regressions that absolute thresholds
miss.  A model scoring 88% accuracy still passes an ``assert acc >= 80%``
gate, but if it historically scored 95%, something is seriously wrong.

These tests verify all three detection methods (zscore, iqr, percentile),
their edge cases (short history, constant values, unknown methods), and
that result details contain the right diagnostic information.
"""
from __future__ import annotations

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.monitor.anomaly import assert_no_test_anomaly

# ---------------------------------------------------------------------------
# Z-score method
# ---------------------------------------------------------------------------


class TestZscore:
    """Z-score anomaly detection tests.

    Z = (current - mean) / std.  Anomalous if |Z| > threshold (default 3.0).
    A threshold of 3.0 means ~0.3% probability under normality.
    """

    def test_normal_value_passes(self):
        # SCENARIO: Current value is close to the historical mean
        # WHY: Normal fluctuations must not trigger false alarms
        # EXPECTED: result.passed is True, z_score is small
        history = [10.0, 10.1, 9.9, 10.2, 9.8, 10.0, 10.1, 9.9]
        result = assert_no_test_anomaly(history, 10.05, method="zscore", threshold=3.0)

        assert result.passed is True
        assert result.details["method"] == "zscore"
        assert abs(result.details["z_score"]) < 3.0

    def test_outlier_fails(self):
        # SCENARIO: Current value is far from the mean (e.g., latency spike)
        # WHY: A 5x deviation from mean must be caught as anomalous
        # EXPECTED: MltkAssertionError raised, z_score > threshold
        history = [10.0, 10.1, 9.9, 10.2, 9.8, 10.0, 10.1, 9.9]
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_no_test_anomaly(history, 50.0, method="zscore", threshold=3.0)

        details = exc_info.value.result.details
        assert details["is_anomaly"] is True
        assert "z_score" in details

    def test_details_contain_z_score(self):
        # SCENARIO: Verify that result details include z_score, mean, std
        # WHY: Diagnostic details are essential for debugging anomaly alerts
        # EXPECTED: details has z_score, mean, std keys
        history = [5.0, 5.1, 4.9, 5.0, 5.2, 4.8, 5.0, 5.1]
        result = assert_no_test_anomaly(history, 5.05, method="zscore")

        assert "z_score" in result.details
        assert "mean" in result.details
        assert "std" in result.details


# ---------------------------------------------------------------------------
# IQR method
# ---------------------------------------------------------------------------


class TestIQR:
    """IQR (Interquartile Range) anomaly detection tests.

    Outliers are outside [Q1 - k*IQR, Q3 + k*IQR] where k = threshold.
    Standard Tukey fence uses k=1.5.  More robust than Z-score for skewed data.
    """

    def test_within_range_passes(self):
        # SCENARIO: Current value is within the IQR whiskers
        # WHY: Values inside the box-plot whiskers are not outliers
        # EXPECTED: result.passed is True
        history = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        result = assert_no_test_anomaly(history, 5.5, method="iqr", threshold=1.5)

        assert result.passed is True
        assert result.details["method"] == "iqr"

    def test_outside_range_fails(self):
        # SCENARIO: Current value is far outside the IQR whiskers
        # WHY: A value 10x the range above Q3 is a clear outlier
        # EXPECTED: MltkAssertionError raised
        history = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_no_test_anomaly(history, 100.0, method="iqr", threshold=1.5)

        details = exc_info.value.result.details
        assert details["is_anomaly"] is True
        assert "lower_bound" in details
        assert "upper_bound" in details


# ---------------------------------------------------------------------------
# Percentile method
# ---------------------------------------------------------------------------


class TestPercentile:
    """Percentile-based anomaly detection tests.

    Flags values below the Nth or above the (100-N)th percentile.
    Non-parametric -- no normality assumption.
    """

    def test_normal_passes(self):
        # SCENARIO: Current value is near the median (50th percentile)
        # WHY: Middle-of-the-distribution values should never be flagged
        # EXPECTED: result.passed is True
        history = list(range(1, 101))  # 1..100
        result = assert_no_test_anomaly(history, 50.0, method="percentile", threshold=5.0)

        assert result.passed is True
        assert result.details["method"] == "percentile"

    def test_extreme_fails(self):
        # SCENARIO: Current value is far below the 5th percentile
        # WHY: A value at the 0.1th percentile (or below min) is extreme
        # EXPECTED: MltkAssertionError raised
        history = list(range(50, 150))  # 50..149
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_no_test_anomaly(history, 1.0, method="percentile", threshold=5.0)

        details = exc_info.value.result.details
        assert details["is_anomaly"] is True
        assert "lower_percentile" in details
        assert "upper_percentile" in details


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and error handling for anomaly detection."""

    def test_short_history_passes(self):
        # SCENARIO: History has fewer than 3 values
        # WHY: Statistical methods need a minimum sample size to be
        #   meaningful.  With 2 data points, std/IQR/percentile are
        #   unreliable.  Should pass with INFO severity rather than
        #   crash or give misleading results.
        # EXPECTED: result.passed is True, severity is INFO
        result = assert_no_test_anomaly([1.0, 2.0], 1.5, method="zscore")

        assert result.passed is True
        assert result.details["history_length"] == 2

    def test_constant_history_zscore(self):
        # SCENARIO: All historical values are identical (std = 0)
        # WHY: Division by zero in Z-score formula.  If history is always
        #   10.0 and current is 10.0 -> normal.  If current is 10.1 ->
        #   anomalous (any deviation from a constant is significant).
        # EXPECTED: exact match passes, any deviation fails
        history = [10.0, 10.0, 10.0, 10.0, 10.0]

        # Exact match: should pass
        result = assert_no_test_anomaly(history, 10.0, method="zscore")
        assert result.passed is True

        # Any deviation: should fail (infinite z-score)
        with pytest.raises(MltkAssertionError):
            assert_no_test_anomaly(history, 10.1, method="zscore")

    def test_unknown_method_raises(self):
        # SCENARIO: Caller passes an unsupported method name
        # WHY: Typos like "z-score" or "Z_SCORE" must produce a clear
        #   error, not silently pass or crash with a confusing traceback
        # EXPECTED: MltkAssertionError with "Unknown method" in message
        history = [1.0, 2.0, 3.0, 4.0, 5.0]
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_no_test_anomaly(history, 3.0, method="invalid_method")

        assert "Unknown method" in str(exc_info.value)

    def test_result_has_duration(self):
        # SCENARIO: The @timed_assertion decorator populates duration_ms
        # WHY: All timed assertions must report wall-clock execution time
        # EXPECTED: duration_ms >= 0.0
        history = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        result = assert_no_test_anomaly(history, 4.5, method="zscore")

        assert result.duration_ms >= 0.0
