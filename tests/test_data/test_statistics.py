"""Tests for mltk.data.statistics — column-level statistical assertions.

Statistical assertions protect the distribution shape of features. A feature
can pass schema and range checks while still being statistically broken:
all values could be valid but clustered at 0 when they should be ~50,
or have a tiny stdev meaning the feature carries no useful signal.

These tests cover mean, median, stdev, and quantile assertions.
"""

import numpy as np
import pandas as pd
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.data.statistics import (
    assert_column_mean,
    assert_column_median,
    assert_column_stdev,
    assert_quantiles,
)

# ---------------------------------------------------------------------------
# assert_column_mean
# ---------------------------------------------------------------------------


class TestAssertColumnMean:
    """Tests for assert_column_mean — mean value bounds."""

    def test_mean_within_bounds(self) -> None:
        """SCENARIO: Age column with realistic mean around 35.
        WHY: Verify the happy path — a well-distributed column should pass.
        EXPECTED: pass=True, actual_mean stored in details.
        """
        df = pd.DataFrame({"age": [25, 30, 35, 40, 45]})
        result = assert_column_mean(df, "age", min_val=30.0, max_val=40.0)
        assert result.passed is True
        assert result.details["actual_mean"] == pytest.approx(35.0)

    def test_mean_below_min(self) -> None:
        """SCENARIO: Model output scores all near 0.1 but min_val=0.4.
        WHY: Catch a miscalibrated model where outputs are far too low.
        EXPECTED: MltkAssertionError raised, message mentions 'outside'.
        """
        df = pd.DataFrame({"score": [0.05, 0.10, 0.12, 0.08, 0.11]})
        with pytest.raises(MltkAssertionError) as exc:
            assert_column_mean(df, "score", min_val=0.4, max_val=0.8)
        assert "outside" in str(exc.value)

    def test_mean_above_max(self) -> None:
        """SCENARIO: Response time mean is 950ms but max_val=500ms.
        WHY: Catch latency regression — mean is way above acceptable threshold.
        EXPECTED: MltkAssertionError raised.
        """
        df = pd.DataFrame({"response_ms": [900, 950, 1000, 970, 930]})
        with pytest.raises(MltkAssertionError) as exc:
            assert_column_mean(df, "response_ms", min_val=0.0, max_val=500.0)
        assert "outside" in str(exc.value)

    def test_mean_no_bounds_raises(self) -> None:
        """SCENARIO: Caller forgets to provide any bound.
        WHY: An unbounded mean check is meaningless — fail loudly.
        EXPECTED: ValueError before any assertion logic runs.
        """
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="at least one bound"):
            assert_column_mean(df, "x")

    def test_mean_only_min_bound(self) -> None:
        """SCENARIO: Caller wants to ensure mean is at least 10.0, no upper limit.
        WHY: One-sided bounds are a valid use case — ensure partial check works.
        EXPECTED: pass=True when mean exceeds min_val with no max_val set.
        """
        df = pd.DataFrame({"metric": [15, 20, 25, 30, 35]})
        result = assert_column_mean(df, "metric", min_val=10.0)
        assert result.passed is True

    def test_mean_only_max_bound(self) -> None:
        """SCENARIO: Error rate must stay below 0.05 mean — no lower limit.
        WHY: One-sided upper bound is a valid monitoring pattern.
        EXPECTED: pass=True when mean is below max_val.
        """
        df = pd.DataFrame({"error_rate": [0.01, 0.02, 0.03, 0.01, 0.02]})
        result = assert_column_mean(df, "error_rate", max_val=0.05)
        assert result.passed is True

    def test_mean_duration_recorded(self) -> None:
        """SCENARIO: Verify the @timed_assertion decorator fires correctly.
        WHY: All assertions must carry timing data for performance monitoring.
        EXPECTED: duration_ms > 0.
        """
        df = pd.DataFrame({"v": [1, 2, 3, 4, 5]})
        result = assert_column_mean(df, "v", min_val=0.0, max_val=10.0)
        assert result.duration_ms > 0


# ---------------------------------------------------------------------------
# assert_column_median
# ---------------------------------------------------------------------------


class TestAssertColumnMedian:
    """Tests for assert_column_median — median value bounds."""

    def test_median_within_bounds(self) -> None:
        """SCENARIO: Income column with median around $50K, expected $40K-$60K.
        WHY: Median is more robust to outlier salaries than mean — check it.
        EXPECTED: pass=True, actual_median stored in details.
        """
        df = pd.DataFrame({"income": [30000, 45000, 50000, 55000, 200000]})
        result = assert_column_median(df, "income", min_val=40000.0, max_val=60000.0)
        assert result.passed is True
        assert result.details["actual_median"] == pytest.approx(50000.0)

    def test_median_below_min(self) -> None:
        """SCENARIO: User session length median dropped to 5s (expected ≥20s).
        WHY: A sudden drop in median session length signals product regression.
        EXPECTED: MltkAssertionError raised.
        """
        df = pd.DataFrame({"session_s": [2, 3, 5, 6, 7]})
        with pytest.raises(MltkAssertionError) as exc:
            assert_column_median(df, "session_s", min_val=20.0, max_val=300.0)
        assert "outside" in str(exc.value)

    def test_median_no_bounds_raises(self) -> None:
        """SCENARIO: Neither bound given.
        WHY: Must mirror assert_column_mean behavior — fail clearly.
        EXPECTED: ValueError.
        """
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="at least one bound"):
            assert_column_median(df, "x")


# ---------------------------------------------------------------------------
# assert_column_stdev
# ---------------------------------------------------------------------------


class TestAssertColumnStdev:
    """Tests for assert_column_stdev — standard deviation bounds."""

    def test_stdev_within_bounds(self) -> None:
        """SCENARIO: Normalized feature should have stdev near 1.0 (range 0.8-1.2).
        WHY: Verify that a standard-scaling step is working correctly.
        EXPECTED: pass=True.
        """
        rng = np.random.default_rng(0)
        values = rng.normal(loc=0.0, scale=1.0, size=1000).tolist()
        df = pd.DataFrame({"normalized": values})
        result = assert_column_stdev(df, "normalized", min_val=0.8, max_val=1.2)
        assert result.passed is True

    def test_stdev_too_high(self) -> None:
        """SCENARIO: Pixel intensity feature has stdev=120 but max allowed is 50.
        WHY: Too-wide a spread means the feature may have mixed multiple modalities.
        EXPECTED: MltkAssertionError raised.
        """
        df = pd.DataFrame({"pixel": [0, 50, 100, 150, 200, 250]})
        with pytest.raises(MltkAssertionError) as exc:
            assert_column_stdev(df, "pixel", min_val=0.0, max_val=50.0)
        assert "outside" in str(exc.value)

    def test_stdev_too_low(self) -> None:
        """SCENARIO: Constant-ish feature has stdev=0.001 but min required is 1.0.
        WHY: Near-zero stdev = zero-variance feature. Model can't learn from it.
        EXPECTED: MltkAssertionError raised.
        """
        df = pd.DataFrame({"constant": [5.001, 5.000, 5.001, 5.000, 5.001]})
        with pytest.raises(MltkAssertionError) as exc:
            assert_column_stdev(df, "constant", min_val=1.0)
        assert "outside" in str(exc.value)

    def test_stdev_no_bounds_raises(self) -> None:
        """SCENARIO: Neither bound given.
        WHY: Consistent with mean/median — fail loudly rather than silently.
        EXPECTED: ValueError.
        """
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError, match="at least one bound"):
            assert_column_stdev(df, "x")


# ---------------------------------------------------------------------------
# assert_quantiles
# ---------------------------------------------------------------------------


class TestAssertQuantiles:
    """Tests for assert_quantiles — multi-point distribution shape checks."""

    def test_quantiles_all_pass(self) -> None:
        """SCENARIO: Age distribution in training data should match known population stats.
        WHY: Quantile checks verify the whole shape of the distribution at once.
        EXPECTED: pass=True, all quantiles within their bounds.
        """
        rng = np.random.default_rng(42)
        ages = rng.normal(loc=35, scale=10, size=500).clip(18, 70).tolist()
        df = pd.DataFrame({"age": ages})
        result = assert_quantiles(df, "age", quantiles={
            0.10: (18.0, 28.0),
            0.50: (30.0, 40.0),
            0.90: (44.0, 58.0),
        })
        assert result.passed is True
        assert result.details["failures"] == 0

    def test_quantiles_one_fails(self) -> None:
        """SCENARIO: The 95th percentile of latency jumped above the SLA bound.
        WHY: The mean can look fine while tail latency blows up — quantile checks catch it.
        EXPECTED: MltkAssertionError raised, message identifies the failing quantile.
        """
        # 5 values at 490ms out of 20 total → p75 and above will be 490ms.
        # p50 is within [1, 60] (value = 10), p95 exceeds max bound of 400ms.
        latency = [10] * 15 + [490] * 5  # 20 values; p50=10, p75+=490
        df = pd.DataFrame({"latency_ms": latency})
        with pytest.raises(MltkAssertionError) as exc:
            assert_quantiles(df, "latency_ms", quantiles={
                0.50: (1.0, 60.0),
                0.95: (1.0, 400.0),
            })
        assert "out of bounds" in str(exc.value)

    def test_quantiles_empty_df(self) -> None:
        """SCENARIO: Upstream pipeline produced zero rows — quantile check receives empty df.
        WHY: Empty DataFrames should fail clearly, not raise a numpy/pandas error.
        EXPECTED: MltkAssertionError raised with informative message.
        """
        df = pd.DataFrame({"value": []})
        with pytest.raises(MltkAssertionError) as exc:
            assert_quantiles(df, "value", quantiles={0.5: (0.0, 100.0)})
        assert "empty" in str(exc.value).lower()

    def test_quantiles_multiple_failures_reported(self) -> None:
        """SCENARIO: Both Q25 and Q75 are out of bounds in a miscalibrated feature.
        WHY: All failing quantiles should be reported in one result, not just the first.
        EXPECTED: MltkAssertionError, failure count == 2 in details.
        """
        df = pd.DataFrame({"score": [0.01, 0.02, 0.03, 0.04, 0.05]})
        with pytest.raises(MltkAssertionError) as exc:
            assert_quantiles(df, "score", quantiles={
                0.25: (0.5, 1.0),   # actual ~0.02, will fail
                0.75: (0.5, 1.0),   # actual ~0.04, will fail
            })
        result = exc.value.result
        assert result.details["failures"] == 2

    def test_quantiles_details_contain_actual_values(self) -> None:
        """SCENARIO: Passing quantile check stores the actual computed values.
        WHY: Details must be present for debugging and audit trail, even on pass.
        EXPECTED: pass=True, actual_values dict has entries for each quantile.
        """
        df = pd.DataFrame({"x": [10, 20, 30, 40, 50]})
        result = assert_quantiles(df, "x", quantiles={0.50: (20.0, 40.0)})
        assert result.passed is True
        assert "Q0.50" in result.details["actual_values"]
