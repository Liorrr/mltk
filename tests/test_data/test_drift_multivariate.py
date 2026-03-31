"""Tests for mltk.data.drift — MMD-based multivariate drift.

Univariate drift tests (KS, PSI) check one feature at a time.
Multivariate drift catches shifts in the joint distribution —
mean shifts, covariance changes, and correlation breakdowns that
per-feature tests miss entirely.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.data.drift import assert_no_multivariate_drift


class TestMultivariateDriftBasic:
    """MMD-based multivariate drift — core scenarios."""

    def test_identical_data_passes(self) -> None:
        """PASS: Same data produces high p-value.

        Scenario: Weekly batch matches training data exactly.
        """
        rng = np.random.default_rng(42)
        data = rng.standard_normal((500, 5))
        result = assert_no_multivariate_drift(
            data, data.copy()
        )
        assert result.passed is True

    def test_mean_shift_detected(self) -> None:
        """FAIL: Shifted mean triggers drift detection.

        Scenario: Feature means moved +2 across all dims —
        model inputs are far from training distribution.
        """
        rng = np.random.default_rng(42)
        ref = rng.standard_normal((500, 5))
        cur = ref + 2.0
        with pytest.raises(MltkAssertionError):
            assert_no_multivariate_drift(ref, cur)

    def test_covariance_shift_detected(self) -> None:
        """FAIL: Changed correlation caught by MMD.

        Scenario: Features that were independent are now
        correlated — KS per-feature would miss this.
        """
        rng = np.random.default_rng(42)
        ref = rng.standard_normal((500, 3))
        cur = ref.copy()
        cur[:, 1] = 0.9 * cur[:, 0] + 0.1 * cur[:, 1]
        with pytest.raises(MltkAssertionError):
            assert_no_multivariate_drift(ref, cur)

    def test_subtle_shift_with_large_sample(self) -> None:
        """FAIL: Small shift detectable with enough data.

        Scenario: Mean shifted by 0.3 — subtle but real.
        Large sample gives MMD enough power.
        """
        rng = np.random.default_rng(42)
        ref = rng.standard_normal((2000, 4))
        cur = rng.standard_normal((2000, 4)) + 0.3
        with pytest.raises(MltkAssertionError):
            assert_no_multivariate_drift(ref, cur)


class TestMultivariateDriftEdgeCases:
    """Edge cases and error handling."""

    def test_dimension_mismatch_fails(self) -> None:
        """FAIL: ref has 5 cols, cur has 3 — cannot compare.

        Scenario: Pipeline bug dropped two features.
        """
        rng = np.random.default_rng(42)
        ref = rng.standard_normal((500, 5))
        cur = rng.standard_normal((500, 3))
        with pytest.raises(
            (MltkAssertionError, ValueError)
        ):
            assert_no_multivariate_drift(ref, cur)

    def test_small_sample_warning(self) -> None:
        """Very small sample handled gracefully.

        Scenario: Only 5 rows available — permutation
        test has low power but should not crash.
        """
        rng = np.random.default_rng(42)
        ref = rng.standard_normal((5, 3))
        cur = rng.standard_normal((5, 3))
        result = assert_no_multivariate_drift(ref, cur)
        assert isinstance(result.passed, bool)

    def test_single_column(self) -> None:
        """Single column degenerates to univariate case.

        Scenario: Only one feature — MMD still works.
        """
        rng = np.random.default_rng(42)
        ref = rng.standard_normal((300, 1))
        cur = rng.standard_normal((300, 1))
        result = assert_no_multivariate_drift(ref, cur)
        assert result.passed is True

    def test_unknown_kernel_fails(self) -> None:
        """FAIL: Invalid kernel name rejected.

        Scenario: Typo in kernel parameter.
        """
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 3))
        with pytest.raises(
            (MltkAssertionError, ValueError)
        ):
            assert_no_multivariate_drift(
                data, data, kernel="invalid"
            )


class TestMultivariateDriftConfig:
    """Custom configuration and parameters."""

    def test_custom_sigma(self) -> None:
        """Custom sigma (RBF bandwidth) is respected.

        Scenario: Domain expert knows the right bandwidth.
        """
        rng = np.random.default_rng(42)
        data = rng.standard_normal((300, 4))
        result = assert_no_multivariate_drift(
            data, data.copy(), sigma=1.0
        )
        assert result.passed is True

    def test_custom_n_permutations(self) -> None:
        """Custom permutation count is respected.

        Scenario: Fewer permutations for faster CI runs.
        """
        rng = np.random.default_rng(42)
        data = rng.standard_normal((200, 3))
        result = assert_no_multivariate_drift(
            data, data.copy(), n_permutations=50
        )
        assert result.passed is True

    def test_custom_threshold(self) -> None:
        """Custom p-value threshold overrides default.

        Scenario: Stricter threshold for critical pipeline.
        """
        rng = np.random.default_rng(42)
        data = rng.standard_normal((300, 3))
        result = assert_no_multivariate_drift(
            data, data.copy(), threshold=0.01
        )
        assert result.passed is True


class TestMultivariateDriftInputFormats:
    """Input format handling — DataFrames, NaN rows."""

    def test_dataframe_input(self) -> None:
        """DataFrames accepted alongside ndarrays.

        Scenario: Data comes from pandas pipeline.
        """
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            rng.standard_normal((300, 3)),
            columns=["a", "b", "c"],
        )
        result = assert_no_multivariate_drift(
            df, df.copy()
        )
        assert result.passed is True

    def test_nan_rows_handled(self) -> None:
        """Rows with NaN are dropped before comparison.

        Scenario: Missing values in production batch.
        """
        rng = np.random.default_rng(42)
        ref = rng.standard_normal((300, 3))
        cur = ref.copy()
        cur[0, 0] = np.nan
        cur[5, 2] = np.nan
        result = assert_no_multivariate_drift(ref, cur)
        assert result.passed is True


class TestMultivariateDriftDetails:
    """Result details contain MMD statistic and p-value."""

    def test_details_include_p_value(self) -> None:
        """p-value present in result details.

        Scenario: QA report needs the raw p-value.
        """
        rng = np.random.default_rng(42)
        data = rng.standard_normal((300, 4))
        result = assert_no_multivariate_drift(
            data, data.copy()
        )
        assert "p_value" in result.details
        assert 0.0 <= result.details["p_value"] <= 1.0

    def test_details_include_mmd_statistic(self) -> None:
        """MMD statistic present in result details.

        Scenario: Logging raw MMD for trend monitoring.
        """
        rng = np.random.default_rng(42)
        data = rng.standard_normal((300, 4))
        result = assert_no_multivariate_drift(
            data, data.copy()
        )
        assert "statistic" in result.details or \
            "mmd" in result.details

    def test_subsampling_large_data(self) -> None:
        """Large datasets are subsampled for performance.

        Scenario: 50K-row production batch — MMD is O(n^2),
        so subsampling is critical.
        """
        rng = np.random.default_rng(42)
        ref = rng.standard_normal((5000, 3))
        cur = rng.standard_normal((5000, 3))
        result = assert_no_multivariate_drift(ref, cur)
        assert isinstance(result.passed, bool)


class TestMultivariateDriftHighDim:
    """High-dimensional and degenerate-dimension cases."""

    def test_mmd_high_dimensional_data(self) -> None:
        """FAIL: 50+ features, drift still detected.

        Scenario: Wide feature set (e.g. embeddings);
        mean shift of +1.5 must be caught even at d=50.
        """
        rng = np.random.default_rng(42)
        ref = rng.standard_normal((400, 50))
        cur = rng.standard_normal((400, 50)) + 1.5
        with pytest.raises(MltkAssertionError):
            assert_no_multivariate_drift(ref, cur)

    def test_mmd_identical_distributions_high_dim(
        self,
    ) -> None:
        """PASS: Same distribution in 50-dim passes.

        Scenario: High-dim data from same source — no
        false positive expected.
        """
        rng = np.random.default_rng(42)
        ref = rng.standard_normal((400, 50))
        cur = rng.standard_normal((400, 50))
        result = assert_no_multivariate_drift(
            ref, cur
        )
        assert result.passed is True

    def test_mmd_single_feature(self) -> None:
        """FAIL: 1-D data with big shift detected.

        Scenario: Single feature — degenerate case
        that must still work (not just multivariate).
        """
        rng = np.random.default_rng(42)
        ref = rng.standard_normal((300, 1))
        cur = rng.standard_normal((300, 1)) + 3.0
        with pytest.raises(MltkAssertionError):
            assert_no_multivariate_drift(ref, cur)

    def test_mmd_very_small_sample(self) -> None:
        """n=10 handled gracefully, no crash.

        Scenario: Tiny batch — permutation test has
        low power but must not error out.
        """
        rng = np.random.default_rng(42)
        ref = rng.standard_normal((10, 3))
        cur = rng.standard_normal((10, 3))
        result = assert_no_multivariate_drift(
            ref, cur
        )
        assert isinstance(result.passed, bool)

    def test_mmd_bandwidth_selection(self) -> None:
        """Auto bandwidth vs explicit sigma agree.

        Scenario: Identical data should pass regardless
        of whether sigma is auto-selected or explicit.
        """
        rng = np.random.default_rng(42)
        data = rng.standard_normal((300, 4))
        auto = assert_no_multivariate_drift(
            data, data.copy()
        )
        explicit = assert_no_multivariate_drift(
            data, data.copy(), sigma=1.0
        )
        assert auto.passed is True
        assert explicit.passed is True
