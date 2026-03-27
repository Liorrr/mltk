"""Tests for mltk.data.synthetic -- synthetic data quality assertions.

Synthetic data quality has three dimensions: fidelity (does it look real?),
novelty (is it new, not memorized?), and privacy (is it far enough from real
records?). Each test class targets one assertion and covers pass, fail, edge
cases, and configuration options.
"""

import numpy as np
import pandas as pd
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.data.synthetic import (
    assert_correlation_preserved,
    assert_dcr_safe,
    assert_marginal_fidelity,
    assert_synthetic_novelty,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def real_df() -> pd.DataFrame:
    """Real dataset with correlated features for testing."""
    rng = np.random.default_rng(42)
    n = 500
    age = rng.normal(35, 10, n)
    income = age * 1000 + rng.normal(0, 5000, n)
    score = rng.uniform(0, 100, n)
    return pd.DataFrame({"age": age, "income": income, "score": score})


@pytest.fixture
def good_synthetic_df() -> pd.DataFrame:
    """Synthetic data drawn from the same generating process -- should pass."""
    rng = np.random.default_rng(99)
    n = 500
    age = rng.normal(35, 10, n)
    income = age * 1000 + rng.normal(0, 5000, n)
    score = rng.uniform(0, 100, n)
    return pd.DataFrame({"age": age, "income": income, "score": score})


@pytest.fixture
def bad_synthetic_df() -> pd.DataFrame:
    """Synthetic data with broken correlations and shifted distributions."""
    rng = np.random.default_rng(77)
    n = 500
    age = rng.normal(60, 5, n)  # shifted age
    income = rng.normal(50000, 10000, n)  # independent of age
    score = rng.uniform(0, 100, n)
    return pd.DataFrame({"age": age, "income": income, "score": score})


# ===========================================================================
# 1. Marginal Fidelity
# ===========================================================================


class TestMarginalFidelity:
    """Marginal fidelity: does each synthetic column match the real distribution?"""

    def test_same_distribution_passes(self) -> None:
        """PASS: Synthetic drawn from identical distribution.

        Scenario: Generator perfectly learned the marginal for this column.
        """
        rng = np.random.default_rng(42)
        real = pd.Series(rng.normal(0, 1, 1000))
        synth = pd.Series(np.random.default_rng(99).normal(0, 1, 1000))
        result = assert_marginal_fidelity(real, synth)
        assert result.passed is True
        assert result.details["statistic"] < 0.1
        assert result.details["method"] == "ks"
        assert result.details["n_real"] == 1000
        assert result.details["n_synthetic"] == 1000

    def test_shifted_distribution_fails(self) -> None:
        """FAIL: Mean-shifted distribution detected as unfaithful.

        Scenario: Generator produces ages centered at 60 instead of 35 --
        the marginal is clearly wrong.
        """
        rng = np.random.default_rng(42)
        real = pd.Series(rng.normal(0, 1, 1000))
        synth = pd.Series(np.random.default_rng(99).normal(5, 1, 1000))
        with pytest.raises(MltkAssertionError) as exc:
            assert_marginal_fidelity(real, synth)
        assert "fidelity failed" in str(exc.value)

    def test_psi_method_works(self) -> None:
        """PASS: PSI method on identical distributions gives low divergence.

        PSI is the industry standard for monitoring distribution shifts in
        financial models -- it should also work for synthetic data validation.
        """
        rng = np.random.default_rng(42)
        real = pd.Series(rng.normal(0, 1, 1000))
        synth = pd.Series(np.random.default_rng(99).normal(0, 1, 1000))
        result = assert_marginal_fidelity(real, synth, method="psi")
        assert result.passed is True
        assert result.details["method"] == "psi"
        assert result.details["statistic"] < 0.1

    def test_psi_method_detects_shift(self) -> None:
        """FAIL: PSI detects a shifted distribution."""
        rng = np.random.default_rng(42)
        real = pd.Series(rng.normal(0, 1, 1000))
        synth = pd.Series(np.random.default_rng(99).normal(5, 1, 1000))
        with pytest.raises(MltkAssertionError):
            assert_marginal_fidelity(real, synth, method="psi")

    def test_unknown_method_fails(self) -> None:
        """FAIL: Invalid method name produces a clear error."""
        real = pd.Series([1.0, 2.0, 3.0])
        synth = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(MltkAssertionError) as exc:
            assert_marginal_fidelity(real, synth, method="invalid")
        assert "Unknown method" in str(exc.value)

    def test_custom_threshold(self) -> None:
        """Custom max_divergence overrides the default."""
        rng = np.random.default_rng(42)
        real = pd.Series(rng.normal(0, 1, 500))
        synth = pd.Series(np.random.default_rng(99).normal(0.3, 1, 500))
        # Lenient threshold should pass
        result = assert_marginal_fidelity(real, synth, max_divergence=0.5)
        assert result.passed is True

    def test_empty_series_fails(self) -> None:
        """FAIL: Empty input produces a clear error, not a crash."""
        real = pd.Series(dtype=float)
        synth = pd.Series([1.0, 2.0])
        with pytest.raises(MltkAssertionError) as exc:
            assert_marginal_fidelity(real, synth)
        assert "empty" in str(exc.value).lower()


# ===========================================================================
# 2. Correlation Preserved
# ===========================================================================


class TestCorrelationPreserved:
    """Correlation preservation: are column relationships maintained?"""

    def test_same_correlations_pass(
        self, real_df: pd.DataFrame, good_synthetic_df: pd.DataFrame
    ) -> None:
        """PASS: Synthetic data preserves the age-income correlation.

        Scenario: age * 1000 + noise in both datasets -- the linear
        relationship is maintained.
        """
        result = assert_correlation_preserved(
            real_df, good_synthetic_df, max_delta=0.15
        )
        assert result.passed is True
        assert "frobenius_norm" in result.details
        assert "worst_pair" in result.details

    def test_broken_correlations_fail(
        self, real_df: pd.DataFrame, bad_synthetic_df: pd.DataFrame
    ) -> None:
        """FAIL: Independent columns where real data had strong correlation.

        Scenario: Real data has age-income correlation, but synthetic
        generator made them independent -- models will miss this relationship.
        """
        with pytest.raises(MltkAssertionError) as exc:
            assert_correlation_preserved(real_df, bad_synthetic_df, max_delta=0.05)
        assert "NOT preserved" in str(exc.value)

    def test_worst_pair_reported(
        self, real_df: pd.DataFrame, bad_synthetic_df: pd.DataFrame
    ) -> None:
        """The worst column pair is correctly identified in the result details.

        This helps users know exactly where the synthetic generator failed.
        """
        from mltk.core.result import Severity

        result = assert_correlation_preserved(
            real_df, bad_synthetic_df, max_delta=0.05, severity=Severity.WARNING
        )
        assert result.passed is False
        assert "worst_pair" in result.details
        # The age-income pair should be the most broken
        worst = result.details["worst_pair"]
        assert "age" in worst
        assert "income" in worst

    def test_column_subset(
        self, real_df: pd.DataFrame, good_synthetic_df: pd.DataFrame
    ) -> None:
        """Specifying a column subset only compares those columns."""
        result = assert_correlation_preserved(
            real_df, good_synthetic_df, columns=["age", "income"]
        )
        assert result.passed is True
        assert result.details["n_columns"] == 2

    def test_single_column_fails_gracefully(self) -> None:
        """FAIL: Need at least 2 columns to compute correlations.

        A single column has no pairs -- the check should fail with a
        clear message, not crash.
        """
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(MltkAssertionError) as exc:
            assert_correlation_preserved(df, df)
        assert "at least 2" in str(exc.value).lower()


# ===========================================================================
# 3. Synthetic Novelty
# ===========================================================================


class TestSyntheticNovelty:
    """Novelty: are synthetic rows genuinely new, not memorized copies?"""

    def test_all_unique_passes(self) -> None:
        """PASS: Completely unique synthetic rows -- zero copy rate.

        Scenario: Well-functioning generator that creates entirely new records.
        """
        real = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        synth = pd.DataFrame({"a": [7, 8, 9], "b": [10, 11, 12]})
        result = assert_synthetic_novelty(real, synth)
        assert result.passed is True
        assert result.details["copy_rate"] == 0.0
        assert result.details["n_copies"] == 0

    def test_all_copies_fails(self) -> None:
        """FAIL: Every synthetic row is a copy of a real row.

        Scenario: Generator just memorized and regurgitated the training data.
        This completely defeats the purpose of synthetic data generation.
        """
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        with pytest.raises(MltkAssertionError) as exc:
            assert_synthetic_novelty(df, df, max_copy_rate=0.05)
        assert "too many copies" in str(exc.value).lower()
        result = exc.value.result
        assert result.details["copy_rate"] == 1.0
        assert result.details["n_copies"] == 3

    def test_partial_copies_near_threshold(self) -> None:
        """Partial copies near the threshold -- tests boundary behavior.

        10 synthetic rows, 1 is a copy -> copy_rate = 0.1
        With max_copy_rate=0.1, this should pass (<=).
        With max_copy_rate=0.05, this should fail.
        """
        real = pd.DataFrame({"a": range(5), "b": range(5, 10)})
        synth_rows = [{"a": i + 100, "b": i + 200} for i in range(9)]
        synth_rows.append({"a": 0, "b": 5})  # exact copy of first real row
        synth = pd.DataFrame(synth_rows)

        # At threshold -- should pass (copy_rate = 0.1 <= 0.1)
        result = assert_synthetic_novelty(real, synth, max_copy_rate=0.1)
        assert result.passed is True
        assert result.details["n_copies"] == 1

        # Below threshold -- should fail
        with pytest.raises(MltkAssertionError):
            assert_synthetic_novelty(real, synth, max_copy_rate=0.05)

    def test_column_subset(self) -> None:
        """Column subset restricts which columns are compared for copies."""
        real = pd.DataFrame({"a": [1, 2], "b": [10, 20], "c": [100, 200]})
        # Same 'a' and 'b' values but different 'c' -- only a copy if checking a+b
        synth = pd.DataFrame({"a": [1, 2], "b": [10, 20], "c": [999, 888]})

        # Checking all columns: not copies (c differs)
        result = assert_synthetic_novelty(real, synth)
        assert result.passed is True
        assert result.details["n_copies"] == 0

        # Checking only a+b: they ARE copies
        with pytest.raises(MltkAssertionError):
            assert_synthetic_novelty(real, synth, columns=["a", "b"], max_copy_rate=0.0)

    def test_empty_synthetic_passes(self) -> None:
        """Empty synthetic DataFrame trivially has zero copies."""
        real = pd.DataFrame({"a": [1, 2, 3]})
        synth = pd.DataFrame({"a": pd.Series(dtype=int)})
        result = assert_synthetic_novelty(real, synth)
        assert result.passed is True
        assert result.details["n_synthetic"] == 0


# ===========================================================================
# 4. DCR Safety
# ===========================================================================


class TestDCRSafe:
    """DCR safety: are synthetic records far enough from real records?"""

    def test_well_separated_passes(self) -> None:
        """PASS: Synthetic data is far from real data in feature space.

        Scenario: Real data clustered around 0, synthetic around 100 --
        very different records, no privacy risk.
        """
        rng = np.random.default_rng(42)
        real = pd.DataFrame({
            "x": rng.normal(0, 1, 100),
            "y": rng.normal(0, 1, 100),
        })
        synth = pd.DataFrame({
            "x": rng.normal(100, 1, 100),
            "y": rng.normal(100, 1, 100),
        })
        result = assert_dcr_safe(real, synth, min_dcr=0.01)
        assert result.passed is True
        assert result.details["median_dcr"] > 0.01
        assert "p5_dcr" in result.details

    def test_near_identical_fails(self) -> None:
        """FAIL: Synthetic data is dangerously close to real data.

        Scenario: Generator produced records that are near-copies with tiny
        noise -- an attacker could match them back to real individuals.
        """
        rng = np.random.default_rng(42)
        real = pd.DataFrame({
            "x": rng.normal(0, 1, 100),
            "y": rng.normal(0, 1, 100),
        })
        # Add tiny noise -- these are essentially the same records
        synth = real.copy()
        synth["x"] += rng.normal(0, 0.0001, 100)
        synth["y"] += rng.normal(0, 0.0001, 100)

        with pytest.raises(MltkAssertionError) as exc:
            assert_dcr_safe(real, synth, min_dcr=0.05)
        assert "DCR too low" in str(exc.value)

    def test_sampling_works(self) -> None:
        """Sampling limits the number of synthetic rows evaluated.

        With sample_size=10 on a 100-row dataset, only 10 rows should be
        evaluated -- n_sampled should reflect this.
        """
        rng = np.random.default_rng(42)
        real = pd.DataFrame({
            "x": rng.normal(0, 1, 50),
            "y": rng.normal(0, 1, 50),
        })
        synth = pd.DataFrame({
            "x": rng.normal(100, 1, 100),
            "y": rng.normal(100, 1, 100),
        })
        result = assert_dcr_safe(real, synth, min_dcr=0.01, sample_size=10)
        assert result.passed is True
        assert result.details["n_sampled"] == 10

    def test_full_evaluation_when_small(self) -> None:
        """When synthetic data is smaller than sample_size, evaluate all rows."""
        rng = np.random.default_rng(42)
        real = pd.DataFrame({"x": rng.normal(0, 1, 50)})
        synth = pd.DataFrame({"x": rng.normal(100, 1, 20)})
        result = assert_dcr_safe(real, synth, min_dcr=0.01, sample_size=2000)
        assert result.details["n_sampled"] == 20

    def test_column_subset(self) -> None:
        """Column subset restricts which features are used for distance."""
        rng = np.random.default_rng(42)
        # x is well-separated, z is near-identical
        real = pd.DataFrame({
            "x": rng.normal(0, 1, 50),
            "z": rng.normal(0, 1, 50),
        })
        synth = pd.DataFrame({
            "x": rng.normal(100, 1, 50),
            "z": real["z"] + rng.normal(0, 0.0001, 50),  # near-copy
        })

        # Using only x: should pass (well separated)
        result = assert_dcr_safe(real, synth, columns=["x"], min_dcr=0.01)
        assert result.passed is True

    def test_single_column(self) -> None:
        """DCR works with a single numeric column."""
        real = pd.DataFrame({"val": [0.0, 1.0, 2.0, 3.0, 4.0]})
        synth = pd.DataFrame({"val": [10.0, 11.0, 12.0]})
        result = assert_dcr_safe(real, synth, min_dcr=0.01)
        assert result.passed is True
        assert result.details["n_sampled"] == 3

    def test_empty_columns_parameter(self) -> None:
        """Passing columns=[] gives no shared numeric columns error."""
        real = pd.DataFrame({"x": [1, 2, 3]})
        synth = pd.DataFrame({"x": [4, 5, 6]})
        with pytest.raises(MltkAssertionError) as exc:
            assert_dcr_safe(real, synth, columns=[])
        assert "No shared numeric columns" in str(exc.value)

    def test_constant_column_handled(self) -> None:
        """Constant columns (zero range) don't cause division by zero.

        A column where all values are identical has zero range. The
        normalization must handle this gracefully.
        """
        real = pd.DataFrame({"x": [5.0] * 50, "y": np.arange(50, dtype=float)})
        synth = pd.DataFrame({"x": [5.0] * 30, "y": np.arange(30, dtype=float) + 100})
        result = assert_dcr_safe(real, synth, min_dcr=0.01)
        assert result.passed is True


# ===========================================================================
# Cross-cutting edge cases
# ===========================================================================


class TestEdgeCases:
    """Edge cases that apply across multiple assertions."""

    def test_single_row_dataframes(self) -> None:
        """Single-row DataFrames should work without errors."""
        real = pd.DataFrame({"a": [1.0], "b": [2.0]})
        synth = pd.DataFrame({"a": [3.0], "b": [4.0]})

        # Novelty: different rows -> passes
        result = assert_synthetic_novelty(real, synth)
        assert result.passed is True

        # DCR: should compute distance between the single rows
        result = assert_dcr_safe(real, synth, min_dcr=0.0)
        assert result.passed is True
        assert result.details["n_sampled"] == 1

    def test_marginal_fidelity_has_timing(self) -> None:
        """All assertions should have duration_ms populated by @timed_assertion."""
        rng = np.random.default_rng(42)
        real = pd.Series(rng.normal(0, 1, 1000))
        synth = pd.Series(np.random.default_rng(99).normal(0, 1, 1000))
        result = assert_marginal_fidelity(real, synth)
        assert result.duration_ms >= 0

    def test_novelty_no_shared_columns(self) -> None:
        """No shared columns between DataFrames should fail clearly."""
        real = pd.DataFrame({"a": [1, 2, 3]})
        synth = pd.DataFrame({"b": [4, 5, 6]})
        with pytest.raises(MltkAssertionError) as exc:
            assert_synthetic_novelty(real, synth)
        assert "No shared columns" in str(exc.value)
