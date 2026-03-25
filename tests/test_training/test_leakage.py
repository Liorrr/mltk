"""Tests for mltk.training.leakage — data and feature leakage detection."""

import numpy as np
import pandas as pd
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.training.leakage import (
    assert_no_target_leakage,
    assert_no_train_test_overlap,
    assert_temporal_split,
)


class TestNoTrainTestOverlap:
    """Train/test data leakage detection."""

    def test_no_overlap(self) -> None:
        """PASS: Train and test have completely separate rows."""
        train = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        test = pd.DataFrame({"id": [4, 5, 6], "value": [40, 50, 60]})
        result = assert_no_train_test_overlap(train, test, key_cols=["id"])
        assert result.passed is True

    def test_overlap_detected(self) -> None:
        """FAIL: Rows appear in both train and test — data leakage."""
        train = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
        test = pd.DataFrame({"id": [3, 4, 5], "value": [30, 40, 50]})
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_train_test_overlap(train, test, key_cols=["id"])
        assert "leakage" in str(exc.value).lower()


class TestTemporalSplit:
    """Temporal split validation."""

    def test_valid_temporal_split(self) -> None:
        """PASS: All train data before all test data."""
        train = pd.DataFrame({"date": ["2026-01-01", "2026-01-02"]})
        test = pd.DataFrame({"date": ["2026-02-01", "2026-02-02"]})
        result = assert_temporal_split(train, test, time_col="date")
        assert result.passed is True

    def test_temporal_leakage(self) -> None:
        """FAIL: Train contains dates after test start — temporal leakage."""
        train = pd.DataFrame({"date": ["2026-01-01", "2026-03-01"]})
        test = pd.DataFrame({"date": ["2026-02-01", "2026-02-15"]})
        with pytest.raises(MltkAssertionError) as exc:
            assert_temporal_split(train, test, time_col="date")
        assert "temporal" in str(exc.value).lower()


class TestNoTargetLeakage:
    """Feature-target correlation leakage detection."""

    def test_no_leakage(self) -> None:
        """PASS: Features are not suspiciously correlated with target."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "feature_a": rng.normal(0, 1, 100),
            "feature_b": rng.normal(5, 2, 100),
            "target": rng.integers(0, 2, 100),
        })
        result = assert_no_target_leakage(df, target_col="target")
        assert result.passed is True

    def test_leakage_detected(self) -> None:
        """FAIL: Feature is almost identical to target — proxy leakage."""
        rng = np.random.default_rng(42)
        target = rng.normal(0, 1, 100)
        df = pd.DataFrame({
            "leaky_feature": target + rng.normal(0, 0.01, 100),
            "safe_feature": rng.normal(5, 2, 100),
            "target": target,
        })
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_target_leakage(df, target_col="target", corr_threshold=0.95)
        assert "leakage" in str(exc.value).lower()
