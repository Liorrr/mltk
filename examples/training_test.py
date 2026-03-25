"""Example: Training bug detection with mltk."""
import pandas as pd
import pytest

from mltk.training import assert_no_train_test_overlap, assert_temporal_split


@pytest.mark.ml_data
def test_no_data_leakage():
    train = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})
    test = pd.DataFrame({"id": [4, 5, 6], "value": [40, 50, 60]})
    assert_no_train_test_overlap(train, test, key_cols=["id"])

@pytest.mark.ml_data
def test_temporal_split_correct():
    train = pd.DataFrame({"date": ["2026-01-01", "2026-01-15"]})
    test = pd.DataFrame({"date": ["2026-02-01", "2026-02-15"]})
    assert_temporal_split(train, test, time_col="date")
