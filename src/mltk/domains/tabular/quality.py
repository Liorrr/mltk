"""Tabular data quality — convenience wrappers for DataFrame validation."""

from __future__ import annotations

import pandas as pd

from mltk.core.assertion import timed_assertion
from mltk.core.result import TestResult
from mltk.data.labels import assert_label_balance as _label_balance


@timed_assertion
def assert_class_balance(
    df: pd.DataFrame,
    label_col: str,
    max_ratio: float = 10.0,
) -> TestResult:
    """Assert class balance in a DataFrame column.

    Convenience wrapper around mltk.data.labels.assert_label_balance
    for tabular workflows.

    Args:
        df: DataFrame containing the label column.
        label_col: Name of the label column.
        max_ratio: Maximum majority/minority ratio.

    Returns:
        TestResult with balance details.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"label": ["cat", "cat", "dog", "dog", "cat"]})
        >>> assert_class_balance(df, label_col="label", max_ratio=5.0)
    """
    return _label_balance(df[label_col], max_ratio=max_ratio)
