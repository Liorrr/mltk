"""Training-serving skew detection — compare pipeline outputs.

Training-serving skew occurs when the feature engineering or preprocessing
applied during training differs from what runs in production. Even subtle
differences (e.g., mean imputation with slightly different values, different
normalisation constants) cause silent accuracy degradation that is hard to
trace back to the root cause.
"""

from __future__ import annotations

import numpy as np

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


@timed_assertion
def assert_no_training_serving_skew(
    train_output: np.ndarray | list,
    serve_output: np.ndarray | list,
    tolerance: float = 0.01,
) -> TestResult:
    """Assert training and serving pipelines produce the same output for the same input.

    Compares outputs element-wise. The maximum absolute difference across all
    elements must be <= tolerance. This catches feature engineering differences
    between the train and serve code paths, such as:
    - Different imputation values
    - Different normalisation constants
    - Different one-hot encoding ordering
    - Floating-point accumulation differences

    Args:
        train_output: Output array from the training pipeline.
        serve_output: Output array from the serving/inference pipeline.
        tolerance: Maximum allowed absolute difference per element. Default 0.01.

    Returns:
        TestResult with max_diff and element count details.

    Example:
        >>> train_out = preprocess_train(sample)
        >>> serve_out = preprocess_serve(sample)
        >>> assert_no_training_serving_skew(train_out, serve_out, tolerance=0.01)
    """
    train_arr = np.asarray(train_output, dtype=float).ravel()
    serve_arr = np.asarray(serve_output, dtype=float).ravel()

    if train_arr.shape != serve_arr.shape:
        return assert_true(
            False, name="training.serving_skew",
            message=(
                f"Shape mismatch: train={train_arr.shape} vs serve={serve_arr.shape}"
            ),
            severity=Severity.CRITICAL,
            train_shape=list(train_arr.shape),
            serve_shape=list(serve_arr.shape),
        )

    abs_diff = np.abs(train_arr - serve_arr)
    max_diff = float(abs_diff.max()) if abs_diff.size > 0 else 0.0
    mean_diff = float(abs_diff.mean()) if abs_diff.size > 0 else 0.0
    num_skewed = int(np.sum(abs_diff > tolerance))

    passed = max_diff <= tolerance

    message = (
        f"No skew detected: max_diff={max_diff:.6f} <= tolerance={tolerance} "
        f"({len(train_arr)} elements)"
        if passed
        else f"Training-serving skew: max_diff={max_diff:.6f} > tolerance={tolerance} "
        f"({num_skewed}/{len(train_arr)} elements exceed threshold)"
    )

    return assert_true(
        passed, name="training.serving_skew", message=message,
        severity=Severity.CRITICAL,
        max_diff=max_diff,
        mean_diff=mean_diff,
        tolerance=tolerance,
        num_skewed=num_skewed,
        num_elements=len(train_arr),
    )
