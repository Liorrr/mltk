"""Model metrics assertions -- validate model performance against thresholds.

Catches the most dangerous ML evaluation bug: using the wrong metric.
A model on 99% negative data gets 99% accuracy by always predicting negative.
Use F1/AUC instead. These assertions enforce minimum quality gates.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult

# Metrics where lower is better (error metrics)
_LOWER_IS_BETTER = {"mse", "rmse", "mae"}

_SUPPORTED_METRICS = {
    "accuracy",
    "f1",
    "precision",
    "recall",
    "auc",
    "mse",
    "rmse",
    "mae",
    "r2",
}


def _compute_metric(
    y_true: Any,
    y_pred: Any,
    metric: str,
    average: str = "weighted",
) -> float:
    """Compute a metric using sklearn.

    Args:
        y_true: Ground truth labels/values.
        y_pred: Model predictions.
        metric: Metric name from _SUPPORTED_METRICS.
        average: Averaging strategy for multiclass (weighted/macro/micro).

    Returns:
        Computed metric value as a float.
    """
    try:
        from sklearn import metrics as skm
    except ImportError as err:
        raise ImportError(
            "scikit-learn is required for model metrics. "
            "Install with: pip install mltk[sklearn]"
        ) from err

    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    if metric == "accuracy":
        return float(skm.accuracy_score(y_t, y_p))
    elif metric == "f1":
        return float(skm.f1_score(y_t, y_p, average=average, zero_division=0))
    elif metric == "precision":
        return float(skm.precision_score(y_t, y_p, average=average, zero_division=0))
    elif metric == "recall":
        return float(skm.recall_score(y_t, y_p, average=average, zero_division=0))
    elif metric == "auc":
        return float(skm.roc_auc_score(y_t, y_p))
    elif metric == "mse":
        return float(skm.mean_squared_error(y_t, y_p))
    elif metric == "rmse":
        return float(np.sqrt(skm.mean_squared_error(y_t, y_p)))
    elif metric == "mae":
        return float(skm.mean_absolute_error(y_t, y_p))
    elif metric == "r2":
        return float(skm.r2_score(y_t, y_p))
    else:
        raise ValueError(f"Unknown metric: '{metric}'")


@timed_assertion
def assert_metric(
    y_true: Any,
    y_pred: Any,
    metric: str = "accuracy",
    threshold: float = 0.8,
    average: str = "weighted",
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
    """Assert a model metric meets a minimum threshold.

    Args:
        y_true: Ground truth labels/values.
        y_pred: Model predictions.
        metric: Metric name (accuracy, f1, precision, recall, auc, mse, rmse, mae, r2).
        threshold: Required value. For error metrics (mse/rmse/mae), this is the maximum.
        average: Averaging for multiclass (weighted/macro/micro).
        severity: Severity level for the assertion (default CRITICAL).

    Returns:
        TestResult with actual metric value and threshold.

    Example:
        >>> y_true = [1, 0, 1, 1, 0]
        >>> y_pred = [1, 0, 1, 0, 0]
        >>> assert_metric(y_true, y_pred, metric="accuracy", threshold=0.7)
    """
    if metric not in _SUPPORTED_METRICS:
        return assert_true(
            False,
            name="model.metric",
            message=f"Unknown metric: '{metric}'. Supported: {sorted(_SUPPORTED_METRICS)}",
            severity=severity,
        )

    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    if len(y_t) == 0 or len(y_p) == 0:
        return assert_true(
            False,
            name="model.metric",
            message="Cannot compute metrics on empty arrays",
            severity=severity,
        )

    value = _compute_metric(y_t, y_p, metric, average)

    if metric in _LOWER_IS_BETTER:
        passed = value <= threshold
        comparison = "<="
    else:
        passed = value >= threshold
        comparison = ">="

    message = (
        f"{metric}={value:.4f} {comparison} {threshold}"
        if passed
        else f"{metric}={value:.4f} does not meet threshold {threshold}"
    )

    return assert_true(
        passed,
        name=f"model.metric.{metric}",
        message=message,
        severity=severity,
        metric=metric,
        value=value,
        threshold=threshold,
        average=average,
    )
