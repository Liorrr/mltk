"""Model regression testing -- detect when model quality drops from baseline.

67% of organizations detect silent model degradation more than 6 months late.
save_baseline + assert_no_regression creates an automated safety net that catches
regressions immediately in CI/CD.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult
from mltk.model.metrics import _compute_metric


def save_baseline(
    y_true: Any,
    y_pred: Any,
    metrics: list[str],
    path: str | Path,
    average: str = "weighted",
) -> dict[str, Any]:
    """Compute and save model metrics as a JSON baseline.

    Args:
        y_true: Ground truth labels/values.
        y_pred: Model predictions.
        metrics: List of metric names to compute.
        path: Output file path for JSON baseline.
        average: Averaging for multiclass metrics.

    Returns:
        Dict with computed metrics and metadata (sample_count, timestamp).

    Example:
        >>> save_baseline([1,0,1], [1,0,0], ["accuracy","f1"], "baseline.json")
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    computed = {}
    for m in metrics:
        computed[m] = _compute_metric(y_t, y_p, m, average)

    baseline = {
        "metrics": computed,
        "sample_count": len(y_t),
        "timestamp": datetime.now().isoformat(),
    }

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(baseline, indent=2), encoding="utf-8")

    return baseline


def _load_baseline_value(
    baseline: float | dict[str, Any] | str | Path,
    metric: str,
) -> float:
    """Extract baseline metric value from various input types.

    Args:
        baseline: A float value, a dict with metrics, or a Path to a JSON file.
        metric: The metric name to extract.

    Returns:
        The baseline metric value as a float.

    Raises:
        KeyError: If metric is not found in the baseline dict.
        FileNotFoundError: If baseline path does not exist.
    """
    if isinstance(baseline, (int, float)):
        return float(baseline)

    if isinstance(baseline, dict):
        if metric in baseline:
            return float(baseline[metric])
        if "metrics" in baseline and metric in baseline["metrics"]:
            return float(baseline["metrics"][metric])
        raise KeyError(f"Metric '{metric}' not found in baseline dict")

    # Assume path to JSON
    path = Path(baseline)
    if not path.exists():
        raise FileNotFoundError(f"Baseline file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    return _load_baseline_value(data, metric)


@timed_assertion
def assert_no_regression(
    y_true: Any,
    y_pred: Any,
    baseline: float | dict[str, Any] | str | Path,
    metric: str = "accuracy",
    tolerance: float = 0.02,
    average: str = "weighted",
) -> TestResult:
    """Assert current model metrics have not regressed from baseline.

    Args:
        y_true: Ground truth labels/values.
        y_pred: Current model predictions.
        baseline: Baseline value (float), dict of metrics, or path to JSON.
        metric: Which metric to compare.
        tolerance: Max allowed drop (0.02 = 2% regression allowed).
        average: Averaging for multiclass metrics.

    Returns:
        TestResult with current vs baseline comparison.

    Example:
        >>> assert_no_regression([1,0,1], [1,0,1], baseline=0.9, metric="accuracy")
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    if len(y_t) == 0 or len(y_p) == 0:
        return assert_true(
            False,
            name="model.no_regression",
            message="Cannot compute metrics on empty arrays",
            severity=Severity.CRITICAL,
        )

    try:
        baseline_value = _load_baseline_value(baseline, metric)
    except (KeyError, FileNotFoundError) as e:
        return assert_true(
            False,
            name="model.no_regression",
            message=str(e),
            severity=Severity.CRITICAL,
        )

    current_value = _compute_metric(y_t, y_p, metric, average)
    min_allowed = baseline_value - tolerance
    passed = current_value >= min_allowed

    message = (
        f"{metric}: current={current_value:.4f} vs baseline={baseline_value:.4f} "
        f"(tolerance={tolerance})"
        if passed
        else f"Regression: {metric} dropped from {baseline_value:.4f} to "
        f"{current_value:.4f} (tolerance={tolerance}, min={min_allowed:.4f})"
    )

    return assert_true(
        passed,
        name="model.no_regression",
        message=message,
        severity=Severity.CRITICAL,
        metric=metric,
        current_value=current_value,
        baseline_value=baseline_value,
        tolerance=tolerance,
        min_allowed=min_allowed,
        regression_detected=not passed,
    )
