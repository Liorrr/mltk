"""Overfitting detection -- compare train vs test metric gaps.

The most common silent failure in ML: a model that memorized training data
but generalises poorly. These assertions enforce that train/test gaps stay
bounded and that the label distribution hasn't shifted between splits.
"""

from __future__ import annotations

import numpy as np

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


@timed_assertion
def assert_no_overfitting(
    train_score: float,
    test_score: float,
    max_gap: float = 0.1,
    metric_name: str = "accuracy",
) -> TestResult:
    """Assert the gap between training and test metrics is bounded.

    Overfitting = train_score significantly higher than test_score.
    gap = train_score - test_score
    Fails if gap > max_gap.

    Args:
        train_score: Metric value on the training set.
        test_score: Metric value on the held-out test set.
        max_gap: Maximum allowed gap (train - test). Default 0.1 (10 pp).
        metric_name: Human-readable metric label used in messages.

    Returns:
        TestResult with train_score, test_score, gap, max_gap, metric_name.

    Example:
        >>> assert_no_overfitting(train_score=0.95, test_score=0.88, max_gap=0.1)
    """
    gap = train_score - test_score
    passed = gap <= max_gap

    message = (
        f"{metric_name}: train={train_score:.4f}, test={test_score:.4f}, "
        f"gap={gap:.4f} <= max_gap={max_gap}"
        if passed
        else f"Overfitting detected: {metric_name} gap={gap:.4f} exceeds max_gap={max_gap} "
        f"(train={train_score:.4f}, test={test_score:.4f})"
    )

    return assert_true(
        passed,
        name="model.no_overfitting",
        message=message,
        severity=Severity.CRITICAL,
        train_score=train_score,
        test_score=test_score,
        gap=gap,
        max_gap=max_gap,
        metric_name=metric_name,
    )


@timed_assertion
def assert_label_drift(
    train_labels: list | np.ndarray,
    test_labels: list | np.ndarray,
    max_drift: float = 0.1,
) -> TestResult:
    """Assert label distribution hasn't shifted between train and test sets.

    Computes total variation distance between label distributions.
    TV = 0.5 * sum(|P(y) - Q(y)|) for all unique labels.
    Fails if TV > max_drift.

    Args:
        train_labels: Label array from the training split.
        test_labels: Label array from the test split.
        max_drift: Maximum allowed total variation distance. Default 0.1.

    Returns:
        TestResult with tv_distance, max_drift, train_distribution, test_distribution.

    Example:
        >>> assert_label_drift([0, 0, 1, 1], [0, 1, 1, 1], max_drift=0.2)
    """
    train_arr = np.asarray(train_labels)
    test_arr = np.asarray(test_labels)

    # Gather union of all unique labels
    all_labels = np.union1d(np.unique(train_arr), np.unique(test_arr))

    train_total = len(train_arr)
    test_total = len(test_arr)

    train_dist: dict[str, float] = {}
    test_dist: dict[str, float] = {}

    tv = 0.0
    for label in all_labels:
        p = float(np.sum(train_arr == label)) / train_total if train_total > 0 else 0.0
        q = float(np.sum(test_arr == label)) / test_total if test_total > 0 else 0.0
        train_dist[str(label)] = p
        test_dist[str(label)] = q
        tv += abs(p - q)

    tv_distance = 0.5 * tv
    passed = tv_distance <= max_drift

    message = (
        f"Label distribution stable: TV={tv_distance:.4f} <= max_drift={max_drift}"
        if passed
        else f"Label drift detected: TV={tv_distance:.4f} exceeds max_drift={max_drift}"
    )

    return assert_true(
        passed,
        name="model.label_drift",
        message=message,
        severity=Severity.CRITICAL,
        tv_distance=tv_distance,
        max_drift=max_drift,
        train_distribution=train_dist,
        test_distribution=test_dist,
    )
