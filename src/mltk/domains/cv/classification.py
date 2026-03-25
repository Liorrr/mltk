"""Image classification testing -- top-K accuracy for multi-class models."""

from __future__ import annotations

from typing import Any

import numpy as np

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


@timed_assertion
def assert_topk_accuracy(
    y_true: Any,
    y_probs: Any,
    k: int = 5,
    threshold: float = 0.9,
) -> TestResult:
    """Assert top-K accuracy meets threshold.

    Args:
        y_true: Ground truth class indices (1D array).
        y_probs: Predicted probabilities (N x num_classes matrix).
        k: Number of top predictions to consider.
        threshold: Minimum required top-K accuracy.

    Returns:
        TestResult with accuracy and K value.
    """
    labels = np.asarray(y_true).flatten()
    probs = np.asarray(y_probs, dtype=np.float64)

    if len(labels) == 0:
        return assert_true(
            False, name="cv.topk_accuracy",
            message="No samples to evaluate", severity=Severity.CRITICAL,
        )

    # Get top-K predicted classes per sample
    top_k_preds = np.argsort(probs, axis=1)[:, -k:]

    # Check if true label is in top-K
    correct = sum(
        1 for i, label in enumerate(labels) if label in top_k_preds[i]
    )
    accuracy = correct / len(labels)

    passed = accuracy >= threshold
    message = (
        f"Top-{k} accuracy: {accuracy:.4f} >= {threshold}"
        if passed
        else f"Top-{k} accuracy: {accuracy:.4f} < {threshold}"
    )

    return assert_true(
        passed, name="cv.topk_accuracy", message=message,
        severity=Severity.CRITICAL,
        accuracy=accuracy, k=k, threshold=threshold,
        correct=correct, total=len(labels),
    )
