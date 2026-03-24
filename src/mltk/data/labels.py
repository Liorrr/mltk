"""Label quality testing -- verify class balance and coverage.

Labels are the ground truth that ML models learn from. Imbalanced labels
produce models that predict the majority class and ignore minorities.
Missing labels mean the model is blind to entire categories. These checks
catch label quality issues before they corrupt model training.
"""

from __future__ import annotations

import pandas as pd

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


@timed_assertion
def assert_label_balance(
    labels: pd.Series,
    max_ratio: float = 10.0,
) -> TestResult:
    """Assert class distribution is not too imbalanced.

    Ratio = count(majority_class) / count(minority_class).

    Args:
        labels: Series containing class labels.
        max_ratio: Maximum allowed majority/minority ratio.

    Returns:
        TestResult with class counts and ratio.

    Example:
        >>> assert_label_balance(df["label"], max_ratio=10.0)
    """
    if len(labels) == 0:
        return assert_true(
            False,
            name="data.label_balance",
            message="Cannot check balance on empty labels",
            severity=Severity.CRITICAL,
        )

    counts = labels.value_counts()
    majority_class = counts.index[0]
    minority_class = counts.index[-1]
    majority_count = int(counts.iloc[0])
    minority_count = int(counts.iloc[-1])

    ratio = majority_count / max(minority_count, 1)
    passed = ratio <= max_ratio

    message = (
        f"Label balance ratio {ratio:.1f} within limit {max_ratio}"
        if passed
        else f"Label imbalance: ratio {ratio:.1f} exceeds max {max_ratio} "
        f"('{majority_class}'={majority_count}, '{minority_class}'={minority_count})"
    )

    return assert_true(
        passed,
        name="data.label_balance",
        message=message,
        severity=Severity.CRITICAL,
        class_counts=counts.to_dict(),
        majority_class=str(majority_class),
        minority_class=str(minority_class),
        ratio=ratio,
        max_ratio=max_ratio,
    )


@timed_assertion
def assert_label_coverage(
    labels: pd.Series,
    expected_labels: set[str] | None = None,
    min_samples: int = 1,
) -> TestResult:
    """Assert all expected label classes are present with sufficient samples.

    Args:
        labels: Series containing class labels.
        expected_labels: Required classes. None = check all observed classes.
        min_samples: Minimum samples required per class.

    Returns:
        TestResult with coverage details.

    Example:
        >>> assert_label_coverage(df["label"], expected_labels={"cat", "dog"}, min_samples=10)
    """
    counts = labels.value_counts()
    observed = set(counts.index.astype(str))

    # Determine which labels to check
    check_labels = expected_labels if expected_labels is not None else observed

    missing: list[str] = []
    insufficient: dict[str, int] = {}

    for label in check_labels:
        # Check if label exists (compare as string)
        matching = [k for k in counts.index if str(k) == label]
        if not matching:
            missing.append(label)
        else:
            count = int(counts[matching[0]])
            if count < min_samples:
                insufficient[label] = count

    passed = len(missing) == 0 and len(insufficient) == 0

    errors = []
    if missing:
        errors.append(f"Missing labels: {sorted(missing)}")
    if insufficient:
        for label, count in insufficient.items():
            errors.append(f"'{label}': {count} samples (need {min_samples})")

    message = (
        f"All {len(check_labels)} label(s) present with >= {min_samples} samples"
        if passed
        else "; ".join(errors)
    )

    return assert_true(
        passed,
        name="data.label_coverage",
        message=message,
        severity=Severity.CRITICAL,
        class_counts=counts.to_dict(),
        missing_labels=sorted(missing),
        insufficient_labels=insufficient,
        total_classes=len(observed),
        expected_classes=len(check_labels),
    )
