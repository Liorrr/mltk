"""Anomaly detection for test metrics — catch relative changes, not just absolute failures.

A standard assertion like ``assert accuracy >= 0.80`` catches absolute failures:
the model accuracy dropped below a hard threshold.  But what if your model
*usually* scores 0.95 and today it scores 0.88?  It still passes the 0.80
threshold, but something is clearly wrong -- an 7-point drop from baseline
is a significant regression that warrants investigation.

Anomaly detection catches **relative** changes by comparing the current
metric value against its own history.  Three statistical methods are
supported:

- **Z-score**: How many standard deviations is the current value from the
  historical mean?  A Z-score of 3+ is very unusual (0.3% probability
  under normality).

- **IQR (Interquartile Range)**: The "box" in a box plot.  Values outside
  ``[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`` are considered outliers.  More robust
  than Z-score when the history is skewed or has heavy tails.

- **Percentile**: Flags values below the Nth or above the (100-N)th
  percentile of the history.  Intuitive and non-parametric.

Typical usage::

    # Track inference latency over the last 30 runs
    latency_history = [12.1, 11.8, 12.3, 12.0, 11.9, ...]
    current_latency = 45.2  # something is wrong today

    assert_no_test_anomaly(latency_history, current_latency)
    # -> MltkAssertionError: Z-score = 8.42 exceeds threshold 3.0

"""
from __future__ import annotations

import numpy as np

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


def _zscore_check(
    history: np.ndarray,
    current: float,
    threshold: float,
) -> tuple[bool, dict]:
    """Z-score anomaly detection.

    The Z-score measures how many standard deviations the *current* value
    is from the historical mean::

        z = (current - mean) / std

    If ``|z| > threshold`` (default 3.0), the value is anomalous.

    A threshold of 3.0 corresponds to roughly 0.3% probability under a
    normal distribution -- meaning the value would occur by chance only
    3 times in 1000 observations.

    Args:
        history: Array of historical metric values.
        current: The current observation to test.
        threshold: Maximum acceptable absolute Z-score.

    Returns:
        Tuple of (passed, details_dict).
    """
    mean = float(np.mean(history))
    std = float(np.std(history, ddof=0))

    if std == 0.0:
        # Constant history: any deviation at all is anomalous
        is_anomaly = current != mean
        z_score = 0.0 if not is_anomaly else float("inf")
    else:
        z_score = (current - mean) / std
        is_anomaly = abs(z_score) > threshold

    passed = not is_anomaly
    details = {
        "method": "zscore",
        "z_score": round(z_score, 4) if not np.isinf(z_score) else float("inf"),
        "mean": round(mean, 6),
        "std": round(std, 6),
        "threshold": threshold,
        "current": current,
        "is_anomaly": is_anomaly,
    }
    return passed, details


def _iqr_check(
    history: np.ndarray,
    current: float,
    threshold: float,
) -> tuple[bool, dict]:
    """Interquartile range (IQR) anomaly detection.

    The IQR is the range between the 25th and 75th percentiles (the
    "box" in a box plot).  Values outside the "whiskers" are outliers::

        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

    The *threshold* parameter controls the multiplier (default 1.5 is
    the standard Tukey fence; 3.0 detects only extreme outliers).

    This method is more robust than Z-score when the history has skewed
    distributions or heavy tails.

    Args:
        history: Array of historical metric values.
        current: The current observation to test.
        threshold: IQR multiplier for the fence (default 1.5).

    Returns:
        Tuple of (passed, details_dict).
    """
    q1 = float(np.percentile(history, 25))
    q3 = float(np.percentile(history, 75))
    iqr = q3 - q1

    lower = q1 - threshold * iqr
    upper = q3 + threshold * iqr

    is_anomaly = current < lower or current > upper
    passed = not is_anomaly

    details = {
        "method": "iqr",
        "q1": round(q1, 6),
        "q3": round(q3, 6),
        "iqr": round(iqr, 6),
        "lower_bound": round(lower, 6),
        "upper_bound": round(upper, 6),
        "threshold": threshold,
        "current": current,
        "is_anomaly": is_anomaly,
    }
    return passed, details


def _percentile_check(
    history: np.ndarray,
    current: float,
    threshold: float,
) -> tuple[bool, dict]:
    """Percentile-based anomaly detection.

    Flags the current value as anomalous if it falls below the
    *threshold*-th percentile or above the *(100 - threshold)*-th
    percentile of the history.

    For example, with ``threshold=5.0``, values below the 5th percentile
    or above the 95th percentile are flagged.  This is intuitive and
    non-parametric (no normality assumption).

    Args:
        history: Array of historical metric values.
        current: The current observation to test.
        threshold: Percentile cutoff (e.g. 5.0 means flag below 5th
            or above 95th percentile).

    Returns:
        Tuple of (passed, details_dict).
    """
    lower_pct = float(np.percentile(history, threshold))
    upper_pct = float(np.percentile(history, 100 - threshold))

    is_anomaly = current < lower_pct or current > upper_pct
    passed = not is_anomaly

    details = {
        "method": "percentile",
        "lower_percentile": round(lower_pct, 6),
        "upper_percentile": round(upper_pct, 6),
        "threshold_pct": threshold,
        "current": current,
        "is_anomaly": is_anomaly,
    }
    return passed, details


@timed_assertion
def assert_no_test_anomaly(
    history: list[float],
    current: float,
    method: str = "zscore",
    threshold: float = 3.0,
) -> TestResult:
    """Assert that the current test metric is not anomalous compared to history.

    **Why anomaly detection on test results matters:**

    A test that usually takes 2 seconds but suddenly takes 45 seconds is
    anomalous -- even if it still passes.  A model that usually achieves
    95% accuracy but drops to 88% is anomalous -- even if 88% is above
    your hard threshold of 80%.

    Normal assertion thresholds catch **absolute** failures
    (``accuracy < 80%``).  Anomaly detection catches **relative** changes
    (``accuracy dropped 7% from baseline``).  Both are needed for robust
    ML testing.

    **Supported methods:**

    - ``"zscore"`` (default): Z-score = ``(current - mean) / std``.
      Anomalous if ``|Z| > threshold``.  Default threshold: 3.0.
      Best for roughly normal distributions.

    - ``"iqr"``: Interquartile range.  Anomalous if the value falls
      outside ``[Q1 - threshold*IQR, Q3 + threshold*IQR]``.
      Default threshold: 1.5 (standard Tukey fence).
      More robust to skewed data.

    - ``"percentile"``: Anomalous if the current value is below the
      *threshold*-th percentile or above the *(100 - threshold)*-th
      percentile.  Default threshold: 5.0 (flags bottom/top 5%).

    Args:
        history: Previous metric values (at least 3 recommended for
            meaningful statistics).
        current: The current observation to test.
        method: Detection method -- ``"zscore"``, ``"iqr"``, or
            ``"percentile"``.
        threshold: Sensitivity threshold (interpretation depends on
            *method*; see above).

    Returns:
        :class:`~mltk.core.result.TestResult` with detection details
        including the computed statistic, bounds, and anomaly flag.

    Raises:
        :class:`~mltk.core.assertion.MltkAssertionError`: When an
            anomaly is detected (CRITICAL severity).

    Example:
        >>> history = [12.1, 11.8, 12.3, 12.0, 11.9, 12.2, 12.1]
        >>> assert_no_test_anomaly(history, 12.0)  # normal -> passes
        >>> assert_no_test_anomaly(history, 45.2)  # extreme -> raises
    """
    name = "monitor.anomaly.test_metric"

    # Edge case: insufficient history for meaningful statistics
    if len(history) < 3:
        return assert_true(
            True,
            name=name,
            message=f"Not enough history ({len(history)} < 3) for anomaly detection",
            severity=Severity.INFO,
            method=method,
            threshold=threshold,
            current=current,
            history_length=len(history),
        )

    arr = np.array(history, dtype=np.float64)

    if method == "zscore":
        passed, details = _zscore_check(arr, current, threshold)
    elif method == "iqr":
        passed, details = _iqr_check(arr, current, threshold)
    elif method == "percentile":
        passed, details = _percentile_check(arr, current, threshold)
    else:
        return assert_true(
            False,
            name=name,
            message=f"Unknown method: '{method}'. Supported: 'zscore', 'iqr', 'percentile'",
            severity=Severity.CRITICAL,
            method=method,
            threshold=threshold,
            current=current,
        )

    if passed:
        message = (
            f"No anomaly detected ({method}): "
            f"current={current} is within expected range"
        )
    else:
        if method == "zscore":
            message = (
                f"Anomaly detected ({method}): "
                f"Z-score={details['z_score']} exceeds threshold {threshold}"
            )
        elif method == "iqr":
            message = (
                f"Anomaly detected ({method}): "
                f"current={current} outside "
                f"[{details['lower_bound']:.4f}, {details['upper_bound']:.4f}]"
            )
        else:
            message = (
                f"Anomaly detected ({method}): "
                f"current={current} outside "
                f"[{details['lower_percentile']:.4f}, {details['upper_percentile']:.4f}]"
            )

    return assert_true(
        passed,
        name=name,
        message=message,
        severity=Severity.CRITICAL,
        **details,
        history_length=len(history),
    )
