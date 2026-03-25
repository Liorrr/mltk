"""Production monitoring — detect metric degradation and SLA compliance.

Catches the silent killer: models that slowly degrade over weeks/months.
67% of organizations detect this >6 months late.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


@timed_assertion
def assert_no_degradation(
    metric_history: list[float],
    window: int = 7,
    max_decline: float = 0.05,
) -> TestResult:
    """Assert metric has not degraded over a sliding window.

    Args:
        metric_history: Time-ordered metric values (oldest first).
        window: Number of recent values to compare against earlier values.
        max_decline: Maximum allowed decline from window start to end.

    Returns:
        TestResult with degradation details.

    Example:
        >>> history = [0.95, 0.94, 0.93, 0.92, 0.91, 0.90, 0.89, 0.88]
        >>> assert_no_degradation(history, window=4, max_decline=0.05)
    """
    if len(metric_history) < window:
        return assert_true(
            True, name="monitor.degradation",
            message=f"Not enough history ({len(metric_history)} < {window} window)",
            severity=Severity.INFO,
        )

    arr = np.array(metric_history)
    recent = arr[-window:]
    earlier = arr[:-window] if len(arr) > window else arr[:1]

    recent_mean = float(recent.mean())
    earlier_mean = float(earlier.mean())
    decline = earlier_mean - recent_mean

    passed = decline <= max_decline
    message = (
        f"Metric stable: decline={decline:.4f} <= {max_decline}"
        if passed
        else f"Degradation detected: decline={decline:.4f} > {max_decline} "
        f"(earlier={earlier_mean:.4f}, recent={recent_mean:.4f})"
    )

    return assert_true(
        passed, name="monitor.degradation", message=message,
        severity=Severity.CRITICAL,
        decline=decline, max_decline=max_decline,
        recent_mean=recent_mean, earlier_mean=earlier_mean,
        window=window, history_length=len(metric_history),
    )


@timed_assertion
def assert_sla(
    latency_p99: float | None = None,
    error_rate: float | None = None,
    thresholds: dict[str, float] | None = None,
) -> TestResult:
    """Assert SLA compliance for latency and error rate.

    Args:
        latency_p99: Observed P99 latency in milliseconds.
        error_rate: Observed error rate (0.0-1.0).
        thresholds: Dict with 'latency_p99_ms' and/or 'error_rate' limits.

    Returns:
        TestResult with SLA compliance details.

    Example:
        >>> assert_sla(latency_p99=120.0, error_rate=0.005)
        >>> assert_sla(latency_p99=600.0, thresholds={"latency_p99_ms": 500.0})
    """
    if thresholds is None:
        thresholds = {"latency_p99_ms": 500.0, "error_rate": 0.01}

    violations: list[str] = []
    details: dict[str, Any] = {}

    if latency_p99 is not None:
        max_latency = thresholds.get("latency_p99_ms", 500.0)
        details["latency_p99"] = latency_p99
        details["max_latency"] = max_latency
        if latency_p99 > max_latency:
            violations.append(f"P99 latency {latency_p99:.1f}ms > {max_latency}ms")

    if error_rate is not None:
        max_errors = thresholds.get("error_rate", 0.01)
        details["error_rate"] = error_rate
        details["max_error_rate"] = max_errors
        if error_rate > max_errors:
            violations.append(f"Error rate {error_rate:.4f} > {max_errors}")

    passed = len(violations) == 0
    message = (
        "SLA compliant"
        if passed
        else f"SLA breach: {'; '.join(violations)}"
    )

    return assert_true(
        passed, name="monitor.sla", message=message,
        severity=Severity.CRITICAL, violations=violations, **details,
    )


@timed_assertion
def assert_no_output_drift(
    ref_outputs: list[float] | np.ndarray,
    cur_outputs: list[float] | np.ndarray,
    method: str = "ks",
    threshold: float = 0.05,
) -> TestResult:
    """Assert model output distribution hasn't drifted.

    Compares output distributions using KS test (default) or PSI.
    This catches behavioral drift — when model predictions shift even
    if input features look stable. Useful for monitoring regression
    score distributions or classification probability outputs.

    Supported methods:
    - "ks"  — KS test p-value; pass if p > threshold (default threshold: 0.05)
    - "psi" — Population Stability Index; pass if PSI < threshold (default: 0.05)

    Args:
        ref_outputs: Baseline model outputs (e.g., from a reference window).
        cur_outputs: Current model outputs to compare against baseline.
        method: Comparison method — "ks" (default) or "psi".
        threshold: Significance threshold; interpretation depends on method.

    Returns:
        TestResult with drift statistics including statistic, method,
        threshold, and drift_detected flag.

    Example:
        >>> assert_no_output_drift(ref_scores, cur_scores)
        >>> assert_no_output_drift(ref_scores, cur_scores, method="psi", threshold=0.1)
    """
    name = "monitor.output_drift"

    ref_arr = np.asarray(ref_outputs, dtype=np.float64).ravel()
    cur_arr = np.asarray(cur_outputs, dtype=np.float64).ravel()

    if len(ref_arr) == 0 or len(cur_arr) == 0:
        return assert_true(
            True,
            name=name,
            message="No outputs to compare (empty array)",
            severity=Severity.CRITICAL,
            method=method,
            threshold=threshold,
            statistic=0.0,
            drift_detected=False,
        )

    if method == "ks":
        from scipy.stats import ks_2samp

        stat, p_value = ks_2samp(ref_arr, cur_arr)
        passed = bool(p_value > threshold)
        message = (
            f"Output KS test: p={p_value:.4f} (threshold: {threshold})"
            if passed
            else f"Output drift detected: KS p={p_value:.4f} < {threshold}"
        )
        return assert_true(
            passed,
            name=name,
            message=message,
            severity=Severity.CRITICAL,
            method=method,
            statistic=float(stat),
            p_value=float(p_value),
            threshold=threshold,
            drift_detected=not passed,
        )

    elif method == "psi":
        # Compute PSI using shared bin edges derived from the reference
        bins = np.linspace(ref_arr.min(), ref_arr.max(), 11)
        ref_hist = np.histogram(ref_arr, bins=bins)[0].astype(float) / len(ref_arr)
        cur_hist = np.histogram(cur_arr, bins=bins)[0].astype(float) / len(cur_arr)
        ref_hist = np.clip(ref_hist, 1e-6, None)
        cur_hist = np.clip(cur_hist, 1e-6, None)
        psi_value = float(np.sum((cur_hist - ref_hist) * np.log(cur_hist / ref_hist)))
        passed = psi_value < threshold
        message = (
            f"Output PSI: {psi_value:.4f} (threshold: {threshold})"
            if passed
            else f"Output drift detected: PSI={psi_value:.4f} >= {threshold}"
        )
        return assert_true(
            passed,
            name=name,
            message=message,
            severity=Severity.CRITICAL,
            method=method,
            statistic=psi_value,
            threshold=threshold,
            drift_detected=not passed,
        )

    else:
        return assert_true(
            False,
            name=name,
            message=f"Unknown method: '{method}'. Supported: 'ks', 'psi'",
            severity=Severity.CRITICAL,
            method=method,
            threshold=threshold,
        )
