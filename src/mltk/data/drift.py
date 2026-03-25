"""Data drift detection -- detect when data distributions shift from training baseline.

Drift is silent. A model doesn't crash when data drifts -- it just produces
increasingly wrong predictions. Drift detection is the bridge between
"model works on test set" and "model works in production."

Supports 7 methods:
- KS test: non-parametric, best for continuous numeric features
- PSI: industry standard for financial models, interpretable buckets
- KL divergence: information-theoretic, captures distributional shape
- Chi-squared: designed for categorical features
- Jensen-Shannon: symmetric, bounded [0,1], Evidently's default for categorical
- Wasserstein: proportional to mean shift, Evidently's default for numeric n>1000
- Auto: auto-selects best method based on sample size and dtype
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult

# Default thresholds per method (pass if below/above these)
_DEFAULT_THRESHOLDS: dict[str, float] = {
    "ks": 0.05,         # p-value: pass if p > 0.05
    "psi": 0.1,         # PSI: pass if PSI < 0.1
    "kl": 0.1,          # KL divergence: pass if KL < 0.1
    "chi2": 0.05,       # p-value: pass if p > 0.05
    "js": 0.1,          # Jensen-Shannon: pass if JS < 0.1 (bounded [0,1])
    "wasserstein": 0.1,  # Wasserstein: pass if W < 0.1
    "auto": 0.05,       # Auto: uses method-specific threshold
}


@timed_assertion
def assert_no_drift(
    reference: pd.Series,
    current: pd.Series,
    method: str = "ks",
    threshold: float | None = None,
) -> TestResult:
    """Assert no significant distribution drift between reference and current data.

    Args:
        reference: Baseline distribution (e.g., training data).
        current: Current distribution to compare against baseline.
        method: Detection method -- "ks", "psi", "kl", or "chi2".
        threshold: Custom threshold. If None, uses method-specific default.

    Returns:
        TestResult with drift statistics.

    Example:
        >>> assert_no_drift(train_df["income"], prod_df["income"], method="psi")
    """
    if method not in _DEFAULT_THRESHOLDS:
        return assert_true(
            False,
            name="data.drift",
            message=f"Unknown method: '{method}'. Supported: {list(_DEFAULT_THRESHOLDS.keys())}",
            severity=Severity.CRITICAL,
        )

    thresh = threshold if threshold is not None else _DEFAULT_THRESHOLDS[method]

    # Chi2 works on categorical data -- skip float conversion
    if method == "chi2":
        ref_clean = reference.dropna()
        cur_clean = current.dropna()
        if len(ref_clean) == 0 or len(cur_clean) == 0:
            return assert_true(
                False,
                name="data.drift",
                message="Cannot compute drift on empty arrays",
                severity=Severity.CRITICAL,
            )
        return _drift_chi2(ref_clean, cur_clean, thresh)

    ref_arr = np.asarray(reference.dropna(), dtype=np.float64)
    cur_arr = np.asarray(current.dropna(), dtype=np.float64)

    if len(ref_arr) == 0 or len(cur_arr) == 0:
        return assert_true(
            False,
            name="data.drift",
            message="Cannot compute drift on empty arrays",
            severity=Severity.CRITICAL,
        )

    if method == "auto":
        # Auto-select: Wasserstein for numeric n>1000, KS otherwise
        if len(ref_arr) > 1000:
            return _drift_wasserstein(ref_arr, cur_arr, _DEFAULT_THRESHOLDS["wasserstein"])
        return _drift_ks(ref_arr, cur_arr, _DEFAULT_THRESHOLDS["ks"])
    elif method == "ks":
        return _drift_ks(ref_arr, cur_arr, thresh)
    elif method == "psi":
        return _drift_psi(ref_arr, cur_arr, thresh)
    elif method == "kl":
        return _drift_kl(ref_arr, cur_arr, thresh)
    elif method == "js":
        return _drift_js(ref_arr, cur_arr, thresh)
    else:  # wasserstein
        return _drift_wasserstein(ref_arr, cur_arr, thresh)


def _drift_ks(ref: np.ndarray, cur: np.ndarray, threshold: float) -> TestResult:
    """KS test: compare empirical CDFs.

    Args:
        ref: Reference distribution array.
        cur: Current distribution array.
        threshold: P-value threshold; pass if p > threshold.

    Returns:
        TestResult with KS statistic and p-value.
    """
    from mltk._rust import ks_test

    stat, p_value = ks_test(ref.tolist(), cur.tolist())
    passed = p_value > threshold

    return assert_true(
        passed,
        name="data.drift.ks",
        message=(
            f"KS test: p={p_value:.4f} (threshold: {threshold})"
            if passed
            else f"Drift detected: KS p={p_value:.4f} < {threshold}"
        ),
        severity=Severity.CRITICAL,
        method="ks",
        statistic=stat,
        p_value=p_value,
        threshold=threshold,
        drift_detected=not passed,
    )


def _drift_psi(ref: np.ndarray, cur: np.ndarray, threshold: float) -> TestResult:
    """PSI: Population Stability Index.

    Args:
        ref: Reference distribution array.
        cur: Current distribution array.
        threshold: PSI threshold; pass if PSI < threshold.

    Returns:
        TestResult with PSI value.
    """
    from mltk._rust import psi

    psi_value = psi(ref.tolist(), cur.tolist(), bins=10)
    passed = psi_value < threshold

    return assert_true(
        passed,
        name="data.drift.psi",
        message=(
            f"PSI: {psi_value:.4f} (threshold: {threshold})"
            if passed
            else f"Drift detected: PSI={psi_value:.4f} >= {threshold}"
        ),
        severity=Severity.CRITICAL,
        method="psi",
        statistic=psi_value,
        threshold=threshold,
        drift_detected=not passed,
    )


def _drift_kl(ref: np.ndarray, cur: np.ndarray, threshold: float) -> TestResult:
    """KL divergence via histogram comparison.

    Args:
        ref: Reference distribution array.
        cur: Current distribution array.
        threshold: KL divergence threshold; pass if KL < threshold.

    Returns:
        TestResult with KL divergence value.
    """
    bins = np.linspace(
        min(ref.min(), cur.min()),
        max(ref.max(), cur.max()),
        11,
    )
    ref_hist = np.histogram(ref, bins=bins)[0].astype(float) / len(ref)
    cur_hist = np.histogram(cur, bins=bins)[0].astype(float) / len(cur)

    # Clip to avoid log(0)
    ref_hist = np.clip(ref_hist, 1e-6, None)
    cur_hist = np.clip(cur_hist, 1e-6, None)

    kl_value = float(np.sum(ref_hist * np.log(ref_hist / cur_hist)))
    passed = kl_value < threshold

    return assert_true(
        passed,
        name="data.drift.kl",
        message=(
            f"KL divergence: {kl_value:.4f} (threshold: {threshold})"
            if passed
            else f"Drift detected: KL={kl_value:.4f} >= {threshold}"
        ),
        severity=Severity.CRITICAL,
        method="kl",
        statistic=kl_value,
        threshold=threshold,
        drift_detected=not passed,
    )


def _drift_chi2(
    reference: pd.Series, current: pd.Series, threshold: float
) -> TestResult:
    """Chi-squared test for categorical drift.

    Args:
        reference: Reference categorical series.
        current: Current categorical series.
        threshold: P-value threshold; pass if p > threshold.

    Returns:
        TestResult with chi-squared statistic and p-value.
    """
    try:
        from scipy.stats import chi2_contingency
    except ImportError as err:
        raise ImportError(
            "scipy is required for chi2 drift detection. "
            "Install with: pip install mltk[scipy]"
        ) from err

    ref_counts = reference.value_counts()
    cur_counts = current.value_counts()

    # Align both distributions on the same category set (union of observed categories)
    all_categories = sorted(set(ref_counts.index) | set(cur_counts.index))
    ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
    cur_aligned = [cur_counts.get(cat, 0) for cat in all_categories]

    # Build 2xK contingency table for chi-squared independence test
    contingency = np.array([ref_aligned, cur_aligned])
    stat, p_value, _, _ = chi2_contingency(contingency)
    passed = bool(p_value > threshold)

    return assert_true(
        passed,
        name="data.drift.chi2",
        message=(
            f"Chi2 test: p={p_value:.4f} (threshold: {threshold})"
            if passed
            else f"Drift detected: Chi2 p={p_value:.4f} < {threshold}"
        ),
        severity=Severity.CRITICAL,
        method="chi2",
        statistic=float(stat),
        p_value=float(p_value),
        threshold=threshold,
        drift_detected=not passed,
    )


def _drift_js(ref: np.ndarray, cur: np.ndarray, threshold: float) -> TestResult:
    """Jensen-Shannon divergence: symmetric, bounded [0,1].

    Args:
        ref: Reference distribution array.
        cur: Current distribution array.
        threshold: JS threshold; pass if JS < threshold.

    Returns:
        TestResult with JS divergence value.
    """
    bins = np.linspace(min(ref.min(), cur.min()), max(ref.max(), cur.max()), 11)
    ref_hist = np.histogram(ref, bins=bins)[0].astype(float) / len(ref)
    cur_hist = np.histogram(cur, bins=bins)[0].astype(float) / len(cur)
    ref_hist = np.clip(ref_hist, 1e-10, None)
    cur_hist = np.clip(cur_hist, 1e-10, None)
    m = 0.5 * (ref_hist + cur_hist)
    js_value = float(
        0.5 * np.sum(ref_hist * np.log(ref_hist / m))
        + 0.5 * np.sum(cur_hist * np.log(cur_hist / m))
    )
    passed = js_value < threshold
    return assert_true(
        passed, name="data.drift.js",
        message=(f"JS: {js_value:.4f} (threshold: {threshold})" if passed
                 else f"Drift detected: JS={js_value:.4f} >= {threshold}"),
        severity=Severity.CRITICAL,
        method="js", statistic=js_value, threshold=threshold,
        drift_detected=not passed,
    )


def _drift_wasserstein(ref: np.ndarray, cur: np.ndarray, threshold: float) -> TestResult:
    """Wasserstein (Earth Mover's) distance.

    Args:
        ref: Reference distribution array.
        cur: Current distribution array.
        threshold: Wasserstein threshold; pass if W < threshold.

    Returns:
        TestResult with Wasserstein distance value.
    """
    try:
        from scipy.stats import wasserstein_distance
    except ImportError as err:
        raise ImportError(
            "scipy required for Wasserstein. Install: pip install mltk[scipy]"
        ) from err
    w_value = float(wasserstein_distance(ref, cur))
    passed = w_value < threshold
    return assert_true(
        passed, name="data.drift.wasserstein",
        message=(f"Wasserstein: {w_value:.4f} (threshold: {threshold})" if passed
                 else f"Drift detected: W={w_value:.4f} >= {threshold}"),
        severity=Severity.CRITICAL,
        method="wasserstein", statistic=w_value, threshold=threshold,
        drift_detected=not passed,
    )
