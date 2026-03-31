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
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
    """Assert no significant distribution drift between reference and current data.

    Args:
        reference: Baseline distribution (e.g., training data).
        current: Current distribution to compare against baseline.
        method: Detection method -- "ks", "psi", "kl", or "chi2".
        threshold: Custom threshold. If None, uses method-specific default.
        severity: Severity level for the assertion (default CRITICAL).

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
            severity=severity,
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
                severity=severity,
            )
        return _drift_chi2(ref_clean, cur_clean, thresh, severity)

    ref_arr = np.asarray(reference.dropna(), dtype=np.float64)
    cur_arr = np.asarray(current.dropna(), dtype=np.float64)

    if len(ref_arr) == 0 or len(cur_arr) == 0:
        return assert_true(
            False,
            name="data.drift",
            message="Cannot compute drift on empty arrays",
            severity=severity,
        )

    if method == "auto":
        # Auto-select: Wasserstein for numeric n>1000, KS otherwise
        if len(ref_arr) > 1000:
            return _drift_wasserstein(
                ref_arr,
                cur_arr,
                _DEFAULT_THRESHOLDS["wasserstein"],
                severity,
            )
        return _drift_ks(
            ref_arr, cur_arr, _DEFAULT_THRESHOLDS["ks"], severity
        )
    elif method == "ks":
        return _drift_ks(ref_arr, cur_arr, thresh, severity)
    elif method == "psi":
        return _drift_psi(ref_arr, cur_arr, thresh, severity)
    elif method == "kl":
        return _drift_kl(ref_arr, cur_arr, thresh, severity)
    elif method == "js":
        return _drift_js(ref_arr, cur_arr, thresh, severity)
    else:  # wasserstein
        return _drift_wasserstein(ref_arr, cur_arr, thresh, severity)


def _drift_ks(
    ref: np.ndarray,
    cur: np.ndarray,
    threshold: float,
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
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
        severity=severity,
        method="ks",
        statistic=stat,
        p_value=p_value,
        threshold=threshold,
        drift_detected=not passed,
    )


def _drift_psi(
    ref: np.ndarray,
    cur: np.ndarray,
    threshold: float,
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
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
        severity=severity,
        method="psi",
        statistic=psi_value,
        threshold=threshold,
        drift_detected=not passed,
    )


def _drift_kl(
    ref: np.ndarray,
    cur: np.ndarray,
    threshold: float,
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
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
        severity=severity,
        method="kl",
        statistic=kl_value,
        threshold=threshold,
        drift_detected=not passed,
    )


def _drift_chi2(
    reference: pd.Series,
    current: pd.Series,
    threshold: float,
    severity: Severity = Severity.CRITICAL,
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
        severity=severity,
        method="chi2",
        statistic=float(stat),
        p_value=float(p_value),
        threshold=threshold,
        drift_detected=not passed,
    )


def _drift_js(
    ref: np.ndarray,
    cur: np.ndarray,
    threshold: float,
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
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
        severity=severity,
        method="js", statistic=js_value, threshold=threshold,
        drift_detected=not passed,
    )


def _drift_wasserstein(
    ref: np.ndarray,
    cur: np.ndarray,
    threshold: float,
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
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
        severity=severity,
        method="wasserstein", statistic=w_value, threshold=threshold,
        drift_detected=not passed,
    )


# -----------------------------------------------------------
# Multivariate drift (MMD) — helpers + public assertion
# -----------------------------------------------------------

def _to_array_2d(
    data: np.ndarray | pd.DataFrame,
) -> np.ndarray:
    """Convert input to a 2-D float64 array, dropping NaN rows.

    Args:
        data: Input array or DataFrame.

    Returns:
        Cleaned 2-D numpy array with dtype float64.
    """
    if isinstance(data, pd.DataFrame):
        arr = data.to_numpy(dtype=np.float64)
    else:
        arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    # Drop rows containing any NaN
    mask = ~np.isnan(arr).any(axis=1)
    return arr[mask]


def _subsample(
    data: np.ndarray,
    max_n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Random subsample without replacement.

    Args:
        data: 2-D array to subsample from.
        max_n: Maximum number of rows to keep.
        rng: Numpy random generator instance.

    Returns:
        Subsampled array (unchanged if already small enough).
    """
    if len(data) <= max_n:
        return data
    idx = rng.choice(len(data), size=max_n, replace=False)
    return data[idx]


def _median_heuristic(
    ref: np.ndarray,
    cur: np.ndarray,
    rng: np.random.Generator,
    subsample: int = 500,
) -> float:
    """Median heuristic for RBF bandwidth selection.

    Computes the median of pairwise Euclidean distances on a
    random subsample of the pooled data.

    Args:
        ref: Reference array (2-D).
        cur: Current array (2-D).
        rng: Numpy random generator instance.
        subsample: Max samples for distance computation.

    Returns:
        Bandwidth sigma (median distance), minimum 1e-8.
    """
    pooled = np.vstack([ref, cur])
    pooled = _subsample(pooled, subsample, rng)
    # Pairwise squared distances via expansion trick
    sq_norms = np.sum(pooled ** 2, axis=1)
    # D2[i,j] = ||x_i||^2 + ||x_j||^2 - 2 * x_i . x_j
    d2 = (
        sq_norms[:, None]
        + sq_norms[None, :]
        - 2.0 * pooled @ pooled.T
    )
    np.maximum(d2, 0.0, out=d2)
    # Extract upper triangle (no diagonal)
    triu_idx = np.triu_indices(len(pooled), k=1)
    dists = np.sqrt(d2[triu_idx])
    sigma = float(np.median(dists))
    return max(sigma, 1e-8)


def _rbf_kernel_matrix(
    X: np.ndarray,
    Y: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """RBF (Gaussian) kernel matrix between X and Y.

    K[i,j] = exp(-||x_i - y_j||^2 / (2 * sigma^2))

    Args:
        X: First array, shape (m, d).
        Y: Second array, shape (n, d).
        sigma: Bandwidth parameter.

    Returns:
        Kernel matrix of shape (m, n).
    """
    gamma = 1.0 / (2.0 * sigma * sigma)
    sq_x = np.sum(X ** 2, axis=1)
    sq_y = np.sum(Y ** 2, axis=1)
    d2 = sq_x[:, None] + sq_y[None, :] - 2.0 * X @ Y.T
    np.maximum(d2, 0.0, out=d2)
    return np.exp(-gamma * d2)


def _multi_bandwidth_mmd2(
    ref: np.ndarray,
    cur: np.ndarray,
    sigmas: list[float],
) -> float:
    """Unbiased MMD^2 averaged over multiple bandwidths.

    Uses the diagonal-excluded unbiased estimator:
    MMD^2 = K_xx/(m(m-1)) + K_yy/(n(n-1)) - 2*K_xy/(m*n)

    Args:
        ref: Reference array, shape (m, d).
        cur: Current array, shape (n, d).
        sigmas: List of bandwidth values to average over.

    Returns:
        Average unbiased MMD^2 across all bandwidths.
    """
    m = len(ref)
    n = len(cur)
    total = 0.0
    for sigma in sigmas:
        k_xx = _rbf_kernel_matrix(ref, ref, sigma)
        k_yy = _rbf_kernel_matrix(cur, cur, sigma)
        k_xy = _rbf_kernel_matrix(ref, cur, sigma)
        # Zero diagonal for unbiased estimator
        np.fill_diagonal(k_xx, 0.0)
        np.fill_diagonal(k_yy, 0.0)
        mmd2 = (
            k_xx.sum() / (m * (m - 1))
            + k_yy.sum() / (n * (n - 1))
            - 2.0 * k_xy.sum() / (m * n)
        )
        total += mmd2
    return total / len(sigmas)


@timed_assertion
def assert_no_multivariate_drift(
    reference: np.ndarray | pd.DataFrame,
    current: np.ndarray | pd.DataFrame,
    threshold: float = 0.05,
    n_permutations: int = 200,
    max_samples: int = 500,
    kernel: str = "rbf",
    sigma: float | None = None,
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
    """Assert no multivariate distribution drift (MMD test).

    Unlike per-feature drift tests (KS, PSI), MMD detects joint
    distributional shifts -- including changes in feature
    correlations and covariance structure that per-feature tests
    miss entirely. Based on Gretton et al. (JMLR 2012).

    Uses an RBF kernel with multi-bandwidth averaging
    (0.5*sigma, 1*sigma, 2*sigma) and a permutation test for
    the p-value.

    Args:
        reference: Baseline data, shape (m, d).
        current: Current data to compare, shape (n, d).
        threshold: P-value threshold; pass if p > threshold.
        n_permutations: Permutation count for the p-value.
        max_samples: Max rows per dataset (random subsample).
        kernel: Kernel type. Only "rbf" is supported.
        sigma: Bandwidth override. None = median heuristic.
        severity: Severity level (default CRITICAL).

    Returns:
        TestResult with MMD^2 statistic, p-value, and details.

    Example:
        >>> ref = np.random.randn(200, 3)
        >>> cur = np.random.randn(200, 3)
        >>> assert_no_multivariate_drift(ref, cur)
    """
    name = "data.multivariate_drift.mmd"

    # --- Validate kernel ---------------------------------
    if kernel != "rbf":
        return assert_true(
            False,
            name=name,
            message=(
                f"Unsupported kernel: '{kernel}'. "
                "Supported: ['rbf']"
            ),
            severity=severity,
        )

    # --- Convert and clean -------------------------------
    ref = _to_array_2d(reference)
    cur = _to_array_2d(current)

    # --- Edge case: too few samples ----------------------
    if len(ref) < 2 or len(cur) < 2:
        return assert_true(
            False,
            name=name,
            message=(
                "Need >= 2 samples per dataset. "
                f"Got ref={len(ref)}, cur={len(cur)}."
            ),
            severity=severity,
        )

    # --- Edge case: dimension mismatch -------------------
    if ref.shape[1] != cur.shape[1]:
        return assert_true(
            False,
            name=name,
            message=(
                "Dimension mismatch: "
                f"ref has {ref.shape[1]} features, "
                f"cur has {cur.shape[1]}."
            ),
            severity=severity,
        )

    # --- Subsample for performance -----------------------
    rng = np.random.default_rng(42)
    ref = _subsample(ref, max_samples, rng)
    cur = _subsample(cur, max_samples, rng)

    # --- Bandwidth selection -----------------------------
    if sigma is not None:
        base_sigma = float(sigma)
    else:
        base_sigma = _median_heuristic(ref, cur, rng)
    sigmas = [0.5 * base_sigma, base_sigma, 2.0 * base_sigma]

    # --- Observed MMD^2 ----------------------------------
    m = len(ref)
    n = len(cur)
    observed_mmd2 = _multi_bandwidth_mmd2(ref, cur, sigmas)

    # --- Permutation test for p-value --------------------
    pooled = np.vstack([ref, cur])
    count_ge = 0
    for _ in range(n_permutations):
        perm = rng.permutation(m + n)
        perm_ref = pooled[perm[:m]]
        perm_cur = pooled[perm[m:]]
        perm_mmd2 = _multi_bandwidth_mmd2(
            perm_ref, perm_cur, sigmas
        )
        if perm_mmd2 >= observed_mmd2:
            count_ge += 1
    p_value = (count_ge + 1) / (n_permutations + 1)

    passed = p_value > threshold
    return assert_true(
        passed,
        name=name,
        message=(
            f"MMD test: p={p_value:.4f} "
            f"(threshold: {threshold})"
            if passed
            else f"Multivariate drift detected: "
            f"MMD p={p_value:.4f} < {threshold}"
        ),
        severity=severity,
        method="mmd",
        kernel=kernel,
        statistic=float(observed_mmd2),
        p_value=float(p_value),
        threshold=threshold,
        sigma=base_sigma,
        bandwidths=sigmas,
        n_permutations=n_permutations,
        ref_samples=m,
        cur_samples=n,
        n_features=ref.shape[1],
        drift_detected=not passed,
    )
