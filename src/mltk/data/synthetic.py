"""Synthetic data quality assertions -- validate that generated data is faithful, useful, and safe.

Synthetic data is increasingly used to augment scarce training sets, protect
privacy in shared datasets, and enable testing without real user data. But
synthetic data is only valuable if it satisfies three properties:

1. **Fidelity**: It looks like the real data (same distributions, same correlations).
2. **Novelty**: It creates genuinely new records, not copies of the originals.
3. **Privacy**: It doesn't leak information about individual real records.

These four assertions cover all three properties:
- ``assert_marginal_fidelity`` checks single-column distribution fidelity.
- ``assert_correlation_preserved`` checks multi-column relationship fidelity.
- ``assert_synthetic_novelty`` checks that records are not memorized copies.
- ``assert_dcr_safe`` checks that records are not dangerously close to real ones.

Together they answer: "Is this synthetic data good enough to use in place of
real data, and safe enough to share outside the organization?"
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult

# ---------------------------------------------------------------------------
# 1. Marginal Fidelity
# ---------------------------------------------------------------------------


@timed_assertion
def assert_marginal_fidelity(
    real: pd.Series,
    synthetic: pd.Series,
    method: str = "ks",
    max_divergence: float = 0.1,
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
    """Assert that a synthetic column follows the same distribution as the real column.

    **Why this matters:** Marginal fidelity is the most basic quality check for
    synthetic data. It asks: "Does each individual column in the synthetic
    dataset look like the corresponding column in the real dataset?" If a
    single column is wildly off -- for example, synthetic ages are uniformly
    distributed while real ages follow a normal distribution centered at 35 --
    then any model trained on the synthetic data will learn the wrong feature
    distributions. This is the first thing to check because if marginals are
    wrong, nothing downstream (correlations, utility) can be right.

    Two methods are supported:

    - **KS (Kolmogorov-Smirnov)**: Compares the empirical cumulative
      distribution functions (ECDFs) of the two samples. The KS statistic is
      the maximum vertical distance between the two ECDFs. A small statistic
      means the distributions are similar. This is non-parametric and makes no
      assumptions about the shape of the distribution. Best for continuous
      numeric data.

    - **PSI (Population Stability Index)**: Bins both distributions and
      measures how much the bin proportions have shifted. PSI < 0.1 is
      considered stable, 0.1-0.2 is moderate, > 0.2 is significant. Widely
      used in financial model monitoring. Works well for both continuous and
      discretized data.

    Args:
        real: The real (ground truth) data column.
        synthetic: The synthetically generated data column.
        method: Comparison method -- ``"ks"`` (default) or ``"psi"``.
        max_divergence: Maximum allowed divergence. For KS, this is the max
            KS statistic. For PSI, the max PSI value. Default 0.1.
        severity: Severity level for the assertion.

    Returns:
        TestResult with details: ``statistic``, ``method``, ``max_divergence``,
        ``n_real``, ``n_synthetic``.

    Example:
        >>> import pandas as pd, numpy as np
        >>> rng = np.random.default_rng(42)
        >>> real = pd.Series(rng.normal(0, 1, 1000))
        >>> synth = pd.Series(rng.normal(0, 1, 1000))
        >>> result = assert_marginal_fidelity(real, synth)
        >>> assert result.passed
    """
    valid_methods = ("ks", "psi")
    if method not in valid_methods:
        return assert_true(
            False,
            name="data.synthetic.marginal_fidelity",
            message=f"Unknown method: '{method}'. Supported: {list(valid_methods)}",
            severity=severity,
        )

    real_arr = np.asarray(real.dropna(), dtype=np.float64)
    synth_arr = np.asarray(synthetic.dropna(), dtype=np.float64)

    if len(real_arr) == 0 or len(synth_arr) == 0:
        return assert_true(
            False,
            name="data.synthetic.marginal_fidelity",
            message="Cannot compute marginal fidelity on empty arrays",
            severity=severity,
        )

    if method == "ks":
        statistic = _ks_statistic(real_arr, synth_arr)
    else:  # psi
        statistic = _psi_value(real_arr, synth_arr)

    passed = statistic < max_divergence

    return assert_true(
        passed,
        name="data.synthetic.marginal_fidelity",
        message=(
            f"Marginal fidelity ({method}): {statistic:.4f} < {max_divergence}"
            if passed
            else (
                f"Marginal fidelity failed: {method}={statistic:.4f} "
                f">= {max_divergence}"
            )
        ),
        severity=severity,
        statistic=float(statistic),
        method=method,
        max_divergence=max_divergence,
        n_real=len(real_arr),
        n_synthetic=len(synth_arr),
    )


# ---------------------------------------------------------------------------
# 2. Correlation Preserved
# ---------------------------------------------------------------------------


@timed_assertion
def assert_correlation_preserved(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    max_delta: float = 0.1,
    columns: list[str] | None = None,
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
    """Assert that pairwise column correlations in synthetic data match real data.

    **Why this matters:** Individual columns can each look perfect (pass
    marginal fidelity) while the *relationships between columns* are
    completely wrong. Imagine real data where age and income are strongly
    correlated -- older people tend to earn more. A synthetic generator might
    produce realistic-looking age distributions and realistic-looking income
    distributions, but make them statistically independent. Any model trained
    on that synthetic data would miss the age-income relationship, leading to
    poor predictions on real data.

    This assertion computes the Pearson correlation matrix for both the real
    and synthetic DataFrames, then measures the difference using the
    **Frobenius norm** -- the square root of the sum of squared element-wise
    differences. This single number captures how much the entire correlation
    structure has shifted. The norm is then normalized by the number of unique
    column pairs so the threshold is interpretable regardless of how many
    columns are being compared.

    The assertion also identifies the **worst pair** -- the two columns with
    the largest absolute correlation difference -- so you know exactly where
    the synthetic generator is failing.

    Args:
        real_df: The real (ground truth) DataFrame.
        synthetic_df: The synthetically generated DataFrame.
        max_delta: Maximum allowed normalized Frobenius norm of the correlation
            matrix difference. Default 0.1.
        columns: Subset of columns to compare. If None, uses all numeric
            columns present in both DataFrames.
        severity: Severity level for the assertion.

    Returns:
        TestResult with details: ``frobenius_norm``, ``normalized_diff``,
        ``max_delta``, ``n_columns``, ``worst_pair``.

    Example:
        >>> result = assert_correlation_preserved(real_df, synthetic_df, max_delta=0.15)
        >>> print(result.details["worst_pair"])
    """
    # Select numeric columns present in both DataFrames
    if columns is not None:
        cols = [c for c in columns if c in real_df.columns and c in synthetic_df.columns]
    else:
        real_numeric = set(real_df.select_dtypes(include=[np.number]).columns)
        synth_numeric = set(synthetic_df.select_dtypes(include=[np.number]).columns)
        cols = sorted(real_numeric & synth_numeric)

    if len(cols) < 2:
        return assert_true(
            False,
            name="data.synthetic.correlation_preserved",
            message=(
                f"Need at least 2 shared numeric columns to compare correlations, "
                f"found {len(cols)}"
            ),
            severity=severity,
        )

    corr_real = real_df[cols].corr().values
    corr_synth = synthetic_df[cols].corr().values

    diff = corr_real - corr_synth
    frobenius = float(np.linalg.norm(diff))

    # Number of unique off-diagonal pairs: n*(n-1)/2
    n = len(cols)
    n_pairs = n * (n - 1) // 2
    normalized = frobenius / n_pairs if n_pairs > 0 else frobenius

    # Find worst pair (largest absolute difference in correlation)
    worst_val = 0.0
    worst_pair = (cols[0], cols[1])
    for i in range(n):
        for j in range(i + 1, n):
            delta = abs(diff[i, j])
            if delta > worst_val:
                worst_val = delta
                worst_pair = (cols[i], cols[j])

    passed = normalized <= max_delta

    return assert_true(
        passed,
        name="data.synthetic.correlation_preserved",
        message=(
            f"Correlation preserved: normalized diff={normalized:.4f} <= {max_delta}"
            if passed
            else (
                f"Correlation NOT preserved: normalized diff={normalized:.4f} "
                f"> {max_delta} (worst pair: {worst_pair[0]}-{worst_pair[1]})"
            )
        ),
        severity=severity,
        frobenius_norm=float(frobenius),
        normalized_diff=float(normalized),
        max_delta=max_delta,
        n_columns=n,
        worst_pair=f"{worst_pair[0]}-{worst_pair[1]}",
    )


# ---------------------------------------------------------------------------
# 3. Synthetic Novelty
# ---------------------------------------------------------------------------


@timed_assertion
def assert_synthetic_novelty(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    max_copy_rate: float = 0.05,
    columns: list[str] | None = None,
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
    """Assert that synthetic data is not just a copy of the real data.

    **Why this matters:** The whole point of synthetic data is to generate
    *new* records that look realistic. If a generator memorizes training rows
    and outputs exact duplicates, it defeats both purposes of synthetic data:

    - **Privacy**: Copied rows ARE real records -- sharing them leaks PII.
    - **Augmentation**: Duplicating existing data doesn't add new information;
      it just inflates sample size while the effective diversity stays the same.

    This assertion hashes every row in both datasets, then counts how many
    synthetic row hashes appear in the real row hash set. The **copy rate** is
    the fraction of synthetic rows that are exact duplicates of some real row.

    A good synthetic generator should produce a copy rate near zero. A rate
    above 5% (default threshold) suggests the generator has memorized too much
    of the training data. Common causes include overfitting a GAN, using too
    few training epochs with a VAE, or simply copying rows with minor noise.

    Args:
        real_df: The real (ground truth) DataFrame.
        synthetic_df: The synthetically generated DataFrame.
        max_copy_rate: Maximum allowed fraction of synthetic rows that are
            exact copies of real rows. Default 0.05 (5%).
        columns: Subset of columns to consider when comparing rows.
            If None, uses all columns present in both DataFrames.
        severity: Severity level for the assertion.

    Returns:
        TestResult with details: ``copy_rate``, ``n_copies``, ``n_synthetic``,
        ``n_real``, ``max_copy_rate``.

    Example:
        >>> result = assert_synthetic_novelty(real_df, synth_df, max_copy_rate=0.01)
        >>> assert result.details["copy_rate"] < 0.01
    """
    if columns is not None:
        cols = [c for c in columns if c in real_df.columns and c in synthetic_df.columns]
    else:
        cols = sorted(set(real_df.columns) & set(synthetic_df.columns))

    if len(cols) == 0:
        return assert_true(
            False,
            name="data.synthetic.novelty",
            message="No shared columns between real and synthetic DataFrames",
            severity=severity,
        )

    n_synthetic = len(synthetic_df)
    if n_synthetic == 0:
        return assert_true(
            True,
            name="data.synthetic.novelty",
            message="Synthetic DataFrame is empty -- trivially novel",
            severity=severity,
            copy_rate=0.0,
            n_copies=0,
            n_synthetic=0,
            n_real=len(real_df),
            max_copy_rate=max_copy_rate,
        )

    # Hash rows by converting each row to a tuple of values
    real_hashes = set()
    for row in real_df[cols].itertuples(index=False, name=None):
        real_hashes.add(row)

    n_copies = 0
    for row in synthetic_df[cols].itertuples(index=False, name=None):
        if row in real_hashes:
            n_copies += 1

    copy_rate = n_copies / n_synthetic

    passed = copy_rate <= max_copy_rate

    return assert_true(
        passed,
        name="data.synthetic.novelty",
        message=(
            f"Synthetic novelty: copy rate={copy_rate:.4f} <= {max_copy_rate}"
            if passed
            else (
                f"Synthetic data has too many copies: {n_copies}/{n_synthetic} "
                f"({copy_rate:.2%}) > {max_copy_rate:.2%}"
            )
        ),
        severity=severity,
        copy_rate=float(copy_rate),
        n_copies=n_copies,
        n_synthetic=n_synthetic,
        n_real=len(real_df),
        max_copy_rate=max_copy_rate,
    )


# ---------------------------------------------------------------------------
# 4. DCR Safety (Distance to Closest Record)
# ---------------------------------------------------------------------------


@timed_assertion
def assert_dcr_safe(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    min_dcr: float = 0.05,
    sample_size: int = 2000,
    columns: list[str] | None = None,
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
    """Assert that synthetic records are not dangerously close to real records.

    **Why this matters:** A synthetic row can be *unique* (pass the novelty
    check) but still be suspiciously *close* to a real record. Imagine a real
    medical record with age=45, blood_pressure=130, cholesterol=220. A
    synthetic row with age=45.001, blood_pressure=130.002, cholesterol=219.998
    is technically unique -- the hashes differ -- but a motivated attacker
    could easily match it back to the real patient.

    **Distance to Closest Record (DCR)** measures this risk. For each
    synthetic row, it finds the nearest real row using L2 (Euclidean) distance
    in normalized feature space, and takes that minimum distance. The
    distribution of these minimum distances tells you how private the synthetic
    data is:

    - **Median DCR**: The typical closest-record distance. If this is very
      small, most synthetic rows are uncomfortably close to some real row.
    - **5th percentile DCR (p5)**: The worst-case tail. Even if the median is
      fine, a low 5th percentile means some synthetic rows are near-copies.

    Columns are normalized to [0, 1] using min-max scaling before computing
    distances, so the threshold is interpretable across datasets with
    different feature scales. A ``min_dcr`` of 0.05 means the median synthetic
    row must be at least 5% of the feature range away from the nearest real
    row.

    For efficiency, a random sample of synthetic rows is evaluated rather than
    the full dataset. This is statistically valid because DCR is computed
    per-row and sampled rows are independent.

    Args:
        real_df: The real (ground truth) DataFrame.
        synthetic_df: The synthetically generated DataFrame.
        min_dcr: Minimum acceptable median DCR. Default 0.05.
        sample_size: Maximum number of synthetic rows to evaluate. Default 2000.
        columns: Subset of numeric columns to use. If None, uses all numeric
            columns present in both DataFrames.
        severity: Severity level for the assertion.

    Returns:
        TestResult with details: ``median_dcr``, ``mean_dcr``,
        ``min_dcr_threshold``, ``p5_dcr``, ``n_sampled``, ``n_real``.

    Example:
        >>> result = assert_dcr_safe(real_df, synth_df, min_dcr=0.1)
        >>> print(f"Median DCR: {result.details['median_dcr']:.4f}")
    """
    # Select numeric columns present in both DataFrames
    if columns is not None:
        cols = [c for c in columns if c in real_df.columns and c in synthetic_df.columns]
    else:
        real_numeric = set(real_df.select_dtypes(include=[np.number]).columns)
        synth_numeric = set(synthetic_df.select_dtypes(include=[np.number]).columns)
        cols = sorted(real_numeric & synth_numeric)

    if len(cols) == 0:
        return assert_true(
            False,
            name="data.synthetic.dcr_safe",
            message="No shared numeric columns between real and synthetic DataFrames",
            severity=severity,
        )

    real_vals = real_df[cols].to_numpy(dtype=np.float64)
    synth_vals = synthetic_df[cols].to_numpy(dtype=np.float64)

    if len(real_vals) == 0 or len(synth_vals) == 0:
        return assert_true(
            False,
            name="data.synthetic.dcr_safe",
            message="Cannot compute DCR on empty DataFrames",
            severity=severity,
        )

    # Min-max normalize each column to [0, 1]
    col_min = real_vals.min(axis=0)
    col_max = real_vals.max(axis=0)
    col_range = col_max - col_min
    # Avoid division by zero for constant columns -- treat as 1.0
    col_range[col_range == 0] = 1.0

    real_norm = (real_vals - col_min) / col_range
    synth_norm = (synth_vals - col_min) / col_range

    # Sample synthetic rows for efficiency
    n_synth = len(synth_norm)
    n_sample = min(sample_size, n_synth)
    if n_sample < n_synth:
        rng = np.random.default_rng(42)
        indices = rng.choice(n_synth, size=n_sample, replace=False)
        synth_sample = synth_norm[indices]
    else:
        synth_sample = synth_norm

    # Vectorized DCR: for each synthetic row, compute L2 distance to all real
    # rows, take the minimum. Uses broadcasting: (n_sample, 1, d) - (1, n_real, d)
    # would be memory-heavy for large datasets, so we iterate over synthetic
    # rows with vectorized distance to all real rows.
    min_distances = np.empty(n_sample, dtype=np.float64)
    for i in range(n_sample):
        # (n_real, d) - (d,) broadcasts to (n_real, d), then L2 per row
        diffs = real_norm - synth_sample[i]
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        min_distances[i] = dists.min()

    median_dcr = float(np.median(min_distances))
    mean_dcr = float(np.mean(min_distances))
    p5_dcr = float(np.percentile(min_distances, 5))

    passed = median_dcr >= min_dcr

    return assert_true(
        passed,
        name="data.synthetic.dcr_safe",
        message=(
            f"DCR safe: median={median_dcr:.4f} >= {min_dcr}"
            if passed
            else (
                f"DCR too low: median={median_dcr:.4f} < {min_dcr} "
                f"(5th percentile={p5_dcr:.4f})"
            )
        ),
        severity=severity,
        median_dcr=median_dcr,
        mean_dcr=mean_dcr,
        min_dcr_threshold=min_dcr,
        p5_dcr=p5_dcr,
        n_sampled=n_sample,
        n_real=len(real_vals),
    )


# ---------------------------------------------------------------------------
# Internal helpers -- pure numpy, no new dependencies
# ---------------------------------------------------------------------------


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the two-sample KS statistic (max ECDF difference).

    Uses scipy if available (faster C implementation), falls back to pure
    numpy otherwise.
    """
    try:
        from scipy.stats import ks_2samp

        stat, _ = ks_2samp(a, b)
        return float(stat)
    except ImportError:
        pass

    # Pure numpy fallback: merge, sort, build ECDFs, find max difference
    n_a, n_b = len(a), len(b)
    combined = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(np.sort(a), combined, side="right") / n_a
    cdf_b = np.searchsorted(np.sort(b), combined, side="right") / n_b
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _psi_value(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """Compute Population Stability Index between two distributions."""
    breakpoints = np.linspace(
        min(ref.min(), cur.min()),
        max(ref.max(), cur.max()),
        bins + 1,
    )
    ref_pcts = np.histogram(ref, bins=breakpoints)[0].astype(float) / len(ref)
    cur_pcts = np.histogram(cur, bins=breakpoints)[0].astype(float) / len(cur)
    # Clip to avoid log(0)
    ref_pcts = np.clip(ref_pcts, 1e-6, None)
    cur_pcts = np.clip(cur_pcts, 1e-6, None)
    return float(np.sum((cur_pcts - ref_pcts) * np.log(cur_pcts / ref_pcts)))
