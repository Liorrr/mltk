"""Feature attribution stability -- verify that explanations are reproducible.

Feature attribution methods like SHAP and LIME explain model predictions by
assigning an importance score to each input feature. These methods typically
use randomized sampling internally, which means running them twice on the
same data can produce different attribution vectors.

If your "model explanations" change between runs, they are noise, not signal.
Two complementary stability checks are provided:

1. **Top-K overlap** -- do the same features appear at the top of both runs?
   This is the "headline" check: if your top-5 features keep shuffling, no
   stakeholder can trust the explanation.

2. **Cosine similarity** -- do the full attribution vectors point in the same
   direction?  This catches magnitude shifts that top-K misses.  Feature "age"
   might be #1 in both runs, but with importance 0.8 vs 0.2 -- cosine catches
   that, top-K does not.

Both assertions work with raw numpy arrays, not feature-name dicts. This
makes them method-agnostic: feed in SHAP values, LIME weights, integrated
gradients, or any other attribution format.
"""

from __future__ import annotations

import numpy as np

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


@timed_assertion
def assert_top_k_stable(
    attributions_a: np.ndarray,
    attributions_b: np.ndarray,
    k: int = 5,
    min_overlap: float = 0.8,
) -> TestResult:
    """Assert the top-K most important features are consistent across runs.

    Feature attribution methods assign an importance score to every input
    feature.  This assertion checks whether the *set* of top-K features
    (by absolute importance) is the same in two independent runs.

    Overlap is measured as ``|top_a intersect top_b| / K``.  A value of 1.0
    means perfect agreement; 0.0 means completely different top features.

    Absolute values are used because importance is about magnitude, not sign.
    A feature with attribution -0.9 is just as important as one with +0.9.

    When ``k`` exceeds the number of features, it is clamped so that the
    check remains meaningful instead of raising an error.

    Args:
        attributions_a: Attribution vector from run A (1-D array, one value
            per feature).
        attributions_b: Attribution vector from run B (same shape as A).
        k: Number of top features to compare.
        min_overlap: Minimum required overlap ratio in ``[0, 1]``.

    Returns:
        TestResult with details: ``overlap``, ``min_overlap``, ``k``,
        ``top_k_a``, ``top_k_b``, ``common_features``.

    Example:
        >>> import numpy as np
        >>> a = np.array([0.5, 0.1, 0.9, 0.3, 0.8])
        >>> b = np.array([0.6, 0.2, 0.85, 0.25, 0.7])
        >>> assert_top_k_stable(a, b, k=3, min_overlap=0.8)
    """
    a = np.asarray(attributions_a, dtype=np.float64).ravel()
    b = np.asarray(attributions_b, dtype=np.float64).ravel()

    n_features = len(a)

    if n_features == 0:
        return assert_true(
            False,
            name="model.attribution.top_k_stable",
            message="Cannot check top-K stability on empty attributions",
            severity=Severity.CRITICAL,
        )

    # Clamp k to the number of available features
    effective_k = min(k, n_features)

    # Top-K indices by absolute importance (descending)
    top_k_a = set(np.argsort(np.abs(a))[-effective_k:].tolist())
    top_k_b = set(np.argsort(np.abs(b))[-effective_k:].tolist())

    common = top_k_a & top_k_b
    overlap = len(common) / effective_k

    passed = overlap >= min_overlap

    message = (
        f"Top-{effective_k} overlap={overlap:.2f} >= {min_overlap} "
        f"({len(common)}/{effective_k} features in common)"
        if passed
        else f"Unstable: top-{effective_k} overlap={overlap:.2f} < {min_overlap} "
        f"(only {len(common)}/{effective_k} features in common)"
    )

    return assert_true(
        passed,
        name="model.attribution.top_k_stable",
        message=message,
        severity=Severity.CRITICAL,
        overlap=overlap,
        min_overlap=min_overlap,
        k=effective_k,
        top_k_a=sorted(top_k_a),
        top_k_b=sorted(top_k_b),
        common_features=len(common),
    )


@timed_assertion
def assert_attribution_cosine_stability(
    attributions_a: np.ndarray,
    attributions_b: np.ndarray,
    min_cosine: float = 0.9,
) -> TestResult:
    """Assert attribution vectors are directionally stable via cosine similarity.

    Top-K overlap tells you whether the *same* features rank at the top, but
    it ignores magnitude.  Cosine similarity captures the full vector geometry:
    two vectors with cosine 0.99 are nearly identical in shape; cosine 0.5
    means the "explanation" looks fundamentally different even if predictions
    are unchanged.

    For a single sample (1-D arrays), cosine is computed directly:

        cosine = dot(a, b) / (||a|| * ||b||)

    For multiple samples (2-D arrays of shape ``[n_samples, n_features]``),
    cosine is computed per sample (row) and the mean is reported.  This
    gives an aggregate stability measure across the dataset.

    Zero vectors are handled gracefully -- their cosine is defined as 0.0,
    which will fail any reasonable ``min_cosine`` threshold.

    Args:
        attributions_a: Attribution array from run A.  Shape ``(n_features,)``
            for single-sample or ``(n_samples, n_features)`` for multi-sample.
        attributions_b: Attribution array from run B (same shape as A).
        min_cosine: Minimum cosine similarity to pass.

    Returns:
        TestResult with details: ``cosine_similarity``, ``min_cosine``,
        and ``n_features`` (or ``n_samples`` and ``n_features`` for 2-D).

    Example:
        >>> import numpy as np
        >>> a = np.array([0.5, 0.1, 0.9, 0.3])
        >>> b = np.array([0.52, 0.09, 0.88, 0.31])
        >>> assert_attribution_cosine_stability(a, b, min_cosine=0.99)
    """
    a = np.asarray(attributions_a, dtype=np.float64)
    b = np.asarray(attributions_b, dtype=np.float64)

    if a.size == 0:
        return assert_true(
            False,
            name="model.attribution.cosine_stable",
            message="Cannot compute cosine similarity on empty attributions",
            severity=Severity.CRITICAL,
        )

    # --- 2-D: per-sample cosine, then average ---
    if a.ndim == 2:
        n_samples, n_features = a.shape
        cosines = np.empty(n_samples, dtype=np.float64)
        for i in range(n_samples):
            cosines[i] = _cosine(a[i], b[i])
        cosine_sim = float(cosines.mean())

        passed = cosine_sim >= min_cosine
        message = (
            f"Cosine stability={cosine_sim:.4f} >= {min_cosine} "
            f"(mean over {n_samples} samples)"
            if passed
            else f"Unstable: cosine={cosine_sim:.4f} < {min_cosine} "
            f"(mean over {n_samples} samples)"
        )

        return assert_true(
            passed,
            name="model.attribution.cosine_stable",
            message=message,
            severity=Severity.CRITICAL,
            cosine_similarity=cosine_sim,
            min_cosine=min_cosine,
            n_samples=n_samples,
            n_features=n_features,
        )

    # --- 1-D: single vector ---
    a_flat = a.ravel()
    b_flat = b.ravel()
    n_features = len(a_flat)

    cosine_sim = float(_cosine(a_flat, b_flat))

    passed = cosine_sim >= min_cosine
    message = (
        f"Cosine stability={cosine_sim:.4f} >= {min_cosine} "
        f"({n_features} features)"
        if passed
        else f"Unstable: cosine={cosine_sim:.4f} < {min_cosine} "
        f"({n_features} features)"
    )

    return assert_true(
        passed,
        name="model.attribution.cosine_stable",
        message=message,
        severity=Severity.CRITICAL,
        cosine_similarity=cosine_sim,
        min_cosine=min_cosine,
        n_features=n_features,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors.

    Returns 0.0 when either vector has zero norm, rather than producing
    NaN or raising an error.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
