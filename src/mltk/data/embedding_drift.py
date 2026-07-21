"""Embedding drift detection for text, image, and multimodal ML systems.

Detects when embedding distributions shift from training baseline.
Supports centroid distance (cosine, euclidean) and MMD methods.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


@timed_assertion
def assert_no_embedding_drift(
    reference: Any,
    current: Any,
    method: str = "cosine",
    threshold: float = 0.1,
    seed: int = 0,
) -> TestResult:
    """Assert no significant drift in embedding space.

    Args:
        reference: Reference embeddings (N, D) array.
        current: Current embeddings (M, D) array.
        method: Detection method -- "cosine", "euclidean", or "mmd".
        threshold: Maximum allowed distance/divergence.
        seed: Random seed for deterministic MMD subsampling.

    Returns:
        TestResult with drift statistics.

    Example:
        >>> ref = np.random.randn(100, 768)
        >>> cur = np.random.randn(100, 768)
        >>> assert_no_embedding_drift(ref, cur, method="cosine", threshold=0.2)
    """
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)

    if ref.size == 0 or cur.size == 0:
        return assert_true(
            False, name="data.embedding_drift",
            message="Cannot compute embedding drift on empty arrays",
            severity=Severity.CRITICAL,
        )

    if method == "cosine":
        distance = _cosine_centroid_distance(ref, cur)
    elif method == "euclidean":
        distance = _euclidean_centroid_distance(ref, cur)
    elif method == "mmd":
        distance = _mmd(ref, cur, seed=seed)
    else:
        return assert_true(
            False, name="data.embedding_drift",
            message=f"Unknown method: '{method}'. Supported: cosine, euclidean, mmd",
            severity=Severity.CRITICAL,
        )

    passed = distance < threshold
    message = (
        f"Embedding drift ({method}): {distance:.6f} < {threshold}"
        if passed
        else f"Embedding drift detected ({method}): {distance:.6f} >= {threshold}"
    )

    return assert_true(
        passed, name=f"data.embedding_drift.{method}", message=message,
        severity=Severity.CRITICAL,
        method=method, distance=distance, threshold=threshold,
        ref_shape=list(ref.shape), cur_shape=list(cur.shape),
    )


def _cosine_centroid_distance(ref: np.ndarray, cur: np.ndarray) -> float:
    """Cosine distance between centroids of two embedding sets."""
    try:
        from mltk._rust import centroid_cosine_distance as _rust_centroid

        return _rust_centroid(ref.tolist(), cur.tolist())
    except ImportError:
        pass

    # numpy fallback
    ref_centroid = ref.mean(axis=0)
    cur_centroid = cur.mean(axis=0)
    cos_sim = np.dot(ref_centroid, cur_centroid) / (
        np.linalg.norm(ref_centroid) * np.linalg.norm(cur_centroid) + 1e-10
    )
    return float(1.0 - cos_sim)


def _euclidean_centroid_distance(ref: np.ndarray, cur: np.ndarray) -> float:
    """Euclidean distance between centroids."""
    return float(np.linalg.norm(ref.mean(axis=0) - cur.mean(axis=0)))


def _sample_rows(
    values: np.ndarray,
    max_rows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a deterministic random row sample, or the full array when smaller."""
    if len(values) <= max_rows:
        return values
    indices = rng.choice(len(values), size=max_rows, replace=False)
    return values[indices]


def _mmd(
    ref: np.ndarray,
    cur: np.ndarray,
    gamma: float | None = None,
    seed: int = 0,
) -> float:
    """Maximum Mean Discrepancy with RBF kernel.

    Uses the median heuristic for kernel bandwidth if gamma is not specified.
    """
    rng = np.random.default_rng(seed)

    if gamma is None:
        # Median heuristic
        combined = np.vstack([
            _sample_rows(ref, 100, rng),
            _sample_rows(cur, 100, rng),
        ])
        dists = np.sqrt(((combined[:, None] - combined[None, :]) ** 2).sum(axis=2))
        median_dist = float(np.median(dists[dists > 0]))
        gamma = 1.0 / (2.0 * median_dist ** 2) if median_dist > 0 else 1.0

    def rbf_kernel(x: np.ndarray, y: np.ndarray) -> float:
        dists_sq = ((x[:, None] - y[None, :]) ** 2).sum(axis=2)
        return float(np.exp(-gamma * dists_sq).mean())

    # Subsample for efficiency
    n = min(len(ref), 200)
    m = min(len(cur), 200)
    ref_sub = _sample_rows(ref, n, rng)
    cur_sub = _sample_rows(cur, m, rng)

    mmd_val = (
        rbf_kernel(ref_sub, ref_sub)
        - 2 * rbf_kernel(ref_sub, cur_sub)
        + rbf_kernel(cur_sub, cur_sub)
    )
    return float(max(0, mmd_val))
