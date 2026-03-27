"""Distributed training validation — verify multi-GPU/multi-node correctness.

Distributed training bugs are silent by design: each rank believes it is
correct. Gradient desync, batch size misconfiguration, and world_size
mismatches only surface as unexpectedly slow convergence or diverging runs
across replicas. These assertions take plain numpy arrays — framework agnostic.
"""

from __future__ import annotations

import numpy as np

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


@timed_assertion
def assert_effective_batch_size(
    local_batch_size: int,
    world_size: int,
    expected_batch_size: int,
) -> TestResult:
    """Assert effective batch size equals local_batch_size * world_size.

    A mismatch means gradient accumulation is wrong or world_size is
    misconfigured. Common bug: forgetting to scale learning rate when changing
    world_size (linear scaling rule requires lr *= world_size).

    Args:
        local_batch_size: Batch size per GPU/process.
        world_size: Total number of GPUs/processes in the job.
        expected_batch_size: The batch size the training recipe was designed for.

    Returns:
        TestResult with ``effective_batch_size``, ``local_batch_size``, and
        ``world_size`` details.

    Example:
        >>> assert_effective_batch_size(local_batch_size=32, world_size=4, expected_batch_size=128)
    """
    effective = local_batch_size * world_size
    passed = effective == expected_batch_size
    message = (
        f"Effective batch size OK: {local_batch_size} x {world_size} = {effective}"
        if passed
        else (
            f"Batch size mismatch: {local_batch_size} x {world_size} = {effective}, "
            f"expected {expected_batch_size}. "
            f"Check gradient accumulation steps or world_size configuration."
        )
    )

    return assert_true(
        passed,
        name="training.effective_batch_size",
        message=message,
        severity=Severity.CRITICAL,
        effective_batch_size=effective,
        local_batch_size=local_batch_size,
        world_size=world_size,
        expected_batch_size=expected_batch_size,
    )


@timed_assertion
def assert_gradient_sync(
    grads_rank0: list[np.ndarray],
    grads_rank1: list[np.ndarray],
    tolerance: float = 1e-5,
) -> TestResult:
    """Assert gradients are synchronized across ranks after all-reduce.

    Compares gradient arrays from two ranks element-wise. After a correct
    all-reduce (DDP/FSDP/Horovod), gradients must be nearly identical within
    floating-point tolerance. Diverged gradients mean ranks are learning
    different things — model replicas will desynchronize over time.

    Reports the maximum absolute difference found and which layer indices
    exceed the tolerance threshold.

    Args:
        grads_rank0: List of gradient numpy arrays from rank 0, one per layer.
        grads_rank1: List of gradient numpy arrays from rank 1, one per layer.
        tolerance: Maximum allowed element-wise absolute difference.

    Returns:
        TestResult with ``max_diff``, ``diverged_layers``, ``num_layers``, and
        ``tolerance`` details.

    Example:
        >>> grads0 = [np.array([0.01, 0.02]), np.array([0.1, -0.05])]
        >>> grads1 = [np.array([0.01, 0.02]), np.array([0.1, -0.05])]
        >>> assert_gradient_sync(grads0, grads1, tolerance=1e-5)
    """
    if len(grads_rank0) != len(grads_rank1):
        return assert_true(
            False,
            name="training.gradient_sync",
            message=(
                f"Rank gradient list length mismatch: "
                f"rank0={len(grads_rank0)}, rank1={len(grads_rank1)}"
            ),
            severity=Severity.CRITICAL,
            max_diff=float("inf"),
            diverged_layers=[],
            num_layers=max(len(grads_rank0), len(grads_rank1)),
            tolerance=tolerance,
        )

    diverged_layers: list[int] = []
    max_diff: float = 0.0

    for i, (g0, g1) in enumerate(zip(grads_rank0, grads_rank1, strict=False)):
        arr0 = np.asarray(g0, dtype=float)
        arr1 = np.asarray(g1, dtype=float)
        if arr0.shape != arr1.shape:
            return assert_true(
                False,
                name="training.gradient_sync",
                message=(
                    f"Layer {i} shape mismatch: "
                    f"rank0={arr0.shape} vs rank1={arr1.shape}"
                ),
                severity=Severity.CRITICAL,
                max_diff=float("inf"),
                diverged_layers=[i],
                num_layers=len(grads_rank0),
                tolerance=tolerance,
            )
        layer_max_diff = float(np.max(np.abs(arr0 - arr1)))
        if layer_max_diff > max_diff:
            max_diff = layer_max_diff
        if layer_max_diff > tolerance:
            diverged_layers.append(i)

    passed = len(diverged_layers) == 0
    message = (
        f"Gradient sync OK: {len(grads_rank0)} layers within tolerance={tolerance} "
        f"(max_diff={max_diff:.2e})"
        if passed
        else (
            f"Gradient desync: layers {diverged_layers} exceed tolerance={tolerance} "
            f"(max_diff={max_diff:.2e} across {len(grads_rank0)} layers)"
        )
    )

    return assert_true(
        passed,
        name="training.gradient_sync",
        message=message,
        severity=Severity.CRITICAL,
        max_diff=max_diff,
        diverged_layers=diverged_layers,
        num_layers=len(grads_rank0),
        tolerance=tolerance,
    )


@timed_assertion
def assert_n_rank_gradient_sync(
    grads_by_rank: list[list[np.ndarray]],
    tolerance: float = 1e-5,
) -> TestResult:
    """Assert gradients are synchronized across N ranks after all-reduce.

    Generalizes :func:`assert_gradient_sync` beyond two ranks. Compares every
    pair of ranks element-wise; a correct all-reduce must produce identical
    gradients on all participants. Any divergence indicates a broken collective
    or missing synchronization barrier.

    Args:
        grads_by_rank: List of per-rank gradient lists. Each inner list has one
            numpy array per layer; all ranks must have the same number of
            layers with matching shapes.
        tolerance: Maximum allowed element-wise absolute difference between any
            pair of ranks.

    Returns:
        TestResult with ``max_diff``, ``diverged_pairs`` (list of
        ``(rank_i, rank_j, layer, diff)`` tuples), ``num_ranks``, and
        ``num_layers`` details.

    Example:
        >>> import numpy as np
        >>> g = [np.array([0.01, 0.02])]
        >>> assert_n_rank_gradient_sync([g, g, g], tolerance=1e-5)
    """
    num_ranks = len(grads_by_rank)

    if num_ranks < 1:
        return assert_true(
            False,
            name="training.n_rank_gradient_sync",
            message="No ranks provided — need at least 1.",
            severity=Severity.CRITICAL,
            max_diff=float("inf"),
            diverged_pairs=[],
            num_ranks=0,
            num_layers=0,
        )

    num_layers = len(grads_by_rank[0])
    for r in range(1, num_ranks):
        if len(grads_by_rank[r]) != num_layers:
            return assert_true(
                False,
                name="training.n_rank_gradient_sync",
                message=(
                    f"Layer count mismatch: rank 0 has {num_layers} layers, "
                    f"rank {r} has {len(grads_by_rank[r])} layers."
                ),
                severity=Severity.CRITICAL,
                max_diff=float("inf"),
                diverged_pairs=[],
                num_ranks=num_ranks,
                num_layers=num_layers,
            )

    diverged_pairs: list[tuple[int, int, int, float]] = []
    max_diff: float = 0.0

    for ri in range(num_ranks):
        for rj in range(ri + 1, num_ranks):
            for layer_idx in range(num_layers):
                arr_i = np.asarray(grads_by_rank[ri][layer_idx], dtype=float)
                arr_j = np.asarray(grads_by_rank[rj][layer_idx], dtype=float)
                if arr_i.shape != arr_j.shape:
                    return assert_true(
                        False,
                        name="training.n_rank_gradient_sync",
                        message=(
                            f"Shape mismatch at layer {layer_idx}: "
                            f"rank {ri} shape={arr_i.shape} vs "
                            f"rank {rj} shape={arr_j.shape}."
                        ),
                        severity=Severity.CRITICAL,
                        max_diff=float("inf"),
                        diverged_pairs=[],
                        num_ranks=num_ranks,
                        num_layers=num_layers,
                    )
                layer_diff = float(np.max(np.abs(arr_i - arr_j)))
                if layer_diff > max_diff:
                    max_diff = layer_diff
                if layer_diff > tolerance:
                    diverged_pairs.append((ri, rj, layer_idx, layer_diff))

    passed = len(diverged_pairs) == 0
    message = (
        f"N-rank gradient sync OK: {num_ranks} ranks, {num_layers} layers "
        f"within tolerance={tolerance} (max_diff={max_diff:.2e})"
        if passed
        else (
            f"Gradient desync across ranks: {len(diverged_pairs)} diverged pairs "
            f"exceed tolerance={tolerance} (max_diff={max_diff:.2e} across "
            f"{num_ranks} ranks, {num_layers} layers)"
        )
    )

    return assert_true(
        passed,
        name="training.n_rank_gradient_sync",
        message=message,
        severity=Severity.CRITICAL,
        max_diff=max_diff,
        diverged_pairs=diverged_pairs,
        num_ranks=num_ranks,
        num_layers=num_layers,
    )


@timed_assertion
def assert_gradient_alignment(
    grads_a: list[np.ndarray],
    grads_b: list[np.ndarray],
    min_cosine: float = 0.9,
) -> TestResult:
    """Assert gradient vectors from two ranks are directionally aligned.

    Flattens all layer gradients into a single vector per rank, then computes
    cosine similarity. Low cosine similarity means the ranks are optimizing in
    substantially different directions — a sign of data-parallel misconfiguration,
    stale weights, or a communication failure that corrupted gradient values
    without zeroing them.

    Args:
        grads_a: Gradient arrays from rank A, one per layer.
        grads_b: Gradient arrays from rank B, one per layer.
        min_cosine: Minimum acceptable cosine similarity (default 0.9).

    Returns:
        TestResult with ``cosine_similarity`` and ``min_cosine`` details.

    Example:
        >>> import numpy as np
        >>> g = [np.array([1.0, 2.0, 3.0])]
        >>> assert_gradient_alignment(g, g, min_cosine=0.9)
    """
    flat_a = np.concatenate([np.asarray(g, dtype=float).ravel() for g in grads_a])
    flat_b = np.concatenate([np.asarray(g, dtype=float).ravel() for g in grads_b])

    norm_a = float(np.linalg.norm(flat_a))
    norm_b = float(np.linalg.norm(flat_b))

    if norm_a == 0.0 or norm_b == 0.0:
        cosine = 0.0
    else:
        cosine = float(np.dot(flat_a, flat_b) / (norm_a * norm_b))

    passed = cosine >= min_cosine
    message = (
        f"Gradient alignment OK: cosine={cosine:.4f} >= {min_cosine}"
        if passed
        else (
            f"Gradient misalignment: cosine={cosine:.4f} < {min_cosine}. "
            f"Ranks are optimizing in divergent directions."
        )
    )

    return assert_true(
        passed,
        name="training.gradient_alignment",
        message=message,
        severity=Severity.CRITICAL,
        cosine_similarity=cosine,
        min_cosine=min_cosine,
    )


@timed_assertion
def assert_weight_divergence(
    weights_a: list[np.ndarray],
    weights_b: list[np.ndarray],
    max_l2_distance: float = 0.01,
) -> TestResult:
    """Assert weight vectors from two checkpoints or ranks are close in L2 space.

    Flattens all layer weights into a single vector per source, then computes
    the L2 (Euclidean) distance. A large distance means model replicas have
    drifted apart — common when all-reduce is silently dropped or gradient
    accumulation differs across ranks.

    Args:
        weights_a: Weight arrays from checkpoint/rank A, one per layer.
        weights_b: Weight arrays from checkpoint/rank B, one per layer.
        max_l2_distance: Maximum allowed L2 distance (default 0.01).

    Returns:
        TestResult with ``l2_distance``, ``max_l2_distance``, and ``num_params``
        details.

    Example:
        >>> import numpy as np
        >>> w = [np.array([0.5, -0.3, 0.1])]
        >>> assert_weight_divergence(w, w, max_l2_distance=0.01)
    """
    flat_a = np.concatenate([np.asarray(w, dtype=float).ravel() for w in weights_a])
    flat_b = np.concatenate([np.asarray(w, dtype=float).ravel() for w in weights_b])
    num_params = len(flat_a)

    l2 = float(np.linalg.norm(flat_a - flat_b))

    passed = l2 <= max_l2_distance
    message = (
        f"Weight divergence OK: L2={l2:.6f} <= {max_l2_distance} "
        f"({num_params} params)"
        if passed
        else (
            f"Weight divergence exceeded: L2={l2:.6f} > {max_l2_distance} "
            f"({num_params} params). Model replicas have drifted apart."
        )
    )

    return assert_true(
        passed,
        name="training.weight_divergence",
        message=message,
        severity=Severity.CRITICAL,
        l2_distance=l2,
        max_l2_distance=max_l2_distance,
        num_params=num_params,
    )


@timed_assertion
def assert_gradient_clipped(
    gradients: list[np.ndarray],
    max_norm: float,
) -> TestResult:
    """Assert the global gradient norm is within the clipping threshold.

    Computes the L2 norm of all gradients concatenated into a single vector.
    If gradient clipping is configured but not applied (e.g., called after
    backward but before clip), the global norm may exceed ``max_norm``,
    indicating a training pipeline ordering bug.

    Args:
        gradients: List of gradient numpy arrays, one per layer.
        max_norm: Maximum allowed global L2 norm.

    Returns:
        TestResult with ``global_norm``, ``max_norm``, and ``num_layers``
        details.

    Example:
        >>> import numpy as np
        >>> g = [np.array([0.1, 0.2]), np.array([-0.05])]
        >>> assert_gradient_clipped(g, max_norm=1.0)
    """
    flat = np.concatenate([np.asarray(g, dtype=float).ravel() for g in gradients])
    global_norm = float(np.linalg.norm(flat))
    num_layers = len(gradients)

    passed = global_norm <= max_norm
    message = (
        f"Gradient clipping OK: global_norm={global_norm:.6f} <= {max_norm} "
        f"({num_layers} layers)"
        if passed
        else (
            f"Gradient not clipped: global_norm={global_norm:.6f} > {max_norm} "
            f"({num_layers} layers). Clipping may not have been applied."
        )
    )

    return assert_true(
        passed,
        name="training.gradient_clipped",
        message=message,
        severity=Severity.CRITICAL,
        global_norm=global_norm,
        max_norm=max_norm,
        num_layers=num_layers,
    )
