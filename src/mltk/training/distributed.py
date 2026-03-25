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
