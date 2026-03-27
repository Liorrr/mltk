"""Pipeline reproducibility testing -- ensure deterministic training and artifact integrity."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


@timed_assertion
def assert_reproducible(
    func: Callable[..., Any],
    *args: Any,
    seed: int = 42,
    runs: int = 3,
    tolerance: float = 0.001,
) -> TestResult:
    """Assert function produces identical output across runs with same seed.

    Args:
        func: Function to test (e.g., train_model).
        *args: Arguments to pass to func.
        seed: Random seed to set before each run.
        runs: Number of runs to compare.
        tolerance: Max allowed difference between outputs (for numeric).

    Returns:
        TestResult with reproducibility details.

    Example:
        >>> import numpy as np
        >>> def train(data): return np.mean(data)
        >>> assert_reproducible(train, [1.0, 2.0, 3.0], seed=42, runs=3)
    """
    outputs = []
    for _run in range(runs):
        random.seed(seed)
        np.random.seed(seed)

        # Optional framework seeding — silently skip if not installed
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

        try:
            import tensorflow as tf

            tf.random.set_seed(seed)
        except ImportError:
            pass

        result = func(*args)
        outputs.append(result)

    # Compare outputs
    base = outputs[0]
    max_diff = 0.0
    all_match = True

    for i in range(1, len(outputs)):
        try:
            if isinstance(base, np.ndarray):
                diff = float(np.max(np.abs(np.asarray(base) - np.asarray(outputs[i]))))
            elif isinstance(base, (int, float)):
                diff = abs(float(base) - float(outputs[i]))
            else:
                diff = 0.0 if base == outputs[i] else 1.0
            max_diff = max(max_diff, diff)
            if diff > tolerance:
                all_match = False
        except (TypeError, ValueError):
            if base != outputs[i]:
                all_match = False
                max_diff = 1.0

    message = (
        f"Reproducible: max_diff={max_diff:.6f} <= {tolerance} across {runs} runs"
        if all_match
        else f"Non-reproducible: max_diff={max_diff:.6f} > {tolerance}"
    )

    return assert_true(
        all_match,
        name="pipeline.reproducible",
        message=message,
        severity=Severity.CRITICAL,
        seed=seed,
        runs=runs,
        tolerance=tolerance,
        max_diff=max_diff,
    )


@timed_assertion
def assert_checksum(
    path: str | Path,
    expected_hash: str,
) -> TestResult:
    """Assert file matches expected SHA-256 hash.

    Args:
        path: Path to file to verify.
        expected_hash: Expected SHA-256 hex digest (with or without 'sha256:' prefix).

    Returns:
        TestResult with hash comparison.

    Example:
        >>> assert_checksum("model.onnx", "sha256:abc123...")
    """
    p = Path(path)
    if not p.exists():
        return assert_true(
            False,
            name="pipeline.checksum",
            message=f"File not found: {path}",
            severity=Severity.CRITICAL,
        )

    # Compute hash
    sha = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    actual = sha.hexdigest()

    # Strip prefix
    expected = expected_hash.removeprefix("sha256:")

    passed = actual == expected
    message = (
        f"Checksum OK: {actual[:16]}..."
        if passed
        else f"Checksum mismatch: expected {expected[:16]}..., got {actual[:16]}..."
    )

    return assert_true(
        passed,
        name="pipeline.checksum",
        message=message,
        severity=Severity.CRITICAL,
        actual_hash=actual,
        expected_hash=expected,
    )
