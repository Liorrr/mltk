"""Checkpoint/resume validation — verify training can resume correctly.

Checkpoint bugs are expensive: a training run that silently fails to save
optimizer state (only saving model weights) will appear to resume but will
restart learning rate schedules, momentum buffers, and scaler state from
scratch — causing a visible loss spike or invisible convergence regression.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult

_DEFAULT_REQUIRED_KEYS = ["epoch", "model_state"]


@timed_assertion
def assert_checkpoint_complete(
    path: str | Path,
    required_keys: list[str] | None = None,
) -> TestResult:
    """Assert checkpoint file exists and contains required keys.

    Supports JSON checkpoints. Validates that:
    1. The file exists at ``path``.
    2. The file parses as valid JSON.
    3. All ``required_keys`` are present at the top level of the JSON object.

    Args:
        path: Path to the checkpoint file (JSON format).
        required_keys: Keys that must be present in the checkpoint. Defaults to
            ``["epoch", "model_state"]``.

    Returns:
        TestResult with file path, found keys, and any missing keys.

    Example:
        >>> assert_checkpoint_complete("/runs/ckpt_epoch5.json")
    """
    if required_keys is None:
        required_keys = _DEFAULT_REQUIRED_KEYS

    checkpoint_path = Path(path)

    if not checkpoint_path.exists():
        return assert_true(
            False,
            name="training.checkpoint_complete",
            message=f"Checkpoint file not found: {checkpoint_path}",
            severity=Severity.CRITICAL,
            path=str(checkpoint_path),
            required_keys=required_keys,
        )

    # Attempt to parse as JSON
    try:
        with checkpoint_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        return assert_true(
            False,
            name="training.checkpoint_complete",
            message=f"Checkpoint is not valid JSON: {exc}",
            severity=Severity.CRITICAL,
            path=str(checkpoint_path),
            required_keys=required_keys,
        )
    except Exception as exc:  # noqa: BLE001
        return assert_true(
            False,
            name="training.checkpoint_complete",
            message=f"Failed to read checkpoint: {exc}",
            severity=Severity.CRITICAL,
            path=str(checkpoint_path),
            required_keys=required_keys,
        )

    if not isinstance(data, dict):
        return assert_true(
            False,
            name="training.checkpoint_complete",
            message="Checkpoint JSON is not a top-level object (dict expected)",
            severity=Severity.CRITICAL,
            path=str(checkpoint_path),
            required_keys=required_keys,
        )

    found_keys = list(data.keys())
    missing_keys = [k for k in required_keys if k not in data]
    passed = len(missing_keys) == 0

    message = (
        f"Checkpoint complete: all required keys present {required_keys}"
        if passed
        else f"Checkpoint incomplete: missing keys {missing_keys} "
        f"(found: {found_keys})"
    )

    return assert_true(
        passed,
        name="training.checkpoint_complete",
        message=message,
        severity=Severity.CRITICAL,
        path=str(checkpoint_path),
        required_keys=required_keys,
        missing_keys=missing_keys,
        found_keys=found_keys,
    )


@timed_assertion
def assert_resume_loss_continuous(
    pre_losses: list[float],
    post_losses: list[float],
    max_gap: float = 0.5,
) -> TestResult:
    """Assert loss continuity after checkpoint resume.

    Compares the last pre-checkpoint loss to the first post-checkpoint loss.
    A gap larger than ``max_gap`` indicates that the checkpoint did not
    properly restore optimizer state (learning rate schedule, momentum buffers,
    gradient scaler) — causing a loss spike at resume time.

    Args:
        pre_losses: Loss values recorded before the checkpoint (e.g., last N
            training steps before saving). Must be non-empty.
        post_losses: Loss values recorded after resuming from the checkpoint
            (e.g., first N training steps after loading). Must be non-empty.
        max_gap: Maximum allowed absolute difference between the last
            pre-checkpoint loss and first post-checkpoint loss. Default 0.5.

    Returns:
        TestResult with last pre-loss, first post-loss, and measured gap.

    Example:
        >>> assert_resume_loss_continuous(pre=[1.2, 1.1, 1.0], post=[1.05, 0.98])
    """
    if not pre_losses:
        return assert_true(
            False,
            name="training.resume_loss_continuous",
            message="pre_losses is empty — cannot determine pre-checkpoint loss",
            severity=Severity.CRITICAL,
            max_gap=max_gap,
        )

    if not post_losses:
        return assert_true(
            False,
            name="training.resume_loss_continuous",
            message="post_losses is empty — cannot determine post-resume loss",
            severity=Severity.CRITICAL,
            max_gap=max_gap,
        )

    last_pre = float(pre_losses[-1])
    first_post = float(post_losses[0])

    # Guard against NaN/inf in provided losses
    if not np.isfinite(last_pre):
        return assert_true(
            False,
            name="training.resume_loss_continuous",
            message=f"Last pre-checkpoint loss is non-finite: {last_pre}",
            severity=Severity.CRITICAL,
            last_pre_loss=last_pre,
            first_post_loss=first_post,
            max_gap=max_gap,
        )

    if not np.isfinite(first_post):
        return assert_true(
            False,
            name="training.resume_loss_continuous",
            message=f"First post-resume loss is non-finite: {first_post}",
            severity=Severity.CRITICAL,
            last_pre_loss=last_pre,
            first_post_loss=first_post,
            max_gap=max_gap,
        )

    gap = abs(first_post - last_pre)
    passed = gap <= max_gap

    message = (
        f"Loss continuous after resume: |{first_post:.4f} - {last_pre:.4f}| "
        f"= {gap:.4f} <= {max_gap} threshold"
        if passed
        else f"Loss discontinuity after resume: |{first_post:.4f} - {last_pre:.4f}| "
        f"= {gap:.4f} > {max_gap} threshold — "
        f"optimizer state likely not restored from checkpoint"
    )

    return assert_true(
        passed,
        name="training.resume_loss_continuous",
        message=message,
        severity=Severity.CRITICAL,
        last_pre_loss=round(last_pre, 6),
        first_post_loss=round(first_post, 6),
        gap=round(gap, 6),
        max_gap=max_gap,
        pre_steps=len(pre_losses),
        post_steps=len(post_losses),
    )
