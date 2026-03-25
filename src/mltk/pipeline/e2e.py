"""End-to-end pipeline testing -- validate full data→train→predict pipeline."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


@timed_assertion
def assert_pipeline(
    steps: list[Callable[..., Any]],
    input_data: Any,
    expected_output_type: type | None = None,
) -> TestResult:
    """Assert an end-to-end pipeline runs without errors.

    Args:
        steps: List of callables to run in sequence.
        input_data: Initial input passed to first step.
        expected_output_type: Expected type of final output. None = skip type check.

    Returns:
        TestResult with pipeline execution details.

    Example:
        >>> steps = [lambda x: x * 2, lambda x: x + 1]
        >>> assert_pipeline(steps, input_data=5, expected_output_type=int)
    """
    current = input_data
    completed_steps = 0

    for i, step in enumerate(steps):
        try:
            current = step(current)
            completed_steps += 1
        except Exception as e:
            return assert_true(
                False,
                name="pipeline.e2e",
                message=f"Pipeline failed at step {i} ({step.__name__}): {type(e).__name__}: {e}",
                severity=Severity.CRITICAL,
                failed_step=i,
                step_name=step.__name__,
                completed_steps=completed_steps,
                total_steps=len(steps),
            )

    # Type check if requested
    if expected_output_type is not None and not isinstance(current, expected_output_type):
        return assert_true(
            False,
            name="pipeline.e2e",
            message=(
                f"Pipeline output type mismatch: expected {expected_output_type.__name__}, "
                f"got {type(current).__name__}"
            ),
            severity=Severity.CRITICAL,
            completed_steps=completed_steps,
            total_steps=len(steps),
        )

    return assert_true(
        True,
        name="pipeline.e2e",
        message=f"Pipeline completed: {completed_steps}/{len(steps)} steps",
        severity=Severity.CRITICAL,
        completed_steps=completed_steps,
        total_steps=len(steps),
    )
