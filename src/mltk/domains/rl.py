"""Reinforcement learning assertions -- reward validation and episode quality.

Reinforcement learning (RL) trains agents by trial-and-error using reward
signals. Unlike supervised learning where labels are fixed, RL reward
functions are hand-designed and notoriously buggy. A broken reward function
wastes GPU-days of training on an agent that learns the wrong behavior
(reward hacking) or learns nothing at all.

These assertions catch the two most common RL bugs:

1. **Unbounded rewards**: A reward that shoots to infinity (or negative
   infinity) destabilizes training. Gradient updates become enormous,
   weights explode, and the agent diverges. This is the RL equivalent
   of a NaN loss in supervised learning.

2. **Low cumulative reward**: If the agent finishes episodes with low
   total reward, it is not solving the task. This is the RL equivalent
   of low accuracy -- the agent has not learned useful behavior.

Both assertions work on raw reward arrays from episodes, requiring no
special RL framework. They integrate into CI pipelines to gate model
promotions: only agents that achieve bounded, sufficient rewards proceed
to deployment.
"""

from __future__ import annotations

import numpy as np

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult

__all__ = [
    "assert_reward_bounded",
    "assert_cumulative_reward",
]


@timed_assertion
def assert_reward_bounded(
    rewards: np.ndarray | list,
    min_reward: float | None = None,
    max_reward: float | None = None,
) -> TestResult:
    """Assert that all rewards fall within specified bounds.

    Reward functions are hand-designed and often contain edge-case bugs
    that produce extreme values. An unbounded reward causes:
    - Exploding gradients during policy optimization
    - Numerical overflow in advantage estimation (GAE)
    - Reward hacking where the agent exploits the unbounded signal

    This assertion checks every reward value against [min_reward, max_reward].
    At least one bound must be provided.

    Args:
        rewards: Array of reward values from one or more episodes.
        min_reward: Minimum allowed reward (inclusive). None to skip lower bound.
        max_reward: Maximum allowed reward (inclusive). None to skip upper bound.

    Returns:
        TestResult with details: ``actual_min``, ``actual_max``,
        ``n_violations``, ``n_total``.

    Raises:
        MltkAssertionError: If any reward violates the bounds (CRITICAL severity).

    Example:
        >>> import numpy as np
        >>> rewards = np.array([0.1, 0.5, -0.2, 0.8])
        >>> result = assert_reward_bounded(rewards, min_reward=-1.0, max_reward=1.0)
        >>> result.passed
        True
    """
    if min_reward is None and max_reward is None:
        return assert_true(
            False,
            name="rl.reward_bounded",
            message="At least one of min_reward or max_reward must be provided",
            severity=Severity.CRITICAL,
        )

    r = np.asarray(rewards, dtype=float).ravel()
    n_total = len(r)

    if n_total == 0:
        return assert_true(
            False,
            name="rl.reward_bounded",
            message="Cannot check bounds on empty rewards array",
            severity=Severity.CRITICAL,
        )

    actual_min = float(np.min(r))
    actual_max = float(np.max(r))

    # Count violations
    violations = np.zeros(n_total, dtype=bool)
    if min_reward is not None:
        violations |= r < min_reward
    if max_reward is not None:
        violations |= r > max_reward
    n_violations = int(np.sum(violations))

    passed = n_violations == 0

    # Build descriptive message
    bounds_str = ""
    if min_reward is not None and max_reward is not None:
        bounds_str = f"[{min_reward}, {max_reward}]"
    elif min_reward is not None:
        bounds_str = f"[{min_reward}, inf)"
    else:
        bounds_str = f"(-inf, {max_reward}]"

    message = (
        f"All {n_total} rewards within {bounds_str} "
        f"(actual range: [{actual_min:.4f}, {actual_max:.4f}])"
        if passed
        else (
            f"{n_violations}/{n_total} rewards outside {bounds_str} "
            f"(actual range: [{actual_min:.4f}, {actual_max:.4f}])"
        )
    )

    return assert_true(
        passed,
        name="rl.reward_bounded",
        message=message,
        severity=Severity.CRITICAL,
        actual_min=actual_min,
        actual_max=actual_max,
        n_violations=n_violations,
        n_total=n_total,
    )

@timed_assertion
def assert_cumulative_reward(
    rewards: np.ndarray | list,
    min_cumulative: float,
) -> TestResult:
    """Assert that cumulative episode reward meets a minimum threshold.

    Cumulative reward is the sum of all step rewards in an episode. It is
    the primary measure of whether an RL agent has learned to solve its
    task. Low cumulative reward means:
    - The agent has not learned useful behavior
    - The reward function may be too sparse (agent never discovers reward)
    - The environment may be broken (agent cannot reach goal states)

    This is the RL equivalent of checking model accuracy -- a necessary
    (but not sufficient) condition for deployment.

    Args:
        rewards: Array of per-step reward values from an episode.
        min_cumulative: Minimum required sum of rewards.

    Returns:
        TestResult with details: ``cumulative_reward``, ``min_cumulative``,
        ``n_steps``, ``mean_reward``.

    Raises:
        MltkAssertionError: If cumulative_reward < min_cumulative (CRITICAL severity).

    Example:
        >>> import numpy as np
        >>> # Agent earns small rewards each step, totaling 8.0
        >>> rewards = np.array([1.0, 2.0, 1.5, 0.5, 3.0])
        >>> result = assert_cumulative_reward(rewards, min_cumulative=5.0)
        >>> result.passed
        True
    """
    r = np.asarray(rewards, dtype=float).ravel()
    n_steps = len(r)

    if n_steps == 0:
        return assert_true(
            False,
            name="rl.cumulative_reward",
            message="Cannot compute cumulative reward on empty rewards array",
            severity=Severity.CRITICAL,
        )

    cumulative = float(np.sum(r))
    mean_reward = float(np.mean(r))

    passed = cumulative >= min_cumulative
    message = (
        f"Cumulative reward: {cumulative:.4f} >= {min_cumulative} "
        f"({n_steps} steps, mean={mean_reward:.4f})"
        if passed
        else (
            f"Cumulative reward too low: {cumulative:.4f} < {min_cumulative} "
            f"({n_steps} steps, mean={mean_reward:.4f})"
        )
    )

    return assert_true(
        passed,
        name="rl.cumulative_reward",
        message=message,
        severity=Severity.CRITICAL,
        cumulative_reward=cumulative,
        min_cumulative=min_cumulative,
        n_steps=n_steps,
        mean_reward=mean_reward,
    )
