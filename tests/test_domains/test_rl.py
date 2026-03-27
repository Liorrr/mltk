"""Tests for mltk.domains.rl -- reinforcement learning reward assertions.

RL reward functions are hand-designed and notoriously error-prone. These
tests validate two critical properties:

1. Reward bounding: rewards must stay within a defined range. Unbounded
   rewards cause exploding gradients, diverging training, and wasted compute.

2. Cumulative reward: the total reward an agent earns in an episode must
   exceed a minimum. Low cumulative reward means the agent has not learned
   to solve its task.
"""

import numpy as np
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.domains.rl import assert_cumulative_reward, assert_reward_bounded


class TestRewardBounded:
    """Reward bounding tests.

    Reward functions must produce values within a defined range.
    Violations indicate reward function bugs (e.g., division by zero
    producing infinity, or edge cases producing extreme negatives).
    """

    def test_within_bounds_passes(self) -> None:
        """PASS: All rewards within [-1, 1].

        WHY: A well-designed reward function for a navigation task might
        give -1 for hitting a wall, +1 for reaching the goal, and small
        values in between. All values staying in range means the reward
        function is working correctly.
        Expected: n_violations=0, passes.
        """
        rewards = np.array([0.1, -0.5, 0.8, -1.0, 1.0, 0.0])
        result = assert_reward_bounded(rewards, min_reward=-1.0, max_reward=1.0)
        assert result.passed is True
        assert result.details["n_violations"] == 0
        assert result.details["n_total"] == 6

    def test_out_of_bounds_fails(self) -> None:
        """FAIL: Reward exceeds upper bound.

        WHY: A reward of 100.0 in a [-1, 1] system means the reward
        function has a bug (e.g., missing clamp, division producing
        large values). This causes gradient explosions during training.
        Expected: MltkAssertionError raised.
        """
        rewards = np.array([0.5, 100.0, -0.3])
        with pytest.raises(MltkAssertionError):
            assert_reward_bounded(rewards, min_reward=-1.0, max_reward=1.0)

    def test_only_min_bound(self) -> None:
        """PASS: Only lower bound checked, all rewards above it.

        WHY: Some reward functions have a natural lower bound (e.g.,
        distance-based rewards are >= 0) but no upper limit. Checking
        only the lower bound catches underflow bugs without constraining
        the upside.
        Expected: Passes with only min_reward set.
        """
        rewards = [0.0, 1.0, 5.0, 100.0]
        result = assert_reward_bounded(rewards, min_reward=0.0)
        assert result.passed is True
        assert result.details["actual_min"] == 0.0

    def test_only_max_bound(self) -> None:
        """PASS: Only upper bound checked, all rewards below it.

        WHY: Penalty-only reward functions (all rewards <= 0) should
        never produce positive values. Checking only max_reward catches
        sign errors in the reward computation.
        Expected: Passes with only max_reward set.
        """
        rewards = [-5.0, -1.0, -0.1, 0.0]
        result = assert_reward_bounded(rewards, max_reward=0.0)
        assert result.passed is True
        assert result.details["actual_max"] == 0.0

    def test_both_bounds_none_fails(self) -> None:
        """FAIL: No bounds provided -- assertion is meaningless.

        WHY: If neither min nor max is specified, there is nothing to
        check. This is a caller error and should fail immediately with
        a clear message.
        Expected: MltkAssertionError raised.
        """
        with pytest.raises(MltkAssertionError):
            assert_reward_bounded([1.0, 2.0])

    def test_single_step(self) -> None:
        """PASS: Single reward value within bounds.

        WHY: Edge case for single-step episodes or per-step monitoring.
        The assertion should handle n_total=1 without errors.
        Expected: n_total=1, actual_min == actual_max.
        """
        result = assert_reward_bounded([0.5], min_reward=0.0, max_reward=1.0)
        assert result.passed is True
        assert result.details["n_total"] == 1
        assert result.details["actual_min"] == result.details["actual_max"]

    def test_violation_count(self) -> None:
        """FAIL: Multiple violations counted accurately.

        WHY: When debugging a reward function, engineers need to know
        HOW MANY rewards violated bounds, not just that some did. Two
        violations out of five is a different severity than five out of five.
        Expected: n_violations=2 (the -2.0 and 2.0 values).
        """
        rewards = np.array([0.5, -2.0, 0.3, 2.0, -0.1])
        with pytest.raises(MltkAssertionError) as exc:
            assert_reward_bounded(rewards, min_reward=-1.0, max_reward=1.0)
        result = exc.value.result
        assert result.details["n_violations"] == 2


class TestCumulativeReward:
    """Cumulative reward tests.

    Cumulative reward (sum of all step rewards) is the primary measure
    of RL agent performance. An agent that finishes episodes with low
    total reward has not learned to solve the task.
    """

    def test_high_sum_passes(self) -> None:
        """PASS: Cumulative reward exceeds threshold.

        WHY: An agent earning 10.0 total reward in a task where 5.0 is
        the minimum means it has learned effective behavior. This is the
        gate check before promoting a trained agent to production.
        Expected: cumulative_reward=10.0 >= min_cumulative=5.0.
        """
        rewards = np.array([1.0, 2.0, 3.0, 4.0])
        result = assert_cumulative_reward(rewards, min_cumulative=5.0)
        assert result.passed is True
        assert result.details["cumulative_reward"] == 10.0
        assert result.details["n_steps"] == 4

    def test_low_sum_fails(self) -> None:
        """FAIL: Cumulative reward below threshold.

        WHY: An agent that earns only 2.0 when 10.0 is required has not
        learned the task. This blocks the agent from deployment, saving
        users from a broken product.
        Expected: MltkAssertionError raised.
        """
        rewards = [0.5, 0.5, 0.5, 0.5]
        with pytest.raises(MltkAssertionError):
            assert_cumulative_reward(rewards, min_cumulative=10.0)

    def test_single_step(self) -> None:
        """PASS: Single-step episode with sufficient reward.

        WHY: Some environments (e.g., bandit problems) have single-step
        episodes. The assertion should handle n_steps=1 correctly, with
        cumulative_reward == mean_reward == the single value.
        Expected: n_steps=1, mean_reward == cumulative_reward.
        """
        result = assert_cumulative_reward([5.0], min_cumulative=3.0)
        assert result.passed is True
        assert result.details["n_steps"] == 1
        assert result.details["mean_reward"] == result.details["cumulative_reward"]

    def test_all_zeros(self) -> None:
        """FAIL: All-zero rewards with positive threshold.

        WHY: All-zero rewards usually mean the environment is broken --
        the agent never reaches a reward-producing state, or the reward
        function always returns 0. This should fail for any positive
        threshold, alerting engineers to debug the environment.
        Expected: MltkAssertionError raised, cumulative_reward=0.
        """
        rewards = [0.0, 0.0, 0.0, 0.0, 0.0]
        with pytest.raises(MltkAssertionError):
            assert_cumulative_reward(rewards, min_cumulative=1.0)

    def test_negative_rewards(self) -> None:
        """PASS: Negative rewards with negative threshold.

        WHY: Penalty-based environments give negative rewards for every
        step (time cost) and zero for reaching the goal. A cumulative
        reward of -3.0 with a threshold of -5.0 means the agent solved
        the task in fewer steps than the maximum allowed penalty.
        Expected: cumulative_reward=-3.0 >= min_cumulative=-5.0.
        """
        rewards = np.array([-1.0, -1.0, -1.0])
        result = assert_cumulative_reward(rewards, min_cumulative=-5.0)
        assert result.passed is True
        assert result.details["cumulative_reward"] == -3.0
        assert result.details["mean_reward"] == -1.0

    def test_details_include_mean(self) -> None:
        """PASS: Mean reward is included in details.

        WHY: Mean reward per step helps diagnose whether the agent is
        earning consistently small rewards (mean ~0.5 over 20 steps) or
        has a few big spikes (mean ~0.5 but some steps at 5.0). Both
        produce the same cumulative but indicate different behaviors.
        Expected: mean_reward = cumulative / n_steps.
        """
        rewards = [2.0, 4.0, 6.0]
        result = assert_cumulative_reward(rewards, min_cumulative=1.0)
        assert result.passed is True
        assert result.details["mean_reward"] == 4.0
        assert result.details["cumulative_reward"] == 12.0
        assert result.details["n_steps"] == 3
