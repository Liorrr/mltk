"""Tests for mltk.domains.llm.multi_agent — multi-agent coordination assertions."""

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.domains.llm.multi_agent import assert_agent_handoff, assert_no_agent_loop

# ── assert_no_agent_loop ─────────────────────────────────────────────────


class TestNoAgentLoop:
    """Loop detection — catch circular delegation in multi-agent sequences."""

    def test_no_loop_linear_chain(self) -> None:
        # SCENARIO: Four distinct agents hand off work in a straight line.
        # WHY: A linear pipeline (no repeated subsequence) should always pass.
        # EXPECTED: passes — no cycle detected.
        result = assert_no_agent_loop(
            ["router", "classifier", "specialist", "summarizer"],
        )
        assert result.passed is True
        assert result.details["cycle_detected"] is False
        assert result.details["cycle_pattern"] is None
        assert result.details["cycle_count"] == 0
        assert result.details["total_handoffs"] == 4

    def test_simple_ab_loop_detected(self) -> None:
        # SCENARIO: Two agents delegate back and forth 4 times: A->B->A->B->A->B->A->B.
        # WHY: The cycle ["A","B"] repeats 4 times, exceeding the default max_cycles=2.
        #   This is the classic infinite-delegation failure: A asks B, B asks A, repeat.
        # EXPECTED: raises MltkAssertionError with cycle=["A","B"], count=4.
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_no_agent_loop(["A", "B", "A", "B", "A", "B", "A", "B"])
        result = exc_info.value.result
        assert result.details["cycle_detected"] is True
        assert result.details["cycle_pattern"] == ["A", "B"]
        assert result.details["cycle_count"] == 4

    def test_long_chain_no_loop(self) -> None:
        # SCENARIO: A long sequence with varied agents and no repeating cycle.
        # WHY: Length alone does not indicate a loop — variety matters.
        # EXPECTED: passes — no cycle of length >= 2 repeats excessively.
        agents = ["planner", "researcher", "coder", "reviewer", "tester", "deployer"]
        result = assert_no_agent_loop(agents)
        assert result.passed is True
        assert result.details["cycle_detected"] is False

    def test_single_agent_repeated_not_a_loop(self) -> None:
        # SCENARIO: The same agent handles 5 consecutive steps.
        # WHY: A single agent retrying is NOT a multi-agent coordination loop.
        #   The assertion only detects cycles of length >= 2 (inter-agent loops),
        #   so a single agent running repeatedly is a different concern
        #   (handled by assert_step_efficiency instead).
        # EXPECTED: passes — cycle length 1 is not considered a loop.
        result = assert_no_agent_loop(
            ["summarizer", "summarizer", "summarizer", "summarizer", "summarizer"],
        )
        assert result.passed is True
        assert result.details["cycle_detected"] is False

    def test_max_cycles_threshold(self) -> None:
        # SCENARIO: A->B repeats exactly 3 times.  With max_cycles=3, this should
        #   pass (3 <= 3).  With max_cycles=2, it should fail (3 > 2).
        # WHY: The threshold is inclusive — repeating exactly max_cycles times is OK.
        # EXPECTED: passes at max_cycles=3, fails at max_cycles=2.
        agents = ["A", "B", "A", "B", "A", "B"]  # cycle repeats 3 times

        result = assert_no_agent_loop(agents, max_cycles=3)
        assert result.passed is True
        assert result.details["cycle_count"] == 3

        with pytest.raises(MltkAssertionError) as exc_info:
            assert_no_agent_loop(agents, max_cycles=2)
        result = exc_info.value.result
        assert result.details["cycle_detected"] is True
        assert result.details["cycle_count"] == 3

    def test_no_loop_has_timing(self) -> None:
        # SCENARIO: Assertion is wrapped with @timed_assertion.
        # WHY: duration_ms must be populated for performance tracking.
        # EXPECTED: duration_ms > 0.
        result = assert_no_agent_loop(["A", "B", "C"])
        assert result.duration_ms > 0


# ── assert_agent_handoff ─────────────────────────────────────────────────


class TestAgentHandoff:
    """Handoff validation — agent sequence follows the expected flow."""

    def test_exact_match_strict_passes(self) -> None:
        # SCENARIO: Actual flow exactly matches expected flow in strict mode.
        # WHY: When strict=True, the sequences must be identical — same agents,
        #   same order, same length.
        # EXPECTED: passes with no missing agents.
        result = assert_agent_handoff(
            agent_names=["router", "classifier", "specialist"],
            expected_flow=["router", "classifier", "specialist"],
            strict=True,
        )
        assert result.passed is True
        assert result.details["missing_agents"] == []
        assert result.details["strict"] is True

    def test_subsequence_nonstrict_passes(self) -> None:
        # SCENARIO: Expected flow is a subsequence of the actual flow.
        #   Extra "logger" agent appears between router and classifier.
        # WHY: Non-strict mode allows intermediate agents (logging, monitoring)
        #   that don't change the fundamental pipeline structure.
        # EXPECTED: passes — router, classifier, specialist appear in order.
        result = assert_agent_handoff(
            agent_names=["router", "logger", "classifier", "monitor", "specialist"],
            expected_flow=["router", "classifier", "specialist"],
            strict=False,
        )
        assert result.passed is True
        assert result.details["missing_agents"] == []
        assert result.details["strict"] is False

    def test_missing_agent_fails(self) -> None:
        # SCENARIO: Expected flow includes "specialist" but actual skips it.
        # WHY: A missing agent means the pipeline skipped a critical step.
        #   In a support system, skipping the specialist means the customer
        #   gets a summary without expert analysis.
        # EXPECTED: raises MltkAssertionError; "specialist" in missing_agents.
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_agent_handoff(
                agent_names=["router", "classifier", "summarizer"],
                expected_flow=["router", "classifier", "specialist", "summarizer"],
            )
        result = exc_info.value.result
        assert "specialist" in result.details["missing_agents"]

    def test_wrong_order_strict_fails(self) -> None:
        # SCENARIO: All expected agents are present but in wrong order (strict mode).
        # WHY: In strict mode, order matters — classifier before router means
        #   classification happened without routing, which is architecturally wrong.
        # EXPECTED: raises MltkAssertionError.
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_agent_handoff(
                agent_names=["classifier", "router", "specialist"],
                expected_flow=["router", "classifier", "specialist"],
                strict=True,
            )
        assert exc_info.value.result.passed is False

    def test_empty_expected_passes(self) -> None:
        # SCENARIO: Expected flow is empty (no handoff requirements).
        # WHY: An empty expected flow means "any sequence is acceptable" —
        #   useful when testing that agents run at all without caring about order.
        # EXPECTED: passes regardless of actual agent sequence.
        result = assert_agent_handoff(
            agent_names=["router", "classifier"],
            expected_flow=[],
        )
        assert result.passed is True
        assert result.details["missing_agents"] == []

    def test_handoff_has_timing(self) -> None:
        # SCENARIO: Assertion is wrapped with @timed_assertion.
        # WHY: duration_ms must be populated for performance tracking.
        # EXPECTED: duration_ms > 0.
        result = assert_agent_handoff(
            agent_names=["A", "B"],
            expected_flow=["A", "B"],
        )
        assert result.duration_ms > 0
