"""Tests for trace-based agentic assertions (tool_chain, forbidden_actions, step_efficiency)."""

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.domains.llm.agentic import (
    assert_no_forbidden_actions,
    assert_step_efficiency,
    assert_tool_chain,
)
from mltk.domains.llm.trace import AgentTrace, ToolCall


def _make_trace(*tool_names: str) -> AgentTrace:
    """Helper — build an AgentTrace from a list of tool name strings."""
    return AgentTrace(
        tool_calls=[ToolCall(name=n, arguments={}) for n in tool_names],
    )


class TestToolChain:
    """assert_tool_chain — verify expected tools appear in an agent trace."""

    def test_unordered_match_pass(self) -> None:
        # SCENARIO: Trace contains both expected tools in a different order.
        # WHY: Default (non-strict) mode only checks presence, not order.
        # EXPECTED: passes; missing list is empty.
        trace = AgentTrace(tool_calls=[
            ToolCall(name="calculator", arguments={"expr": "2+2"}, result="4"),
            ToolCall(name="search", arguments={"q": "test"}),
        ])
        result = assert_tool_chain(trace, expected_tools=["search", "calculator"])
        assert result.passed is True
        assert result.details["missing"] == []
        assert result.details["strict_order"] is False

    def test_strict_order_pass(self) -> None:
        # SCENARIO: Trace has extra tools between expected ones, but order matches.
        # WHY: Subsequence check allows interleaved tools as long as order is correct.
        # EXPECTED: passes with strict_order=True.
        trace = _make_trace("search", "summarize", "calculator", "format")
        result = assert_tool_chain(
            trace,
            expected_tools=["search", "calculator", "format"],
            strict_order=True,
        )
        assert result.passed is True
        assert result.details["missing"] == []
        assert result.details["strict_order"] is True

    def test_missing_tool_fail(self) -> None:
        # SCENARIO: Expected tool "calculator" is absent from the trace.
        # WHY: Agent skipped a required step.
        # EXPECTED: raises MltkAssertionError; "calculator" in missing.
        trace = _make_trace("search", "format")
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_tool_chain(trace, expected_tools=["search", "calculator"])
        result = exc_info.value.result
        assert result.passed is False
        assert "calculator" in result.details["missing"]

    def test_strict_order_wrong_sequence_fail(self) -> None:
        # SCENARIO: Both tools present but in wrong order for strict mode.
        # WHY: "calculator" appears before "search" but expected order is search->calculator.
        # EXPECTED: raises MltkAssertionError.
        trace = _make_trace("calculator", "search")
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_tool_chain(
                trace,
                expected_tools=["search", "calculator"],
                strict_order=True,
            )
        result = exc_info.value.result
        assert result.passed is False
        assert result.details["strict_order"] is True
        # "calculator" is consumed first; then "search" is found;
        # but "calculator" was expected AFTER "search", so "calculator" is missing.
        assert len(result.details["missing"]) > 0

    def test_empty_expected_pass(self) -> None:
        # SCENARIO: No tools expected — any trace (even non-empty) is fine.
        # WHY: An empty expected list means there are no requirements.
        # EXPECTED: passes.
        trace = _make_trace("search")
        result = assert_tool_chain(trace, expected_tools=[])
        assert result.passed is True

    def test_empty_trace_with_expected_fail(self) -> None:
        # SCENARIO: Trace is empty but tools were expected.
        # WHY: Agent did nothing when it should have called tools.
        # EXPECTED: raises MltkAssertionError; all expected tools missing.
        trace = AgentTrace(tool_calls=[])
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_tool_chain(trace, expected_tools=["search", "calculator"])
        result = exc_info.value.result
        assert sorted(result.details["missing"]) == ["calculator", "search"]

    def test_empty_trace_empty_expected_pass(self) -> None:
        # SCENARIO: Both trace and expected are empty.
        # WHY: No tools expected, none called — correct behavior.
        # EXPECTED: passes.
        trace = AgentTrace(tool_calls=[])
        result = assert_tool_chain(trace, expected_tools=[])
        assert result.passed is True

    def test_has_timing(self) -> None:
        # SCENARIO: @timed_assertion decorator populates duration_ms.
        # WHY: All assertions must record timing for performance tracking.
        # EXPECTED: duration_ms > 0.
        trace = _make_trace("search")
        result = assert_tool_chain(trace, expected_tools=["search"])
        assert result.duration_ms > 0


class TestNoForbiddenActions:
    """assert_no_forbidden_actions — verify no forbidden tools were called."""

    def test_no_forbidden_pass(self) -> None:
        # SCENARIO: Trace uses only safe tools, none in the forbidden list.
        # WHY: Agent stayed within allowed actions.
        # EXPECTED: passes; forbidden_found is empty.
        trace = AgentTrace(tool_calls=[
            ToolCall(name="search", arguments={"q": "weather"}),
            ToolCall(name="calculator", arguments={"expr": "1+1"}, result="2"),
        ])
        result = assert_no_forbidden_actions(
            trace, forbidden_tools=["delete_database", "send_email"],
        )
        assert result.passed is True
        assert result.details["forbidden_found"] == []
        assert result.details["total_calls"] == 2

    def test_forbidden_tool_found_fail(self) -> None:
        # SCENARIO: Agent called "delete_database" which is forbidden.
        # WHY: Safety guardrail violation.
        # EXPECTED: raises MltkAssertionError; "delete_database" in forbidden_found.
        trace = AgentTrace(tool_calls=[
            ToolCall(name="search", arguments={"q": "users"}),
            ToolCall(name="delete_database", arguments={"target": "users"}),
        ])
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_no_forbidden_actions(
                trace, forbidden_tools=["delete_database", "send_email"],
            )
        result = exc_info.value.result
        assert "delete_database" in result.details["forbidden_found"]
        assert result.details["total_calls"] == 2

    def test_multiple_forbidden_found_fail(self) -> None:
        # SCENARIO: Agent called two different forbidden tools.
        # WHY: Both violations should be reported.
        # EXPECTED: raises MltkAssertionError; both forbidden tools listed.
        trace = _make_trace("search", "delete_database", "send_email", "format")
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_no_forbidden_actions(
                trace, forbidden_tools=["delete_database", "send_email"],
            )
        result = exc_info.value.result
        assert sorted(result.details["forbidden_found"]) == [
            "delete_database", "send_email",
        ]

    def test_empty_trace_pass(self) -> None:
        # SCENARIO: Trace has no tool calls.
        # WHY: No calls means no forbidden actions can have occurred.
        # EXPECTED: passes.
        trace = AgentTrace(tool_calls=[])
        result = assert_no_forbidden_actions(
            trace, forbidden_tools=["delete_database"],
        )
        assert result.passed is True
        assert result.details["forbidden_found"] == []
        assert result.details["total_calls"] == 0

    def test_empty_forbidden_list_pass(self) -> None:
        # SCENARIO: No tools are forbidden.
        # WHY: With no restrictions, any trace is acceptable.
        # EXPECTED: passes.
        trace = _make_trace("search", "delete_database")
        result = assert_no_forbidden_actions(trace, forbidden_tools=[])
        assert result.passed is True

    def test_has_timing(self) -> None:
        # SCENARIO: @timed_assertion decorator populates duration_ms.
        # EXPECTED: duration_ms > 0.
        trace = _make_trace("search")
        result = assert_no_forbidden_actions(trace, forbidden_tools=["delete"])
        assert result.duration_ms > 0


class TestStepEfficiency:
    """assert_step_efficiency — verify agent completed within step budget."""

    def test_under_limit_pass(self) -> None:
        # SCENARIO: Agent used 2 steps with a budget of 5.
        # WHY: Well under budget — efficient agent.
        # EXPECTED: passes with actual_steps=2, max_steps=5.
        trace = AgentTrace(tool_calls=[
            ToolCall(name="search", arguments={"q": "test"}),
            ToolCall(name="calculator", arguments={"expr": "2+2"}, result="4"),
        ])
        result = assert_step_efficiency(trace, max_steps=5)
        assert result.passed is True
        assert result.details["actual_steps"] == 2
        assert result.details["max_steps"] == 5

    def test_exact_limit_pass(self) -> None:
        # SCENARIO: Agent used exactly the maximum allowed steps.
        # WHY: Boundary condition — should still pass (<= comparison).
        # EXPECTED: passes with actual_steps == max_steps.
        trace = _make_trace("a", "b", "c")
        result = assert_step_efficiency(trace, max_steps=3)
        assert result.passed is True
        assert result.details["actual_steps"] == 3

    def test_over_limit_fail(self) -> None:
        # SCENARIO: Agent used 4 steps but budget was 2.
        # WHY: Excessive tool use indicates looping or inefficiency.
        # EXPECTED: raises MltkAssertionError; actual_steps > max_steps.
        trace = _make_trace("a", "b", "c", "d")
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_step_efficiency(trace, max_steps=2)
        result = exc_info.value.result
        assert result.details["actual_steps"] == 4
        assert result.details["max_steps"] == 2

    def test_empty_trace_pass(self) -> None:
        # SCENARIO: Trace is empty (0 steps), max_steps=0.
        # WHY: Zero steps within a budget of zero is valid.
        # EXPECTED: passes with actual_steps=0.
        trace = AgentTrace(tool_calls=[])
        result = assert_step_efficiency(trace, max_steps=0)
        assert result.passed is True
        assert result.details["actual_steps"] == 0

    def test_empty_trace_nonzero_limit_pass(self) -> None:
        # SCENARIO: Trace is empty but budget allows steps.
        # WHY: Agent solved the task with zero tool calls — perfectly efficient.
        # EXPECTED: passes.
        trace = AgentTrace(tool_calls=[])
        result = assert_step_efficiency(trace, max_steps=10)
        assert result.passed is True

    def test_has_timing(self) -> None:
        # SCENARIO: @timed_assertion decorator populates duration_ms.
        # EXPECTED: duration_ms > 0.
        trace = _make_trace("search")
        result = assert_step_efficiency(trace, max_steps=5)
        assert result.duration_ms > 0
