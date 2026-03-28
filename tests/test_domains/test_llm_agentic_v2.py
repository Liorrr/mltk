"""Tests for extended agentic assertions (redundant, hallucinated, cost, recovery).

These assertions target failure modes that emerge when AI agents run
autonomously — degenerate loops, invented tool names, runaway costs, and
missing error-recovery strategies.
"""

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.domains.llm.agentic import (
    assert_cost_budget,
    assert_error_recovery,
    assert_no_hallucinated_tools,
    assert_no_redundant_calls,
)
from mltk.domains.llm.trace import AgentTrace, ToolCall


def _make_trace(*tool_names: str) -> AgentTrace:
    """Helper -- build an AgentTrace from a list of tool name strings."""
    return AgentTrace(
        tool_calls=[ToolCall(name=n, arguments={}) for n in tool_names],
    )


# -----------------------------------------------------------------------
# assert_no_redundant_calls
# -----------------------------------------------------------------------


class TestNoRedundantCalls:
    """assert_no_redundant_calls -- detect stuck agent loops.

    Agents sometimes enter degenerate loops where they call the same tool
    repeatedly because they cannot parse the result or their planner keeps
    re-triggering the identical action.  This assertion catches those loops
    by checking that no single tool appears more than *max_repeat* times
    in a row within the trace.
    """

    def test_no_repeats_pass(self) -> None:
        # SCENARIO: Every tool call is different -- no consecutive repeats.
        # WHY: A varied tool sequence indicates healthy agent reasoning.
        # EXPECTED: passes; redundant_tools is empty, max_consecutive is 1.
        trace = _make_trace("search", "calculator", "format")
        result = assert_no_redundant_calls(trace, max_repeat=2)
        assert result.passed is True
        assert result.details["redundant_tools"] == []
        assert result.details["max_consecutive"] == 1
        assert result.details["max_repeat"] == 2

    def test_consecutive_repeats_fail(self) -> None:
        # SCENARIO: "search" is called 3 times in a row with max_repeat=2.
        # WHY: Three consecutive identical calls strongly suggests a stuck loop.
        # EXPECTED: raises MltkAssertionError; redundant_tools lists "search"
        #   with count=3 and the starting index.
        trace = _make_trace("search", "search", "search", "calculator")
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_no_redundant_calls(trace, max_repeat=2)
        result = exc_info.value.result
        assert result.passed is False
        assert len(result.details["redundant_tools"]) == 1
        entry = result.details["redundant_tools"][0]
        assert entry["tool"] == "search"
        assert entry["count"] == 3
        assert entry["start_index"] == 0
        assert result.details["max_consecutive"] == 3

    def test_ignore_tools_works(self) -> None:
        # SCENARIO: "think" is called 5 times in a row, but it is in ignore_tools.
        # WHY: Some tools (internal reasoning steps, loggers) repeat by design
        #   and should not trigger a redundancy failure.
        # EXPECTED: passes; the "think" run is excluded from the scan.
        trace = _make_trace("think", "think", "think", "think", "think", "search")
        result = assert_no_redundant_calls(
            trace, max_repeat=2, ignore_tools=["think"],
        )
        assert result.passed is True
        assert result.details["redundant_tools"] == []

    def test_non_consecutive_repeats_ok(self) -> None:
        # SCENARIO: "search" appears 4 times total but never more than once in a row.
        # WHY: Non-consecutive uses of the same tool are perfectly normal --
        #   different stages of reasoning may call the same tool with different inputs.
        # EXPECTED: passes; max_consecutive is 1.
        trace = _make_trace("search", "calc", "search", "format", "search", "calc", "search")
        result = assert_no_redundant_calls(trace, max_repeat=2)
        assert result.passed is True
        assert result.details["max_consecutive"] == 1

    def test_empty_trace_pass(self) -> None:
        # SCENARIO: Trace has no tool calls at all.
        # WHY: No calls means no redundancy.
        # EXPECTED: passes; max_consecutive is 0.
        trace = AgentTrace(tool_calls=[])
        result = assert_no_redundant_calls(trace, max_repeat=2)
        assert result.passed is True
        assert result.details["max_consecutive"] == 0

    def test_has_timing(self) -> None:
        # SCENARIO: @timed_assertion decorator populates duration_ms.
        # EXPECTED: duration_ms > 0.
        trace = _make_trace("search")
        result = assert_no_redundant_calls(trace, max_repeat=2)
        assert result.duration_ms > 0


# -----------------------------------------------------------------------
# assert_no_hallucinated_tools
# -----------------------------------------------------------------------


class TestNoHallucinatedTools:
    """assert_no_hallucinated_tools -- catch invented tool names.

    LLMs sometimes generate calls to tool names that do not exist -- calling
    "google_search" when the actual tool is "search", or inventing "send_sms"
    when no messaging capability was provided.  This assertion ensures every
    tool name in the trace belongs to the set of tools the agent was actually
    given access to.
    """

    def test_all_known_pass(self) -> None:
        # SCENARIO: Every tool call uses a name from the known_tools list.
        # WHY: Agent correctly used only its available tools.
        # EXPECTED: passes; hallucinated list is empty.
        trace = _make_trace("search", "calculator")
        result = assert_no_hallucinated_tools(
            trace, known_tools=["search", "calculator", "format"],
        )
        assert result.passed is True
        assert result.details["hallucinated"] == []
        assert result.details["total_calls"] == 2

    def test_hallucinated_tool_fail(self) -> None:
        # SCENARIO: Agent called "google_search" which is not in known_tools.
        # WHY: The agent invented a tool name -- this call did nothing, but
        #   downstream logic may assume the search succeeded.
        # EXPECTED: raises MltkAssertionError; "google_search" in hallucinated.
        trace = _make_trace("search", "google_search")
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_no_hallucinated_tools(
                trace, known_tools=["search", "calculator"],
            )
        result = exc_info.value.result
        assert result.passed is False
        assert "google_search" in result.details["hallucinated"]
        assert result.details["total_calls"] == 2

    def test_empty_trace_pass(self) -> None:
        # SCENARIO: Trace has no tool calls.
        # WHY: No calls means no opportunity to hallucinate.
        # EXPECTED: passes; hallucinated list is empty, total_calls is 0.
        trace = AgentTrace(tool_calls=[])
        result = assert_no_hallucinated_tools(
            trace, known_tools=["search"],
        )
        assert result.passed is True
        assert result.details["hallucinated"] == []
        assert result.details["total_calls"] == 0

    def test_details_list_correct(self) -> None:
        # SCENARIO: Multiple hallucinated tool names, some repeated.
        # WHY: The details list should contain each unique hallucinated name
        #   exactly once, sorted alphabetically.
        # EXPECTED: hallucinated == ["fake_api", "imaginary_db"] (sorted, deduplicated).
        trace = _make_trace("search", "fake_api", "fake_api", "imaginary_db")
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_no_hallucinated_tools(
                trace, known_tools=["search", "calculator"],
            )
        result = exc_info.value.result
        assert result.details["hallucinated"] == ["fake_api", "imaginary_db"]
        assert result.details["total_calls"] == 4

    def test_has_timing(self) -> None:
        # SCENARIO: @timed_assertion decorator populates duration_ms.
        # EXPECTED: duration_ms > 0.
        trace = _make_trace("search")
        result = assert_no_hallucinated_tools(trace, known_tools=["search"])
        assert result.duration_ms > 0


# -----------------------------------------------------------------------
# assert_cost_budget
# -----------------------------------------------------------------------


class TestCostBudget:
    """assert_cost_budget -- enforce resource consumption limits.

    Production agents can burn through API credits at alarming speed.
    This assertion validates that the agent's recorded token count and
    execution duration stay within the specified budget.  Designed for
    CI/CD pipelines: replay traces from staging, verify cost bounds,
    then promote to production.
    """

    def test_under_budget_pass(self) -> None:
        # SCENARIO: Both tokens and duration are under their respective limits.
        # WHY: Agent operated efficiently within all budget constraints.
        # EXPECTED: passes; both exceeded flags are False.
        trace = AgentTrace(
            tool_calls=[ToolCall(name="search", arguments={"q": "test"})],
            total_tokens=500,
            total_duration_ms=1000.0,
        )
        result = assert_cost_budget(
            trace, max_total_tokens=1000, max_duration_ms=2000.0,
        )
        assert result.passed is True
        assert result.details["token_budget_exceeded"] is False
        assert result.details["duration_budget_exceeded"] is False
        assert result.details["total_tokens"] == 500
        assert result.details["total_duration_ms"] == 1000.0

    def test_tokens_over_fail(self) -> None:
        # SCENARIO: Token consumption exceeds the budget.
        # WHY: Agent used too many tokens -- likely looping or generating
        #   excessively long outputs.
        # EXPECTED: raises MltkAssertionError; token_budget_exceeded is True.
        trace = AgentTrace(
            tool_calls=[ToolCall(name="search", arguments={})],
            total_tokens=1500,
            total_duration_ms=500.0,
        )
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_cost_budget(trace, max_total_tokens=1000)
        result = exc_info.value.result
        assert result.passed is False
        assert result.details["token_budget_exceeded"] is True
        assert result.details["total_tokens"] == 1500
        assert result.details["max_total_tokens"] == 1000

    def test_duration_over_fail(self) -> None:
        # SCENARIO: Execution duration exceeds the budget.
        # WHY: Agent ran too long -- could indicate a stuck loop or excessively
        #   slow tool calls.
        # EXPECTED: raises MltkAssertionError; duration_budget_exceeded is True.
        trace = AgentTrace(
            tool_calls=[ToolCall(name="search", arguments={})],
            total_tokens=100,
            total_duration_ms=5000.0,
        )
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_cost_budget(trace, max_duration_ms=2000.0)
        result = exc_info.value.result
        assert result.passed is False
        assert result.details["duration_budget_exceeded"] is True
        assert result.details["total_duration_ms"] == 5000.0
        assert result.details["max_duration_ms"] == 2000.0

    def test_both_checked(self) -> None:
        # SCENARIO: Both budgets provided and both within limits.
        # WHY: Verifies that the assertion checks BOTH constraints, not just one.
        # EXPECTED: passes; details contain all four budget fields.
        trace = AgentTrace(
            tool_calls=[ToolCall(name="calc", arguments={"expr": "1+1"})],
            total_tokens=200,
            total_duration_ms=800.0,
        )
        result = assert_cost_budget(
            trace, max_total_tokens=500, max_duration_ms=1000.0,
        )
        assert result.passed is True
        assert result.details["max_total_tokens"] == 500
        assert result.details["max_duration_ms"] == 1000.0
        assert result.details["token_budget_exceeded"] is False
        assert result.details["duration_budget_exceeded"] is False

    def test_no_budget_raises_value_error(self) -> None:
        # SCENARIO: Neither max_total_tokens nor max_duration_ms is provided.
        # WHY: An assertion with no constraints checks nothing -- caller error.
        # EXPECTED: raises ValueError.
        trace = AgentTrace(tool_calls=[], total_tokens=100)
        with pytest.raises(ValueError, match="At least one budget"):
            assert_cost_budget(trace)

    def test_has_timing(self) -> None:
        # SCENARIO: @timed_assertion decorator populates duration_ms.
        # EXPECTED: duration_ms > 0.
        trace = AgentTrace(tool_calls=[], total_tokens=10)
        result = assert_cost_budget(trace, max_total_tokens=100)
        assert result.duration_ms > 0


# -----------------------------------------------------------------------
# assert_error_recovery
# -----------------------------------------------------------------------


class TestErrorRecovery:
    """assert_error_recovery -- verify agents recover from tool failures.

    Good agents adapt when a tool call fails: they try a different query,
    fall back to an alternative tool, or gracefully inform the user.  Bad
    agents hammer the same failing tool in a loop, burning budget and
    producing no useful output.  This assertion checks that the longest
    streak of consecutive errors stays within bounds.
    """

    def test_no_errors_pass(self) -> None:
        # SCENARIO: All tool calls succeeded -- no errors at all.
        # WHY: A clean trace trivially satisfies error recovery.
        # EXPECTED: passes; max_error_streak is 0, total_errors is 0.
        trace = AgentTrace(tool_calls=[
            ToolCall(name="search", arguments={"q": "test"}, result="ok"),
            ToolCall(name="calculator", arguments={"expr": "2+2"}, result="4"),
        ])
        result = assert_error_recovery(trace, max_consecutive_errors=3)
        assert result.passed is True
        assert result.details["max_error_streak"] == 0
        assert result.details["total_errors"] == 0
        assert result.details["total_calls"] == 2

    def test_short_streak_ok(self) -> None:
        # SCENARIO: Two consecutive errors followed by a success.
        # WHY: Short error streaks (1-2) are normal -- networks time out, APIs
        #   return transient errors.  The agent recovered on the third attempt.
        # EXPECTED: passes with max_error_streak=2 (within default limit of 3).
        trace = AgentTrace(tool_calls=[
            ToolCall(name="search", arguments={"q": "test"}, error="timeout"),
            ToolCall(name="search", arguments={"q": "test"}, error="timeout"),
            ToolCall(name="search", arguments={"q": "test v2"}, result="found"),
        ])
        result = assert_error_recovery(trace, max_consecutive_errors=3)
        assert result.passed is True
        assert result.details["max_error_streak"] == 2
        assert result.details["total_errors"] == 2

    def test_long_streak_fail(self) -> None:
        # SCENARIO: Four consecutive errors with max_consecutive_errors=3.
        # WHY: The agent failed to recover -- it kept retrying the same failing
        #   tool without changing strategy.
        # EXPECTED: raises MltkAssertionError; max_error_streak is 4.
        trace = AgentTrace(tool_calls=[
            ToolCall(name="api", arguments={}, error="500"),
            ToolCall(name="api", arguments={}, error="500"),
            ToolCall(name="api", arguments={}, error="500"),
            ToolCall(name="api", arguments={}, error="500"),
        ])
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_error_recovery(trace, max_consecutive_errors=3)
        result = exc_info.value.result
        assert result.passed is False
        assert result.details["max_error_streak"] == 4
        assert result.details["max_consecutive_errors"] == 3
        assert result.details["total_errors"] == 4
        assert result.details["total_calls"] == 4

    def test_errors_interspersed_with_success_ok(self) -> None:
        # SCENARIO: Multiple errors scattered through the trace, but never more
        #   than one in a row (each error is followed by a success).
        # WHY: The agent hits individual failures but always recovers before
        #   the next call.  This is healthy behavior.
        # EXPECTED: passes; max_error_streak is 1, total_errors is 3.
        trace = AgentTrace(tool_calls=[
            ToolCall(name="search", arguments={}, error="not found"),
            ToolCall(name="search", arguments={}, result="ok"),
            ToolCall(name="calc", arguments={}, error="div by zero"),
            ToolCall(name="calc", arguments={}, result="42"),
            ToolCall(name="format", arguments={}, error="bad template"),
            ToolCall(name="format", arguments={}, result="done"),
        ])
        result = assert_error_recovery(trace, max_consecutive_errors=3)
        assert result.passed is True
        assert result.details["max_error_streak"] == 1
        assert result.details["total_errors"] == 3
        assert result.details["total_calls"] == 6

    def test_empty_trace_pass(self) -> None:
        # SCENARIO: Trace has no tool calls at all.
        # WHY: No calls means no errors.
        # EXPECTED: passes; all counters are 0.
        trace = AgentTrace(tool_calls=[])
        result = assert_error_recovery(trace, max_consecutive_errors=3)
        assert result.passed is True
        assert result.details["max_error_streak"] == 0
        assert result.details["total_errors"] == 0
        assert result.details["total_calls"] == 0

    def test_has_timing(self) -> None:
        # SCENARIO: @timed_assertion decorator populates duration_ms.
        # EXPECTED: duration_ms > 0.
        trace = _make_trace("search")
        result = assert_error_recovery(trace, max_consecutive_errors=3)
        assert result.duration_ms > 0


# -----------------------------------------------------------------------
# Hardening: parametrized, edge-case tests (appended)
# -----------------------------------------------------------------------


class TestRedundantCallsParametrized:
    """Parametrize max_repeat threshold for redundant call detection."""

    @pytest.mark.parametrize("max_repeat", [1, 2, 5])
    def test_exact_threshold_passes(
        self, max_repeat: int,
    ) -> None:
        """PASS: Consecutive run exactly at max_repeat is allowed.

        A run of length == max_repeat should pass because the
        assertion triggers on strictly-greater-than.
        """
        names = ["search"] * max_repeat + ["calc"]
        trace = _make_trace(*names)
        result = assert_no_redundant_calls(
            trace, max_repeat=max_repeat,
        )
        assert result.passed is True
        assert result.details["max_consecutive"] == max_repeat

    @pytest.mark.parametrize("max_repeat", [1, 2, 5])
    def test_one_over_threshold_fails(
        self, max_repeat: int,
    ) -> None:
        """FAIL: Consecutive run one above max_repeat triggers.

        max_repeat + 1 consecutive identical calls must be
        flagged as redundant.
        """
        names = ["search"] * (max_repeat + 1) + ["calc"]
        trace = _make_trace(*names)
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_no_redundant_calls(
                trace, max_repeat=max_repeat,
            )
        result = exc_info.value.result
        assert result.passed is False
        entry = result.details["redundant_tools"][0]
        assert entry["tool"] == "search"
        assert entry["count"] == max_repeat + 1


class TestHallucinatedToolsEmptyKnown:
    """Hallucinated tools with an empty known_tools list."""

    def test_all_tools_hallucinated(self) -> None:
        """FAIL: Every tool is hallucinated when known_tools is
        empty.

        An agent given zero tools should not call anything.
        Any call is a hallucination by definition.
        """
        trace = _make_trace("search", "calc", "format")
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_no_hallucinated_tools(
                trace, known_tools=[],
            )
        result = exc_info.value.result
        assert result.passed is False
        assert sorted(result.details["hallucinated"]) == [
            "calc", "format", "search",
        ]
        assert result.details["total_calls"] == 3

    def test_empty_trace_empty_known_passes(self) -> None:
        """PASS: No calls + no known tools = trivially correct."""
        trace = AgentTrace(tool_calls=[])
        result = assert_no_hallucinated_tools(
            trace, known_tools=[],
        )
        assert result.passed is True


class TestCostBudgetDurationOnly:
    """Cost budget with only duration (no token budget)."""

    def test_duration_only_under_budget(self) -> None:
        """PASS: Duration within budget, tokens unchecked.

        When only max_duration_ms is provided, token usage
        should be ignored entirely.
        """
        trace = AgentTrace(
            tool_calls=[
                ToolCall(name="search", arguments={}),
            ],
            total_tokens=999999,
            total_duration_ms=500.0,
        )
        result = assert_cost_budget(
            trace, max_duration_ms=1000.0,
        )
        assert result.passed is True
        assert result.details["duration_budget_exceeded"] is False
        assert result.details["max_total_tokens"] is None

    def test_duration_only_over_budget(self) -> None:
        """FAIL: Duration exceeds budget, tokens irrelevant."""
        trace = AgentTrace(
            tool_calls=[
                ToolCall(name="search", arguments={}),
            ],
            total_tokens=1,
            total_duration_ms=3000.0,
        )
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_cost_budget(
                trace, max_duration_ms=2000.0,
            )
        result = exc_info.value.result
        assert result.details["duration_budget_exceeded"] is True


class TestErrorRecoveryAlternating:
    """Error recovery with alternating success/error pattern."""

    def test_alternating_success_error(self) -> None:
        """PASS: Alternating error-success pattern, streak = 1.

        E-S-E-S-E-S has many errors but the agent recovers
        after each one. Max streak is 1.
        """
        trace = AgentTrace(tool_calls=[
            ToolCall(
                name="api", arguments={}, error="fail",
            ),
            ToolCall(
                name="api", arguments={}, result="ok",
            ),
            ToolCall(
                name="api", arguments={}, error="fail",
            ),
            ToolCall(
                name="api", arguments={}, result="ok",
            ),
            ToolCall(
                name="api", arguments={}, error="fail",
            ),
            ToolCall(
                name="api", arguments={}, result="ok",
            ),
        ])
        result = assert_error_recovery(
            trace, max_consecutive_errors=1,
        )
        assert result.passed is True
        assert result.details["max_error_streak"] == 1
        assert result.details["total_errors"] == 3
        assert result.details["total_calls"] == 6

    def test_double_error_then_success_pattern(self) -> None:
        """PASS: Two errors then success, repeated. Streak = 2.

        E-E-S-E-E-S pattern has max streak of 2 which passes
        when max_consecutive_errors=2.
        """
        trace = AgentTrace(tool_calls=[
            ToolCall(
                name="a", arguments={}, error="err",
            ),
            ToolCall(
                name="a", arguments={}, error="err",
            ),
            ToolCall(
                name="a", arguments={}, result="ok",
            ),
            ToolCall(
                name="b", arguments={}, error="err",
            ),
            ToolCall(
                name="b", arguments={}, error="err",
            ),
            ToolCall(
                name="b", arguments={}, result="ok",
            ),
        ])
        result = assert_error_recovery(
            trace, max_consecutive_errors=2,
        )
        assert result.passed is True
        assert result.details["max_error_streak"] == 2
        assert result.details["total_errors"] == 4
