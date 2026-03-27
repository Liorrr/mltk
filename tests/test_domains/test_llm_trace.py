"""Tests for mltk.domains.llm.trace — AgentTrace and ToolCall dataclasses."""

import pytest

from mltk.domains.llm.trace import AgentTrace, ToolCall


class TestToolCallDefaults:
    """ToolCall creation with default values."""

    def test_name_only(self) -> None:
        # SCENARIO: Create a ToolCall providing only the required name field.
        # WHY: All optional fields should fall back to sensible defaults.
        # EXPECTED: arguments={}, result=None, error=None, duration_ms=0.0.
        tc = ToolCall(name="search")
        assert tc.name == "search"
        assert tc.arguments == {}
        assert tc.result is None
        assert tc.error is None
        assert tc.duration_ms == 0.0

    def test_default_arguments_are_independent(self) -> None:
        # SCENARIO: Two ToolCalls created without explicit arguments dicts.
        # WHY: default_factory must return a new dict each time — no shared mutable state.
        # EXPECTED: Mutating one dict does not affect the other.
        tc1 = ToolCall(name="a")
        tc2 = ToolCall(name="b")
        tc1.arguments["key"] = "value"
        assert "key" not in tc2.arguments


class TestToolCallFull:
    """ToolCall creation with all fields populated."""

    def test_all_fields(self) -> None:
        # SCENARIO: Create a ToolCall with every field explicitly set.
        # WHY: Ensures all attributes are stored and accessible.
        # EXPECTED: Each attribute matches the value passed at construction.
        tc = ToolCall(
            name="calculator",
            arguments={"a": 10, "b": 5, "op": "add"},
            result="15",
            error=None,
            duration_ms=42.5,
        )
        assert tc.name == "calculator"
        assert tc.arguments == {"a": 10, "b": 5, "op": "add"}
        assert tc.result == "15"
        assert tc.error is None
        assert tc.duration_ms == 42.5

    def test_with_error(self) -> None:
        # SCENARIO: ToolCall that failed during execution.
        # WHY: error field captures failure info; result may still be None.
        # EXPECTED: error is the string message, result is None.
        tc = ToolCall(
            name="database_query",
            arguments={"sql": "SELECT *"},
            result=None,
            error="Connection refused",
            duration_ms=1200.0,
        )
        assert tc.error == "Connection refused"
        assert tc.result is None
        assert tc.duration_ms == 1200.0


class TestAgentTraceEmpty:
    """AgentTrace creation with empty tool_calls."""

    def test_empty_trace(self) -> None:
        # SCENARIO: Agent trace with no tool calls (pure reasoning, no actions).
        # WHY: An empty trace is valid — the agent may have answered without tools.
        # EXPECTED: tool_calls=[], total_tokens=0, total_duration_ms=0.0, metadata={}.
        trace = AgentTrace()
        assert trace.tool_calls == []
        assert trace.total_tokens == 0
        assert trace.total_duration_ms == 0.0
        assert trace.metadata == {}

    def test_empty_trace_properties(self) -> None:
        # SCENARIO: Properties on an empty trace.
        # WHY: Properties must handle the empty case without errors.
        # EXPECTED: tool_names=[], step_count=0, failed_calls=[].
        trace = AgentTrace()
        assert trace.tool_names == []
        assert trace.step_count == 0
        assert trace.failed_calls == []


class TestAgentTraceWithCalls:
    """AgentTrace with multiple tool_calls."""

    @pytest.fixture
    def multi_call_trace(self) -> AgentTrace:
        """Trace with three tool calls: two successful, one failed."""
        return AgentTrace(
            tool_calls=[
                ToolCall(name="search", arguments={"query": "weather"}, result="Sunny, 25C"),
                ToolCall(name="calculator", arguments={"expr": "2+2"}, result="4"),
                ToolCall(
                    name="database_query",
                    arguments={"sql": "SELECT 1"},
                    error="timeout",
                    duration_ms=5000.0,
                ),
            ],
            total_tokens=350,
            total_duration_ms=6200.0,
            metadata={"model": "gpt-4", "session_id": "abc-123"},
        )

    def test_tool_calls_stored(self, multi_call_trace: AgentTrace) -> None:
        # SCENARIO: Verify tool_calls list is stored and has expected length.
        # WHY: The core data structure must faithfully record all calls.
        # EXPECTED: 3 tool calls in order.
        assert len(multi_call_trace.tool_calls) == 3
        assert multi_call_trace.tool_calls[0].name == "search"
        assert multi_call_trace.tool_calls[1].name == "calculator"
        assert multi_call_trace.tool_calls[2].name == "database_query"

    def test_aggregate_fields(self, multi_call_trace: AgentTrace) -> None:
        # SCENARIO: Verify aggregate fields (tokens, duration, metadata).
        # WHY: These top-level fields are independent of individual tool calls.
        # EXPECTED: Values match what was passed at construction.
        assert multi_call_trace.total_tokens == 350
        assert multi_call_trace.total_duration_ms == 6200.0
        assert multi_call_trace.metadata == {"model": "gpt-4", "session_id": "abc-123"}


class TestToolNamesProperty:
    """AgentTrace.tool_names property."""

    def test_returns_names_in_order(self) -> None:
        # SCENARIO: Three tool calls made in a specific order.
        # WHY: tool_names must preserve call order for chain verification.
        # EXPECTED: ["search", "calculator", "formatter"] in that exact order.
        trace = AgentTrace(
            tool_calls=[
                ToolCall(name="search"),
                ToolCall(name="calculator"),
                ToolCall(name="formatter"),
            ]
        )
        assert trace.tool_names == ["search", "calculator", "formatter"]

    def test_duplicates_preserved(self) -> None:
        # SCENARIO: Agent called the same tool twice.
        # WHY: tool_names should reflect actual call sequence, including repeats.
        # EXPECTED: ["search", "search"] — both occurrences present.
        trace = AgentTrace(
            tool_calls=[ToolCall(name="search"), ToolCall(name="search")]
        )
        assert trace.tool_names == ["search", "search"]

    def test_empty(self) -> None:
        # SCENARIO: No tool calls at all.
        # WHY: Empty list, not an error.
        # EXPECTED: [].
        assert AgentTrace().tool_names == []


class TestStepCountProperty:
    """AgentTrace.step_count property."""

    def test_zero_steps(self) -> None:
        # SCENARIO: Empty trace.
        # EXPECTED: step_count == 0.
        assert AgentTrace().step_count == 0

    def test_three_steps(self) -> None:
        # SCENARIO: Trace with three tool calls.
        # EXPECTED: step_count == 3.
        trace = AgentTrace(
            tool_calls=[ToolCall(name="a"), ToolCall(name="b"), ToolCall(name="c")]
        )
        assert trace.step_count == 3

    def test_single_step(self) -> None:
        # SCENARIO: Trace with one tool call.
        # EXPECTED: step_count == 1.
        trace = AgentTrace(tool_calls=[ToolCall(name="only")])
        assert trace.step_count == 1


class TestFailedCallsProperty:
    """AgentTrace.failed_calls property."""

    def test_no_failures(self) -> None:
        # SCENARIO: All tool calls succeeded (error=None).
        # WHY: failed_calls should return empty list when nothing failed.
        # EXPECTED: [].
        trace = AgentTrace(
            tool_calls=[
                ToolCall(name="search", result="ok"),
                ToolCall(name="calculator", result="42"),
            ]
        )
        assert trace.failed_calls == []

    def test_one_failure(self) -> None:
        # SCENARIO: One of three calls has an error.
        # WHY: Only calls with error set (not None) should be returned.
        # EXPECTED: Single failed ToolCall with name="db".
        trace = AgentTrace(
            tool_calls=[
                ToolCall(name="search", result="ok"),
                ToolCall(name="db", error="connection refused"),
                ToolCall(name="calculator", result="7"),
            ]
        )
        failed = trace.failed_calls
        assert len(failed) == 1
        assert failed[0].name == "db"
        assert failed[0].error == "connection refused"

    def test_all_failures(self) -> None:
        # SCENARIO: Every tool call failed.
        # WHY: All calls should appear in failed_calls.
        # EXPECTED: 2 failed calls.
        trace = AgentTrace(
            tool_calls=[
                ToolCall(name="a", error="err1"),
                ToolCall(name="b", error="err2"),
            ]
        )
        assert len(trace.failed_calls) == 2

    def test_empty_string_error_is_failure(self) -> None:
        # SCENARIO: error is an empty string (not None).
        # WHY: The filter is `error is not None`; empty string is truthy for "is not None".
        # EXPECTED: Counts as a failure.
        trace = AgentTrace(tool_calls=[ToolCall(name="x", error="")])
        assert len(trace.failed_calls) == 1


class TestFromDictSimple:
    """AgentTrace.from_dict with simple/direct format."""

    def test_simple_format(self) -> None:
        # SCENARIO: Standard dict with "tool_calls" key and top-level metadata.
        # WHY: This is the canonical serialisation format.
        # EXPECTED: All fields parsed correctly.
        data = {
            "tool_calls": [
                {"name": "search", "arguments": {"query": "weather"}, "result": "Sunny"},
                {"name": "calculator", "arguments": {"expr": "1+1"}, "result": "2"},
            ],
            "total_tokens": 200,
            "total_duration_ms": 1500.0,
            "metadata": {"model": "gpt-4"},
        }
        trace = AgentTrace.from_dict(data)
        assert trace.step_count == 2
        assert trace.tool_names == ["search", "calculator"]
        assert trace.total_tokens == 200
        assert trace.total_duration_ms == 1500.0
        assert trace.metadata == {"model": "gpt-4"}
        assert trace.tool_calls[0].result == "Sunny"
        assert trace.tool_calls[0].arguments == {"query": "weather"}

    def test_simple_with_error(self) -> None:
        # SCENARIO: One tool call has an error field in the dict.
        # WHY: Errors must survive the from_dict round-trip.
        # EXPECTED: failed_calls includes the errored call.
        data = {
            "tool_calls": [
                {"name": "search", "arguments": {}, "error": "rate limited"},
            ],
        }
        trace = AgentTrace.from_dict(data)
        assert len(trace.failed_calls) == 1
        assert trace.failed_calls[0].error == "rate limited"

    def test_simple_with_duration_ms(self) -> None:
        # SCENARIO: Tool call dict includes duration_ms.
        # WHY: Per-call timing must be parsed.
        # EXPECTED: duration_ms is a float on the ToolCall.
        data = {
            "tool_calls": [
                {"name": "search", "arguments": {}, "duration_ms": 123.4},
            ],
        }
        trace = AgentTrace.from_dict(data)
        assert abs(trace.tool_calls[0].duration_ms - 123.4) < 1e-9


class TestFromDictOpenAI:
    """AgentTrace.from_dict with OpenAI function-calling format."""

    def test_openai_function_calling(self) -> None:
        # SCENARIO: OpenAI-style nested format with "function" key and JSON-string arguments.
        # WHY: OpenAI API returns tool calls in this nested format; from_dict must handle it.
        # EXPECTED: name and arguments extracted from the nested structure, JSON string decoded.
        data = {
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "London", "units": "celsius"}',
                    },
                },
            ],
            "total_tokens": 100,
        }
        trace = AgentTrace.from_dict(data)
        assert trace.step_count == 1
        assert trace.tool_names == ["get_weather"]
        assert trace.tool_calls[0].arguments == {"location": "London", "units": "celsius"}
        assert trace.total_tokens == 100

    def test_openai_multiple_calls(self) -> None:
        # SCENARIO: Multiple OpenAI-format tool calls in one trace.
        # WHY: Agents often issue parallel function calls.
        # EXPECTED: Both calls parsed, arguments decoded.
        data = {
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": '{"query": "AI news"}',
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": '{"expression": "2*3"}',
                    },
                },
            ],
        }
        trace = AgentTrace.from_dict(data)
        assert trace.tool_names == ["search", "calculator"]
        assert trace.tool_calls[0].arguments == {"query": "AI news"}
        assert trace.tool_calls[1].arguments == {"expression": "2*3"}

    def test_openai_invalid_json_arguments(self) -> None:
        # SCENARIO: OpenAI arguments is a malformed JSON string.
        # WHY: Graceful degradation — should not crash, should default to empty dict.
        # EXPECTED: arguments == {} after failed JSON parse.
        data = {
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "broken",
                        "arguments": "{not valid json",
                    },
                },
            ],
        }
        trace = AgentTrace.from_dict(data)
        assert trace.tool_calls[0].name == "broken"
        assert trace.tool_calls[0].arguments == {}


class TestFromDictFlatList:
    """AgentTrace.from_dict with a flat list (no wrapper dict)."""

    def test_flat_list(self) -> None:
        # SCENARIO: Caller passes a bare list of tool-call dicts instead of a wrapping dict.
        # WHY: Convenience format — from_dict wraps it as {"tool_calls": [...]}.
        # EXPECTED: Parsed identically to the wrapped format.
        data = [
            {"name": "search", "arguments": {"q": "hello"}},
            {"name": "format", "arguments": {"style": "markdown"}},
        ]
        trace = AgentTrace.from_dict(data)
        assert trace.step_count == 2
        assert trace.tool_names == ["search", "format"]
        assert trace.tool_calls[0].arguments == {"q": "hello"}

    def test_flat_list_empty(self) -> None:
        # SCENARIO: Empty list passed to from_dict.
        # WHY: Should produce an empty trace, not an error.
        # EXPECTED: step_count == 0, no tool calls.
        trace = AgentTrace.from_dict([])
        assert trace.step_count == 0
        assert trace.tool_calls == []

    def test_flat_list_defaults(self) -> None:
        # SCENARIO: Flat list — total_tokens and total_duration_ms are not available.
        # WHY: A bare list has no place for aggregate fields; they should default.
        # EXPECTED: total_tokens=0, total_duration_ms=0.0, metadata={}.
        trace = AgentTrace.from_dict([{"name": "x"}])
        assert trace.total_tokens == 0
        assert trace.total_duration_ms == 0.0
        assert trace.metadata == {}


class TestFromDictMissingFields:
    """AgentTrace.from_dict with missing or optional fields."""

    def test_empty_dict(self) -> None:
        # SCENARIO: Completely empty dict — no tool_calls key at all.
        # WHY: Should gracefully produce an empty trace.
        # EXPECTED: step_count=0, defaults for all aggregate fields.
        trace = AgentTrace.from_dict({})
        assert trace.step_count == 0
        assert trace.total_tokens == 0
        assert trace.total_duration_ms == 0.0
        assert trace.metadata == {}

    def test_missing_tool_call_name(self) -> None:
        # SCENARIO: A tool call dict without a "name" key.
        # WHY: Name defaults to empty string — the call is still recorded.
        # EXPECTED: ToolCall with name="" is created.
        data = {"tool_calls": [{"arguments": {"x": 1}}]}
        trace = AgentTrace.from_dict(data)
        assert trace.step_count == 1
        assert trace.tool_calls[0].name == ""
        assert trace.tool_calls[0].arguments == {"x": 1}

    def test_missing_arguments(self) -> None:
        # SCENARIO: Tool call dict has name but no arguments key.
        # WHY: arguments defaults to empty dict.
        # EXPECTED: arguments == {}.
        data = {"tool_calls": [{"name": "search"}]}
        trace = AgentTrace.from_dict(data)
        assert trace.tool_calls[0].arguments == {}

    def test_missing_optional_top_level_fields(self) -> None:
        # SCENARIO: Dict has tool_calls but no total_tokens, total_duration_ms, or metadata.
        # WHY: All top-level fields are optional and should default.
        # EXPECTED: total_tokens=0, total_duration_ms=0.0, metadata={}.
        data = {
            "tool_calls": [{"name": "ping", "arguments": {}}],
        }
        trace = AgentTrace.from_dict(data)
        assert trace.total_tokens == 0
        assert trace.total_duration_ms == 0.0
        assert trace.metadata == {}

    def test_missing_result_and_error(self) -> None:
        # SCENARIO: Tool call dict has no result and no error keys.
        # WHY: Both are optional — result=None, error=None.
        # EXPECTED: No crash, failed_calls is empty.
        data = {"tool_calls": [{"name": "search", "arguments": {"q": "test"}}]}
        trace = AgentTrace.from_dict(data)
        assert trace.tool_calls[0].result is None
        assert trace.tool_calls[0].error is None
        assert trace.failed_calls == []

    def test_missing_duration_ms_defaults_to_zero(self) -> None:
        # SCENARIO: Tool call dict without duration_ms.
        # WHY: Should default to 0.0, not raise.
        # EXPECTED: duration_ms == 0.0.
        data = {"tool_calls": [{"name": "search"}]}
        trace = AgentTrace.from_dict(data)
        assert trace.tool_calls[0].duration_ms == 0.0
