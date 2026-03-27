# Agent Trace Testing

Test AI agent execution traces -- verify tool chains, detect forbidden actions, and enforce step efficiency.

**Module:** `mltk.domains.llm`

---

## AgentTrace Data Model

Agent traces capture the full sequence of tool calls made during an AI agent execution. The data model consists of two dataclasses:

### ToolCall

A single tool/function call within an agent trace.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | *(required)* | Tool name (e.g., `"search"`, `"calculator"`) |
| `arguments` | `dict[str, Any]` | `{}` | Arguments passed to the tool |
| `result` | `str \| None` | `None` | Tool output, if available |
| `error` | `str \| None` | `None` | Error message if the call failed |
| `duration_ms` | `float` | `0.0` | Execution time in milliseconds |

```python
from mltk.domains.llm.trace import ToolCall

# Successful call
call = ToolCall(
    name="search",
    arguments={"query": "weather forecast"},
    result="Sunny, 25C",
    duration_ms=120.5,
)

# Failed call
failed = ToolCall(
    name="database_query",
    arguments={"sql": "SELECT *"},
    error="Connection refused",
    duration_ms=5000.0,
)
```

### AgentTrace

A complete execution trace collecting all tool calls, token usage, wall-clock duration, and metadata.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tool_calls` | `list[ToolCall]` | `[]` | Ordered list of tool calls |
| `total_tokens` | `int` | `0` | Total tokens consumed |
| `total_duration_ms` | `float` | `0.0` | Total wall-clock time in milliseconds |
| `metadata` | `dict[str, Any]` | `{}` | Arbitrary metadata attached to the trace |

#### Properties

| Property | Return Type | Description |
|----------|-------------|-------------|
| `tool_names` | `list[str]` | List of tool names in call order |
| `step_count` | `int` | Number of tool calls |
| `failed_calls` | `list[ToolCall]` | Tool calls that resulted in errors (`error is not None`) |

```python
from mltk.domains.llm.trace import AgentTrace, ToolCall

trace = AgentTrace(
    tool_calls=[
        ToolCall(name="search", arguments={"q": "AI news"}, result="..."),
        ToolCall(name="summarize", arguments={"text": "..."}, result="Summary here"),
        ToolCall(name="database_query", error="timeout"),
    ],
    total_tokens=450,
    total_duration_ms=3200.0,
    metadata={"model": "gpt-4", "session_id": "abc-123"},
)

trace.tool_names   # ["search", "summarize", "database_query"]
trace.step_count   # 3
trace.failed_calls # [ToolCall(name="database_query", error="timeout", ...)]
```

### AgentTrace.from_dict

Construct an `AgentTrace` from a plain dictionary or list. Handles three common serialisation formats automatically.

#### Format 1: Simple dict

The canonical format with a `"tool_calls"` key and optional top-level fields.

```python
trace = AgentTrace.from_dict({
    "tool_calls": [
        {"name": "search", "arguments": {"query": "weather"}},
        {"name": "calculator", "arguments": {"expr": "2+2"}, "result": "4"},
    ],
    "total_tokens": 200,
    "total_duration_ms": 1500.0,
    "metadata": {"model": "gpt-4"},
})
```

#### Format 2: OpenAI function-calling

Nested `"function"` key with JSON-string arguments, as returned by the OpenAI API.

```python
trace = AgentTrace.from_dict({
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
})
# Arguments are automatically decoded from JSON string to dict.
trace.tool_calls[0].arguments  # {"location": "London", "units": "celsius"}
```

#### Format 3: Flat list

A bare list of tool-call dicts, automatically wrapped as `{"tool_calls": [...]}`.

```python
trace = AgentTrace.from_dict([
    {"name": "search", "arguments": {"q": "hello"}},
    {"name": "format", "arguments": {"style": "markdown"}},
])
trace.step_count  # 2
```

---

## Assertions

### assert_tool_chain

Verify the agent called tools in the expected order. Checks that the expected tool sequence appears as a contiguous subsequence within the trace.

```python
from mltk.domains.llm.agentic import assert_tool_chain
from mltk.domains.llm.trace import AgentTrace

trace = AgentTrace.from_dict([
    {"name": "search", "arguments": {"query": "AI regulation"}},
    {"name": "summarize", "arguments": {"text": "..."}},
    {"name": "format", "arguments": {"style": "markdown"}},
])

# Pass: tools appear in expected order
assert_tool_chain(trace, expected_chain=["search", "summarize", "format"])

# Fail: "format" should not come before "summarize"
assert_tool_chain(trace, expected_chain=["search", "format", "summarize"])
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `trace` | `AgentTrace` | *(required)* | The agent execution trace to verify |
| `expected_chain` | `list[str]` | *(required)* | Expected ordered sequence of tool names |

#### Returns

`TestResult` with details:

- `expected_chain` -- the tool chain that was expected
- `actual_chain` -- the actual `tool_names` from the trace
- `matched` -- whether the expected chain was found in order

---

### assert_no_forbidden_actions

Verify the agent did not call any tools from a forbidden list. Useful for enforcing safety boundaries (e.g., no file deletion, no direct database writes).

```python
from mltk.domains.llm.agentic import assert_no_forbidden_actions
from mltk.domains.llm.trace import AgentTrace

trace = AgentTrace.from_dict([
    {"name": "search", "arguments": {"query": "user data"}},
    {"name": "summarize", "arguments": {"text": "..."}},
])

# Pass: no forbidden tools were called
assert_no_forbidden_actions(trace, forbidden=["delete_file", "drop_table", "send_email"])

# Fail: "delete_file" is in the trace
bad_trace = AgentTrace.from_dict([
    {"name": "search", "arguments": {}},
    {"name": "delete_file", "arguments": {"path": "/etc/passwd"}},
])
assert_no_forbidden_actions(bad_trace, forbidden=["delete_file", "drop_table"])
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `trace` | `AgentTrace` | *(required)* | The agent execution trace to verify |
| `forbidden` | `list[str]` | *(required)* | List of tool names that must not appear in the trace |

#### Returns

`TestResult` with details:

- `forbidden_tools` -- the forbidden list that was checked
- `violations` -- list of forbidden tool names that were found in the trace
- `violation_count` -- number of violations detected

---

### assert_step_efficiency

Verify the agent completed the task within a step budget. Catches agents that loop excessively, retry unnecessarily, or take roundabout paths to a solution.

```python
from mltk.domains.llm.agentic import assert_step_efficiency
from mltk.domains.llm.trace import AgentTrace

trace = AgentTrace.from_dict([
    {"name": "search", "arguments": {"query": "capital of France"}},
    {"name": "format", "arguments": {"text": "Paris"}},
])

# Pass: 2 steps is within the budget of 5
assert_step_efficiency(trace, max_steps=5)

# Fail: 2 steps exceeds a budget of 1
assert_step_efficiency(trace, max_steps=1)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `trace` | `AgentTrace` | *(required)* | The agent execution trace to verify |
| `max_steps` | `int` | *(required)* | Maximum allowed number of tool calls |

#### Returns

`TestResult` with details:

- `actual_steps` -- number of tool calls in the trace
- `max_steps` -- the budget that was enforced
- `efficiency_ratio` -- `actual_steps / max_steps` (lower is better)

---

## Integration with pytest

Use `AgentTrace` and the trace assertions inside standard pytest test functions.

```python
import pytest
from mltk.domains.llm.trace import AgentTrace
from mltk.domains.llm.agentic import (
    assert_tool_chain,
    assert_no_forbidden_actions,
    assert_step_efficiency,
)


@pytest.fixture()
def agent_trace() -> AgentTrace:
    """Simulate an agent trace from a recorded session."""
    return AgentTrace.from_dict({
        "tool_calls": [
            {"name": "search", "arguments": {"query": "python best practices"}},
            {"name": "summarize", "arguments": {"text": "...long article..."}},
            {"name": "format", "arguments": {"style": "bullet_points"}},
        ],
        "total_tokens": 520,
        "total_duration_ms": 2400.0,
        "metadata": {"model": "gpt-4"},
    })


def test_tool_chain_correct(agent_trace: AgentTrace) -> None:
    """Agent follows the expected search -> summarize -> format pipeline."""
    result = assert_tool_chain(agent_trace, expected_chain=["search", "summarize", "format"])
    assert result.passed is True


def test_no_dangerous_tools(agent_trace: AgentTrace) -> None:
    """Agent must not call destructive tools."""
    result = assert_no_forbidden_actions(
        agent_trace,
        forbidden=["delete_file", "drop_table", "execute_code", "send_email"],
    )
    assert result.passed is True


def test_step_budget(agent_trace: AgentTrace) -> None:
    """Agent should complete the task in 5 steps or fewer."""
    result = assert_step_efficiency(agent_trace, max_steps=5)
    assert result.passed is True
```

### Testing with OpenAI traces

Parse real OpenAI API responses directly:

```python
def test_openai_trace() -> None:
    """Verify a trace captured from an OpenAI function-calling response."""
    openai_response = {
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "Tokyo"}',
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "format_report",
                    "arguments": '{"data": "Sunny, 28C", "format": "markdown"}',
                },
            },
        ],
        "total_tokens": 180,
    }
    trace = AgentTrace.from_dict(openai_response)

    assert_tool_chain(trace, expected_chain=["get_weather", "format_report"])
    assert_no_forbidden_actions(trace, forbidden=["delete_account"])
    assert_step_efficiency(trace, max_steps=10)
```
