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

---

## Extended Trace Assertions

The following assertions go beyond basic tool-chain validation and target the failure modes that emerge in real-world agentic deployments: infinite retry loops, hallucinated tool names, runaway costs, and error cascades. These are the bugs that do not raise exceptions -- they silently burn through token budgets and wall-clock time while producing no useful work.

### assert_no_redundant_calls

Agents stuck in retry loops are one of the most expensive failure modes in production. When an LLM receives an error from a tool, a well-designed agent retries once or twice and then pivots to an alternative strategy. A broken agent, however, repeats the same failing call indefinitely: `search` -> `search` -> `search` -> `search` -> ... until the token budget is exhausted or the orchestrator times out.

This is not hypothetical. Retry storms are the leading cause of unexpected spend in agentic pipelines. The agent's reasoning looks plausible on each individual step ("let me try again"), but the overall trace reveals a loop that produces no forward progress.

`assert_no_redundant_calls` scans the trace for consecutive runs of the same tool and fails if any tool appears more than `max_repeat` times in a row. You can optionally exclude tools that are expected to repeat (e.g., a `think` or `log` tool that legitimately fires on every step).

```python
from mltk.domains.llm.agentic import assert_no_redundant_calls
from mltk.domains.llm.trace import AgentTrace

# An agent that got stuck retrying a failed search
trace = AgentTrace.from_dict([
    {"name": "search", "arguments": {"q": "annual revenue"}, "error": "timeout"},
    {"name": "search", "arguments": {"q": "annual revenue"}, "error": "timeout"},
    {"name": "search", "arguments": {"q": "annual revenue"}, "error": "timeout"},
    {"name": "search", "arguments": {"q": "annual revenue"}, "error": "timeout"},
    {"name": "search", "arguments": {"q": "annual revenue"}, "error": "timeout"},
    {"name": "summarize", "arguments": {"text": ""}, "result": "No data found"},
])

# Fail: "search" appears 5 times consecutively, exceeding max_repeat=3
result = assert_no_redundant_calls(trace, max_repeat=3)
assert result.passed is False
assert result.details["redundant_tools"] == ["search"]
assert result.details["max_consecutive"] == 5

# Pass: allow "think" to repeat (it is an internal reasoning tool)
trace_with_think = AgentTrace.from_dict([
    {"name": "think", "arguments": {"thought": "step 1"}},
    {"name": "think", "arguments": {"thought": "step 2"}},
    {"name": "think", "arguments": {"thought": "step 3"}},
    {"name": "think", "arguments": {"thought": "step 4"}},
    {"name": "search", "arguments": {"q": "revenue"}},
])
result = assert_no_redundant_calls(trace_with_think, max_repeat=2, ignore_tools=["think"])
assert result.passed is True
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `trace` | `AgentTrace` | *(required)* | The agent execution trace to verify |
| `max_repeat` | `int` | `3` | Maximum allowed consecutive calls to the same tool |
| `ignore_tools` | `list[str]` | `[]` | Tool names excluded from redundancy checking (e.g., internal reasoning tools) |

#### Returns

`TestResult` with details:

- `redundant_tools` -- list of tool names that exceeded the consecutive-call limit
- `max_consecutive` -- the longest consecutive run found for any single tool
- `max_repeat` -- the threshold that was enforced

---

### assert_no_hallucinated_tools

LLMs sometimes invent tool names that do not exist. When an agent is provided with tools named `search`, `calculator`, and `send_email`, it may spontaneously call `web_search`, `math_eval`, or `email_sender` instead. These hallucinated tool names are syntactically valid function calls, but they map to nothing in the tool registry. Depending on the orchestration framework, the result is either a silent no-op (the call is swallowed) or a generic error that the agent misinterprets and retries.

This failure mode is particularly dangerous because it is invisible in logs that only track errors. The tool call was "successful" from the LLM's perspective -- it generated valid JSON with a plausible function name. But no work was performed.

`assert_no_hallucinated_tools` compares every tool name in the trace against a set of known, registered tool names. Any tool call whose name is not in the known set is flagged as a hallucination.

```python
from mltk.domains.llm.agentic import assert_no_hallucinated_tools
from mltk.domains.llm.trace import AgentTrace

# Define the tools the agent was actually given
registered_tools = ["search", "calculator", "format_output"]

# An agent that hallucinated a tool name
trace = AgentTrace.from_dict([
    {"name": "search", "arguments": {"q": "population of France"}},
    {"name": "web_lookup", "arguments": {"url": "https://example.com"}},  # hallucinated
    {"name": "calculator", "arguments": {"expr": "67000000 / 551695"}},
    {"name": "format_output", "arguments": {"style": "markdown"}},
])

# Fail: "web_lookup" is not a registered tool
result = assert_no_hallucinated_tools(trace, known_tools=registered_tools)
assert result.passed is False
assert result.details["hallucinated"] == ["web_lookup"]
assert result.details["known_tools"] == registered_tools

# Pass: all tool names match registered tools
clean_trace = AgentTrace.from_dict([
    {"name": "search", "arguments": {"q": "GDP of Japan"}},
    {"name": "calculator", "arguments": {"expr": "4.2e12 / 125e6"}},
    {"name": "format_output", "arguments": {"style": "table"}},
])
result = assert_no_hallucinated_tools(clean_trace, known_tools=registered_tools)
assert result.passed is True
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `trace` | `AgentTrace` | *(required)* | The agent execution trace to verify |
| `known_tools` | `list[str]` | *(required)* | Exhaustive list of tool names the agent is authorized to call |

#### Returns

`TestResult` with details:

- `hallucinated` -- list of tool names found in the trace that are not in `known_tools`
- `known_tools` -- the authorized tool list that was checked against
- `total_calls` -- total number of tool calls in the trace
- `hallucination_count` -- number of hallucinated tool calls

---

### assert_cost_budget

In March 2024, a developer reported spending $13,000 in 40 minutes when an autonomous coding agent entered an infinite loop against a paid API. The agent was functioning correctly at the individual-step level -- each API call returned a valid response -- but the orchestrator never terminated the session. There was no cost gate.

Agentic systems need hard budget limits enforced outside the LLM's reasoning loop. The LLM cannot be trusted to monitor its own spend, because it has no reliable awareness of token counts or wall-clock time. Budget enforcement must happen at the trace level: after the run completes (or is interrupted), verify that the total resource consumption stayed within bounds.

`assert_cost_budget` checks two independent dimensions: total tokens consumed and total wall-clock duration. Either or both can be specified. This assertion is designed to run in CI/CD pipelines as a cost gate -- if an agent trace from a test run exceeds the budget, the build fails before the agent is deployed to production.

```python
from mltk.domains.llm.agentic import assert_cost_budget
from mltk.domains.llm.trace import AgentTrace

# A trace from a normal agent run
trace = AgentTrace(
    tool_calls=[
        ToolCall(name="search", arguments={"q": "quarterly earnings"}, result="..."),
        ToolCall(name="summarize", arguments={"text": "..."}, result="Summary"),
        ToolCall(name="format", arguments={"style": "pdf"}, result="formatted.pdf"),
    ],
    total_tokens=12_500,
    total_duration_ms=45_000.0,  # 45 seconds
)

# Pass: within both budgets
result = assert_cost_budget(trace, max_total_tokens=50_000, max_duration_ms=120_000)
assert result.passed is True
assert result.details["tokens_exceeded"] is False
assert result.details["duration_exceeded"] is False

# A runaway trace that burned through tokens
expensive_trace = AgentTrace(
    tool_calls=[ToolCall(name="analyze", arguments={}, result="...")] * 200,
    total_tokens=850_000,
    total_duration_ms=600_000.0,  # 10 minutes
)

# Fail: token budget exceeded
result = assert_cost_budget(expensive_trace, max_total_tokens=100_000)
assert result.passed is False
assert result.details["tokens_exceeded"] is True
assert result.details["actual_tokens"] == 850_000
assert result.details["max_total_tokens"] == 100_000

# Fail: duration budget exceeded (even if tokens are within budget)
result = assert_cost_budget(expensive_trace, max_duration_ms=300_000)
assert result.passed is False
assert result.details["duration_exceeded"] is True
assert result.details["actual_duration_ms"] == 600_000.0
assert result.details["max_duration_ms"] == 300_000
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `trace` | `AgentTrace` | *(required)* | The agent execution trace to verify |
| `max_total_tokens` | `int \| None` | `None` | Maximum allowed total token consumption. Skipped if `None`. |
| `max_duration_ms` | `float \| None` | `None` | Maximum allowed wall-clock duration in milliseconds. Skipped if `None`. |

> At least one of `max_total_tokens` or `max_duration_ms` must be provided.

#### Returns

`TestResult` with details:

- `tokens_exceeded` -- `True` if `actual_tokens > max_total_tokens`
- `actual_tokens` -- total tokens from the trace
- `max_total_tokens` -- the token budget that was enforced (or `None` if not set)
- `duration_exceeded` -- `True` if `actual_duration_ms > max_duration_ms`
- `actual_duration_ms` -- total duration from the trace
- `max_duration_ms` -- the duration budget that was enforced (or `None` if not set)

---

### assert_error_recovery

A well-designed agent encounters an error, adjusts its approach, and continues. A poorly designed agent encounters an error and hammers the same failing tool call repeatedly, generating a long streak of identical errors before eventually giving up or exhausting its budget. The difference between the two is error recovery -- the ability to detect failure, try a different tool or different arguments, and make forward progress.

`assert_error_recovery` scans the trace for consecutive tool calls that all resulted in errors (i.e., `ToolCall.error is not None`). If the longest streak of consecutive errors exceeds `max_consecutive_errors`, the assertion fails. This catches agents that lack fallback logic: instead of pivoting after one or two failures, they blindly retry the same broken path.

Note the distinction from `assert_no_redundant_calls`: that assertion checks for the same tool name repeating, while this assertion checks for any sequence of errors regardless of which tools failed. An agent could call `search` (error), then `database_query` (error), then `api_call` (error) -- three different tools, but a streak of 3 consecutive errors that indicates the agent is flailing without a recovery strategy.

```python
from mltk.domains.llm.agentic import assert_error_recovery
from mltk.domains.llm.trace import AgentTrace

# An agent with poor error recovery: 4 errors in a row before succeeding
trace = AgentTrace.from_dict([
    {"name": "search", "arguments": {"q": "data"}, "result": "found 10 results"},
    {"name": "database_query", "arguments": {"sql": "SELECT *"}, "error": "connection refused"},
    {"name": "database_query", "arguments": {"sql": "SELECT *"}, "error": "connection refused"},
    {"name": "api_call", "arguments": {"url": "/data"}, "error": "404 not found"},
    {"name": "file_read", "arguments": {"path": "/tmp/data.csv"}, "error": "file not found"},
    {"name": "search", "arguments": {"q": "fallback data"}, "result": "found 5 results"},
    {"name": "summarize", "arguments": {"text": "..."}, "result": "Summary complete"},
])

# Fail: 4 consecutive errors exceeds max_consecutive_errors=2
result = assert_error_recovery(trace, max_consecutive_errors=2)
assert result.passed is False
assert result.details["max_error_streak"] == 4
assert result.details["total_errors"] == 4

# Pass: allow up to 4 consecutive errors
result = assert_error_recovery(trace, max_consecutive_errors=4)
assert result.passed is True

# A well-behaved agent that recovers quickly
good_trace = AgentTrace.from_dict([
    {"name": "search", "arguments": {"q": "report"}, "result": "found"},
    {"name": "database_query", "arguments": {"sql": "..."}, "error": "timeout"},
    {"name": "search", "arguments": {"q": "report fallback"}, "result": "cached result"},
    {"name": "summarize", "arguments": {"text": "..."}, "result": "done"},
])

# Pass: only 1 consecutive error, agent recovered immediately
result = assert_error_recovery(good_trace, max_consecutive_errors=2)
assert result.passed is True
assert result.details["max_error_streak"] == 1
assert result.details["total_errors"] == 1
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `trace` | `AgentTrace` | *(required)* | The agent execution trace to verify |
| `max_consecutive_errors` | `int` | `3` | Maximum allowed streak of consecutive failed tool calls |

#### Returns

`TestResult` with details:

- `max_error_streak` -- the longest run of consecutive tool calls with errors
- `total_errors` -- total number of failed tool calls in the entire trace
- `max_consecutive_errors` -- the threshold that was enforced
- `step_count` -- total number of tool calls in the trace

---

## Extended pytest Integration

Combining the core and extended assertions gives comprehensive coverage of an agent's behavior in a single test file.

```python
import pytest
from mltk.domains.llm.trace import AgentTrace, ToolCall
from mltk.domains.llm.agentic import (
    assert_tool_chain,
    assert_no_forbidden_actions,
    assert_step_efficiency,
    assert_no_redundant_calls,
    assert_no_hallucinated_tools,
    assert_cost_budget,
    assert_error_recovery,
)

REGISTERED_TOOLS = ["search", "calculator", "summarize", "format_output"]


@pytest.fixture()
def agent_trace() -> AgentTrace:
    """Simulate a recorded agent session."""
    return AgentTrace(
        tool_calls=[
            ToolCall(name="search", arguments={"q": "climate data"}, result="..."),
            ToolCall(name="search", arguments={"q": "climate data 2024"}, result="..."),
            ToolCall(name="calculator", arguments={"expr": "avg(temps)"}, result="18.3"),
            ToolCall(name="summarize", arguments={"text": "..."}, result="Summary"),
            ToolCall(name="format_output", arguments={"style": "chart"}, result="chart.png"),
        ],
        total_tokens=15_200,
        total_duration_ms=32_000.0,
    )


def test_no_retry_loops(agent_trace: AgentTrace) -> None:
    """Agent should not call the same tool more than 3 times in a row."""
    result = assert_no_redundant_calls(agent_trace, max_repeat=3)
    assert result.passed is True


def test_no_hallucinated_tools(agent_trace: AgentTrace) -> None:
    """Every tool call must target a registered tool."""
    result = assert_no_hallucinated_tools(agent_trace, known_tools=REGISTERED_TOOLS)
    assert result.passed is True


def test_cost_budget(agent_trace: AgentTrace) -> None:
    """Agent must stay within 50K tokens and 2 minutes."""
    result = assert_cost_budget(
        agent_trace,
        max_total_tokens=50_000,
        max_duration_ms=120_000,
    )
    assert result.passed is True


def test_error_recovery(agent_trace: AgentTrace) -> None:
    """Agent must not produce more than 2 consecutive errors."""
    result = assert_error_recovery(agent_trace, max_consecutive_errors=2)
    assert result.passed is True
```
