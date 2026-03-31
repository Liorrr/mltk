# MCP Evaluation

An AI agent that calls tools correctly in isolation but
sends `"limit": "ten"` to an MCP server expecting an
integer is broken. An agent that reads `file:///etc/passwd`
when it only needed `file:///workspace/README.md` is a
security incident. An agent stuck in a five-retry loop
on a failing API wastes tokens and time.

These are not theoretical risks. MCP (Model Context
Protocol) connects agents to databases, file systems,
APIs, and code editors through a universal standard --
the "USB-C for AI." With 97 million+ SDK downloads and
Linux Foundation governance (January 2025), MCP is the
dominant agent-tool integration protocol.

But agents regularly make schema-invalid calls,
over-fetch resources, and retry endlessly. These are
testable bugs. mltk makes them pytest assertions.

**No other testing tool validates agent tool calls
against the server's declared JSON Schema. No other
tool asserts on MCP resource access patterns. mltk is
the first.**

**Module:** `mltk.domains.llm.mcp`

**ML Lifecycle Stage:** Agent evaluation / CI gate /
Integration testing / Security audit

**Bugs caught:**

- Tool calls with wrong argument types (string where
  the schema requires integer)
- Missing required parameters that the server declared
  mandatory
- Arguments out of range (`limit: 999` when the schema
  says `maximum: 100`)
- Agents reading resources they should never access
  (path traversal, unauthorized URIs)
- Agents stuck retrying the same failing tool call
  with identical arguments
- Context window overflow from unchecked resource
  accumulation
- Cross-server tool routing errors in multi-server
  setups

---

## Why MCP Evaluation Matters

### The protocol

MCP was released by Anthropic in November 2024 as an
open standard for connecting LLM agents to external
systems. It uses JSON-RPC 2.0 message framing over
stdio, HTTP+SSE, or WebSocket transports.

Before MCP, every agent framework built bespoke
connectors -- one integration per tool, per framework.
MCP standardizes this: any MCP-compliant server
connects to any MCP-compliant client.

The specification defines five core concepts:

| Concept | What it does |
|---------|-------------|
| **Tools** | Functions the server exposes with JSON Schema parameter definitions |
| **Resources** | Read-only data URIs (`file://`, `db://`, custom schemes) |
| **Prompts** | Server-curated prompt templates with arguments |
| **Sampling** | Server-initiated LLM generation requests |
| **Roots** | Directory/URI boundaries the server may access |

**Citation:** MCP Specification v1.0
(modelcontextprotocol.io/specification). Linux
Foundation Technical Advisory Committee vote, January
2025. Python SDK: github.com/modelcontextprotocol/
python-sdk.

### The adoption

MCP has native support in Claude, Cursor, VS Code
Copilot, Cline, and OpenCode. The awesome-mcp-servers
list catalogs 6,000+ MCP servers. Every major IDE and
AI coding tool has shipped MCP support.

**Citation:** "awesome-mcp-servers" (github.com/
punkpeye/awesome-mcp-servers). Cursor MCP docs
(cursor.com/docs/mcp). VS Code Copilot MCP docs
(code.visualstudio.com/docs/copilot/mcp).

### The testing gap

None of this adoption came with testing tools. An agent
discovers tools via `tools/list`, calls them via
`tools/call`, reads resources via `resources/read` --
and nobody validates that the agent's arguments match
the server's schema, or that the agent accessed only
the resources it should have, or that it handled errors
without getting stuck in a retry loop.

---

## The Testing Gap in Detail

### DeepEval: LLM-as-Judge

DeepEval (8.6K GitHub stars) shipped the first
MCP-specific metrics in early 2025. Their approach
centers on two LLM-as-Judge metrics:

```python
# DeepEval's approach -- requires GPT-4o
from deepeval.metrics import MCPUseMetric

metric = MCPUseMetric(threshold=0.7)
# Sends the trace to GPT-4o for scoring
```

The LLM judge decides whether the agent "used MCP
tools appropriately." This is expensive (~$0.03 per
evaluation), non-deterministic (different runs give
different scores), requires network access (cannot run
in air-gapped CI), and cannot validate structural
correctness (the judge does not check JSON Schema).

**Citation:** DeepEval MCP docs (docs.confident-ai.com/
docs/mcp). DeepEval GitHub (github.com/confident-ai/
deepeval).

### mltk: contract-based

mltk takes the opposite approach. The tool's JSON
Schema IS the test contract. `jsonschema.validate()`
checks structural conformance deterministically, with
no LLM, no network, no cost per evaluation.

| Capability | mltk | DeepEval | RAGAS | Promptfoo |
|-----------|:----:|:--------:|:-----:|:---------:|
| JSON Schema validation | **yes** | no | no | no |
| Resource access assertions | **yes** | no | no | no |
| Server namespace routing | **yes** | no | no | no |
| Context window assertions | **yes** | no | no | no |
| Works offline (no LLM) | **yes** | no | no | partial |
| pytest native | **yes** | yes | no | no |
| MCP session completion | planned | yes | no | no |

**Citation:** RAGAS docs (docs.ragas.io). Promptfoo
docs (promptfoo.dev/docs). Inspect AI
(github.com/UKGovernmentBEIS/inspect_ai). None offer
MCP-specific assertions.

### Why contract-based is better for structural bugs

A JSON Schema is a formal specification. When the
server declares `"type": "integer", "maximum": 100`,
there is exactly one correct answer to "is 999 valid?"
-- no. An LLM judge might say "the agent passed a
reasonable value" because it does not have the schema
context. Contract-based validation is deterministic,
complete, and free.

LLM-as-Judge is the right tool for subjective quality
("did the agent accomplish the user's goal?"). mltk
supports this as a planned extension (P2). But for
structural correctness, the schema is the test.

---

## Data Model

MCP testing requires extending the existing `AgentTrace`
data model. The design is additive and backward-
compatible: all 10 existing agentic assertions continue
to work on `McpTrace` without modification.

### McpToolCall

Extends `ToolCall` with MCP-specific fields.

```python
from dataclasses import dataclass, field
from mltk.domains.llm.trace import ToolCall

@dataclass
class McpToolCall(ToolCall):
    """Tool call with MCP server context."""
    server: str = ""
    tool_schema: dict = field(
        default_factory=dict,
    )
    is_error: bool = False
    context_tokens_before: int = 0
```

| Field | Type | Purpose |
|-------|------|---------|
| `server` | `str` | MCP server namespace (e.g., `"filesystem"`) |
| `tool_schema` | `dict` | The `inputSchema` from `tools/list` |
| `is_error` | `bool` | Whether the server returned `isError: true` |
| `context_tokens_before` | `int` | Running token count before this call |

### McpResourceAccess

A single MCP resource read. Resources are read-only
data URIs -- files, database rows, API responses. They
are a distinct MCP concept, not modeled in the base
`AgentTrace`.

```python
@dataclass
class McpResourceAccess:
    """A single MCP resource read."""
    uri: str
    server: str = ""
    result: str | None = None
    error: str | None = None
    content_tokens: int = 0
    duration_ms: float = 0.0
```

| Field | Type | Purpose |
|-------|------|---------|
| `uri` | `str` | Resource URI (e.g., `"file:///workspace/README.md"`) |
| `server` | `str` | Which MCP server served this resource |
| `result` | `str \| None` | Resource content (if captured) |
| `error` | `str \| None` | Error message (if read failed) |
| `content_tokens` | `int` | Token count of the resource content |
| `duration_ms` | `float` | Read latency |

### McpTrace

Extends `AgentTrace` with resource accesses and model
context limit. Because `McpTrace` subclasses
`AgentTrace`, it passes to every existing assertion
unchanged.

```python
from mltk.domains.llm.trace import AgentTrace

@dataclass
class McpTrace(AgentTrace):
    """Agent trace with MCP-specific data."""
    mcp_tool_calls: list[McpToolCall] = field(
        default_factory=list,
    )
    resource_accesses: list[McpResourceAccess] = (
        field(default_factory=list)
    )
    servers: list[str] = field(
        default_factory=list,
    )
    model_context_limit: int = 0
```

### Why subclass instead of adding fields?

Adding nullable MCP fields to `AgentTrace` pollutes
the base class for every non-MCP user. A subclass is
additive, explicit about what data is available, and
preserves polymorphism. Every function accepting
`AgentTrace` also accepts `McpTrace`.

### Backward compatibility

```python
trace = McpTrace(
    mcp_tool_calls=[
        McpToolCall(
            name="read_file",
            server="filesystem",
            arguments={"path": "/workspace/data.csv"},
            result="col1,col2\n1,2\n",
        ),
    ],
    resource_accesses=[
        McpResourceAccess(
            uri="file:///workspace/data.csv",
        ),
    ],
    total_tokens=15_000,
    total_duration_ms=3200.0,
)

# All existing assertions work unchanged:
from mltk.domains.llm.agentic import (
    assert_tool_chain,
    assert_cost_budget,
    assert_error_recovery,
)

assert_tool_chain(
    trace, expected_tools=["read_file"],
)
assert_cost_budget(
    trace, max_total_tokens=50_000,
)
assert_error_recovery(
    trace, max_consecutive_errors=3,
)
```

---

## Assertions

### assert_mcp_tool_schema_conformance

Validates that arguments passed to an MCP tool conform
to the tool's declared JSON Schema. This is mltk's
primary first-mover assertion for MCP evaluation.

**Why this matters:** Every MCP tool is declared with
an `inputSchema` -- a JSON Schema object describing its
parameters. When an agent calls the tool, it must send
arguments that conform to this schema. DeepEval's
`ToolCallAccuracy` requires the test author to manually
specify every expected argument value. mltk requires
only the schema (which the server already provides).
The schema IS the test.

**Citation:** JSON Schema draft-07 specification
(json-schema.org). jsonschema Python library
(github.com/python-jsonschema/jsonschema, 4K+ stars).
MCP Specification Section 6.1: Tool Declaration
(modelcontextprotocol.io/specification).

#### What it catches

| Failure | Example |
|---------|---------|
| Wrong type | `"limit": "ten"` when schema says `integer` |
| Missing required | Schema has `required: ["query"]`, agent omits `query` |
| Out of range | `"limit": 999` when schema says `maximum: 100` |
| Extra properties | `"unknown": true` when `additionalProperties: false` |
| Wrong nested type | `"filters": "string"` when schema expects `object` |
| Invalid format | `"date_after": "not-a-date"` when `format: "date"` |

#### Signature

```python
from mltk.domains.llm.mcp import (
    assert_mcp_tool_schema_conformance,
)

result = assert_mcp_tool_schema_conformance(
    tool_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "minLength": 1,
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
            },
            "filters": {
                "type": "object",
                "properties": {
                    "date_after": {
                        "type": "string",
                        "format": "date",
                    },
                },
            },
        },
        "required": ["query"],
    },
    actual_args={
        "query": "climate change",
        "limit": 10,
    },
    tool_name="search_documents",
)
```

#### Parameters

| Name | Type | Default | What |
|------|------|---------|------|
| `tool_schema` | `dict` | *(required)* | The `inputSchema` from `tools/list` |
| `actual_args` | `dict` | *(required)* | Arguments the agent passed |
| `tool_name` | `str` | `""` | Tool name for error messages |

#### Returns

`TestResult` with:

- `passed`: whether the arguments conform
- `errors`: list of schema violation messages
- `tool_name`: the tool that was validated
- `schema`: the schema that was checked against

#### How it works

Under the hood, this calls `jsonschema.validate()`:

```python
import jsonschema

try:
    jsonschema.validate(
        instance=actual_args,
        schema=tool_schema,
    )
    # passed = True, errors = []
except jsonschema.ValidationError as e:
    # passed = False, errors = [e.message]
```

The `jsonschema` library validates against JSON Schema
draft-04 through draft-2020-12. It checks types,
required fields, numeric ranges, string patterns,
format constraints, and nested schemas recursively.

#### Failing example

```python
# Agent sends string where integer is required
result = assert_mcp_tool_schema_conformance(
    tool_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {
                "type": "integer",
                "maximum": 100,
            },
        },
        "required": ["query"],
    },
    actual_args={
        "query": "climate change",
        "limit": "ten",  # Wrong type!
    },
    tool_name="search_documents",
)
assert not result.passed
# errors: ["'ten' is not of type 'integer'"]
```

#### Batch validation from a trace

Validate every tool call in an `McpTrace` at once:

```python
trace = McpTrace(
    mcp_tool_calls=[
        McpToolCall(
            name="search",
            server="docs",
            arguments={"query": "ml", "limit": 10},
            tool_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {
                        "type": "integer",
                        "maximum": 100,
                    },
                },
                "required": ["query"],
            },
        ),
        McpToolCall(
            name="read_file",
            server="filesystem",
            arguments={"path": 42},  # Wrong!
            tool_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        ),
    ],
)

for tc in trace.mcp_tool_calls:
    result = assert_mcp_tool_schema_conformance(
        tool_schema=tc.tool_schema,
        actual_args=tc.arguments,
        tool_name=f"{tc.server}::{tc.name}",
    )
    assert result.passed, result.message
```

---

### assert_mcp_tool_selection

Tests whether an agent called the correct set of MCP
tools, with server-namespace awareness. An agent
calling `filesystem::read_file` should not count as
having called `database::read_file`.

**Why this matters:** MCP agents often connect to
multiple servers simultaneously. A coding agent might
use `filesystem::read_file` for source code and
`github::search_repos` for repository lookup. If the
agent routes a file read to the wrong server, the call
may silently succeed with wrong data or fail with a
confusing error. Server-namespace routing is invisible
to existing tool selection assertions.

**Citation:** MCP Specification Section 4: Server
Namespaces (modelcontextprotocol.io/specification).
"MCP: The Missing Manual" -- Simon Willison blog
(January 2025), covering practical multi-server
patterns.

#### What it catches

| Failure | Example |
|---------|---------|
| Wrong server | Agent calls `database::read_file` instead of `filesystem::read_file` |
| Missing tool | Agent never calls a required tool |
| Extra tool | Agent calls tools not needed for the task |
| Cross-server confusion | Two servers expose similar tool names |

#### Signature

```python
from mltk.domains.llm.mcp import (
    assert_mcp_tool_selection,
)

result = assert_mcp_tool_selection(
    trace=trace,
    expected_tools=[
        "filesystem::read_file",
        "github::search_repos",
    ],
    server=None,  # check all servers
)
```

#### Parameters

| Name | Type | Default | What |
|------|------|---------|------|
| `trace` | `McpTrace` | *(required)* | The agent trace |
| `expected_tools` | `list[str]` | *(required)* | Expected tools as `"server::tool"` or `"tool"` |
| `server` | `str \| None` | `None` | Filter to a specific server namespace |

#### Returns

`TestResult` with:

- `passed`: whether precision and recall are both 1.0
- `precision`: fraction of called tools that were
  expected
- `recall`: fraction of expected tools that were
  called
- `f1`: harmonic mean of precision and recall
- `missing`: expected tools not called
- `extra`: called tools not expected

#### Multi-server example

```python
trace = McpTrace(
    mcp_tool_calls=[
        McpToolCall(
            name="read_file",
            server="filesystem",
            arguments={"path": "/src/main.py"},
            result="import os\n...",
        ),
        McpToolCall(
            name="search_repos",
            server="github",
            arguments={"query": "mltk"},
            result='[{"name": "mltk"}]',
        ),
        McpToolCall(
            name="run_query",
            server="database",
            arguments={"sql": "SELECT 1"},
            result="1",
        ),
    ],
)

# Check all servers
result = assert_mcp_tool_selection(
    trace=trace,
    expected_tools=[
        "filesystem::read_file",
        "github::search_repos",
    ],
)
# precision: 0.67 (2/3 expected, 1 extra)
# recall: 1.0 (both expected tools called)

# Filter to filesystem server only
result = assert_mcp_tool_selection(
    trace=trace,
    expected_tools=["filesystem::read_file"],
    server="filesystem",
)
# precision: 1.0, recall: 1.0
```

---

### assert_mcp_resource_access

Tests whether an agent accessed the correct MCP
resources and did not access forbidden ones. This
assertion is unique to mltk -- no other testing tool
models MCP resource access.

**Why this matters:** MCP Resources are read-only data
URIs. An agent that reads `file:///etc/passwd` when the
task only required reading `file:///workspace/README.md`
is a security violation. An agent that reads 50
resources when 2 would suffice is wasting tokens and
context window budget. Resource access patterns are a
distinct testing concern that tool call assertions
cannot cover.

**Citation:** MCP Specification Section 5: Resources
(modelcontextprotocol.io/specification). Trail of Bits
"Breaking MCP" (March 2025, trailofbits.com) -- covers
unauthorized resource access as an attack surface.
OWASP LLM Top 10 2025 (LLM08: Tool Security).

#### What it catches

| Failure | Example |
|---------|---------|
| Forbidden access | Agent reads `file:///etc/passwd` |
| Missing access | Agent never reads the required reference doc |
| Over-fetching | Agent reads 50 resources when 2 suffice |
| Path traversal | Agent reads `file:///../../etc/shadow` |

#### Signature

```python
from mltk.domains.llm.mcp import (
    assert_mcp_resource_access,
)

result = assert_mcp_resource_access(
    trace=trace,
    expected_uris=[
        "file:///workspace/README.md",
        "file:///workspace/docs/api.md",
    ],
    forbidden_uris=[
        "file:///etc/passwd",
        "file:///etc/shadow",
    ],
    max_reads=10,
)
```

#### Parameters

| Name | Type | Default | What |
|------|------|---------|------|
| `trace` | `McpTrace` | *(required)* | The agent trace |
| `expected_uris` | `list[str] \| None` | `None` | URIs that MUST be accessed |
| `forbidden_uris` | `list[str] \| None` | `None` | URIs that MUST NOT be accessed |
| `max_reads` | `int \| None` | `None` | Maximum total resource reads allowed |

#### Returns

`TestResult` with:

- `passed`: whether all checks passed
- `missing_uris`: expected URIs not accessed
- `forbidden_accessed`: forbidden URIs that were
  accessed
- `total_reads`: total resource reads in the trace
- `over_limit`: whether `max_reads` was exceeded

#### Security testing example

An agent tasked with summarizing project documentation
should read docs but never access system files:

```python
trace = McpTrace(
    resource_accesses=[
        McpResourceAccess(
            uri="file:///workspace/README.md",
            content_tokens=500,
        ),
        McpResourceAccess(
            uri="file:///workspace/docs/api.md",
            content_tokens=1200,
        ),
    ],
)

# Passes: read expected docs, nothing forbidden
result = assert_mcp_resource_access(
    trace=trace,
    expected_uris=[
        "file:///workspace/README.md",
    ],
    forbidden_uris=[
        "file:///etc/passwd",
        "file:///etc/shadow",
        "file:///root/.ssh/id_rsa",
    ],
)
assert result.passed
```

#### Over-fetching example

An agent should summarize one file but reads every
file in the workspace:

```python
trace = McpTrace(
    resource_accesses=[
        McpResourceAccess(
            uri=f"file:///workspace/file_{i}.md",
            content_tokens=500,
        )
        for i in range(50)
    ],
)

result = assert_mcp_resource_access(
    trace=trace,
    max_reads=5,
)
assert not result.passed
# total_reads: 50, over_limit: True
```

---

### assert_mcp_context_window

Tests whether an agent's MCP session stayed within the
model's declared context window. Reports utilization
percentage -- "82% utilized" is actionable, "103,421
tokens" is not.

**Why this matters:** MCP sessions accumulate context
from tool results, resource content, and conversation
history. Each model has a hard context limit (200K for
Claude 3.5, 128K for GPT-4o, 32K for smaller models).
An agent that exceeds the limit silently drops context,
which causes forgotten instructions, repeated work, and
incorrect answers. The existing `assert_cost_budget`
checks an arbitrary user-supplied `max_total_tokens`.
This assertion takes the model's actual context window
as the reference.

#### What it catches

| Failure | Example |
|---------|---------|
| Overflow | Agent uses 210K tokens on a 200K model |
| Near-overflow | Agent at 95% utilization (warning) |
| Resource bloat | Agent reads huge resources filling the window |

#### Signature

```python
from mltk.domains.llm.mcp import (
    assert_mcp_context_window,
)

result = assert_mcp_context_window(
    trace=trace,
    model_context_limit=200_000,
    max_utilization=0.9,
)
```

#### Parameters

| Name | Type | Default | What |
|------|------|---------|------|
| `trace` | `McpTrace` | *(required)* | The agent trace |
| `model_context_limit` | `int` | *(required)* | Model's declared context window (tokens) |
| `max_utilization` | `float` | `0.9` | Maximum utilization fraction (0-1) |

#### Returns

`TestResult` with:

- `passed`: whether utilization is under the limit
- `utilization`: fraction of context window used
- `total_tokens`: total tokens in the trace
- `context_limit`: the model's context window
- `headroom_tokens`: remaining tokens before overflow

#### Example: approaching the limit

```python
trace = McpTrace(
    total_tokens=175_000,
    mcp_tool_calls=[
        McpToolCall(
            name="search",
            server="docs",
            arguments={"query": "deployment"},
            result="... (50K tokens of results)",
            context_tokens_before=120_000,
        ),
    ],
    resource_accesses=[
        McpResourceAccess(
            uri="file:///workspace/full-docs.md",
            content_tokens=40_000,
        ),
    ],
)

result = assert_mcp_context_window(
    trace=trace,
    model_context_limit=200_000,
    max_utilization=0.9,
)
# utilization: 0.875 (175K / 200K)
# passed: True (under 0.9 threshold)
# headroom_tokens: 25_000
```

#### Common context limits

| Model | Context window |
|-------|---------------|
| Claude 3.5 Sonnet | 200,000 |
| Claude 3.5 Haiku | 200,000 |
| GPT-4o | 128,000 |
| GPT-4o-mini | 128,000 |
| Gemini 1.5 Pro | 2,000,000 |
| Llama 3.1 405B | 128,000 |

---

### assert_mcp_error_recovery

Tests whether an agent handles MCP tool errors by
changing strategy instead of retrying the same call
with identical arguments. Detects retry loops -- a
common failure mode where agents repeat a failing call
three, five, or ten times without changing anything.

**Why this matters:** MCP servers return
`isError: true` in tool responses when a call fails.
A well-behaved agent should try a different tool,
change its arguments, or report the error to the user.
An agent that retries the identical call is stuck in a
loop, wasting tokens and time. The existing
`assert_error_recovery` checks for consecutive error
streaks. This assertion specifically detects same-tool,
same-arguments retry patterns.

**Citation:** MCP Specification Section 6.3: Tool
Response Format (modelcontextprotocol.io/specification)
-- `isError` field and error handling guidance.

#### What it catches

| Failure | Example |
|---------|---------|
| Same-tool retry | Agent calls `search(query="x")` 5 times, all fail |
| Infinite loop | Agent alternates between two failing tools |
| No strategy change | Agent retries with identical arguments after error |

#### Signature

```python
from mltk.domains.llm.mcp import (
    assert_mcp_error_recovery,
)

result = assert_mcp_error_recovery(
    trace=trace,
    max_same_tool_retries=2,
)
```

#### Parameters

| Name | Type | Default | What |
|------|------|---------|------|
| `trace` | `McpTrace` | *(required)* | The agent trace |
| `max_same_tool_retries` | `int` | `2` | Max identical retries before flagging |

#### Returns

`TestResult` with:

- `passed`: whether no retry loop exceeded the limit
- `max_retries_found`: longest same-tool retry streak
- `retry_details`: list of `{tool, server, args, count}`
  for each retry pattern found

#### Retry loop example

An agent tries to call a rate-limited API five times
with the same arguments:

```python
trace = McpTrace(
    mcp_tool_calls=[
        McpToolCall(
            name="fetch_data",
            server="api",
            arguments={"endpoint": "/users"},
            is_error=True,
            error="429 Too Many Requests",
        ),
        McpToolCall(
            name="fetch_data",
            server="api",
            arguments={"endpoint": "/users"},
            is_error=True,
            error="429 Too Many Requests",
        ),
        McpToolCall(
            name="fetch_data",
            server="api",
            arguments={"endpoint": "/users"},
            is_error=True,
            error="429 Too Many Requests",
        ),
        McpToolCall(
            name="fetch_data",
            server="api",
            arguments={"endpoint": "/users"},
            is_error=True,
            error="429 Too Many Requests",
        ),
        McpToolCall(
            name="fetch_data",
            server="api",
            arguments={"endpoint": "/users"},
            is_error=True,
            error="429 Too Many Requests",
        ),
    ],
)

result = assert_mcp_error_recovery(
    trace=trace,
    max_same_tool_retries=2,
)
assert not result.passed
# max_retries_found: 5
# retry_details: [{tool: "fetch_data",
#   server: "api",
#   args: {"endpoint": "/users"},
#   count: 5}]
```

#### Good recovery example

Agent gets an error, changes strategy:

```python
trace = McpTrace(
    mcp_tool_calls=[
        McpToolCall(
            name="fetch_data",
            server="api",
            arguments={"endpoint": "/users"},
            is_error=True,
            error="429 Too Many Requests",
        ),
        # Agent changes strategy: different endpoint
        McpToolCall(
            name="fetch_data",
            server="api",
            arguments={"endpoint": "/users/cached"},
            result='[{"name": "Alice"}]',
        ),
    ],
)

result = assert_mcp_error_recovery(
    trace=trace,
    max_same_tool_retries=2,
)
assert result.passed
# max_retries_found: 1 (only one attempt per args)
```

---

## Integration with Existing Assertions

`McpTrace` is an `AgentTrace`. All 10 existing agentic
assertions work on MCP traces without modification.

| Existing assertion | Works with McpTrace | What it tests |
|-------------------|:-------------------:|---------------|
| `assert_task_completion` | yes | Token overlap with expected output |
| `assert_tool_selection` | yes | Tool set precision/recall |
| `assert_tool_call_correctness` | yes | Exact arg value match |
| `assert_tool_chain` | yes | Ordered/unordered tool sequence |
| `assert_no_forbidden_actions` | yes | Prohibited tool names |
| `assert_step_efficiency` | yes | Max tool call count |
| `assert_no_redundant_calls` | yes | Consecutive duplicate calls |
| `assert_no_hallucinated_tools` | yes | Tool names not in known set |
| `assert_cost_budget` | yes | Token and time budgets |
| `assert_error_recovery` | yes | Consecutive error streaks |

### Composable test suites

Run MCP-specific and general assertions in one test:

```python
import pytest
from mltk.domains.llm.agentic import (
    assert_tool_chain,
    assert_no_hallucinated_tools,
    assert_cost_budget,
)
from mltk.domains.llm.mcp import (
    assert_mcp_tool_schema_conformance,
    assert_mcp_tool_selection,
    assert_mcp_resource_access,
)


def test_mcp_agent_full(mcp_trace):
    """Full MCP agent evaluation."""

    # Schema conformance (MCP-specific)
    for tc in mcp_trace.mcp_tool_calls:
        r = assert_mcp_tool_schema_conformance(
            tool_schema=tc.tool_schema,
            actual_args=tc.arguments,
            tool_name=f"{tc.server}::{tc.name}",
        )
        assert r.passed, r.message

    # Tool selection (MCP-specific)
    r = assert_mcp_tool_selection(
        trace=mcp_trace,
        expected_tools=[
            "filesystem::read_file",
            "docs::search",
        ],
    )
    assert r.passed, r.message

    # Resource access (MCP-specific)
    r = assert_mcp_resource_access(
        trace=mcp_trace,
        forbidden_uris=["file:///etc/passwd"],
        max_reads=20,
    )
    assert r.passed, r.message

    # Tool chain order (existing)
    assert_tool_chain(
        mcp_trace,
        expected_tools=["read_file", "search"],
    )

    # No hallucinated tools (existing)
    assert_no_hallucinated_tools(
        mcp_trace,
        known_tools=[
            "read_file", "search", "write_file",
        ],
    )

    # Budget (existing)
    assert_cost_budget(
        mcp_trace, max_total_tokens=100_000,
    )
```

---

## Installation

### Core (no MCP assertions)

```bash
pip install mltk
```

All existing assertions work. MCP dataclasses
(`McpTrace`, `McpToolCall`, `McpResourceAccess`) are
available without extra dependencies.

### MCP assertions

```bash
pip install mltk[mcp]
```

Installs `jsonschema>=4.0` (~100KB, no heavy transitive
dependencies). Required for
`assert_mcp_tool_schema_conformance`. The other four
MCP assertions have zero extra dependencies.

### What happens without jsonschema?

If you call `assert_mcp_tool_schema_conformance` without
`jsonschema` installed:

```
ImportError: jsonschema is required for MCP schema
conformance validation. Install with:
    pip install mltk[mcp]
```

The other four assertions
(`assert_mcp_tool_selection`,
`assert_mcp_resource_access`,
`assert_mcp_context_window`,
`assert_mcp_error_recovery`) work without `jsonschema`.

---

## Method Decision Flowchart

```
Testing an MCP agent?
  |
  YES
  |
  v
Need to validate tool arguments against schema?
  YES --> assert_mcp_tool_schema_conformance
         Contract-based. No LLM. jsonschema dep.
  |
  v
Need to verify correct tools were called?
  YES --> assert_mcp_tool_selection
         Server-namespace aware. Precision/recall.
  |
  v
Need to control resource access?
  YES --> assert_mcp_resource_access
         Expected URIs, forbidden URIs, max reads.
  |
  v
Need to check context window budget?
  YES --> assert_mcp_context_window
         Model-aware. Reports utilization %.
  |
  v
Need to detect retry loops?
  YES --> assert_mcp_error_recovery
         Same-tool, same-args retry detection.
  |
  v
Need general agentic assertions too?
  YES --> McpTrace works with all 10 existing
         assertions. Use them together.
```

**Rules of thumb:**

- **Every MCP test suite** should include
  `assert_mcp_tool_schema_conformance`. Schema
  validation is the highest-value, lowest-cost
  assertion. The schema IS the test.
- **Multi-server agents** need
  `assert_mcp_tool_selection` with the `server`
  parameter. Single-server agents can use the
  existing `assert_tool_selection` from the
  agentic module.
- **Security-sensitive agents** need
  `assert_mcp_resource_access` with `forbidden_uris`.
  This is the only assertion in any testing framework
  that validates MCP resource access patterns.
- **Long-running sessions** need
  `assert_mcp_context_window`. Agents that accumulate
  context across many tool calls and resource reads
  can overflow silently.
- **Unreliable tool servers** need
  `assert_mcp_error_recovery`. Rate limits, network
  errors, and server bugs cause retry loops that
  waste tokens.

---

## Pytest Examples

### Basic schema validation

```python
import pytest
from mltk.domains.llm.mcp import (
    assert_mcp_tool_schema_conformance,
)

SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "minLength": 1,
        },
        "limit": {
            "type": "integer",
            "minimum": 1,
            "maximum": 100,
        },
    },
    "required": ["query"],
}


@pytest.mark.parametrize("args,should_pass", [
    ({"query": "ml", "limit": 10}, True),
    ({"query": "ml"}, True),
    ({"limit": 10}, False),            # missing query
    ({"query": "", "limit": 10}, False),  # too short
    ({"query": "ml", "limit": 999}, False),  # > max
    ({"query": "ml", "limit": "ten"}, False),  # type
])
def test_search_schema(args, should_pass):
    result = assert_mcp_tool_schema_conformance(
        tool_schema=SEARCH_SCHEMA,
        actual_args=args,
        tool_name="search_documents",
    )
    assert result.passed == should_pass
```

### Security audit

```python
import pytest
from mltk.domains.llm.mcp import (
    assert_mcp_resource_access,
    assert_mcp_error_recovery,
)

FORBIDDEN_PATHS = [
    "file:///etc/passwd",
    "file:///etc/shadow",
    "file:///root/.ssh/id_rsa",
    "file:///root/.aws/credentials",
]


def test_no_unauthorized_access(mcp_trace):
    """Agent must not access system files."""
    result = assert_mcp_resource_access(
        trace=mcp_trace,
        forbidden_uris=FORBIDDEN_PATHS,
    )
    assert result.passed, (
        f"Accessed forbidden URIs: "
        f"{result.details['forbidden_accessed']}"
    )


def test_no_retry_storms(mcp_trace):
    """Agent must change strategy after errors."""
    result = assert_mcp_error_recovery(
        trace=mcp_trace,
        max_same_tool_retries=3,
    )
    assert result.passed, (
        f"Retry loop detected: "
        f"{result.details['max_retries_found']} "
        f"identical retries"
    )
```

### CI pipeline integration

```python
import json
import pytest
from mltk.domains.llm.mcp import (
    McpTrace,
    McpToolCall,
    McpResourceAccess,
    assert_mcp_tool_schema_conformance,
    assert_mcp_tool_selection,
    assert_mcp_resource_access,
    assert_mcp_context_window,
)


@pytest.fixture
def trace_from_log():
    """Load trace from MCP session log."""
    with open("tests/fixtures/mcp_session.json") as f:
        data = json.load(f)
    return McpTrace(
        mcp_tool_calls=[
            McpToolCall(**tc)
            for tc in data["tool_calls"]
        ],
        resource_accesses=[
            McpResourceAccess(**ra)
            for ra in data["resource_accesses"]
        ],
        total_tokens=data["total_tokens"],
    )


def test_all_schemas_valid(trace_from_log):
    """Every tool call must match its schema."""
    for tc in trace_from_log.mcp_tool_calls:
        if tc.tool_schema:
            r = assert_mcp_tool_schema_conformance(
                tool_schema=tc.tool_schema,
                actual_args=tc.arguments,
                tool_name=(
                    f"{tc.server}::{tc.name}"
                ),
            )
            assert r.passed, r.message


def test_context_budget(trace_from_log):
    """Session must stay under 90% context usage."""
    result = assert_mcp_context_window(
        trace=trace_from_log,
        model_context_limit=200_000,
        max_utilization=0.9,
    )
    assert result.passed, (
        f"Context utilization: "
        f"{result.details['utilization']:.1%}"
    )
```

---

## Design Decisions

### Why contract-based instead of LLM-as-Judge?

LLM-as-Judge (DeepEval's approach) costs ~$0.03 per
evaluation, requires network access, is
non-deterministic, and cannot validate structural
correctness. A JSON Schema is a formal specification
-- `jsonschema.validate()` gives a definitive yes/no
answer. Contract-based testing is deterministic,
complete, free, and runs in air-gapped environments.

LLM-as-Judge is the right approach for subjective
quality evaluation ("did the agent accomplish the
user's goal?"). mltk plans to add this as a future
extension using the existing `judge.py` pattern. But
for structural correctness, the schema is the test.

**Citation:** DeepEval MCPUseMetric documentation
(docs.confident-ai.com/docs/mcp) describes their
LLM-as-Judge approach. JSON Schema specification
(json-schema.org) provides the formal contract
framework.

### Why a subclass instead of optional fields?

Adding nullable MCP fields (`server`, `tool_schema`,
`resource_accesses`) to `AgentTrace` pollutes the base
class. Every user who never touches MCP would see
`None` fields everywhere. A subclass (`McpTrace`)
is additive, self-documenting, and preserves
polymorphism -- any function accepting `AgentTrace`
also accepts `McpTrace`.

### Why trace-based instead of live testing?

An alternative approach is building a full MCP client
harness that spins up real servers, sends real
messages, and validates real responses. This requires
managing stdio processes, async transports, and server
state in CI. High infrastructure complexity for
marginal testing gain.

Trace-based testing is hermetic: capture the MCP
interaction once, then assert on the trace repeatedly
with zero infrastructure. This is consistent with
mltk's existing design across all assertion families.

**Citation:** Trail of Bits "Breaking MCP" (March 2025)
recommended trace-based security analysis over live
harness testing due to the complexity of MCP transport
management.

### Why jsonschema as an optional dependency?

`jsonschema` is ~100KB with minimal transitive
dependencies. But mltk's core install (`numpy` +
`pandas`) should not grow for a feature that only
MCP-focused users need. The `mltk[mcp]` extra group
keeps the core lightweight while giving MCP users
everything they need in one command.

---

## Research Citations

| # | Source | Relevance |
|---|--------|-----------|
| 1 | MCP Spec v1.0 (modelcontextprotocol.io) | Protocol, schemas, resources |
| 2 | Linux Foundation TAC vote (Jan 2025) | Standardization, governance |
| 3 | Python MCP SDK (github.com/modelcontextprotocol) | Implementation reference |
| 4 | DeepEval MCP docs (confident-ai.com) | Competitor analysis |
| 5 | Trail of Bits "Breaking MCP" (Mar 2025) | Security attack surfaces |
| 6 | OWASP LLM Top 10 2025 (LLM08) | Security testing scope |
| 7 | Invariant Labs MCP Whitepaper (Feb 2025) | Prompt injection via tools |
| 8 | JSON Schema draft-07 (json-schema.org) | Conformance validation spec |
| 9 | jsonschema (python-jsonschema/jsonschema) | Implementation dependency |
| 10 | "MCP: Missing Manual" -- Willison (Jan 2025) | Multi-server patterns |
| 11 | "MCP vs function calling" -- W&B (Feb 2025) | Architecture comparison |
| 12 | awesome-mcp-servers (punkpeye) | 6,000+ server ecosystem |
| 13 | Anthropic MCP announcement (Nov 2024) | Protocol origin |
| 14 | Cursor MCP docs (cursor.com/docs/mcp) | Client ecosystem adoption |
| 15 | VS Code Copilot MCP docs (code.visualstudio.com) | Enterprise adoption signal |
