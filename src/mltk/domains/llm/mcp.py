"""MCP evaluation — tool schema conformance, resource access, context window.

MCP (Model Context Protocol) is the emerging standard for connecting LLM
agents to external tools, resources, and data sources.  Think of it as
USB-C for AI: any MCP-compliant server can connect to any MCP-compliant
client.

**Why MCP testing matters:** Agents regularly make schema-invalid tool
calls (wrong types, missing required fields), over-fetch resources, and
retry endlessly on errors.  These are testable, catchable bugs that
current agentic evaluation frameworks largely ignore.

This module provides five assertions that validate MCP-specific behavior
on top of the existing :class:`AgentTrace` infrastructure:

1. **Schema conformance** -- validates tool args against JSON Schema
   (the single biggest gap vs. competitors like DeepEval, which require
   manually specifying expected arg values instead of using the schema
   the server already declares).
2. **Tool selection** -- server-namespace-aware tool matching
   (``"filesystem::read_file"`` routes to the correct server).
3. **Resource access** -- expected/forbidden URI enforcement and read
   limits.
4. **Context window** -- model-aware utilization percentage reporting.
5. **Error recovery** -- detects same-tool-same-args retry loops.

All assertions are offline (no LLM required), deterministic, and
backward-compatible with the base :class:`AgentTrace` dataclass.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult
from mltk.domains.llm.trace import AgentTrace, ToolCall

__all__ = [
    "McpToolCall",
    "McpResourceAccess",
    "McpTrace",
    "assert_mcp_tool_schema_conformance",
    "assert_mcp_tool_selection",
    "assert_mcp_resource_access",
    "assert_mcp_context_window",
    "assert_mcp_error_recovery",
]

# ------------------------------------------------------------------
# Dataclasses
# ------------------------------------------------------------------


@dataclass
class McpToolCall(ToolCall):
    """MCP-specific tool call with server namespace and schema.

    Extends :class:`ToolCall` with fields unique to MCP sessions: the
    originating server name, the tool's declared ``inputSchema``, and a
    running token count at the point this call was made.

    Attributes:
        server: MCP server name (e.g., ``"filesystem"``).
        schema: The tool's ``inputSchema`` from ``tools/list``.
        context_tokens: Running token count at this step.
    """

    server: str = ""
    schema: dict[str, Any] = field(default_factory=dict)
    context_tokens: int = 0


@dataclass
class McpResourceAccess:
    """A resource read within an MCP session.

    MCP Resources are read-only URIs (files, database rows, API
    responses).  Unlike tool calls, resource reads do not invoke
    server-side logic -- they fetch data that the agent then
    incorporates into its context window.

    Attributes:
        uri: Resource URI (e.g., ``"file:///workspace/README.md"``).
        server: Which MCP server provided this resource.
        content_tokens: Tokens consumed by the resource content.
        result: Resource content, if available.
        error: Error message if the read failed.
        duration_ms: Read duration in milliseconds.
    """

    uri: str = ""
    server: str = ""
    content_tokens: int = 0
    result: str | None = None
    error: str | None = None
    duration_ms: float = 0.0


@dataclass
class McpTrace(AgentTrace):
    """MCP-aware agent trace extending :class:`AgentTrace`.

    Adds resource access tracking and a declared model context limit.
    Because ``McpTrace`` subclasses ``AgentTrace``, all existing
    agentic assertions (``assert_tool_chain``, ``assert_cost_budget``,
    etc.) continue to work unchanged.

    Attributes:
        resource_accesses: Ordered list of resource reads in the
            session.
        model_context_limit: Declared context window size for the
            model (e.g., 200000 for Claude, 128000 for GPT-4o).
    """

    resource_accesses: list[McpResourceAccess] = field(
        default_factory=list,
    )
    model_context_limit: int = 0

    @property
    def tool_names(self) -> list[str]:
        """Tool names with server prefix when present.

        For :class:`McpToolCall` instances with a non-empty
        ``server``, the name is returned as ``"server::tool"``.
        Plain :class:`ToolCall` instances use the bare name.
        """
        names: list[str] = []
        for tc in self.tool_calls:
            if isinstance(tc, McpToolCall) and tc.server:
                names.append(f"{tc.server}::{tc.name}")
            else:
                names.append(tc.name)
        return names


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _parse_server_tool(name: str) -> tuple[str, str]:
    """Split a namespaced tool name into ``(server, tool)``.

    Handles both namespaced (``"filesystem::read_file"``) and plain
    (``"search"``) formats.

    Args:
        name: Tool name, optionally prefixed with server namespace.

    Returns:
        Tuple of ``(server, tool_name)``.  Server is ``""`` when
        the name has no ``"::"`` separator.

    Example:
        >>> _parse_server_tool("filesystem::read_file")
        ('filesystem', 'read_file')
        >>> _parse_server_tool("search")
        ('', 'search')
    """
    if "::" in name:
        parts = name.split("::", 1)
        return (parts[0], parts[1])
    return ("", name)


def _tools_match(expected: str, actual: str) -> bool:
    """Check whether an expected tool spec matches an actual tool name.

    Namespaced expected (``"fs::read"``) requires both server and name
    to match.  Plain expected (``"read"``) matches by name only.
    """
    exp_server, exp_name = _parse_server_tool(expected)
    act_server, act_name = _parse_server_tool(actual)
    if exp_server:
        return exp_server == act_server and exp_name == act_name
    return exp_name == act_name


# ------------------------------------------------------------------
# Assertions
# ------------------------------------------------------------------


@timed_assertion
def assert_mcp_tool_schema_conformance(
    tool_schema: dict[str, Any],
    actual_args: dict[str, Any],
    tool_name: str = "",
) -> TestResult:
    """Assert that tool arguments conform to the tool's JSON Schema.

    **Why this is the biggest gap in MCP testing today:** DeepEval's
    ``ToolCallAccuracy`` requires the test author to manually specify
    every expected argument value.  Schema conformance is fundamentally
    different -- the schema IS the test.  The MCP server already
    declares what is valid via ``inputSchema``; this assertion simply
    validates the agent's arguments against that contract.

    **What it catches:**

    - ``{"limit": "ten"}`` when the schema says ``"type": "integer"``
    - Missing a ``required`` field
    - Value out of range (``"limit": 999`` when ``maximum: 100``)
    - Unknown extra field when ``additionalProperties: false``
    - Wrong nested type (``"filters": "str"`` when schema expects
      ``object``)

    Uses ``jsonschema.validate()`` under the hood.  The ``jsonschema``
    package is an optional dependency -- install with
    ``pip install mltk[mcp]`` or ``pip install jsonschema``.

    Args:
        tool_schema: The ``inputSchema`` dict from the MCP tool
            manifest (as returned by ``tools/list``).
        actual_args: The argument dict the agent passed to the tool.
        tool_name: Optional tool name for clearer error messages.

    Returns:
        TestResult with ``validation_errors``, ``tool_name``, and
        ``schema_properties`` details.

    Example:
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "query": {"type": "string"},
        ...         "limit": {"type": "integer", "minimum": 1},
        ...     },
        ...     "required": ["query"],
        ... }
        >>> assert_mcp_tool_schema_conformance(
        ...     tool_schema=schema,
        ...     actual_args={"query": "weather", "limit": 10},
        ...     tool_name="search_documents",
        ... )
    """
    try:
        import jsonschema  # type: ignore[import-untyped]
    except ImportError:
        return assert_true(
            False,
            name="llm.mcp.tool_schema_conformance",
            message=(
                "jsonschema is required for MCP schema conformance "
                "assertions. Install it with: pip install mltk[mcp]  "
                "(or: pip install jsonschema)"
            ),
            severity=Severity.CRITICAL,
            tool_name=tool_name,
            errors=[],
            import_error=True,
        )

    errors: list[str] = []

    try:
        jsonschema.validate(
            instance=actual_args, schema=tool_schema,
        )
    except jsonschema.ValidationError as exc:
        errors.append(exc.message)
    except jsonschema.SchemaError as exc:
        errors.append(
            f"Schema itself is malformed: {exc.message}"
        )

    passed = len(errors) == 0
    label = f" for '{tool_name}'" if tool_name else ""
    schema_props = list(
        tool_schema.get("properties", {}).keys()
    )

    if passed:
        message = (
            f"Schema conformance OK{label}: args validated "
            f"against {len(schema_props)} schema properties"
        )
    else:
        message = (
            f"Schema conformance FAILED{label}: "
            + "; ".join(errors)
        )

    return assert_true(
        passed,
        name="llm.mcp.tool_schema_conformance",
        message=message,
        severity=Severity.CRITICAL,
        errors=errors,
        tool_name=tool_name,
        schema_properties=schema_props,
    )


@timed_assertion
def assert_mcp_tool_selection(
    trace: McpTrace,
    expected_tools: Sequence[str],
    server: str | None = None,
) -> TestResult:
    """Assert that an MCP trace contains the expected tools.

    Extends the base ``assert_tool_selection`` concept with MCP
    server-namespace awareness.  Expected tool names can use the
    ``"server::tool"`` format (e.g., ``"filesystem::read_file"``),
    which matches only calls where *both* server and tool name match.
    Plain names (e.g., ``"search"``) match by tool name alone.

    When ``server`` is provided, only tool calls from that server are
    considered -- calls to other servers are filtered out before
    comparison.

    **Why server namespaces matter:** Two MCP servers can expose tools
    with similar names (e.g., ``filesystem::read`` vs
    ``database::read``).  Without namespace awareness, a test cannot
    distinguish which server the agent intended to route to.

    Args:
        trace: The MCP agent trace to validate.
        expected_tools: Tool names that should appear in the trace.
            Use ``"server::tool"`` for namespace-specific matching.
        server: If set, only consider calls from this server.

    Returns:
        TestResult with ``precision``, ``recall``, ``f1``,
        ``missing_tools``, ``extra_tools``, and ``actual_tools``.

    Example:
        >>> trace = McpTrace(tool_calls=[
        ...     McpToolCall(
        ...         name="read_file", server="filesystem",
        ...         arguments={"path": "/data.csv"},
        ...     ),
        ... ])
        >>> assert_mcp_tool_selection(
        ...     trace,
        ...     expected_tools=["filesystem::read_file"],
        ... )
    """
    # Build actual tool set from trace, respecting server filter.
    actual_qualified: list[str] = []
    for tc in trace.tool_calls:
        tc_server = getattr(tc, "server", "")
        if server is not None and tc_server != server:
            continue
        if tc_server:
            actual_qualified.append(
                f"{tc_server}::{tc.name}"
            )
        else:
            actual_qualified.append(tc.name)

    actual_set = set(actual_qualified)
    expected_set = set(expected_tools)

    # Match expected tools against actual set using namespace rules.
    expected_matched: set[str] = set()
    for exp in expected_tools:
        for act in actual_set:
            if _tools_match(exp, act):
                expected_matched.add(exp)
                break

    missing = sorted(expected_set - expected_matched)

    # Compute extra: actual tools not matched by any expected tool.
    matched_actuals: set[str] = set()
    for act in actual_set:
        for exp in expected_set:
            if _tools_match(exp, act):
                matched_actuals.add(act)
                break
    extra = sorted(actual_set - matched_actuals)

    true_positives = len(expected_matched)
    precision = (
        true_positives / len(actual_set) if actual_set
        else (1.0 if not expected_set else 0.0)
    )
    recall = (
        true_positives / len(expected_set) if expected_set
        else 1.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    passed = not missing and not extra

    if passed:
        message = (
            f"MCP tool selection correct: all "
            f"{len(expected_set)} expected tools found "
            f"(precision={precision:.4f}, "
            f"recall={recall:.4f}, f1={f1:.4f})"
        )
    else:
        parts: list[str] = []
        if missing:
            parts.append(f"missing={missing}")
        if extra:
            parts.append(f"extra={extra}")
        message = (
            "MCP tool selection incorrect: "
            + ", ".join(parts)
            + f" (precision={precision:.4f}, "
            f"recall={recall:.4f}, f1={f1:.4f})"
        )

    return assert_true(
        passed,
        name="llm.mcp.tool_selection",
        message=message,
        severity=Severity.CRITICAL,
        precision=precision,
        recall=recall,
        f1=f1,
        missing_tools=missing,
        extra_tools=extra,
        actual_tools=sorted(actual_set),
        expected_count=len(expected_set),
        actual_count=len(actual_set),
    )


@timed_assertion
def assert_mcp_resource_access(
    trace: McpTrace,
    expected_uris: Sequence[str] | None = None,
    forbidden_uris: Sequence[str] | None = None,
    max_reads: int | None = None,
) -> TestResult:
    """Assert correct resource access patterns in an MCP session.

    MCP Resources are read-only URIs -- files, database rows, API
    responses.  Agents that over-fetch resources waste context window
    budget and may access sensitive data they should not see.

    This assertion checks three independent constraints:

    - **expected_uris**: Every listed URI must have been accessed.
    - **forbidden_uris**: None of these URIs may have been accessed.
    - **max_reads**: Total resource read count must not exceed this.

    At least one constraint must be provided.

    **What it catches:**

    - Agent reads ``file:///etc/passwd`` (forbidden path).
    - Agent accesses 50 resources when the task needed 2 (over-fetch).
    - Agent never reads the reference document it was supposed to use.

    Args:
        trace: The MCP agent trace to validate.
        expected_uris: URIs that MUST be accessed (all required).
        forbidden_uris: URIs that MUST NOT be accessed.
        max_reads: Maximum total resource reads allowed.

    Returns:
        TestResult with ``missing_uris``, ``forbidden_accessed``,
        ``total_reads``, ``max_reads``, and ``accessed_uris``.

    Example:
        >>> trace = McpTrace(resource_accesses=[
        ...     McpResourceAccess(uri="file:///workspace/data.csv"),
        ... ])
        >>> assert_mcp_resource_access(
        ...     trace,
        ...     expected_uris=["file:///workspace/data.csv"],
        ...     forbidden_uris=["file:///etc/passwd"],
        ... )
    """
    if (
        expected_uris is None
        and forbidden_uris is None
        and max_reads is None
    ):
        raise ValueError(
            "At least one constraint required: "
            "expected_uris, forbidden_uris, or max_reads. "
            "Example: assert_mcp_resource_access(trace, "
            "expected_uris=['file:///data.csv'])"
        )

    accessed_uris = {ra.uri for ra in trace.resource_accesses}
    total_reads = len(trace.resource_accesses)
    errors: list[str] = []

    # Check expected URIs.
    missing_uris: list[str] = []
    if expected_uris is not None:
        expected_set = set(expected_uris)
        missing_uris = sorted(expected_set - accessed_uris)
        if missing_uris:
            errors.append(f"missing={missing_uris}")

    # Check forbidden URIs.
    forbidden_accessed: list[str] = []
    if forbidden_uris is not None:
        forbidden_set = set(forbidden_uris)
        forbidden_accessed = sorted(
            accessed_uris & forbidden_set
        )
        if forbidden_accessed:
            errors.append(
                f"forbidden_accessed={forbidden_accessed}"
            )

    # Check max reads.
    reads_exceeded = False
    if max_reads is not None:
        reads_exceeded = total_reads > max_reads
        if reads_exceeded:
            errors.append(
                f"total_reads={total_reads} > "
                f"max={max_reads}"
            )

    passed = len(errors) == 0

    if passed:
        message = (
            f"Resource access OK: {total_reads} read(s), "
            f"{len(accessed_uris)} unique URI(s)"
        )
    else:
        message = (
            "Resource access violation: "
            + "; ".join(errors)
        )

    return assert_true(
        passed,
        name="llm.mcp.resource_access",
        message=message,
        severity=Severity.CRITICAL,
        missing_uris=missing_uris,
        forbidden_accessed=forbidden_accessed,
        total_reads=total_reads,
        max_reads=max_reads,
        accessed_uris=sorted(accessed_uris),
        errors=errors,
    )


@timed_assertion
def assert_mcp_context_window(
    trace: McpTrace,
    model_context_limit: int | None = None,
    max_utilization: float = 0.9,
) -> TestResult:
    """Assert that context window utilization stays within budget.

    MCP sessions accumulate context: tool results, resource content,
    conversation history.  Agents must track their total token
    consumption and avoid exceeding the model's declared context
    limit.

    Unlike ``assert_cost_budget`` (which checks arbitrary token caps),
    this assertion is **model-aware**: it takes the model's declared
    context window as the reference and reports utilization as a
    percentage.  ``"82% utilized"`` is immediately actionable;
    ``"103,421 tokens"`` is not.

    The limit is taken from ``model_context_limit`` if provided, or
    falls back to ``trace.model_context_limit``.

    Args:
        trace: The MCP agent trace to validate.
        model_context_limit: Declared model context window size
            (e.g., 200000 for Claude, 128000 for GPT-4o).  Falls
            back to ``trace.model_context_limit`` if ``None``.
        max_utilization: Maximum acceptable utilization ratio
            (default 0.9 = 90%).

    Returns:
        TestResult with ``utilization``, ``total_tokens``,
        ``context_limit``, and ``max_utilization`` details.

    Example:
        >>> trace = McpTrace(
        ...     total_tokens=90000,
        ...     model_context_limit=200000,
        ... )
        >>> assert_mcp_context_window(trace, max_utilization=0.9)
    """
    limit = model_context_limit or trace.model_context_limit
    if limit <= 0:
        raise ValueError(
            "model_context_limit must be > 0. "
            "Provide it as a parameter (e.g., model_context_limit=200000 "
            "for Claude, 128000 for GPT-4o) or set "
            "trace.model_context_limit before calling."
        )

    utilization = trace.total_tokens / limit
    passed = utilization <= max_utilization

    pct = utilization * 100
    max_pct = max_utilization * 100

    if passed:
        message = (
            f"Context window OK: {pct:.1f}% utilized "
            f"({trace.total_tokens}/{limit} tokens, "
            f"max {max_pct:.0f}%)"
        )
    else:
        message = (
            f"Context window exceeded: {pct:.1f}% utilized "
            f"({trace.total_tokens}/{limit} tokens, "
            f"max {max_pct:.0f}%)"
        )

    return assert_true(
        passed,
        name="llm.mcp.context_window",
        message=message,
        severity=Severity.CRITICAL,
        utilization=utilization,
        total_tokens=trace.total_tokens,
        context_limit=limit,
        max_utilization=max_utilization,
    )


@timed_assertion
def assert_mcp_error_recovery(
    trace: McpTrace,
    max_same_tool_retries: int = 3,
) -> TestResult:
    """Assert the agent does not retry the same tool with same args.

    A well-designed agent recovers from errors by changing strategy:
    different arguments, a different tool, or a graceful fallback.
    A degenerate agent hammers the same failing tool with the same
    arguments in a loop -- wasting tokens, burning API budget, and
    making no forward progress.

    This assertion specifically detects **same-tool, same-args** retry
    loops.  It is more targeted than ``assert_error_recovery`` (which
    counts any consecutive error streak regardless of tool identity)
    and ``assert_no_redundant_calls`` (which counts consecutive calls
    to the same tool regardless of error status).

    **Why this matters for MCP:** MCP servers return ``isError: true``
    in tool responses.  An agent that receives an error should change
    strategy, not blindly retry.  Common failure: the agent retries
    ``read_file`` with the same non-existent path five times.

    Args:
        trace: The MCP agent trace to validate.
        max_same_tool_retries: Maximum allowed consecutive retries
            of the same tool with identical arguments after an error
            (default 3).

    Returns:
        TestResult with ``retry_loops`` (list of dicts describing
        each detected loop), ``max_retries_seen``, and
        ``max_same_tool_retries``.

    Example:
        >>> trace = McpTrace(tool_calls=[
        ...     McpToolCall(
        ...         name="read_file",
        ...         arguments={"path": "/missing.txt"},
        ...         error="File not found",
        ...     ),
        ...     McpToolCall(
        ...         name="read_file",
        ...         arguments={"path": "/missing.txt"},
        ...         error="File not found",
        ...     ),
        ...     McpToolCall(
        ...         name="search",
        ...         arguments={"q": "alternative"},
        ...         result="found it",
        ...     ),
        ... ])
        >>> assert_mcp_error_recovery(
        ...     trace, max_same_tool_retries=3,
        ... )
    """
    retry_loops: list[dict[str, Any]] = []
    max_retries_seen = 0

    calls = trace.tool_calls
    i = 0
    while i < len(calls):
        tc = calls[i]
        if tc.error is None:
            i += 1
            continue

        # Count consecutive calls with same name + args + error.
        run_length = 1
        while i + run_length < len(calls):
            next_tc = calls[i + run_length]
            if (
                next_tc.name == tc.name
                and next_tc.arguments == tc.arguments
                and next_tc.error is not None
            ):
                run_length += 1
            else:
                break

        if run_length > max_retries_seen:
            max_retries_seen = run_length

        if run_length > max_same_tool_retries:
            retry_loops.append({
                "tool": tc.name,
                "arguments": tc.arguments,
                "retries": run_length,
                "start_index": i,
            })

        i += run_length

    passed = len(retry_loops) == 0

    if passed:
        message = (
            f"MCP error recovery OK: max same-tool retries "
            f"is {max_retries_seen} "
            f"(<= {max_same_tool_retries} limit)"
        )
    else:
        loop_summary = ", ".join(
            f"{lp['tool']}x{lp['retries']}"
            for lp in retry_loops
        )
        message = (
            f"MCP error recovery failed: retry loops "
            f"detected: {loop_summary} "
            f"(max_same_tool_retries="
            f"{max_same_tool_retries})"
        )

    return assert_true(
        passed,
        name="llm.mcp.error_recovery",
        message=message,
        severity=Severity.CRITICAL,
        retry_loops=retry_loops,
        max_retries_seen=max_retries_seen,
        max_same_tool_retries=max_same_tool_retries,
        violation_count=len(retry_loops),
    )
