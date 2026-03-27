"""Agentic evaluation — task completion, tool selection, tool call correctness."""

from __future__ import annotations

import numpy as np

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult
from mltk.domains.llm._utils import _tokenize
from mltk.domains.llm.trace import AgentTrace


def _token_overlap(a: str, b: str) -> float:
    """Jaccard-style overlap: |a ∩ b| / |a ∪ b|.

    Returns 1.0 for two empty strings, 0.0 if only one is empty.
    """
    tokens_a = _tokenize(a)
    tokens_b = _tokenize(b)

    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0

    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union


@timed_assertion
def assert_task_completion(
    expected_output: str,
    actual_output: str,
    min_score: float = 0.7,
) -> TestResult:
    """Assert agent completed the task via token overlap between expected and actual.

    Uses Jaccard similarity (|expected ∩ actual| / |expected ∪ actual|) to
    measure how much of the expected output content was produced. A high score
    indicates the agent addressed the task; a low score indicates the agent
    produced irrelevant or incomplete output.

    Args:
        expected_output: The reference output representing a complete task.
        actual_output: The output produced by the agent.
        min_score: Minimum Jaccard overlap required (default 0.7).

    Returns:
        TestResult with task completion score.

    Example:
        >>> assert_task_completion(
        ...     expected_output="Sorted list: [1, 2, 3, 4, 5]",
        ...     actual_output="The sorted list is [1, 2, 3, 4, 5].",
        ...     min_score=0.5,
        ... )
    """
    score = _token_overlap(expected_output, actual_output)
    passed = score >= min_score

    expected_tokens = _tokenize(expected_output)
    actual_tokens = _tokenize(actual_output)
    matched = len(expected_tokens & actual_tokens)

    message = (
        f"Task completion: {score:.4f} >= {min_score} "
        f"({matched} tokens in common)"
        if passed
        else f"Incomplete task: {score:.4f} < {min_score} "
        f"({matched} tokens in common)"
    )

    return assert_true(
        passed, name="llm.agentic.task_completion", message=message,
        severity=Severity.CRITICAL,
        score=score, min_score=min_score,
        expected_tokens=len(expected_tokens),
        actual_tokens=len(actual_tokens),
        common_tokens=matched,
    )


@timed_assertion
def assert_tool_selection(
    expected_tools: list[str],
    actual_tools: list[str],
) -> TestResult:
    """Assert agent selected the correct tools.

    Checks that all expected tools were called and no unexpected tools were
    used. Reports missing tools, extra tools, precision, and recall so the
    caller can understand exactly how the agent deviated from the expected
    tool-use plan.

    A test passes only when there are zero missing tools AND zero extra tools.

    Args:
        expected_tools: Tools the agent should have called.
        actual_tools: Tools the agent actually called.

    Returns:
        TestResult with precision, recall, missing and extra tool lists.

    Example:
        >>> assert_tool_selection(
        ...     expected_tools=["search", "calculator"],
        ...     actual_tools=["search", "calculator"],
        ... )
    """
    expected_set = set(expected_tools)
    actual_set = set(actual_tools)

    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)

    true_positives = len(expected_set & actual_set)
    if actual_set:
        precision = true_positives / len(actual_set)
    else:
        precision = 1.0 if not expected_set else 0.0
    recall = true_positives / len(expected_set) if expected_set else 1.0

    passed = not missing and not extra

    if passed:
        message = (
            f"Tool selection correct: all {len(expected_set)} expected tools called, "
            f"no unexpected tools (precision={precision:.4f}, recall={recall:.4f})"
        )
    else:
        parts = []
        if missing:
            parts.append(f"missing={missing}")
        if extra:
            parts.append(f"extra={extra}")
        message = (
            "Incorrect tool selection — "
            + ", ".join(parts)
            + f" (precision={precision:.4f}, recall={recall:.4f})"
        )

    return assert_true(
        passed, name="llm.agentic.tool_selection", message=message,
        severity=Severity.CRITICAL,
        precision=precision,
        recall=recall,
        missing_tools=missing,
        extra_tools=extra,
        expected_count=len(expected_set),
        actual_count=len(actual_set),
    )


@timed_assertion
def assert_tool_call_correctness(
    expected_args: dict,
    actual_args: dict,
    tolerance: float = 0.01,
) -> TestResult:
    """Assert tool was called with correct arguments.

    For numeric arguments (int, float, numpy numeric): checks the absolute
    difference is within ``tolerance``.
    For all other argument types: requires exact equality.

    All keys present in ``expected_args`` must also appear in ``actual_args``
    with correct values. Extra keys in ``actual_args`` are treated as errors.

    Args:
        expected_args: The argument dictionary the tool should have been called with.
        actual_args: The argument dictionary the tool was actually called with.
        tolerance: Absolute tolerance for numeric comparisons (default 0.01).

    Returns:
        TestResult with per-argument mismatch details.

    Example:
        >>> assert_tool_call_correctness(
        ...     expected_args={"temperature": 0.7, "model": "gpt-4"},
        ...     actual_args={"temperature": 0.705, "model": "gpt-4"},
        ...     tolerance=0.01,
        ... )
    """
    mismatches: list[str] = []
    missing_keys = sorted(set(expected_args) - set(actual_args))
    extra_keys = sorted(set(actual_args) - set(expected_args))

    for key in missing_keys:
        mismatches.append(f"missing key '{key}' (expected {expected_args[key]!r})")

    for key in extra_keys:
        mismatches.append(f"unexpected key '{key}' = {actual_args[key]!r}")

    for key in expected_args:
        if key not in actual_args:
            continue  # already reported as missing

        exp_val = expected_args[key]
        act_val = actual_args[key]

        is_numeric = isinstance(exp_val, (int, float)) or (
            hasattr(np, "floating") and isinstance(exp_val, (np.integer, np.floating))
        )

        if is_numeric:
            diff = abs(float(exp_val) - float(act_val))
            if diff > tolerance:
                mismatches.append(
                    f"'{key}': expected {exp_val} ± {tolerance}, "
                    f"got {act_val} (diff={diff:.6f})"
                )
        else:
            if exp_val != act_val:
                mismatches.append(
                    f"'{key}': expected {exp_val!r}, got {act_val!r}"
                )

    passed = not mismatches
    total_checked = len(expected_args) + len(extra_keys)

    if passed:
        message = (
            f"Tool args correct: {len(expected_args)} argument(s) verified "
            f"(numeric tolerance={tolerance})"
        )
    else:
        message = (
            f"Tool args incorrect: {len(mismatches)} mismatch(es) — "
            + "; ".join(mismatches)
        )

    return assert_true(
        passed, name="llm.agentic.tool_call_correctness", message=message,
        severity=Severity.CRITICAL,
        mismatches=mismatches,
        mismatch_count=len(mismatches),
        total_args_checked=total_checked,
        tolerance=tolerance,
    )


@timed_assertion
def assert_tool_chain(
    trace: AgentTrace,
    expected_tools: list[str],
    strict_order: bool = False,
) -> TestResult:
    """Assert that an agent trace contains the expected sequence of tool calls.

    In the default (non-strict) mode the assertion checks that every tool in
    ``expected_tools`` was called at least once, regardless of call order.  When
    ``strict_order=True`` the expected tools must appear as a subsequence of the
    actual trace — other tool calls may appear between them, but the relative
    order must match.

    Args:
        trace: The agent execution trace to validate.
        expected_tools: Tool names that should appear in the trace.
        strict_order: When True, enforce that expected tools appear in order
            as a subsequence of the actual tool calls.

    Returns:
        TestResult with expected, actual, missing, and strict_order details.

    Example:
        >>> from mltk.domains.llm.trace import AgentTrace, ToolCall
        >>> trace = AgentTrace(tool_calls=[
        ...     ToolCall(name="search", arguments={"q": "weather"}),
        ...     ToolCall(name="calculator", arguments={"expr": "2+2"}),
        ... ])
        >>> assert_tool_chain(trace, expected_tools=["search", "calculator"])
    """
    actual_tools = trace.tool_names

    if strict_order:
        # Check subsequence: expected tools must appear in order within actual.
        it = iter(actual_tools)
        missing = []
        for tool in expected_tools:
            found = False
            for actual in it:
                if actual == tool:
                    found = True
                    break
            if not found:
                missing.append(tool)
        passed = len(missing) == 0
    else:
        # Set comparison: all expected tools must appear at least once.
        actual_set = set(actual_tools)
        missing = sorted(set(expected_tools) - actual_set)
        passed = len(missing) == 0

    if passed:
        mode = "strict order" if strict_order else "unordered"
        message = (
            f"Tool chain correct ({mode}): all {len(expected_tools)} "
            f"expected tools found in trace"
        )
    else:
        mode = "strict order" if strict_order else "unordered"
        message = (
            f"Tool chain incorrect ({mode}): missing {missing} "
            f"from trace {actual_tools}"
        )

    return assert_true(
        passed, name="llm.agentic.tool_chain", message=message,
        severity=Severity.CRITICAL,
        expected=expected_tools,
        actual=actual_tools,
        missing=missing,
        strict_order=strict_order,
    )


@timed_assertion
def assert_no_forbidden_actions(
    trace: AgentTrace,
    forbidden_tools: list[str],
) -> TestResult:
    """Assert that no tool call in the trace used a forbidden tool.

    Scans every tool call in the trace and fails if any tool name matches an
    entry in ``forbidden_tools``.  Useful for safety guardrails — e.g. ensuring
    an agent never calls ``"delete_database"`` or ``"send_email"`` in a
    sandboxed evaluation.

    Args:
        trace: The agent execution trace to validate.
        forbidden_tools: Tool names that must NOT appear in the trace.

    Returns:
        TestResult with forbidden_found list and total_calls count.

    Example:
        >>> from mltk.domains.llm.trace import AgentTrace, ToolCall
        >>> trace = AgentTrace(tool_calls=[
        ...     ToolCall(name="search", arguments={"q": "weather"}),
        ... ])
        >>> assert_no_forbidden_actions(trace, forbidden_tools=["delete_database"])
    """
    forbidden_set = set(forbidden_tools)
    forbidden_found = sorted(
        {tc.name for tc in trace.tool_calls if tc.name in forbidden_set}
    )
    total_calls = trace.step_count
    passed = len(forbidden_found) == 0

    if passed:
        message = (
            f"No forbidden actions: {total_calls} tool call(s) checked, "
            f"none in {sorted(forbidden_set)}"
        )
    else:
        message = (
            f"Forbidden actions detected: {forbidden_found} "
            f"found in {total_calls} tool call(s)"
        )

    return assert_true(
        passed, name="llm.agentic.no_forbidden_actions", message=message,
        severity=Severity.CRITICAL,
        forbidden_found=forbidden_found,
        total_calls=total_calls,
    )


@timed_assertion
def assert_step_efficiency(
    trace: AgentTrace,
    max_steps: int,
) -> TestResult:
    """Assert that the agent completed its task within a step budget.

    Checks that ``trace.step_count <= max_steps``.  A "step" is one tool call
    in the trace.  Helps catch agents that loop excessively or explore
    unnecessary paths, which wastes tokens and latency in production.

    Args:
        trace: The agent execution trace to validate.
        max_steps: Maximum number of tool calls allowed.

    Returns:
        TestResult with actual_steps and max_steps details.

    Example:
        >>> from mltk.domains.llm.trace import AgentTrace, ToolCall
        >>> trace = AgentTrace(tool_calls=[
        ...     ToolCall(name="search", arguments={"q": "weather"}),
        ... ])
        >>> assert_step_efficiency(trace, max_steps=5)
    """
    actual_steps = trace.step_count
    passed = actual_steps <= max_steps

    if passed:
        message = (
            f"Step efficiency OK: {actual_steps} step(s) <= "
            f"{max_steps} max"
        )
    else:
        message = (
            f"Step efficiency exceeded: {actual_steps} step(s) > "
            f"{max_steps} max"
        )

    return assert_true(
        passed, name="llm.agentic.step_efficiency", message=message,
        severity=Severity.CRITICAL,
        actual_steps=actual_steps,
        max_steps=max_steps,
    )
