"""Multi-agent coordination — loop detection and handoff validation.

Multi-agent systems orchestrate several specialized agents that delegate work
to each other.  While powerful, this delegation pattern introduces failure
modes that single-agent systems never encounter:

* **Circular delegation (loops)** -- Agent A asks B, B asks C, C asks A, and
  the cycle repeats indefinitely, burning tokens without producing results.
* **Broken handoff sequences** -- A customer-support pipeline should follow
  Router -> Classifier -> Specialist -> Summarizer.  If the Summarizer hands
  back to the Router, the user waits forever.

The two assertions in this module catch these coordination failures early,
during offline evaluation rather than in production where they waste money
and frustrate users.
"""

from __future__ import annotations

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


def _find_shortest_repeating_cycle(
    names: list[str],
) -> tuple[list[str] | None, int]:
    """Find the shortest subsequence that repeats consecutively.

    Scans every candidate cycle length from 2 up to half the sequence length.
    For each candidate length, checks how many times the cycle repeats
    consecutively starting from each possible offset.  Returns the cycle
    pattern and the maximum number of consecutive repetitions found.

    Returns:
        A tuple of ``(cycle_pattern, cycle_count)``.  If no cycle of length
        >= 2 repeats more than once, returns ``(None, 0)``.
    """
    n = len(names)
    best_pattern: list[str] | None = None
    best_count = 0

    for cycle_len in range(2, n // 2 + 1):
        for start in range(n - cycle_len + 1):
            pattern = names[start : start + cycle_len]
            count = 1
            pos = start + cycle_len
            while pos + cycle_len <= n:
                if names[pos : pos + cycle_len] == pattern:
                    count += 1
                    pos += cycle_len
                else:
                    break
            # Only track patterns that actually repeat (count >= 2).
            # A single occurrence is not a cycle.
            if count >= 2 and count > best_count:
                best_count = count
                best_pattern = pattern

    return best_pattern, best_count


@timed_assertion
def assert_no_agent_loop(
    agent_names: list[str],
    max_cycles: int = 2,
) -> TestResult:
    """Assert that a multi-agent delegation sequence contains no runaway loops.

    In multi-agent orchestration, agents hand off work to each other.  A loop
    occurs when the same subsequence of agents repeats consecutively --
    for example ``["planner", "researcher", "planner", "researcher", ...]``
    contains the cycle ``["planner", "researcher"]`` repeating.  Each
    repetition wastes tokens, latency, and API cost with no forward progress.

    This assertion detects the shortest repeating unit in ``agent_names`` and
    fails if it repeats more than ``max_cycles`` times consecutively.

    A single agent appearing multiple times in a row (e.g.
    ``["summarizer", "summarizer", "summarizer"]``) is NOT considered a
    multi-agent loop because the cycle length is 1 -- only cycles of length
    2 or more are flagged.

    Args:
        agent_names: Ordered list of agent names that handled each step
            (e.g. ``["router", "classifier", "specialist", "router",
            "classifier", "specialist"]``).
        max_cycles: Maximum allowed consecutive repetitions of any cycle
            (default 2).  A value of 2 means the same cycle may appear
            twice, but three consecutive repetitions will fail.

    Returns:
        TestResult with details:

        * ``cycle_detected`` -- whether a cycle exceeding the threshold was
          found.
        * ``cycle_pattern`` -- the repeating subsequence, or ``None``.
        * ``cycle_count`` -- how many consecutive times the cycle repeated.
        * ``total_handoffs`` -- length of the agent_names sequence.
        * ``max_cycles`` -- the threshold that was applied.

    Example:
        >>> # No loop -- linear handoff
        >>> assert_no_agent_loop(["router", "classifier", "specialist"])

        >>> # Loop detected -- A->B repeats 4 times
        >>> assert_no_agent_loop(["A", "B", "A", "B", "A", "B", "A", "B"])
        Traceback (most recent call last):
            ...
        mltk.core.assertion.MltkAssertionError: ...
    """
    cycle_pattern, cycle_count = _find_shortest_repeating_cycle(agent_names)
    cycle_detected = cycle_count > max_cycles

    total_handoffs = len(agent_names)

    if not cycle_detected:
        if cycle_pattern is not None:
            message = (
                f"No agent loop: cycle {cycle_pattern} repeats "
                f"{cycle_count} time(s), within threshold of {max_cycles}"
            )
        else:
            message = (
                f"No agent loop: {total_handoffs} handoff(s), "
                f"no repeating cycle detected"
            )
    else:
        message = (
            f"Agent loop detected: cycle {cycle_pattern} repeats "
            f"{cycle_count} time(s), exceeding threshold of {max_cycles}"
        )

    return assert_true(
        not cycle_detected,
        name="llm.multi_agent.no_loop",
        message=message,
        severity=Severity.CRITICAL,
        cycle_detected=cycle_detected,
        cycle_pattern=cycle_pattern,
        cycle_count=cycle_count,
        total_handoffs=total_handoffs,
        max_cycles=max_cycles,
    )


@timed_assertion
def assert_agent_handoff(
    agent_names: list[str],
    expected_flow: list[str],
    strict: bool = False,
) -> TestResult:
    """Assert that agent handoffs follow an expected flow.

    Multi-agent workflows have designed handoff patterns.  A customer-support
    pipeline might be expected to go ``Router -> Classifier -> Specialist ->
    Summarizer``.  If the actual execution skips the Classifier or sends work
    backward from Summarizer to Router, the pipeline is broken.

    This assertion verifies that the actual agent sequence matches the
    expected flow:

    * **Non-strict** (default) -- ``expected_flow`` must appear as a
      *subsequence* of ``agent_names``.  Extra agents may appear between the
      expected ones, but the relative order must be preserved.  This is the
      same logic used by ``assert_tool_chain(strict_order=True)``.
    * **Strict** -- ``agent_names`` must exactly equal ``expected_flow``.

    Args:
        agent_names: Actual ordered list of agents that handled the task.
        expected_flow: Expected ordered sequence of agent names.
        strict: If ``True``, require an exact match.  If ``False``
            (default), require subsequence match.

    Returns:
        TestResult with details:

        * ``expected_flow`` -- the expected sequence.
        * ``actual_flow`` -- the actual sequence.
        * ``missing_agents`` -- agents in expected_flow that were not found
          (in order, accounting for subsequence matching).
        * ``strict`` -- whether strict mode was used.

    Example:
        >>> # Subsequence match passes
        >>> assert_agent_handoff(
        ...     agent_names=["router", "logger", "classifier", "specialist"],
        ...     expected_flow=["router", "classifier", "specialist"],
        ... )

        >>> # Strict match fails when there are extra agents
        >>> assert_agent_handoff(
        ...     agent_names=["router", "logger", "classifier"],
        ...     expected_flow=["router", "classifier"],
        ...     strict=True,
        ... )
        Traceback (most recent call last):
            ...
        mltk.core.assertion.MltkAssertionError: ...
    """
    if strict:
        passed = agent_names == expected_flow
        # In strict mode, missing = expected agents not in actual at all.
        actual_set = set(agent_names)
        missing = [a for a in expected_flow if a not in actual_set]
        # Also fail if lengths differ or order differs, even if all agents
        # are present.  The passed flag already covers this.
    else:
        # Subsequence check: expected_flow must appear in order within
        # agent_names.  Agents not matched are "missing".
        it = iter(agent_names)
        missing = []
        for agent in expected_flow:
            found = False
            for actual in it:
                if actual == agent:
                    found = True
                    break
            if not found:
                missing.append(agent)
        passed = len(missing) == 0

    mode = "strict" if strict else "subsequence"

    if passed:
        message = (
            f"Agent handoff correct ({mode}): all "
            f"{len(expected_flow)} expected agent(s) found"
        )
    else:
        message = (
            f"Agent handoff incorrect ({mode}): "
            f"missing {missing} from actual flow {agent_names}"
        )

    return assert_true(
        passed,
        name="llm.multi_agent.handoff",
        message=message,
        severity=Severity.CRITICAL,
        expected_flow=expected_flow,
        actual_flow=agent_names,
        missing_agents=missing,
        strict=strict,
    )
