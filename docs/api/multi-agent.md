# Multi-Agent Testing

Test coordination, delegation, and loop detection in multi-agent AI systems.

**Module:** `mltk.domains.llm.multi_agent`

---

## Why Test Multi-Agent Systems?

Multi-agent orchestration has become the dominant pattern for complex AI applications in 2025-2026. Frameworks like LangGraph, CrewAI, AutoGen, and Semantic Kernel all build on the same idea: split a problem across specialized agents that delegate, collaborate, and hand off work to each other. A router agent triages incoming requests, a classifier agent determines intent, a specialist agent performs domain-specific work, and a summarizer agent compiles the final output.

This architecture is powerful, but it introduces an entirely new class of bugs that do not exist in single-agent systems. These bugs are coordination failures:

- **Infinite delegation loops.** Agent A delegates to Agent B, which delegates to Agent C, which delegates back to Agent A. The cycle repeats until the token budget is exhausted. No agent crashes. No error is raised. The system just spins, producing no useful output while accumulating cost.

- **Wrong handoff order.** The specialist agent fires before the classifier has determined intent. The summarizer runs before any work has been done. The output looks plausible but is based on incomplete or unprocessed input.

- **Missing agents in the chain.** The security-review agent was supposed to run between the code-generator and the deployment agent, but the orchestrator skipped it. The code ships without review.

- **Silent agent substitution.** The orchestrator routes to the wrong specialist. A medical-knowledge agent handles a legal question. The response is fluent and confident, but factually wrong in a domain-specific way that no downstream agent can catch.

These bugs do not throw exceptions. They do not appear in error logs. They produce output that looks correct to a casual observer. The only way to catch them is to test the coordination layer explicitly: verify that agents are called in the right order, that no cycles exist, and that every required agent participates in the workflow.

---

## assert_no_agent_loop

Multi-agent systems can form circular delegation patterns. Agent A determines it needs help from Agent B. Agent B analyzes the request and decides Agent C is better suited. Agent C examines the task and routes it back to Agent A. This cycle -- `A -> B -> C -> A -> B -> C -> ...` -- repeats indefinitely.

The insidious part is that each individual handoff decision is locally rational. Agent A genuinely does not know how to handle the task. Agent B makes a reasonable routing decision. Agent C's delegation back to Agent A follows the same logic. No single agent is wrong, but the system as a whole is stuck.

Here is a visual comparison:

```
Healthy chain:       Router -> Classifier -> Specialist -> Summarizer  (done)

Delegation loop:     Router -> Classifier -> Specialist -> Router -> Classifier -> Specialist -> ...  (never terminates)
```

`assert_no_agent_loop` takes the sequence of agent names (in execution order) and checks for repeating cycles. A cycle is defined as any subsequence of agents that appears more than `max_cycles` times consecutively. Even a two-agent ping-pong (`A -> B -> A -> B -> A -> B`) is detected.

```python
from mltk.domains.llm.multi_agent import assert_no_agent_loop

# A healthy pipeline with no loops
agents = ["router", "classifier", "specialist", "summarizer"]
result = assert_no_agent_loop(agents, max_cycles=2)
assert result.passed is True
assert result.details["cycle_detected"] is False

# A delegation loop: router -> classifier -> specialist repeats 4 times
looping_agents = [
    "router", "classifier", "specialist",
    "router", "classifier", "specialist",
    "router", "classifier", "specialist",
    "router", "classifier", "specialist",
]

result = assert_no_agent_loop(looping_agents, max_cycles=2)
assert result.passed is False
assert result.details["cycle_detected"] is True
assert result.details["cycle_pattern"] == ["router", "classifier", "specialist"]
assert result.details["cycle_count"] == 4

# Two-agent ping-pong: planner and executor bounce back and forth
pingpong = ["planner", "executor", "planner", "executor", "planner", "executor"]
result = assert_no_agent_loop(pingpong, max_cycles=2)
assert result.passed is False
assert result.details["cycle_pattern"] == ["planner", "executor"]
assert result.details["cycle_count"] == 3
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `agent_names` | `list[str]` | *(required)* | Ordered list of agent names as they were invoked during execution |
| `max_cycles` | `int` | `2` | Maximum allowed repetitions of any cyclic pattern. A value of `2` means the same subsequence can appear at most twice before failing. |

#### Returns

`TestResult` with details:

- `cycle_detected` -- `True` if a repeating cycle was found that exceeds `max_cycles`
- `cycle_pattern` -- the repeating subsequence of agent names (e.g., `["router", "classifier", "specialist"]`), or `None` if no cycle was detected
- `cycle_count` -- number of times the cycle pattern repeated, or `0` if no cycle
- `agent_count` -- total number of agent invocations in the input
- `max_cycles` -- the threshold that was enforced

---

## assert_agent_handoff

Even when agents do not loop, they can fire in the wrong order. A multi-agent system typically has an expected workflow: the router runs first, then the classifier, then the specialist, then the summarizer. If the specialist runs before the classifier, the specialist is operating on unclassified input and may produce incorrect results. If the summarizer runs before the specialist, it is summarizing nothing.

`assert_agent_handoff` verifies that the actual sequence of agent invocations matches an expected workflow. It supports two modes:

- **Strict mode** (`strict=True`): The expected flow must match the actual flow exactly, in order, with no extra agents and no missing agents. This is appropriate for deterministic pipelines where every agent must run exactly once in a fixed order.

- **Relaxed mode** (`strict=False`): The expected agents must appear in the actual flow in the correct relative order, but other agents may appear between them. This is appropriate for systems where optional agents (logging, monitoring, caching) may interleave with the core workflow.

```python
from mltk.domains.llm.multi_agent import assert_agent_handoff

# --- Strict mode: exact match required ---

actual = ["router", "classifier", "specialist", "summarizer"]
expected = ["router", "classifier", "specialist", "summarizer"]

result = assert_agent_handoff(actual, expected_flow=expected, strict=True)
assert result.passed is True

# Fail (strict): extra "logger" agent not in expected flow
actual_with_extra = ["router", "logger", "classifier", "specialist", "summarizer"]
result = assert_agent_handoff(actual_with_extra, expected_flow=expected, strict=True)
assert result.passed is False
assert result.details["actual_flow"] == actual_with_extra
assert result.details["expected_flow"] == expected

# Fail (strict): "classifier" is missing
actual_missing = ["router", "specialist", "summarizer"]
result = assert_agent_handoff(actual_missing, expected_flow=expected, strict=True)
assert result.passed is False
assert result.details["missing_agents"] == ["classifier"]


# --- Relaxed mode: order preserved, extras allowed ---

# Pass (relaxed): "logger" is extra but the expected agents appear in order
result = assert_agent_handoff(actual_with_extra, expected_flow=expected, strict=False)
assert result.passed is True

# Fail (relaxed): "specialist" appears before "classifier" -- wrong order
wrong_order = ["router", "specialist", "classifier", "summarizer"]
result = assert_agent_handoff(wrong_order, expected_flow=expected, strict=False)
assert result.passed is False

# Fail (relaxed): "classifier" never appears at all
result = assert_agent_handoff(actual_missing, expected_flow=expected, strict=False)
assert result.passed is False
assert result.details["missing_agents"] == ["classifier"]
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `agent_names` | `list[str]` | *(required)* | Ordered list of agent names as they were actually invoked |
| `expected_flow` | `list[str]` | *(required)* | The expected sequence of agent names in the correct workflow order |
| `strict` | `bool` | `True` | If `True`, the actual flow must match the expected flow exactly. If `False`, extra agents are allowed as long as the expected agents appear in order. |

#### Returns

`TestResult` with details:

- `expected_flow` -- the expected sequence that was checked against
- `actual_flow` -- the actual sequence of agent invocations
- `missing_agents` -- list of expected agents that were not found in the actual flow
- `extra_agents` -- list of agents in the actual flow that are not in the expected flow (only populated in strict mode)
- `order_correct` -- `True` if the expected agents appeared in the correct relative order

---

## Testing Patterns for Multi-Agent Systems

The two core assertions -- loop detection and handoff verification -- combine with the trace-level assertions from `mltk.domains.llm.agentic` to form comprehensive test suites for multi-agent pipelines. Below are three common patterns.

### Pattern 1: Workflow Validation

The most basic pattern: verify that your multi-agent pipeline runs the right agents in the right order. This is the multi-agent equivalent of `assert_tool_chain` for single-agent traces.

```python
import pytest
from mltk.domains.llm.multi_agent import assert_agent_handoff


def test_customer_support_workflow() -> None:
    """
    Customer support pipeline must follow:
    Router -> Intent Classifier -> Knowledge Retriever -> Response Generator
    """
    actual_agents = [
        "router",
        "intent_classifier",
        "knowledge_retriever",
        "response_generator",
    ]

    result = assert_agent_handoff(
        actual_agents,
        expected_flow=["router", "intent_classifier", "knowledge_retriever", "response_generator"],
        strict=True,
    )
    assert result.passed is True


def test_code_review_pipeline_relaxed() -> None:
    """
    Code review must include: Analyzer -> Security Scanner -> Reviewer.
    Logging and metrics agents may appear between them.
    """
    actual_agents = [
        "code_analyzer",
        "metrics_collector",      # optional, not in expected flow
        "security_scanner",
        "audit_logger",           # optional, not in expected flow
        "code_reviewer",
    ]

    result = assert_agent_handoff(
        actual_agents,
        expected_flow=["code_analyzer", "security_scanner", "code_reviewer"],
        strict=False,
    )
    assert result.passed is True
```

### Pattern 2: Loop Prevention

When agents can dynamically delegate to each other, loops are always a risk. Test that your orchestrator's loop-breaking logic works by simulating chains that approach the cycle limit without exceeding it, and chains that cross it.

```python
import pytest
from mltk.domains.llm.multi_agent import assert_no_agent_loop


def test_refinement_loop_within_budget() -> None:
    """
    A writer -> reviewer -> writer -> reviewer cycle is acceptable
    up to 2 rounds (the reviewer sends the draft back once for revision).
    """
    agents = [
        "writer",
        "reviewer",     # round 1
        "writer",
        "reviewer",     # round 2 -- max allowed
        "publisher",    # final step, breaks the cycle
    ]

    result = assert_no_agent_loop(agents, max_cycles=2)
    assert result.passed is True


def test_refinement_loop_exceeded() -> None:
    """
    If the writer-reviewer cycle exceeds 2 rounds, the orchestrator
    has failed to break the loop.
    """
    agents = [
        "writer", "reviewer",  # round 1
        "writer", "reviewer",  # round 2
        "writer", "reviewer",  # round 3 -- one too many
    ]

    result = assert_no_agent_loop(agents, max_cycles=2)
    assert result.passed is False
    assert result.details["cycle_count"] == 3


def test_three_agent_cycle_detection() -> None:
    """
    Detect a three-agent cycle: planner -> researcher -> planner -> researcher -> ...
    """
    agents = [
        "planner", "researcher", "executor",
        "planner", "researcher", "executor",
        "planner", "researcher", "executor",
    ]

    result = assert_no_agent_loop(agents, max_cycles=2)
    assert result.passed is False
    assert result.details["cycle_pattern"] == ["planner", "researcher", "executor"]
```

### Pattern 3: Handoff + Budget Combined

In production, you care about both correctness (did the right agents run in the right order?) and cost (did the whole pipeline stay within budget?). Combine `assert_agent_handoff` with `assert_cost_budget` to test both dimensions.

This pattern is especially useful in CI/CD pipelines where you run your multi-agent system against test inputs and verify that the result is both correct and economical.

```python
import pytest
from mltk.domains.llm.multi_agent import assert_agent_handoff, assert_no_agent_loop
from mltk.domains.llm.agentic import assert_cost_budget
from mltk.domains.llm.trace import AgentTrace, ToolCall


@pytest.fixture()
def research_pipeline_result():
    """
    Simulate a multi-agent research pipeline that produces
    both an agent execution log and a resource consumption trace.
    """
    agent_log = [
        "query_parser",
        "search_agent",
        "fact_checker",
        "citation_builder",
        "summary_writer",
    ]

    resource_trace = AgentTrace(
        tool_calls=[
            ToolCall(name="parse_query", arguments={"q": "renewable energy trends"}, result="parsed"),
            ToolCall(name="web_search", arguments={"q": "solar capacity 2025"}, result="12 results"),
            ToolCall(name="web_search", arguments={"q": "wind energy growth"}, result="8 results"),
            ToolCall(name="verify_claim", arguments={"claim": "solar grew 30%"}, result="confirmed"),
            ToolCall(name="verify_claim", arguments={"claim": "wind grew 15%"}, result="confirmed"),
            ToolCall(name="build_citations", arguments={"sources": ["..."]}, result="3 citations"),
            ToolCall(name="generate_summary", arguments={"data": "..."}, result="500-word summary"),
        ],
        total_tokens=28_000,
        total_duration_ms=45_000.0,
    )

    return agent_log, resource_trace


def test_research_handoff_correct(research_pipeline_result) -> None:
    """The research pipeline must follow the expected agent order."""
    agent_log, _ = research_pipeline_result

    result = assert_agent_handoff(
        agent_log,
        expected_flow=["query_parser", "search_agent", "fact_checker", "citation_builder", "summary_writer"],
        strict=True,
    )
    assert result.passed is True


def test_research_no_loops(research_pipeline_result) -> None:
    """The research pipeline must not enter any delegation loops."""
    agent_log, _ = research_pipeline_result

    result = assert_no_agent_loop(agent_log, max_cycles=2)
    assert result.passed is True


def test_research_within_budget(research_pipeline_result) -> None:
    """The entire pipeline must stay within 50K tokens and 60 seconds."""
    _, resource_trace = research_pipeline_result

    result = assert_cost_budget(
        resource_trace,
        max_total_tokens=50_000,
        max_duration_ms=60_000,
    )
    assert result.passed is True


def test_research_pipeline_full_validation(research_pipeline_result) -> None:
    """
    Combined validation: correct handoff order, no loops, within budget.
    This is the kind of test you run in CI before deploying a new
    orchestrator configuration.
    """
    agent_log, resource_trace = research_pipeline_result

    # Coordination correctness
    handoff = assert_agent_handoff(
        agent_log,
        expected_flow=["query_parser", "search_agent", "fact_checker", "citation_builder", "summary_writer"],
        strict=True,
    )
    assert handoff.passed, f"Handoff failed: {handoff.details}"

    # No infinite delegation
    loop_check = assert_no_agent_loop(agent_log, max_cycles=2)
    assert loop_check.passed, f"Loop detected: {loop_check.details}"

    # Cost containment
    budget = assert_cost_budget(
        resource_trace,
        max_total_tokens=50_000,
        max_duration_ms=60_000,
    )
    assert budget.passed, f"Budget exceeded: {budget.details}"
```

---

## Integration with pytest

A complete test file combining all multi-agent assertions for a production pipeline.

```python
import pytest
from mltk.domains.llm.multi_agent import assert_no_agent_loop, assert_agent_handoff
from mltk.domains.llm.agentic import assert_cost_budget, assert_error_recovery
from mltk.domains.llm.trace import AgentTrace, ToolCall


EXPECTED_PIPELINE = [
    "intake",
    "triage",
    "specialist",
    "quality_check",
    "delivery",
]


class TestMultiAgentPipeline:
    """Full test suite for a 5-agent processing pipeline."""

    @pytest.fixture()
    def pipeline_run(self):
        """Fixture: a single pipeline execution with agent log and trace."""
        agents = ["intake", "triage", "specialist", "quality_check", "delivery"]
        trace = AgentTrace(
            tool_calls=[
                ToolCall(name="receive_request", arguments={}, result="ticket-001"),
                ToolCall(name="classify_priority", arguments={}, result="high"),
                ToolCall(name="process_ticket", arguments={}, result="resolved"),
                ToolCall(name="run_qa_checks", arguments={}, result="all passed"),
                ToolCall(name="send_response", arguments={}, result="delivered"),
            ],
            total_tokens=18_500,
            total_duration_ms=22_000.0,
        )
        return agents, trace

    def test_correct_handoff_order(self, pipeline_run) -> None:
        agents, _ = pipeline_run
        result = assert_agent_handoff(agents, expected_flow=EXPECTED_PIPELINE, strict=True)
        assert result.passed is True

    def test_no_delegation_loops(self, pipeline_run) -> None:
        agents, _ = pipeline_run
        result = assert_no_agent_loop(agents, max_cycles=2)
        assert result.passed is True

    def test_token_budget(self, pipeline_run) -> None:
        _, trace = pipeline_run
        result = assert_cost_budget(trace, max_total_tokens=50_000)
        assert result.passed is True

    def test_duration_budget(self, pipeline_run) -> None:
        _, trace = pipeline_run
        result = assert_cost_budget(trace, max_duration_ms=60_000)
        assert result.passed is True

    def test_error_recovery(self, pipeline_run) -> None:
        _, trace = pipeline_run
        result = assert_error_recovery(trace, max_consecutive_errors=2)
        assert result.passed is True
```
