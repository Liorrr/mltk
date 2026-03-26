"""Tests for mltk.data.lineage -- data lineage tracking and provenance.

Each test follows the pattern:
  # SCENARIO: <what situation is being tested>
  # WHY: <why this matters / what could go wrong>
  # EXPECTED: <the concrete assertion>
"""

from __future__ import annotations

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.data.lineage import (
    LineageGraph,
    LineageNode,
    assert_lineage_complete,
    track_lineage,
)

# ---------------------------------------------------------------------------
# Test 1 — track_lineage decorator
# ---------------------------------------------------------------------------

def test_track_lineage_decorator() -> None:
    # SCENARIO: A function is decorated with @track_lineage; we call it once
    # WHY: The decorator must transparently record hashes without altering
    #       the function's return value or raising errors
    # EXPECTED: The function returns normally and the graph has exactly 1 node

    graph = LineageGraph()

    @track_lineage(graph, "double")
    def double(x: int) -> int:
        return x * 2

    result = double(5)

    assert result == 10
    assert len(graph.nodes) == 1
    assert graph.nodes[0].name == "double"
    assert len(graph.nodes[0].input_hash) == 12
    assert len(graph.nodes[0].output_hash) == 12
    assert graph.nodes[0].timestamp  # non-empty


# ---------------------------------------------------------------------------
# Test 2 — LineageGraph.add
# ---------------------------------------------------------------------------

def test_lineage_graph_add() -> None:
    # SCENARIO: Manually add transformation steps to a LineageGraph
    # WHY: The .add() method is the fundamental API; if hashing or node
    #       creation is broken, the entire lineage system is broken
    # EXPECTED: Each add() call appends a correctly-structured LineageNode

    graph = LineageGraph()
    graph.add("clean", input_data=[1, 2, 3], output_data=[1, 2])
    graph.add("normalize", input_data=[1, 2], output_data=[0.5, 1.0])

    assert len(graph.nodes) == 2
    assert graph.nodes[0].name == "clean"
    assert graph.nodes[1].name == "normalize"

    # Hashes are 12-char hex substrings
    for node in graph.nodes:
        assert isinstance(node, LineageNode)
        assert len(node.input_hash) == 12
        assert len(node.output_hash) == 12

    # Different inputs produce different hashes
    assert graph.nodes[0].input_hash != graph.nodes[1].input_hash


# ---------------------------------------------------------------------------
# Test 3 — LineageGraph.export
# ---------------------------------------------------------------------------

def test_lineage_export() -> None:
    # SCENARIO: Export the lineage as JSON-serializable dicts
    # WHY: Lineage must be exportable for storage, audit logs, and
    #       downstream tooling; broken export = no observability
    # EXPECTED: export() returns a list of dicts with the expected keys

    graph = LineageGraph()
    graph.add("step_a", input_data="raw", output_data="processed")

    exported = graph.export()

    assert isinstance(exported, list)
    assert len(exported) == 1
    entry = exported[0]
    assert set(entry.keys()) == {"name", "input_hash", "output_hash", "timestamp"}
    assert entry["name"] == "step_a"
    assert isinstance(entry["input_hash"], str)
    assert isinstance(entry["output_hash"], str)
    assert isinstance(entry["timestamp"], str)


# ---------------------------------------------------------------------------
# Test 4 — assert_lineage_complete PASS
# ---------------------------------------------------------------------------

def test_assert_lineage_complete_pass() -> None:
    # SCENARIO: Graph has exactly the expected number of steps
    # WHY: The assertion is the contract enforcement layer — if it gives
    #       false negatives, incomplete pipelines pass silently
    # EXPECTED: TestResult.passed is True, message mentions correct count

    graph = LineageGraph()
    graph.add("load", "file.csv", [1, 2, 3])
    graph.add("transform", [1, 2, 3], [10, 20, 30])

    result = assert_lineage_complete(graph, expected_steps=2)

    assert result.passed is True
    assert "2" in result.message
    assert result.duration_ms >= 0


# ---------------------------------------------------------------------------
# Test 5 — assert_lineage_complete FAIL
# ---------------------------------------------------------------------------

def test_assert_lineage_complete_fail() -> None:
    # SCENARIO: Graph has fewer steps than expected (a transformation was
    #       skipped or silently failed)
    # WHY: Missing steps mean data flowed through an incomplete pipeline;
    #       the assertion must raise to block bad data from reaching training
    # EXPECTED: MltkAssertionError is raised with severity CRITICAL

    graph = LineageGraph()
    graph.add("load", "file.csv", [1, 2, 3])
    # Missing "transform" step

    with pytest.raises(MltkAssertionError) as exc:
        assert_lineage_complete(graph, expected_steps=3)

    assert exc.value.result.passed is False
    assert "1" in exc.value.result.message  # actual count
    assert "3" in exc.value.result.message  # expected count


# ---------------------------------------------------------------------------
# Test 6 — empty graph
# ---------------------------------------------------------------------------

def test_empty_graph() -> None:
    # SCENARIO: A brand-new LineageGraph with no recorded steps
    # WHY: Edge case — calling export on an empty graph must return an empty
    #       list (not crash), and assert_lineage_complete with 0 should pass
    # EXPECTED: export() returns [], assert_lineage_complete(0) passes

    graph = LineageGraph()

    assert graph.nodes == []
    assert graph.export() == []

    result = assert_lineage_complete(graph, expected_steps=0)
    assert result.passed is True


# ---------------------------------------------------------------------------
# Test 7 — decorator preserves function metadata
# ---------------------------------------------------------------------------

def test_decorator_preserves_metadata() -> None:
    # SCENARIO: The @track_lineage decorator is applied to a function
    # WHY: functools.wraps must be used so that introspection tools
    #       (pytest, IDEs, docs) still see the original function name/docs
    # EXPECTED: __name__ and __doc__ are preserved

    graph = LineageGraph()

    @track_lineage(graph, "add_one")
    def add_one(x: int) -> int:
        """Add one to x."""
        return x + 1

    assert add_one.__name__ == "add_one"
    assert add_one.__doc__ == "Add one to x."


# ---------------------------------------------------------------------------
# Test 8 — multiple decorator calls accumulate
# ---------------------------------------------------------------------------

def test_multiple_calls_accumulate() -> None:
    # SCENARIO: A decorated function is called multiple times
    # WHY: Each call should create a separate lineage node; if nodes
    #       are overwritten instead of appended, history is lost
    # EXPECTED: graph has 3 nodes after 3 calls

    graph = LineageGraph()

    @track_lineage(graph, "inc")
    def inc(x: int) -> int:
        return x + 1

    inc(1)
    inc(2)
    inc(3)

    assert len(graph.nodes) == 3
    assert all(n.name == "inc" for n in graph.nodes)
