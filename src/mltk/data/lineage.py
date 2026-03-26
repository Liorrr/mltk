"""Data lineage tracking -- record and verify data transformation provenance."""

from __future__ import annotations

import functools
import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


@dataclass
class LineageNode:
    """A single recorded transformation step in the lineage graph."""

    name: str
    input_hash: str
    output_hash: str
    timestamp: str


class LineageGraph:
    """DAG of data transformations.

    Tracks each step as a :class:`LineageNode` containing the
    transformation name and SHA-256 hashes of inputs/outputs.

    Example:
        >>> g = LineageGraph()
        >>> g.add("clean", input_data=[1, 2], output_data=[1])
        >>> g.export()
        [{'name': 'clean', 'input_hash': '...', 'output_hash': '...', 'timestamp': '...'}]
    """

    def __init__(self) -> None:
        self.nodes: list[LineageNode] = []

    def add(self, name: str, input_data: Any, output_data: Any) -> None:
        """Record a transformation step.

        Args:
            name: Human-readable name for this transformation.
            input_data: The data (or representative value) fed into the step.
            output_data: The data (or representative value) produced by the step.
        """
        input_hash = hashlib.sha256(str(input_data).encode()).hexdigest()[:12]
        output_hash = hashlib.sha256(str(output_data).encode()).hexdigest()[:12]
        ts = datetime.now(timezone.utc).isoformat()
        self.nodes.append(
            LineageNode(
                name=name,
                input_hash=input_hash,
                output_hash=output_hash,
                timestamp=ts,
            )
        )

    def export(self) -> list[dict[str, str]]:
        """Export lineage as a JSON-serializable list of dicts."""
        return [
            {
                "name": node.name,
                "input_hash": node.input_hash,
                "output_hash": node.output_hash,
                "timestamp": node.timestamp,
            }
            for node in self.nodes
        ]


def track_lineage(graph: LineageGraph, name: str) -> Callable:
    """Decorator that records input/output hashes of a function call.

    Wraps the target function so that every invocation appends a
    :class:`LineageNode` to *graph* with SHA-256 hashes of the
    stringified arguments and return value.

    Args:
        graph: The :class:`LineageGraph` to record into.
        name: Label for this transformation step.

    Returns:
        A decorator that can be applied to any callable.

    Example:
        >>> g = LineageGraph()
        >>> @track_lineage(g, "double")
        ... def double(x):
        ...     return x * 2
        >>> double(5)
        10
        >>> len(g.nodes)
        1
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            input_hash = hashlib.sha256(str(args).encode()).hexdigest()[:12]
            result = func(*args, **kwargs)
            output_hash = hashlib.sha256(str(result).encode()).hexdigest()[:12]
            ts = datetime.now(timezone.utc).isoformat()
            graph.nodes.append(
                LineageNode(
                    name=name,
                    input_hash=input_hash,
                    output_hash=output_hash,
                    timestamp=ts,
                )
            )
            return result

        return wrapper

    return decorator


@timed_assertion
def assert_lineage_complete(
    graph: LineageGraph,
    expected_steps: int,
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
    """Assert lineage graph has the expected number of transformation steps.

    Args:
        graph: The :class:`LineageGraph` to validate.
        expected_steps: Expected number of recorded transformations.
        severity: Severity level for the assertion.

    Returns:
        :class:`~mltk.core.result.TestResult` capturing the outcome.
    """
    actual = len(graph.nodes)
    return assert_true(
        actual == expected_steps,
        name="data.lineage.complete",
        message=(
            f"Lineage has {actual} steps (expected {expected_steps})"
        ),
        severity=severity,
        actual_steps=actual,
        expected_steps=expected_steps,
    )
