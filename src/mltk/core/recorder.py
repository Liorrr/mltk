"""Context-local recording of assertion results during pytest tests."""

from __future__ import annotations

import contextvars
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mltk.core.result import TestResult


_current: contextvars.ContextVar[list[TestResult] | None] = contextvars.ContextVar(
    "mltk_assertion_results", default=None
)


def record(result: TestResult) -> None:
    """Record a result when a test has armed the recorder.

    The recorder is inert outside pytest recording. Tasks created during the
    session, including a task shared across tests by an async fixture, append to
    the same list and are captured. Results from threads spawned by a test are
    not captured because threads do not inherit context variables. Assertions
    in a session-scoped fixture land in the first test that requests it.
    Teardown-phase assertions are discarded.
    """
    results = _current.get()
    if results is not None:
        results.append(result)


def arm() -> contextvars.Token[list[TestResult] | None]:
    """Start recording into a fresh list and return the reset token."""
    return _current.set([])


def disarm(token: contextvars.Token[list[TestResult] | None]) -> None:
    """Stop recording and restore the previous context-local state."""
    _current.reset(token)


def drain() -> list[TestResult]:
    """Return recorded results and clear the current list."""
    results = _current.get()
    if results is None:
        return []
    drained = list(results)
    results.clear()
    return drained
