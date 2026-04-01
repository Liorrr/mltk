"""LLM-as-Judge default configuration and convenience wrappers.

Provides a module-level default judge function so users can configure
once and have all subjective assertions use it automatically, without
passing ``judge_fn`` to every call.  Thread-safe via a simple module
variable (CPython GIL protects single-word writes).

This is an **additive convenience layer** -- existing assertions in
``judge.py``, ``safety.py``, and ``rag.py`` keep working unchanged.
No existing modules are modified.

Usage:
    >>> from mltk.domains.llm.judge_defaults import (
    ...     configure_default_judge,
    ...     get_default_judge,
    ...     assert_with_judge,
    ... )
    >>> configure_default_judge(my_llm_fn)
    >>> result = assert_with_judge(
    ...     "faithfulness", answer="...", context="...",
    ... )
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from typing import Any

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult
from mltk.domains.llm._utils import _tokenize

__all__ = [
    "configure_default_judge",
    "get_default_judge",
    "resolve_judge",
    "assert_with_judge",
]

# -----------------------------------------------------------------
# Module-level default judge (thread-safe via lock)
# -----------------------------------------------------------------

_lock = threading.Lock()
_default_judge: Callable[[str], str] | None = None


def configure_default_judge(
    judge_fn: Callable[[str], str] | None,
) -> None:
    """Set the module-level default judge function.

    All calls to ``resolve_judge`` and ``assert_with_judge`` will
    use this function when no explicit ``judge_fn`` is provided.
    Pass ``None`` to clear the default.

    Thread-safe: protected by a module-level lock.

    Args:
        judge_fn: A callable that takes a prompt string and returns
            the judge response, or ``None`` to clear.
    """
    global _default_judge
    with _lock:
        _default_judge = judge_fn


def get_default_judge() -> Callable[[str], str] | None:
    """Retrieve the current default judge function.

    Returns:
        The configured judge callable, or ``None`` if not set.
    """
    with _lock:
        return _default_judge


# -----------------------------------------------------------------
# Judge resolution helper
# -----------------------------------------------------------------

def resolve_judge(
    explicit_judge: Callable[..., Any] | None = None,
    method: str = "auto",
    fallback_method: str = "lexical",
) -> tuple[Callable[..., Any] | None, str]:
    """Resolve which judge and method to use.

    Resolution order:

    1. If *explicit_judge* is provided, use it (method stays as-is
       unless ``"auto"`` in which case it becomes ``"llm"``).
    2. If ``method="auto"`` and no explicit judge, check the module
       default.  If set, use it with ``method="llm"``.
    3. If no judge available at all, fall back to *fallback_method*
       (typically ``"lexical"`` or ``"embedding"``).

    Args:
        explicit_judge: A judge callable passed directly by the
            caller, or ``None``.
        method: The requested evaluation method.  ``"auto"`` means
            let this function decide.
        fallback_method: Method to use when no judge is available
            and method is ``"auto"``.

    Returns:
        A ``(judge_fn_or_none, resolved_method)`` tuple.
    """
    if explicit_judge is not None:
        resolved = "llm" if method == "auto" else method
        return explicit_judge, resolved

    if method == "auto":
        default = get_default_judge()
        if default is not None:
            return default, "llm"
        return None, fallback_method

    # Explicit method requested but no judge -- pass through.
    # The caller is responsible for validating method support.
    return None, method


# -----------------------------------------------------------------
# Convenience assertion wrapper
# -----------------------------------------------------------------

_LEXICAL_ASSERTIONS = frozenset({
    "faithfulness",
    "context_relevancy",
    "answer_relevancy",
    "hallucination",
})


def _lexical_score(
    text_a: str,
    text_b: str,
) -> float:
    """Compute token overlap ratio between two texts.

    Returns the fraction of tokens in *text_a* that also appear in
    *text_b*.  Used as the lexical fallback when no judge is
    available.
    """
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if not tokens_a:
        return 1.0
    return len(tokens_a & tokens_b) / len(tokens_a)


@timed_assertion
def assert_with_judge(
    assertion_name: str,
    text_a: str,
    text_b: str,
    judge_fn: Callable[[str, str], float] | None = None,
    fallback_method: str = "lexical",
    min_score: float = 0.5,
) -> TestResult:
    """Generic judge-or-fallback assertion wrapper.

    Tries to use a judge function first; falls back to lexical
    token overlap if no judge is available.  This is a convenience
    wrapper, NOT a replacement for the specialised assertions in
    ``rag.py`` or ``safety.py``.

    Resolution order for the judge:

    1. *judge_fn* argument (explicit per-call judge).
    2. Module-level default (set via ``configure_default_judge``).
    3. Lexical token overlap fallback (or whatever
       *fallback_method* specifies).

    Args:
        assertion_name: Name for the assertion (e.g.
            ``"faithfulness"``).  Used in the ``TestResult.name``.
        text_a: Primary text (e.g. an answer or claim).
        text_b: Reference text (e.g. context or source).
        judge_fn: Optional callable ``(text_a, text_b) -> float``.
        fallback_method: Method to use when no judge is available.
            Currently only ``"lexical"`` is supported as built-in.
        min_score: Minimum score required to pass.

    Returns:
        TestResult with score, method, and min_score in details.
    """
    # Resolve which scoring approach to use
    resolved_fn: Callable[..., float] | None = judge_fn
    method = "llm" if resolved_fn is not None else "auto"

    if resolved_fn is None:
        default = get_default_judge()
        if default is not None:
            # Wrap the default (str->str) judge into a 2-arg scorer
            def _default_wrapper(a: str, b: str) -> float:
                prompt = (
                    f"Rate how well the following text is "
                    f"supported by the reference.\n\n"
                    f"Text: {a}\n\n"
                    f"Reference: {b}\n\n"
                    f"Return ONLY a score between 0.0 and 1.0."
                )
                raw = default(prompt)  # type: ignore[misc]
                match = re.search(
                    r"(\d+(?:\.\d+)?)", str(raw),
                )
                return float(match.group(1)) if match else 0.0

            resolved_fn = _default_wrapper
            method = "llm"

    if resolved_fn is not None:
        try:
            score = float(resolved_fn(text_a, text_b))
        except Exception as exc:
            return assert_true(
                False,
                name=f"llm.judge_defaults.{assertion_name}",
                message=(
                    f"Judge error: {type(exc).__name__}: {exc}"
                ),
                severity=Severity.CRITICAL,
                method="llm",
                error=str(exc),
            )
        method = "llm"
    else:
        # Fallback to lexical
        if fallback_method != "lexical":
            return assert_true(
                False,
                name=f"llm.judge_defaults.{assertion_name}",
                message=(
                    f"No judge available and fallback method "
                    f"'{fallback_method}' requires one."
                ),
                severity=Severity.CRITICAL,
                method=fallback_method,
            )
        score = _lexical_score(text_a, text_b)
        method = "lexical"

    passed = score >= min_score
    cmp = ">=" if passed else "<"
    message = (
        f"{assertion_name} ({method}): "
        f"{score:.4f} {cmp} {min_score}"
    )

    return assert_true(
        passed,
        name=f"llm.judge_defaults.{assertion_name}",
        message=message,
        severity=Severity.CRITICAL,
        score=round(score, 4),
        min_score=min_score,
        method=method,
    )
