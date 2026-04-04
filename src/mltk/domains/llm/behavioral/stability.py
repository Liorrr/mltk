"""Output stability -- detect non-determinism in model outputs."""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
from typing import Any

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult
from mltk.domains.llm.similarity import _token_f1

_EQUIVALENCE_METHODS = (
    "token_f1",
    "embedding",
    "label_match",
)


def _pairwise_token_f1(
    outputs: list[str],
    threshold: float,
) -> float:
    """Fraction of output pairs above threshold (token F1)."""
    pairs = list(combinations(range(len(outputs)), 2))
    if not pairs:
        return 1.0
    above = sum(
        1
        for i, j in pairs
        if _token_f1(outputs[i], outputs[j]) >= threshold
    )
    return above / len(pairs)


def _pairwise_embedding(
    outputs: list[str],
    threshold: float,
    model_name: str,
) -> float:
    """Fraction of output pairs above threshold (embedding)."""
    from mltk.domains.llm._backends import (
        embedding_cosine_pairs,
    )

    pairs = list(combinations(range(len(outputs)), 2))
    if not pairs:
        return 1.0

    texts_a = [outputs[i] for i, _ in pairs]
    texts_b = [outputs[j] for _, j in pairs]
    scores = embedding_cosine_pairs(
        texts_a, texts_b, model_name,
    )
    above = sum(1 for s in scores if s >= threshold)
    return above / len(pairs)


def _pairwise_label_match(
    outputs: list[str],
) -> float:
    """Fraction of output pairs that are exact matches."""
    pairs = list(combinations(range(len(outputs)), 2))
    if not pairs:
        return 1.0
    matches = sum(
        1
        for i, j in pairs
        if outputs[i].strip() == outputs[j].strip()
    )
    return matches / len(pairs)


@timed_assertion
def assert_output_stability(
    model_fn: Callable[[str], Any],
    inputs: list[str],
    n_runs: int = 5,
    equivalence_method: str = "token_f1",
    min_stability: float = 0.9,
    similarity_threshold: float | None = None,
    embedding_model: str = "all-mpnet-base-v2",
) -> TestResult:
    """Assert a model produces consistent outputs across repeated runs.

    For each input, calls *model_fn* ``n_runs`` times and measures
    pairwise equivalence across all outputs.  Catches
    non-determinism issues (e.g. temperature > 0 variance,
    sampling randomness, race conditions in batched inference).

    Supports three equivalence strategies via *equivalence_method*:

    - ``"token_f1"`` (default) -- token-level F1 between each
      output pair.  Good for free-text LLMs.
    - ``"embedding"`` -- cosine similarity via
      sentence-transformers.  Good for semantic equivalence
      when wording varies.  Requires ``sentence-transformers``.
    - ``"label_match"`` -- exact string match after stripping.
      Good for deterministic classifiers or structured outputs.

    Args:
        model_fn: Callable that takes a prompt string and returns
            the model output (stringified if not already str).
        inputs: List of input prompts to test stability on.
        n_runs: Number of times to run each input (>= 2).
        equivalence_method: One of ``"token_f1"``,
            ``"embedding"``, ``"label_match"``.
        min_stability: Minimum required average stability
            across all inputs (0.0--1.0).
        similarity_threshold: Minimum pairwise score to count
            as equivalent.  Defaults to 0.8 for ``"token_f1"``
            and ``"embedding"``.  Ignored for ``"label_match"``.
        embedding_model: Sentence-transformer model name for
            ``equivalence_method="embedding"``.

    Returns:
        TestResult with stability metrics including
        ``avg_stability``, ``per_input_stability``,
        ``worst_input``, and ``method``.

    Example:
        >>> def my_model(prompt: str) -> str:
        ...     return "Paris"
        >>> assert_output_stability(
        ...     my_model,
        ...     ["What is the capital of France?"],
        ...     n_runs=3,
        ...     equivalence_method="label_match",
        ... )
    """
    name = "llm.behavioral.output_stability"

    if n_runs < 2:
        return assert_true(
            False,
            name=name,
            message="n_runs must be >= 2 for stability testing.",
            severity=Severity.CRITICAL,
        )

    if not inputs:
        return assert_true(
            False,
            name=name,
            message="inputs list is empty.",
            severity=Severity.CRITICAL,
        )

    if equivalence_method not in _EQUIVALENCE_METHODS:
        supported = ", ".join(
            f"'{m}'" for m in _EQUIVALENCE_METHODS
        )
        return assert_true(
            False,
            name=name,
            message=(
                f"Unknown equivalence_method: "
                f"'{equivalence_method}'. "
                f"Supported: {supported}"
            ),
            severity=Severity.CRITICAL,
        )

    # Default threshold for score-based methods
    threshold = similarity_threshold
    if threshold is None:
        threshold = 0.8

    per_input: list[dict[str, object]] = []
    worst_stability = 1.0
    worst_input: str | None = None

    from mltk.domains.llm._backends import normalize_unicode

    for inp in inputs:
        inp = normalize_unicode(inp)
        outputs: list[str] = []
        for _ in range(n_runs):
            try:
                out = str(model_fn(inp))
            except Exception:
                out = ""
            outputs.append(out)

        n_unique = len({o.strip() for o in outputs})

        # Compute pairwise stability
        if equivalence_method == "token_f1":
            stability = _pairwise_token_f1(
                outputs, threshold,
            )
        elif equivalence_method == "embedding":
            try:
                stability = _pairwise_embedding(
                    outputs, threshold, embedding_model,
                )
            except ImportError as exc:
                return assert_true(
                    False,
                    name=name,
                    message=str(exc),
                    severity=Severity.CRITICAL,
                    method=equivalence_method,
                )
        else:
            # label_match
            stability = _pairwise_label_match(outputs)

        per_input.append({
            "input": inp,
            "stability": round(stability, 4),
            "n_runs": n_runs,
            "n_unique_outputs": n_unique,
        })

        if stability < worst_stability:
            worst_stability = stability
            worst_input = inp

    stabilities = [
        float(p["stability"]) for p in per_input
    ]
    avg_stability = (
        sum(stabilities) / len(stabilities)
        if stabilities
        else 0.0
    )

    passed = avg_stability >= min_stability

    message = (
        f"Output stability ({equivalence_method}): "
        f"{avg_stability:.4f} >= {min_stability} "
        f"across {len(inputs)} inputs"
        if passed
        else f"Output stability too low "
        f"({equivalence_method}): "
        f"{avg_stability:.4f} < {min_stability} "
        f"(worst input: {worst_input!r})"
    )

    return assert_true(
        passed,
        name=name,
        message=message,
        severity=Severity.CRITICAL,
        avg_stability=round(avg_stability, 4),
        min_stability=min_stability,
        per_input_stability=per_input,
        worst_input=worst_input,
        method=equivalence_method,
        n_runs=n_runs,
        n_inputs=len(inputs),
    )
