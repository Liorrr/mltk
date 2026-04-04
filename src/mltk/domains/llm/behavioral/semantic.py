"""Semantic equivalence and directional expectation assertions.

Semantic equivalence (MENLI, TACL 2023): bidirectional NLI catches
contradictions that cosine similarity misses (15-30 pct more robust
than BERTScore alone).

Directional expectation (CheckList DIR): a known perturbation
should change the output in a predictable direction -- the
complement of invariance testing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult

# ---------------------------------------------------------------
# Supported methods for semantic equivalence
# ---------------------------------------------------------------

_EQUIVALENCE_METHODS = ("nli", "embedding", "token_f1")


# ---------------------------------------------------------------
# Public assertions
# ---------------------------------------------------------------


@timed_assertion
def assert_semantic_equivalence(
    text_a: str,
    text_b: str,
    method: str = "nli",
    min_score: float = 0.7,
    nli_model: str = (
        "cross-encoder/nli-deberta-v3-base"
    ),
    embedding_model: str = "all-mpnet-base-v2",
) -> TestResult:
    """Assert two texts are semantically equivalent.

    Uses bidirectional NLI (default) to detect both
    equivalence *and* contradiction -- something cosine
    similarity cannot do.  Falls back to embedding cosine
    or zero-dep token F1.

    Args:
        text_a: First text.
        text_b: Second text.
        method: One of ``"nli"`` (bidirectional entailment),
            ``"embedding"`` (cosine similarity), or
            ``"token_f1"`` (zero-dep token overlap).
        min_score: Minimum score to pass.  For ``"nli"``
            this applies to the *minimum* of forward and
            backward entailment probabilities.
        nli_model: Cross-encoder model for ``"nli"``.
        embedding_model: Sentence-transformer model for
            ``"embedding"``.

    Returns:
        TestResult with equivalence score, per-direction
        details, and contradiction flag (NLI only).

    Example:
        >>> assert_semantic_equivalence(
        ...     "The cat sat on the mat.",
        ...     "A feline was resting on the rug.",
        ...     method="token_f1", min_score=0.1,
        ... )
    """
    assertion_name = (
        "llm.behavioral.semantic_equivalence"
    )

    if method not in _EQUIVALENCE_METHODS:
        supported = ", ".join(
            f"'{m}'" for m in _EQUIVALENCE_METHODS
        )
        return assert_true(
            False,
            name=assertion_name,
            message=(
                f"Unknown method: '{method}'. "
                f"Supported: {supported}"
            ),
            severity=Severity.CRITICAL,
            method=method,
        )

    from mltk.domains.llm._backends import (
        normalize_unicode,
    )

    norm_a = normalize_unicode(text_a)
    norm_b = normalize_unicode(text_b)

    # -- NLI (bidirectional entailment) -------------------------
    if method == "nli":
        from mltk.domains.llm._backends import (
            nli_bidirectional,
        )

        result = nli_bidirectional(
            norm_a, norm_b, model_name=nli_model,
        )

        fwd: dict[str, Any] = result["forward"]  # type: ignore[assignment]
        bwd: dict[str, Any] = result["backward"]  # type: ignore[assignment]
        fwd_ent = float(fwd["entailment"])
        bwd_ent = float(bwd["entailment"])
        contradiction = bool(result["contradiction"])
        equivalent = bool(result["equivalent"])

        score = min(fwd_ent, bwd_ent)

        if contradiction:
            passed = False
            message = (
                "Semantic equivalence (nli): "
                "CONTRADICTION detected"
            )
        elif score >= min_score and equivalent:
            passed = True
            message = (
                f"Semantic equivalence (nli): "
                f"{score:.4f} >= {min_score}"
            )
        else:
            passed = False
            message = (
                f"Semantic equivalence (nli): "
                f"{score:.4f} < {min_score}"
            )

        return assert_true(
            passed,
            name=assertion_name,
            message=message,
            severity=Severity.CRITICAL,
            method=method,
            score=round(score, 4),
            min_score=min_score,
            forward_entailment=round(fwd_ent, 4),
            backward_entailment=round(bwd_ent, 4),
            contradiction=contradiction,
            equivalent=equivalent,
        )

    # -- Embedding cosine similarity ----------------------------
    if method == "embedding":
        from mltk.domains.llm._backends import (
            embedding_cosine_single,
        )

        score = embedding_cosine_single(
            norm_a, norm_b, model_name=embedding_model,
        )
        passed = score >= min_score

        message = (
            f"Semantic equivalence (embedding): "
            f"{score:.4f} >= {min_score}"
            if passed
            else f"Semantic equivalence (embedding): "
            f"{score:.4f} < {min_score}"
        )

        return assert_true(
            passed,
            name=assertion_name,
            message=message,
            severity=Severity.CRITICAL,
            method=method,
            score=round(score, 4),
            min_score=min_score,
            contradiction=False,
        )

    # -- Token F1 (zero-dep fallback) --------------------------
    from mltk.domains.llm.similarity import _token_f1

    score = _token_f1(norm_a, norm_b)
    passed = score >= min_score

    message = (
        f"Semantic equivalence (token_f1): "
        f"{score:.4f} >= {min_score}"
        if passed
        else f"Semantic equivalence (token_f1): "
        f"{score:.4f} < {min_score}"
    )

    return assert_true(
        passed,
        name=assertion_name,
        message=message,
        severity=Severity.CRITICAL,
        method=method,
        score=round(score, 4),
        min_score=min_score,
        contradiction=False,
    )


@timed_assertion
def assert_directional_expectation(
    model_fn: Callable[[str], str],
    input_text: str,
    perturbation: Callable[[str], str],
    direction_fn: Callable[[str, str], bool],
    perturbation_name: str | None = None,
) -> TestResult:
    """Assert a perturbation changes output in an expected direction.

    Implements the CheckList DIR pattern: apply a known
    perturbation to the input and verify that the output
    shifts in a predictable, testable direction.  This is
    the complement of invariance testing.

    Args:
        model_fn: ``str -> str`` callable (model under test).
        input_text: The original input prompt.
        perturbation: ``str -> str`` that modifies the input
            in a meaningful way.
        direction_fn: ``(original_out, perturbed_out) -> bool``
            that returns True when the output changed in the
            expected direction.
        perturbation_name: Human-readable label for details
            (e.g. ``"add length constraint"``).

    Returns:
        TestResult with original/perturbed outputs and the
        direction check result.

    Examples:
        Length constraint::

            assert_directional_expectation(
                my_model,
                "Explain gravity",
                perturbation=lambda t: t + " in 100 words",
                direction_fn=lambda o, p: len(p) < len(o),
                perturbation_name="add length constraint",
            )

        Sentiment shift::

            assert_directional_expectation(
                my_model,
                "Review this product",
                perturbation=lambda t: (
                    "Give a negative review: " + t
                ),
                direction_fn=lambda o, p: "bad" in p.lower(),
                perturbation_name="negative sentiment",
            )
    """
    assertion_name = (
        "llm.behavioral.directional_expectation"
    )
    label = perturbation_name or "unnamed"

    from mltk.domains.llm._backends import (
        normalize_unicode,
    )

    normalized_input = normalize_unicode(input_text)

    # Run model on original and perturbed inputs
    try:
        original_output = model_fn(normalized_input)
    except Exception as exc:  # noqa: BLE001
        return assert_true(
            False,
            name=assertion_name,
            message=(
                f"model_fn raised on original input: "
                f"{type(exc).__name__}: {exc}"
            ),
            severity=Severity.CRITICAL,
            perturbation_name=label,
            error=str(exc),
        )

    perturbed_input = perturbation(normalized_input)

    try:
        perturbed_output = model_fn(perturbed_input)
    except Exception as exc:  # noqa: BLE001
        return assert_true(
            False,
            name=assertion_name,
            message=(
                f"model_fn raised on perturbed input: "
                f"{type(exc).__name__}: {exc}"
            ),
            severity=Severity.CRITICAL,
            perturbation_name=label,
            original_output=original_output,
            perturbed_input=perturbed_input,
            error=str(exc),
        )

    # Check directional expectation
    try:
        direction_met = direction_fn(
            original_output, perturbed_output,
        )
    except Exception as exc:  # noqa: BLE001
        return assert_true(
            False,
            name=assertion_name,
            message=(
                f"direction_fn raised: "
                f"{type(exc).__name__}: {exc}"
            ),
            severity=Severity.CRITICAL,
            perturbation_name=label,
            original_output=original_output,
            perturbed_input=perturbed_input,
            perturbed_output=perturbed_output,
            error=str(exc),
        )

    passed = bool(direction_met)

    message = (
        f"Directional expectation "
        f"({label}): direction met"
        if passed
        else f"Directional expectation "
        f"({label}): direction NOT met"
    )

    return assert_true(
        passed,
        name=assertion_name,
        message=message,
        severity=Severity.CRITICAL,
        perturbation_name=label,
        original_output=original_output,
        perturbed_input=perturbed_input,
        perturbed_output=perturbed_output,
        direction_met=passed,
    )
