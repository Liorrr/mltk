"""Behavioral invariance assertions for LLM outputs.

Paraphrase invariance: semantically equivalent inputs should
produce semantically equivalent outputs.

Format invariance: cosmetic formatting changes (case, spacing,
punctuation) should not alter model behavior.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from itertools import combinations
from typing import Any

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult

# -----------------------------------------------------------------
# Default format transforms (zero-dep, built-in)
# -----------------------------------------------------------------

_DEFAULT_FORMAT_TRANSFORMS: list[
    tuple[str, Callable[[str], str]]
] = [
    ("lowercase", lambda t: t.lower()),
    ("uppercase", lambda t: t.upper()),
    ("strip_whitespace", lambda t: t.strip()),
    (
        "no_punctuation",
        lambda t: re.sub(r"[^\w\s]", "", t),
    ),
    (
        "double_space",
        lambda t: re.sub(r"\s+", "  ", t),
    ),
]

# Per-method default similarity thresholds
_METHOD_DEFAULTS: dict[str, float] = {
    "token_f1": 0.50,
    "embedding": 0.80,
    "judge": 0.70,
    "label_match": 1.0,
}

# Auto-method ambiguous zone boundaries
_AUTO_LO = 0.50
_AUTO_HI = 0.90

# -----------------------------------------------------------------
# Equivalence methods (supported names)
# -----------------------------------------------------------------

_EQUIVALENCE_METHODS = (
    "token_f1",
    "embedding",
    "entailment",
    "judge",
    "auto",
    "label_match",
)


# -----------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------


def _is_classifier_output(outputs: list[Any]) -> bool:
    """Detect classifier-style output (short labels).

    Heuristic: all outputs are hashable scalars AND all
    string outputs are short single tokens (no whitespace).
    This avoids false positives on generative models that
    return full sentences.
    """
    for o in outputs:
        if isinstance(o, (int, float, bool)):
            continue
        if isinstance(o, str):
            # Short, single-token labels only
            if " " in o.strip() or len(o) > 50:
                return False
            continue
        return False
    return True


def _pair_score(
    out_a: str,
    out_b: str,
    method: str,
    threshold: float,
    embedding_model: str,
    nli_model: str,
    judge_fn: Callable[[str, str], float] | None,
) -> tuple[float, bool]:
    """Score a single output pair. Returns (score, equivalent).

    For entailment the score is 1.0 if bidirectionally entailed,
    else the minimum forward/backward entailment probability.
    """
    if method == "token_f1":
        from mltk.domains.llm.similarity import _token_f1

        score = _token_f1(out_a, out_b)
        return score, score >= threshold

    if method == "embedding":
        from mltk.domains.llm._backends import (
            embedding_cosine_pairs,
        )

        scores = embedding_cosine_pairs(
            [out_a], [out_b], model_name=embedding_model,
        )
        score = scores[0]
        return score, score >= threshold

    if method == "entailment":
        from mltk.domains.llm._backends import (
            nli_bidirectional,
        )

        result = nli_bidirectional(
            out_a, out_b, model_name=nli_model,
        )
        equiv = bool(result["equivalent"])
        fwd_ent = float(
            result["forward"]["entailment"]  # type: ignore[index]
        )
        bwd_ent = float(
            result["backward"]["entailment"]  # type: ignore[index]
        )
        score = min(fwd_ent, bwd_ent) if not equiv else 1.0
        return score, equiv

    if method == "judge":
        if judge_fn is None:
            raise ValueError(
                "judge_fn is required for method='judge'"
            )
        score = float(judge_fn(out_a, out_b))
        return score, score >= threshold

    if method == "label_match":
        eq = out_a == out_b
        return (1.0 if eq else 0.0), eq

    raise ValueError(f"Unknown method: '{method}'")


def _auto_pair_score(
    out_a: str,
    out_b: str,
    embedding_model: str,
    nli_model: str,
) -> tuple[float, bool]:
    """Auto method: cosine first, NLI for ambiguous zone."""
    from mltk.domains.llm._backends import (
        embedding_cosine_pairs,
        nli_bidirectional,
    )

    cosine = embedding_cosine_pairs(
        [out_a], [out_b], model_name=embedding_model,
    )[0]

    if cosine >= _AUTO_HI:
        return cosine, True
    if cosine < _AUTO_LO:
        return cosine, False

    # Ambiguous zone -- fall back to NLI
    result = nli_bidirectional(
        out_a, out_b, model_name=nli_model,
    )
    equiv = bool(result["equivalent"])
    fwd_ent = float(
        result["forward"]["entailment"]  # type: ignore[index]
    )
    bwd_ent = float(
        result["backward"]["entailment"]  # type: ignore[index]
    )
    score = min(fwd_ent, bwd_ent) if not equiv else 1.0
    return score, equiv


def _resolve_threshold(
    method: str,
    similarity_threshold: float | None,
) -> float:
    """Return the effective threshold for *method*."""
    if similarity_threshold is not None:
        return similarity_threshold
    return _METHOD_DEFAULTS.get(method, 0.50)


# -----------------------------------------------------------------
# Public assertions
# -----------------------------------------------------------------


@timed_assertion
def assert_paraphrase_invariance(
    model_fn: Callable[[str], Any],
    paraphrases: list[str],
    equivalence_method: str = "token_f1",
    min_invariance: float = 0.8,
    similarity_threshold: float | None = None,
    embedding_model: str = "all-mpnet-base-v2",
    nli_model: str = (
        "cross-encoder/nli-deberta-v3-base"
    ),
    judge_fn: (
        Callable[[str, str], float] | None
    ) = None,
) -> TestResult:
    """Assert that paraphrased inputs produce equivalent outputs.

    Runs *model_fn* on each paraphrase, then checks that a
    sufficient fraction of output pairs are semantically
    equivalent according to *equivalence_method*.

    Args:
        model_fn: ``str -> Any`` callable (the model under test).
        paraphrases: List of semantically equivalent inputs.
        equivalence_method: One of ``"token_f1"``,
            ``"embedding"``, ``"entailment"``, ``"judge"``,
            ``"auto"``, ``"label_match"``.
        min_invariance: Minimum fraction of equivalent pairs
            to pass (0-1).
        similarity_threshold: Override per-method default.
        embedding_model: Sentence-transformer model for
            ``"embedding"`` and ``"auto"`` methods.
        nli_model: Cross-encoder model for ``"entailment"``
            and ``"auto"`` methods.
        judge_fn: ``(str, str) -> float`` scorer for
            ``method="judge"``.

    Returns:
        TestResult with invariance rate and per-pair details.

    Example:
        >>> def my_model(prompt):
        ...     return "World War 2 was a global conflict."
        >>> assert_paraphrase_invariance(
        ...     my_model,
        ...     ["What is WW2?", "Tell me about WW2"],
        ... )
    """
    if equivalence_method not in _EQUIVALENCE_METHODS:
        supported = ", ".join(
            f"'{m}'" for m in _EQUIVALENCE_METHODS
        )
        return assert_true(
            False,
            name="llm.behavioral.paraphrase_invariance",
            message=(
                f"Unknown method: '{equivalence_method}'"
                f". Supported: {supported}"
            ),
            severity=Severity.CRITICAL,
            method=equivalence_method,
        )

    if len(paraphrases) < 2:
        return assert_true(
            False,
            name="llm.behavioral.paraphrase_invariance",
            message=(
                "Need >= 2 paraphrases, "
                f"got {len(paraphrases)}"
            ),
            severity=Severity.CRITICAL,
        )

    from mltk.domains.llm._backends import (
        normalize_unicode,
    )

    # Normalize inputs and collect outputs
    normalized = [normalize_unicode(p) for p in paraphrases]
    outputs: list[Any] = [
        model_fn(p) for p in normalized
    ]

    # Auto-detect classifier output -> label_match
    method = equivalence_method
    if method != "label_match" and _is_classifier_output(
        outputs,
    ):
        method = "label_match"

    # Stringify for comparison methods that need text
    str_outputs: list[str] = [str(o) for o in outputs]

    threshold = _resolve_threshold(
        method, similarity_threshold,
    )

    # Compare ALL output pairs
    pairs = list(combinations(range(len(outputs)), 2))
    total_pairs = len(pairs)
    equivalent_count = 0
    pair_details: list[dict[str, Any]] = []
    worst_score = 1.0
    worst_pair: tuple[int, int] = (0, 1)

    for i, j in pairs:
        if method == "auto":
            score, equiv = _auto_pair_score(
                str_outputs[i],
                str_outputs[j],
                embedding_model,
                nli_model,
            )
        else:
            score, equiv = _pair_score(
                str_outputs[i],
                str_outputs[j],
                method,
                threshold,
                embedding_model,
                nli_model,
                judge_fn,
            )

        if equiv:
            equivalent_count += 1

        pair_details.append({
            "pair": (i, j),
            "score": round(score, 4),
            "equivalent": equiv,
        })

        if score < worst_score:
            worst_score = score
            worst_pair = (i, j)

    invariance_rate = (
        equivalent_count / total_pairs
        if total_pairs > 0
        else 0.0
    )
    passed = invariance_rate >= min_invariance

    message = (
        f"Paraphrase invariance ({method}): "
        f"{invariance_rate:.4f} >= {min_invariance} "
        f"({equivalent_count}/{total_pairs} pairs)"
        if passed
        else f"Paraphrase invariance too low ({method}): "
        f"{invariance_rate:.4f} < {min_invariance} "
        f"({equivalent_count}/{total_pairs} pairs)"
    )

    # Per-paraphrase outputs for ProSA requirement
    per_input = [
        {
            "input": paraphrases[idx],
            "output": str_outputs[idx],
        }
        for idx in range(len(paraphrases))
    ]

    return assert_true(
        passed,
        name="llm.behavioral.paraphrase_invariance",
        message=message,
        severity=Severity.CRITICAL,
        method=method,
        invariance_rate=round(invariance_rate, 4),
        min_invariance=min_invariance,
        threshold=threshold,
        equivalent_pairs=equivalent_count,
        total_pairs=total_pairs,
        pair_scores=pair_details,
        worst_pair=worst_pair,
        worst_score=round(worst_score, 4),
        per_input_outputs=per_input,
    )


@timed_assertion
def assert_format_invariance(
    model_fn: Callable[[str], Any],
    input_text: str,
    transforms: (
        list[Callable[[str], str]] | None
    ) = None,
    equivalence_method: str = "token_f1",
    min_invariance: float = 0.9,
    similarity_threshold: float | None = None,
    embedding_model: str = "all-mpnet-base-v2",
) -> TestResult:
    """Assert that formatting changes do not alter model output.

    Applies cosmetic transforms (case, spacing, punctuation) to
    *input_text*, runs *model_fn* on each variant, and checks
    that outputs remain equivalent to the original.

    Args:
        model_fn: ``str -> Any`` callable (the model under test).
        input_text: The original input prompt.
        transforms: List of ``str -> str`` transform functions.
            ``None`` uses built-in defaults (lowercase, uppercase,
            strip whitespace, remove punctuation, double-space).
        equivalence_method: One of ``"token_f1"``,
            ``"embedding"``, ``"entailment"``, ``"label_match"``.
        min_invariance: Minimum fraction of equivalent
            transforms to pass (0-1).
        similarity_threshold: Override per-method default.
        embedding_model: Sentence-transformer model for
            ``"embedding"`` method.

    Returns:
        TestResult with invariance rate and per-transform details.

    Example:
        >>> def my_model(prompt):
        ...     return "Paris is the capital of France."
        >>> assert_format_invariance(
        ...     my_model,
        ...     "What is the capital of France?",
        ... )
    """
    allowed = (
        "token_f1",
        "embedding",
        "entailment",
        "label_match",
    )
    if equivalence_method not in allowed:
        supported = ", ".join(f"'{m}'" for m in allowed)
        return assert_true(
            False,
            name="llm.behavioral.format_invariance",
            message=(
                f"Unknown method: '{equivalence_method}'"
                f". Supported: {supported}"
            ),
            severity=Severity.CRITICAL,
            method=equivalence_method,
        )

    from mltk.domains.llm._backends import (
        normalize_unicode,
    )

    original_input = normalize_unicode(input_text)
    original_output = str(model_fn(original_input))

    # Resolve transforms
    if transforms is not None:
        named_transforms: list[
            tuple[str, Callable[[str], str]]
        ] = [
            (f"custom_{idx}", fn)
            for idx, fn in enumerate(transforms)
        ]
    else:
        named_transforms = list(
            _DEFAULT_FORMAT_TRANSFORMS
        )

    threshold = _resolve_threshold(
        equivalence_method, similarity_threshold,
    )

    total = len(named_transforms)
    equivalent_count = 0
    transform_details: list[dict[str, Any]] = []

    for name, fn in named_transforms:
        transformed_input = fn(original_input)
        transformed_output = str(
            model_fn(transformed_input)
        )

        if equivalence_method == "auto":
            score, equiv = _auto_pair_score(
                original_output,
                transformed_output,
                embedding_model,
                "cross-encoder/nli-deberta-v3-base",
            )
        else:
            score, equiv = _pair_score(
                original_output,
                transformed_output,
                equivalence_method,
                threshold,
                embedding_model,
                "cross-encoder/nli-deberta-v3-base",
                None,
            )

        if equiv:
            equivalent_count += 1

        transform_details.append({
            "transform": name,
            "input": transformed_input,
            "output": transformed_output,
            "score": round(score, 4),
            "equivalent": equiv,
        })

    invariance_rate = (
        equivalent_count / total if total > 0 else 0.0
    )
    passed = invariance_rate >= min_invariance

    message = (
        f"Format invariance ({equivalence_method}): "
        f"{invariance_rate:.4f} >= {min_invariance} "
        f"({equivalent_count}/{total} transforms)"
        if passed
        else f"Format invariance too low "
        f"({equivalence_method}): "
        f"{invariance_rate:.4f} < {min_invariance} "
        f"({equivalent_count}/{total} transforms)"
    )

    return assert_true(
        passed,
        name="llm.behavioral.format_invariance",
        message=message,
        severity=Severity.CRITICAL,
        method=equivalence_method,
        invariance_rate=round(invariance_rate, 4),
        min_invariance=min_invariance,
        threshold=threshold,
        equivalent_transforms=equivalent_count,
        total_transforms=total,
        original_input=input_text,
        original_output=original_output,
        transform_results=transform_details,
    )
