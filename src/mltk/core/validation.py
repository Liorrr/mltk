"""Shared input-validation helpers.

``require_same_length`` is the single contract for paired-sequence assertion
inputs. Before S102, ~20 sites used ``zip(..., strict=False)``, which silently
truncated to the shortest input and reported a metric computed on partial data
with no signal to the caller. Mismatched lengths are a usage error and now fail
loudly, consistent with the ``on_empty="fail"`` default adopted in S101.

Internal fixed-length pairings (e.g. a digit string against a constant weight
table) are invariants, not user contracts — those use ``zip(..., strict=True)``
directly rather than this helper.
"""
from __future__ import annotations

from collections.abc import Sized


def require_same_length(context: str, /, **sequences: Sized) -> None:
    """Raise ``ValueError`` if the named sequences differ in length.

    Args:
        context: Assertion or function name, used to prefix the error.
        **sequences: Named sequences to compare, e.g.
            ``require_same_length("assert_map", predictions=p, ground_truth=g)``.

    Raises:
        ValueError: If two or more sequences have differing lengths. The
            message names every sequence and its length, so the caller can see
            which input is wrong without re-running under a debugger.

    Example:
        >>> require_same_length("assert_map", preds=[1, 2], truth=[1])
        Traceback (most recent call last):
            ...
        ValueError: assert_map: length mismatch -- preds (2), truth (1) must have the same length
    """
    if len(sequences) < 2:
        return
    lengths = {name: len(seq) for name, seq in sequences.items()}
    if len(set(lengths.values())) == 1:
        return
    detail = ", ".join(f"{name} ({length})" for name, length in lengths.items())
    raise ValueError(
        f"{context}: length mismatch -- {detail} must have the same length"
    )
