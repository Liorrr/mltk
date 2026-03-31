"""Quality filtering for synthetic QA pairs.

Scores each ``QAPair`` on self-containment (is the answer
grounded in the context?) and answerability (can the question
be answered from the context?).

In template mode (no ``llm_fn``), all pairs pass with a score
of 1.0 -- template generation is deterministic and inherently
grounded, so filtering would just add overhead.

In LLM mode, the filter sends a scoring prompt to the same
``llm_fn`` callable and parses the numeric score.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any


class QualityFilter:
    """Score and filter synthetic QA pairs for quality.

    Evaluates two dimensions:

    1. **Self-containment**: Is the answer supported by the
       context?  A score of 1.0 means fully grounded.
    2. **Answerability**: Can the question be answered from
       the context?  For out-of-scope questions this should
       be low (and that is expected).

    The final score is the average of both dimensions.

    Args:
        llm_fn: ``str -> str`` callable for LLM-based scoring.
            If ``None``, all pairs receive a score of 1.0
            (template mode bypass).
        threshold: Minimum score (0.0 -- 1.0) for a pair to
            pass quality filtering.  Default 0.6.
        max_retries: Number of times to retry LLM scoring if
            parsing fails.  Default 1.

    Example::

        from mltk.domains.llm.synthetic._quality import (
            QualityFilter,
        )

        filt = QualityFilter(llm_fn=my_llm, threshold=0.7)
        score = filt.score(qa_pair)
        if score >= filt.threshold:
            print("Pair passes quality check")
    """

    _SCORING_PROMPT = (
        "You are a QA quality evaluator.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer: {answer}\n\n"
        "Score this QA pair on two dimensions "
        "(each 0.0 to 1.0):\n"
        "1. self_containment: Is the answer fully "
        "supported by the context?\n"
        "2. answerability: Can this question be answered "
        "from the context?\n\n"
        'Return JSON: {{"self_containment": 0.X, '
        '"answerability": 0.X}}'
    )

    def __init__(
        self,
        llm_fn: Callable[[str], str] | None = None,
        threshold: float = 0.6,
        max_retries: int = 1,
    ) -> None:
        self._llm_fn = llm_fn
        self.threshold = threshold
        self._max_retries = max_retries

    def score(self, pair: object) -> float:
        """Score a ``QAPair`` on quality dimensions.

        Uses duck typing to access ``question``, ``answer``,
        and ``context`` attributes, avoiding circular imports
        with the ``generator`` module.

        Args:
            pair: A ``QAPair`` instance.  Must have
                ``question``, ``answer``, and ``context``
                attributes.

        Returns:
            Float score between 0.0 and 1.0.  Returns 1.0
            in template mode (no ``llm_fn``).
        """
        if self._llm_fn is None:
            return 1.0

        context = getattr(pair, "context", "")
        if isinstance(context, list):
            context = " ".join(context)

        question = getattr(pair, "question", "")
        answer = getattr(pair, "answer", "")

        prompt = self._SCORING_PROMPT.format(
            context=context,
            question=question,
            answer=answer,
        )

        for _ in range(1 + self._max_retries):
            try:
                raw = self._llm_fn(prompt)
                result = _parse_score(raw)
                if result is not None:
                    return result
            except Exception:  # noqa: BLE001
                continue

        # If all retries fail, return 0.0 (fail-safe)
        return 0.0

    def passes(self, pair: object) -> bool:
        """Return ``True`` if *pair* meets the threshold.

        Convenience wrapper around :meth:`score`.

        Args:
            pair: A ``QAPair`` instance.

        Returns:
            ``True`` if the score is >= ``threshold``.
        """
        return self.score(pair) >= self.threshold


# ---------------------------------------------------------------
# Internal parsing helpers
# ---------------------------------------------------------------

def _parse_score(raw: str) -> float | None:
    """Extract numeric scores from LLM response.

    Tries JSON first, then markdown fences, then regex.

    Returns:
        Average of self_containment and answerability,
        or ``None`` if parsing fails.
    """
    if not raw or not raw.strip():
        return None

    # Try JSON parse
    data = _try_json_parse(raw)
    if data is not None:
        return _extract_scores(data)

    # Try extracting from markdown fences
    match = re.search(
        r"```(?:json)?\s*\n?(.*?)\n?\s*```",
        raw,
        re.DOTALL,
    )
    if match:
        try:
            data_inner = json.loads(match.group(1))
            if isinstance(data_inner, dict):
                return _extract_scores(data_inner)
        except (json.JSONDecodeError, ValueError):
            pass

    # Regex fallback
    sc_match = re.search(
        r"self_containment[\"']?\s*:\s*([\d.]+)", raw,
    )
    ans_match = re.search(
        r"answerability[\"']?\s*:\s*([\d.]+)", raw,
    )
    if sc_match and ans_match:
        try:
            sc = float(sc_match.group(1))
            ans = float(ans_match.group(1))
            return _clamp_avg(sc, ans)
        except ValueError:
            pass

    return None


def _try_json_parse(raw: str) -> dict[str, Any] | None:
    """Attempt direct JSON parse."""
    try:
        data = json.loads(raw.strip())
        if isinstance(data, dict):
            return data  # type: ignore[return-value]
        return None
    except (json.JSONDecodeError, ValueError):
        return None


def _extract_scores(data: dict[str, Any]) -> float | None:
    """Average self_containment and answerability scores."""
    try:
        sc = float(data.get("self_containment", 0))
        ans = float(data.get("answerability", 0))
        return _clamp_avg(sc, ans)
    except (TypeError, ValueError):
        return None


def _clamp_avg(a: float, b: float) -> float:
    """Average two scores, clamped to [0.0, 1.0]."""
    avg = (a + b) / 2.0
    return max(0.0, min(1.0, avg))
