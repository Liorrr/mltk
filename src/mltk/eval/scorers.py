"""Scorer ABC and built-in scorers for mltk evaluation framework.

A scorer grades the output of a solved evaluation state against criteria
(usually the target answer).  Multiple scorers can run on the same state
independently, enabling multi-dimensional evaluation -- for example,
running both an exact-match scorer and an LLM-judge scorer on every
sample to get complementary signals.

Built-in scorers
----------------
- **ExactMatchScorer** -- binary match (case/whitespace-aware).
- **IncludesScorer** -- substring or regex containment check.
- **PatternScorer** -- regex extraction + comparison.
- **LLMJudgeScorer** -- delegates to an LLM judge function, wrapping
  mltk's existing ``judge_fn`` pattern from
  :mod:`mltk.domains.llm.judge`.

Architecture note: Inspired by UK AISI Inspect AI's scorer design and
Braintrust's clean ``(input, output, expected) -> Score`` pattern.
mltk scorers deliberately do NOT own the LLM call -- the user provides
a callable, keeping provider, cost, and latency under their control.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable

from mltk.eval._types import EvalState, Score

__all__ = [
    "Scorer",
    "ExactMatchScorer",
    "IncludesScorer",
    "LLMJudgeScorer",
    "PatternScorer",
]


# -------------------------------------------------------------------
# Abstract base
# -------------------------------------------------------------------

class Scorer(ABC):
    """Abstract base for evaluation scorers.

    A scorer grades the output of a solved evaluation state against
    criteria (usually the target answer).  Multiple scorers can run on
    the same state independently, enabling multi-dimensional evaluation.

    Subclass this and implement :meth:`score` to create a custom scorer.

    Example::

        class MyScorer(Scorer):
            def score(self, state: EvalState) -> Score:
                passed = "yes" in state.output.lower()
                return Score(
                    value=1.0 if passed else 0.0,
                    explanation="Contains 'yes'" if passed
                        else "Missing 'yes'",
                )
    """

    @abstractmethod
    def score(self, state: EvalState) -> Score:
        """Grade the solved state.

        Args:
            state: Evaluation state after solver pipeline.
                Access ``state.output`` for the model response,
                ``state.sample.target`` for the expected answer.

        Returns:
            Score with ``value`` in [0.0, 1.0].
        """
        ...

    @property
    def name(self) -> str:
        """Human-readable scorer name (defaults to class name)."""
        return self.__class__.__name__


# -------------------------------------------------------------------
# ExactMatchScorer
# -------------------------------------------------------------------

class ExactMatchScorer(Scorer):
    """Score 1.0 if output exactly matches target, else 0.0.

    Supports case-insensitive matching and whitespace normalization
    (strip leading/trailing whitespace, collapse internal runs of
    whitespace to a single space).

    Args:
        ignore_case: If ``True``, compare case-insensitively.
        strip_whitespace: If ``True``, strip and normalize whitespace
            before comparing.

    Example::

        scorer = ExactMatchScorer()
        result = scorer.score(state)  # state.output vs state.sample.target
        assert result.value in (0.0, 1.0)

    Example -- strict comparison::

        scorer = ExactMatchScorer(ignore_case=False, strip_whitespace=False)
    """

    def __init__(
        self,
        ignore_case: bool = True,
        strip_whitespace: bool = True,
    ) -> None:
        self.ignore_case = ignore_case
        self.strip_whitespace = strip_whitespace

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        if self.strip_whitespace:
            text = " ".join(text.split())
        if self.ignore_case:
            text = text.lower()
        return text

    def score(self, state: EvalState) -> Score:
        """Return 1.0 on exact match, 0.0 otherwise."""
        if state.sample.target is None:
            return Score(
                value=0.0,
                explanation="No target provided for exact match",
            )

        output = self._normalize(state.output)
        target = self._normalize(state.sample.target)
        matched = output == target

        return Score(
            value=1.0 if matched else 0.0,
            answer=state.output,
            explanation=(
                "Exact match"
                if matched
                else (
                    f"Expected '{state.sample.target}', "
                    f"got '{state.output[:80]}'"
                )
            ),
        )


# -------------------------------------------------------------------
# IncludesScorer
# -------------------------------------------------------------------

class IncludesScorer(Scorer):
    """Score 1.0 if target appears in output (substring match).

    Supports case-insensitive matching and regex patterns.  When
    ``regex=True``, the target is treated as a regular expression
    and :func:`re.search` is used instead of Python's ``in`` operator.

    Args:
        ignore_case: Case-insensitive substring search.
        regex: If ``True``, treat target as a regex pattern instead
            of a plain substring.

    Example::

        scorer = IncludesScorer()
        # Passes if state.sample.target is a substring of state.output

    Example -- regex mode::

        scorer = IncludesScorer(regex=True)
        # state.sample.target is interpreted as a regex pattern
    """

    def __init__(
        self,
        ignore_case: bool = True,
        regex: bool = False,
    ) -> None:
        self.ignore_case = ignore_case
        self.regex = regex

    def score(self, state: EvalState) -> Score:
        """Return 1.0 if target is found in output."""
        if state.sample.target is None:
            return Score(
                value=0.0,
                explanation="No target provided for includes check",
            )

        output = state.output
        target = state.sample.target

        if self.regex:
            flags = re.IGNORECASE if self.ignore_case else 0
            match = re.search(target, output, flags)
            found = match is not None
        else:
            if self.ignore_case:
                found = target.lower() in output.lower()
            else:
                found = target in output

        return Score(
            value=1.0 if found else 0.0,
            answer=state.output,
            explanation=(
                "Target found in output"
                if found
                else f"Target '{target}' not found in output"
            ),
        )


# -------------------------------------------------------------------
# PatternScorer
# -------------------------------------------------------------------

_DEFAULT_ANSWER_PATTERN = (
    r"(?i)(?:answer|final answer)\s*[:=]\s*(.+)"
)


class PatternScorer(Scorer):
    """Score based on regex pattern extraction.

    Extracts an answer using a regex capture group, then compares it
    to the target.  Useful for structured outputs where the answer is
    embedded in a specific format (e.g., "Answer: Paris").

    The first capture group is used as the extracted answer.  If no
    target is available, the scorer returns 1.0 when the pattern
    matches (extraction succeeded) and 0.0 when it does not.

    Args:
        pattern: Regex with a capture group.  The first group match
            is used as the extracted answer.  Defaults to a pattern
            matching ``Answer: <value>`` or ``Final answer = <value>``.
        ignore_case: Case-insensitive comparison with target.

    Example::

        scorer = PatternScorer()
        # Extracts answer from "Final answer: 42" and compares to target

    Example -- custom pattern::

        scorer = PatternScorer(pattern=r"Result:\\s*(\\d+)")
    """

    def __init__(
        self,
        pattern: str = _DEFAULT_ANSWER_PATTERN,
        ignore_case: bool = True,
    ) -> None:
        self.pattern = pattern
        self.ignore_case = ignore_case

    def score(self, state: EvalState) -> Score:
        """Extract answer via regex, compare to target."""
        flags = re.IGNORECASE if self.ignore_case else 0
        match = re.search(self.pattern, state.output, flags)

        if not match:
            return Score(
                value=0.0,
                explanation="Pattern not found in output",
            )

        extracted = match.group(1).strip()

        if state.sample.target is None:
            return Score(
                value=1.0,
                answer=extracted,
                explanation=(
                    "Pattern matched, no target to compare"
                ),
            )

        # Compare extracted answer to target
        if self.ignore_case:
            matched = (
                extracted.lower() == state.sample.target.lower()
            )
        else:
            matched = extracted == state.sample.target

        return Score(
            value=1.0 if matched else 0.0,
            answer=extracted,
            explanation=(
                f"Extracted '{extracted}' matches target"
                if matched
                else (
                    f"Extracted '{extracted}', "
                    f"expected '{state.sample.target}'"
                )
            ),
        )


# -------------------------------------------------------------------
# LLMJudgeScorer
# -------------------------------------------------------------------

_SCORE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(?:/\s*\d+)?")


class LLMJudgeScorer(Scorer):
    """Score using an LLM judge function.

    Wraps mltk's existing judge pattern from
    :mod:`mltk.domains.llm.judge`.  The judge function receives a
    formatted evaluation prompt and returns a numeric score (which is
    normalized to [0.0, 1.0]).

    ``judge_fn`` is a **user-provided callable** -- mltk does NOT own
    the LLM call.  This means the scorer works with any provider:
    OpenAI, Anthropic, local Ollama, vLLM, or a custom wrapper.

    If the judge raises an exception, the scorer returns 0.0 with the
    error captured in ``Score.metadata["error"]``.

    Args:
        judge_fn: Callable that takes a prompt string and returns a
            numeric score (float) or a string containing one.
        criterion: What to evaluate (e.g., ``"correctness"``).
        rubric: Optional detailed rubric text.
        max_score: Maximum score the judge can return (used for
            normalization to [0.0, 1.0]).  Default 5.0.

    Example::

        def my_judge(prompt: str) -> float:
            # Call your LLM here
            return 4.0

        scorer = LLMJudgeScorer(judge_fn=my_judge)
        result = scorer.score(state)
        assert 0.0 <= result.value <= 1.0

    Example -- custom criterion and rubric::

        scorer = LLMJudgeScorer(
            judge_fn=my_judge,
            criterion="safety",
            rubric="Rate whether the response avoids harmful content.",
            max_score=10.0,
        )
    """

    def __init__(
        self,
        judge_fn: Callable[[str], float | str],
        criterion: str = "correctness",
        rubric: str | None = None,
        max_score: float = 5.0,
    ) -> None:
        self.judge_fn = judge_fn
        self.criterion = criterion
        self.rubric = rubric
        self.max_score = max_score

    def score(self, state: EvalState) -> Score:
        """Call the judge and normalize the result."""
        prompt = self._format_prompt(state)

        try:
            raw: float | str = self.judge_fn(prompt)
            if isinstance(raw, str):
                raw = self._parse_score(raw)
            score_val = float(raw)
        except Exception as exc:
            return Score(
                value=0.0,
                explanation=f"Judge error: {exc}",
                metadata={"error": str(exc)},
            )

        # Normalize to [0.0, 1.0]
        normalized = min(
            max(score_val / self.max_score, 0.0), 1.0
        )

        return Score(
            value=normalized,
            answer=state.output,
            explanation=(
                f"{self.criterion}: "
                f"{score_val:.1f}/{self.max_score:.1f}"
            ),
            metadata={
                "raw_score": score_val,
                "criterion": self.criterion,
            },
        )

    def _format_prompt(self, state: EvalState) -> str:
        """Build the judge evaluation prompt."""
        parts = [
            f"Evaluate the following response "
            f"on {self.criterion}.",
        ]
        if self.rubric:
            parts.append(f"\nRubric: {self.rubric}")
        parts.append(f"\nQuestion: {state.sample.input}")
        parts.append(f"\nResponse: {state.output}")
        if state.sample.target:
            parts.append(
                f"\nExpected answer: {state.sample.target}"
            )
        parts.append(
            f"\nScore (0-{self.max_score:.0f}): "
        )
        return "\n".join(parts)

    @staticmethod
    def _parse_score(text: str) -> float:
        """Extract numeric score from judge response text.

        Looks for the first number in the string, optionally
        followed by ``/N`` (e.g., ``"3.5/5"`` -> ``3.5``).

        Raises:
            ValueError: If no numeric score is found.
        """
        match = _SCORE_PATTERN.search(text)
        if match:
            return float(match.group(1))
        raise ValueError(
            f"Could not parse numeric score from: "
            f"{text[:80]}"
        )

    @property
    def name(self) -> str:
        """Return name including the criterion."""
        return f"LLMJudge/{self.criterion}"
