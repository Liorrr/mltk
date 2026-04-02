"""Evaluation pipeline data types.

Defines the core data structures shared between solvers,
scorers, and the evaluation task runner. These types form
the contract between pipeline stages.

Architecture: Samples flow in, state accumulates through
the solver pipeline, scorers produce scores, and the task
runner aggregates everything into an EvalResult.

::

    EvalSample --> EvalState --> [Solver...] --> [Scorer...] --> Score
                                                         |
                                                    EvalResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalSample:
    """Single evaluation sample (input + expected output).

    This is the atomic unit of evaluation data. Each sample
    represents one question/prompt and its expected answer.
    Samples are immutable once created -- solvers read from
    them but never modify them.

    Args:
        input: The prompt or question to evaluate.
        target: Expected answer (optional -- some scorers
            don't need a target, e.g., quality judges).
        metadata: Arbitrary metadata (e.g., category,
            difficulty, source dataset).

    Example:
        >>> sample = EvalSample(
        ...     input="What is the capital of France?",
        ...     target="Paris",
        ...     metadata={"category": "geography"},
        ... )
    """

    input: str
    target: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalState:
    """Mutable state flowing through the solver pipeline.

    Each solver receives the state, transforms it (adding
    output, messages, or metadata), and passes it to the
    next solver. Scorers receive the final state read-only.

    The ``completed`` flag allows solvers to short-circuit
    the pipeline. For example, a cache solver might set
    ``completed = True`` if it finds a cached response,
    skipping all downstream solvers.

    Args:
        sample: Original evaluation sample (immutable).
        output: Model/solver output string (populated by
            solver).
        messages: Conversation history (for multi-turn
            evaluations).
        metadata: Solver-added context (e.g., chain-of-thought
            reasoning, intermediate results).
        completed: If True, remaining solvers are skipped.

    Example:
        >>> state = EvalState(
        ...     sample=EvalSample(input="2+2?", target="4"),
        ... )
        >>> state.output = "4"
        >>> state.completed = True
    """

    sample: EvalSample
    output: str = ""
    messages: list[dict[str, str]] = field(
        default_factory=list
    )
    metadata: dict[str, Any] = field(default_factory=dict)
    completed: bool = False


@dataclass
class Score:
    """Result from a single scorer evaluation.

    Scorers produce one Score per sample. The ``value`` field
    is the primary metric -- normalized to [0.0, 1.0] by
    convention. Scores above 0.5 are considered "passing"
    for accuracy calculations.

    Args:
        value: Normalized score in [0.0, 1.0].
        answer: Extracted answer (if scorer extracts one).
        explanation: Human-readable scoring rationale.
        metadata: Additional scorer-specific details.

    Example:
        >>> score = Score(
        ...     value=1.0,
        ...     answer="Paris",
        ...     explanation="Exact match",
        ... )
    """

    value: float
    answer: str = ""
    explanation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Aggregated results from a complete evaluation run.

    Contains per-sample scores from all scorers, aggregated
    metrics (accuracy, mean score), and timing information.
    This is the top-level output from ``EvalTask.run()``.

    Metrics follow the naming convention
    ``"{ScorerName}/{metric}"`` -- for example,
    ``"ExactMatchScorer/accuracy"`` or ``"LLMJudge/correctness/mean"``.

    Args:
        task_name: Name of the evaluation task.
        samples: List of final EvalState per sample.
        scores: Scorer name -> list of Score (one per sample).
        metrics: Aggregated metrics
            (e.g., ``"ExactMatchScorer/accuracy"``).
        duration_ms: Total evaluation time in milliseconds.

    Example:
        >>> result = EvalResult(
        ...     task_name="qa",
        ...     metrics={"ExactMatchScorer/accuracy": 0.95},
        ... )
        >>> result.passed
        True
    """

    task_name: str
    samples: list[EvalState] = field(default_factory=list)
    scores: dict[str, list[Score]] = field(
        default_factory=dict
    )
    metrics: dict[str, float] = field(default_factory=dict)
    duration_ms: float = 0.0

    @property
    def passed(self) -> bool:
        """True if all metrics are above 0.5 (convention).

        An empty metrics dict is considered passing -- there
        is nothing to fail. This matches the convention that
        a score of 0.5 or above is "correct enough" for
        binary accuracy.

        Returns:
            True if every metric value >= 0.5.
        """
        if not self.metrics:
            return True
        return all(v >= 0.5 for v in self.metrics.values())

    @property
    def total_samples(self) -> int:
        """Number of samples that were evaluated.

        Returns:
            Length of the samples list.
        """
        return len(self.samples)
