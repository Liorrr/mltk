"""Solver ABC and built-in solvers for mltk eval pipelines.

Solvers are the core processing units in an evaluation pipeline. Each
solver receives an :class:`EvalState`, optionally calls a model via
the ``generate`` callable, and returns the updated state.

Solvers never call models directly — they receive a ``generate``
function that abstracts the LLM backend. This makes solvers testable
with simple stub functions and backend-agnostic by design.

Architecture inspired by UK AISI Inspect AI's solver pattern, adapted
for mltk's pytest-native, zero-dependency philosophy.

Example:
    Build a chain-of-thought pipeline with few-shot examples::

        from mltk.eval.solvers import (
            ChainOfThoughtSolver,
            FewShotSolver,
            GenerateSolver,
            chain,
        )

        pipeline = chain(
            ChainOfThoughtSolver(),
            FewShotSolver(examples=[("2+2", "4"), ("3+3", "6")]),
            GenerateSolver(),
        )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

from mltk.eval._types import EvalState

__all__ = [
    "Solver",
    "GenerateSolver",
    "ChainOfThoughtSolver",
    "FewShotSolver",
    "chain",
]


class Solver(ABC):
    """Abstract base for evaluation solvers.

    A solver transforms evaluation state through a processing step.
    Solvers form a pipeline: each solver receives the state from the
    previous one and passes it to the next. The ``generate`` callable
    abstracts the LLM — solvers never call models directly.

    Subclasses must implement :meth:`solve`. The :attr:`name` property
    returns the class name by default; override it for custom labels.

    Inspired by UK AISI Inspect AI's solver architecture, adapted
    for mltk's pytest-native, zero-dependency design.

    Example:
        Implement a custom solver that uppercases model output::

            class UpperSolver(Solver):
                def solve(self, state, generate):
                    state.output = state.output.upper()
                    return state
    """

    @abstractmethod
    def solve(
        self,
        state: EvalState,
        generate: Callable[[str], str],
    ) -> EvalState:
        """Process state, optionally calling generate for model output.

        Args:
            state: Current evaluation state.
            generate: Callable that sends a prompt to the model
                and returns the response string.

        Returns:
            Updated EvalState (may be the same object, mutated).
        """
        ...

    @property
    def name(self) -> str:
        """Human-readable solver name for logging and metrics.

        Returns:
            The class name by default. Override in subclasses
            to provide a more descriptive label.
        """
        return self.__class__.__name__


class GenerateSolver(Solver):
    """Send input to model and store the response.

    The most basic solver — takes the sample input (or the last
    message if messages exist), sends it to the model via
    ``generate()``, and stores the response in ``state.output``.

    If ``state.completed`` is already ``True``, this solver is a
    no-op and returns state unchanged.

    Example:
        Basic generation from sample input::

            from mltk.eval._types import EvalSample, EvalState
            from mltk.eval.solvers import GenerateSolver

            state = EvalState(sample=EvalSample(input="What is 2+2?"))
            solver = GenerateSolver()
            result = solver.solve(state, lambda prompt: "4")
            assert result.output == "4"
    """

    def solve(
        self,
        state: EvalState,
        generate: Callable[[str], str],
    ) -> EvalState:
        """Generate a model response from the current prompt.

        Uses the last message content if messages exist, otherwise
        falls back to ``state.sample.input``.

        Args:
            state: Current evaluation state.
            generate: Callable that sends a prompt to the model
                and returns the response string.

        Returns:
            State with ``output`` and ``messages`` updated.
        """
        if state.completed:
            return state

        prompt = state.sample.input
        if state.messages:
            last_content = state.messages[-1].get("content", "")
            if last_content:
                prompt = last_content

        response = generate(prompt)
        state.output = response
        state.messages.append(
            {"role": "assistant", "content": response}
        )
        return state

    @property
    def name(self) -> str:
        """Return solver name.

        Returns:
            ``"GenerateSolver"`` — identifies this as the basic
            generation step in pipeline logs.
        """
        return "GenerateSolver"


class ChainOfThoughtSolver(Solver):
    """Prepend chain-of-thought instruction before generating.

    Adds a system message instructing the model to think
    step-by-step before answering. The ``template`` parameter
    allows customizing the CoT instruction.

    If ``state.completed`` is already ``True``, this solver is a
    no-op and returns state unchanged.

    Args:
        template: CoT instruction string. If ``None``, uses a
            default instruction asking for step-by-step reasoning
            before the final answer.

    Example:
        Add CoT reasoning to a math problem::

            from mltk.eval._types import EvalSample, EvalState
            from mltk.eval.solvers import ChainOfThoughtSolver

            state = EvalState(
                sample=EvalSample(input="What is 15% of 200?"),
            )
            solver = ChainOfThoughtSolver()

            def mock_model(prompt):
                return "15% of 200 = 0.15 * 200 = 30"

            result = solver.solve(state, mock_model)
            assert "30" in result.output
    """

    _DEFAULT_TEMPLATE: str = (
        "Think step by step before answering. "
        "Show your reasoning, then provide "
        "your final answer on the last line."
    )

    def __init__(self, template: str | None = None) -> None:
        self.template: str = template or self._DEFAULT_TEMPLATE

    def solve(
        self,
        state: EvalState,
        generate: Callable[[str], str],
    ) -> EvalState:
        """Inject CoT instruction and generate a response.

        Inserts a system message with the CoT template at the
        front of the message list, then builds a combined prompt
        from all messages plus the sample input.

        Args:
            state: Current evaluation state.
            generate: Callable that sends a prompt to the model
                and returns the response string.

        Returns:
            State with CoT system message, user message, and
            assistant response appended to ``messages``.
        """
        if state.completed:
            return state

        state.messages.insert(0, {
            "role": "system",
            "content": self.template,
        })

        prompt = state.sample.input
        if len(state.messages) > 1:
            parts = [
                m["content"]
                for m in state.messages
                if m.get("content")
            ]
            parts.append(prompt)
            prompt = "\n\n".join(parts)
        else:
            prompt = f"{self.template}\n\n{prompt}"

        response = generate(prompt)
        state.output = response
        state.messages.append(
            {"role": "user", "content": state.sample.input}
        )
        state.messages.append(
            {"role": "assistant", "content": response}
        )
        return state

    @property
    def name(self) -> str:
        """Return solver name.

        Returns:
            ``"ChainOfThoughtSolver"`` — identifies this as a CoT
            reasoning step in pipeline logs.
        """
        return "ChainOfThoughtSolver"


class FewShotSolver(Solver):
    """Prepend few-shot examples before generating.

    Adds demonstration examples (input-output pairs) before
    the actual evaluation input, following the standard
    few-shot prompting pattern.

    If ``state.completed`` is already ``True``, this solver is a
    no-op and returns state unchanged.

    Args:
        examples: List of ``(input, output)`` tuples used as
            demonstrations. Each is formatted using the template.
            Must contain at least one example.
        template: Format string for each example. Must contain
            ``{input}`` and ``{output}`` placeholders. Defaults
            to ``"Q: {input}\\nA: {output}"``.

    Raises:
        ValueError: If ``examples`` is empty.

    Example:
        Few-shot arithmetic::

            from mltk.eval._types import EvalSample, EvalState
            from mltk.eval.solvers import FewShotSolver

            solver = FewShotSolver(
                examples=[("2+2", "4"), ("3+3", "6")],
            )
            state = EvalState(
                sample=EvalSample(input="5+5"),
            )
            result = solver.solve(state, lambda p: "10")
            assert result.output == "10"
    """

    _DEFAULT_TEMPLATE: str = "Q: {input}\nA: {output}"

    def __init__(
        self,
        examples: Sequence[tuple[str, str]],
        template: str | None = None,
    ) -> None:
        if not examples:
            raise ValueError(
                "FewShotSolver requires at least one example"
            )
        self.examples: Sequence[tuple[str, str]] = list(examples)
        self.template: str = template or self._DEFAULT_TEMPLATE

    def solve(
        self,
        state: EvalState,
        generate: Callable[[str], str],
    ) -> EvalState:
        """Build a few-shot prompt and generate a response.

        Formats each example using the template, appends the
        actual input as an incomplete example (no output), and
        sends the combined prompt to the model.

        Args:
            state: Current evaluation state.
            generate: Callable that sends a prompt to the model
                and returns the response string.

        Returns:
            State with the few-shot prompt and model response
            appended to ``messages``.
        """
        if state.completed:
            return state

        formatted = [
            self.template.format(input=inp, output=out)
            for inp, out in self.examples
        ]

        examples_text = "\n\n".join(formatted)
        question = self.template.format(
            input=state.sample.input, output=""
        ).rstrip()
        prompt = f"{examples_text}\n\n{question}"

        response = generate(prompt)
        state.output = response
        state.messages.append(
            {"role": "user", "content": prompt}
        )
        state.messages.append(
            {"role": "assistant", "content": response}
        )
        return state

    @property
    def name(self) -> str:
        """Return solver name.

        Returns:
            ``"FewShotSolver"`` — identifies this as a few-shot
            prompting step in pipeline logs.
        """
        return "FewShotSolver"


class _PipelineSolver(Solver):
    """Internal composite solver that chains multiple solvers.

    Runs each solver in sequence, passing state from one to the
    next. Short-circuits if any solver sets ``state.completed``
    to ``True``.

    This class is not part of the public API. Use :func:`chain`
    to create pipeline solvers.
    """

    def __init__(self, solvers: Sequence[Solver]) -> None:
        self._solvers: tuple[Solver, ...] = tuple(solvers)

    def solve(
        self,
        state: EvalState,
        generate: Callable[[str], str],
    ) -> EvalState:
        """Run each solver in sequence, passing state through.

        Args:
            state: Current evaluation state.
            generate: Callable that sends a prompt to the model
                and returns the response string.

        Returns:
            State after all solvers have processed it (or after
            the first solver that sets ``completed = True``).
        """
        for solver in self._solvers:
            state = solver.solve(state, generate)
            if state.completed:
                break
        return state

    @property
    def name(self) -> str:
        """Return descriptive pipeline name.

        Returns:
            A string like ``"Pipeline(CoT -> FewShot -> Generate)"``
            listing each solver in order.
        """
        inner = " -> ".join(s.name for s in self._solvers)
        return f"Pipeline({inner})"


def chain(*solvers: Solver) -> Solver:
    """Chain multiple solvers into a pipeline.

    Creates a composite solver that runs each solver in sequence,
    passing the state from one to the next. If any solver sets
    ``state.completed = True``, the pipeline short-circuits and
    returns immediately.

    Args:
        *solvers: Solvers to chain in order. Must provide at
            least one solver.

    Returns:
        A composite :class:`Solver` that runs the full pipeline.

    Raises:
        ValueError: If no solvers are provided.

    Example:
        Chain CoT reasoning with generation::

            from mltk.eval.solvers import (
                ChainOfThoughtSolver,
                GenerateSolver,
                chain,
            )

            pipeline = chain(
                ChainOfThoughtSolver(),
                GenerateSolver(),
            )
            assert pipeline.name == (
                "Pipeline(ChainOfThoughtSolver -> GenerateSolver)"
            )
    """
    if not solvers:
        raise ValueError("chain() requires at least one solver")
    return _PipelineSolver(solvers)
