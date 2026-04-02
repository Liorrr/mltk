"""Tests for mltk.eval.solvers — solver pipeline."""

from __future__ import annotations

import pytest

from mltk.eval._types import EvalSample, EvalState
from mltk.eval.solvers import (
    ChainOfThoughtSolver,
    FewShotSolver,
    GenerateSolver,
    Solver,
    chain,
)

# ---------------------------------------------------------------
# Mock model functions
# ---------------------------------------------------------------


def echo_model(prompt: str) -> str:
    """Returns the prompt back."""
    return prompt


def fixed_model(prompt: str) -> str:
    """Always returns '4'."""
    return "4"


def cot_model(prompt: str) -> str:
    """Returns step-by-step + answer."""
    return (
        "Step 1: think\n"
        "Step 2: reason\n"
        "Answer: 4"
    )


def _make_state(
    inp: str = "2+2?",
    target: str | None = "4",
    completed: bool = False,
    messages: list | None = None,
) -> EvalState:
    """Build an EvalState with sensible defaults."""
    return EvalState(
        sample=EvalSample(input=inp, target=target),
        completed=completed,
        messages=messages if messages is not None else [],
    )


# ===============================================================
# GenerateSolver
# ===============================================================


class TestGenerateSolver:
    """GenerateSolver: basic model generation."""

    def test_stores_output(self):
        # SCENARIO: generate solver stores model response
        # WHY: core contract — output must be populated
        # EXPECTED: state.output == model response
        solver = GenerateSolver()
        state = _make_state()
        result = solver.solve(state, fixed_model)
        assert result.output == "4"

    def test_uses_sample_input_as_prompt(self):
        # SCENARIO: no messages — uses sample.input
        # WHY: default prompt source is sample.input
        # EXPECTED: echo_model returns the input back
        solver = GenerateSolver()
        state = _make_state(inp="hello world")
        result = solver.solve(state, echo_model)
        assert result.output == "hello world"

    def test_adds_assistant_message(self):
        # SCENARIO: after generation, message list grows
        # WHY: conversation history must be tracked
        # EXPECTED: last message is assistant role
        solver = GenerateSolver()
        state = _make_state()
        result = solver.solve(state, fixed_model)
        assert len(result.messages) >= 1
        last = result.messages[-1]
        assert last["role"] == "assistant"
        assert last["content"] == "4"

    def test_skips_when_completed(self):
        # SCENARIO: state.completed is True
        # WHY: completed flag = short-circuit
        # EXPECTED: output unchanged, no messages added
        solver = GenerateSolver()
        state = _make_state(completed=True)
        original_output = state.output
        result = solver.solve(state, fixed_model)
        assert result.output == original_output
        assert len(result.messages) == 0

    def test_uses_last_message_if_present(self):
        # SCENARIO: messages list has prior content
        # WHY: multi-turn — last message overrides input
        # EXPECTED: model receives last message content
        solver = GenerateSolver()
        msgs = [
            {"role": "user", "content": "override prompt"}
        ]
        state = _make_state(messages=msgs)
        result = solver.solve(state, echo_model)
        assert result.output == "override prompt"

    def test_falls_back_to_input_on_empty_message(self):
        # SCENARIO: messages exist but last has no content
        # WHY: empty content must not break generation
        # EXPECTED: falls back to sample.input
        solver = GenerateSolver()
        msgs = [{"role": "user", "content": ""}]
        state = _make_state(inp="fallback input")
        state.messages = msgs
        result = solver.solve(state, echo_model)
        assert result.output == "fallback input"

    def test_name_property(self):
        # SCENARIO: check solver name
        # WHY: name is used for logging and pipeline repr
        # EXPECTED: "GenerateSolver"
        solver = GenerateSolver()
        assert solver.name == "GenerateSolver"

    def test_empty_input(self):
        # SCENARIO: sample.input is empty string
        # WHY: edge case — must not raise
        # EXPECTED: model receives empty string
        solver = GenerateSolver()
        state = _make_state(inp="")
        result = solver.solve(state, echo_model)
        assert result.output == ""


# ===============================================================
# ChainOfThoughtSolver
# ===============================================================


class TestChainOfThoughtSolver:
    """ChainOfThoughtSolver: CoT instruction injection."""

    def test_prepends_cot_instruction(self):
        # SCENARIO: default CoT prefixed to prompt
        # WHY: core contract — CoT must appear in prompt
        # EXPECTED: model receives CoT + input
        solver = ChainOfThoughtSolver()
        prompts = []

        def capture(prompt: str) -> str:
            prompts.append(prompt)
            return "result"

        state = _make_state(inp="What is 2+2?")
        solver.solve(state, capture)
        assert len(prompts) == 1
        assert "step by step" in prompts[0].lower()
        assert "What is 2+2?" in prompts[0]

    def test_custom_template(self):
        # SCENARIO: user provides custom CoT template
        # WHY: template customization is part of API
        # EXPECTED: custom template used instead of default
        custom = "Reason carefully before answering."
        solver = ChainOfThoughtSolver(template=custom)
        prompts = []

        def capture(prompt: str) -> str:
            prompts.append(prompt)
            return "done"

        state = _make_state(inp="test")
        solver.solve(state, capture)
        assert "Reason carefully" in prompts[0]

    def test_skips_when_completed(self):
        # SCENARIO: state.completed is True
        # WHY: completed flag = skip
        # EXPECTED: no model call, state unchanged
        solver = ChainOfThoughtSolver()
        called = []

        def track(prompt: str) -> str:
            called.append(1)
            return "x"

        state = _make_state(completed=True)
        result = solver.solve(state, track)
        assert len(called) == 0
        assert result.output == ""

    def test_stores_output(self):
        # SCENARIO: CoT solver stores model response
        # WHY: output must be populated after solving
        # EXPECTED: state.output has model response
        solver = ChainOfThoughtSolver()
        state = _make_state()
        result = solver.solve(state, cot_model)
        assert "Answer: 4" in result.output

    def test_adds_messages(self):
        # SCENARIO: messages list grows after solving
        # WHY: conversation history tracking
        # EXPECTED: system + user + assistant messages
        solver = ChainOfThoughtSolver()
        state = _make_state()
        result = solver.solve(state, fixed_model)
        roles = [m["role"] for m in result.messages]
        assert "system" in roles
        assert "assistant" in roles

    def test_name_property(self):
        # SCENARIO: check solver name
        # WHY: name used for pipeline representation
        # EXPECTED: "ChainOfThoughtSolver"
        solver = ChainOfThoughtSolver()
        assert solver.name == "ChainOfThoughtSolver"


# ===============================================================
# FewShotSolver
# ===============================================================


class TestFewShotSolver:
    """FewShotSolver: example-based prompting."""

    def test_formats_examples_in_prompt(self):
        # SCENARIO: examples appear in prompt
        # WHY: core contract — few-shot examples
        # EXPECTED: both examples in prompt text
        solver = FewShotSolver(
            examples=[("2+2", "4"), ("3+3", "6")]
        )
        prompts = []

        def capture(prompt: str) -> str:
            prompts.append(prompt)
            return "10"

        state = _make_state(inp="5+5")
        solver.solve(state, capture)
        assert "2+2" in prompts[0]
        assert "4" in prompts[0]
        assert "3+3" in prompts[0]
        assert "6" in prompts[0]
        assert "5+5" in prompts[0]

    def test_custom_template(self):
        # SCENARIO: user provides custom format template
        # WHY: template customization is part of API
        # EXPECTED: custom format used for examples
        tpl = "Input: {input} -> Output: {output}"
        solver = FewShotSolver(
            examples=[("a", "b")], template=tpl
        )
        prompts = []

        def capture(prompt: str) -> str:
            prompts.append(prompt)
            return "c"

        state = _make_state(inp="x")
        solver.solve(state, capture)
        assert "Input: a -> Output: b" in prompts[0]

    def test_raises_on_empty_examples(self):
        # SCENARIO: empty examples list
        # WHY: must validate input at construction
        # EXPECTED: ValueError raised
        with pytest.raises(ValueError, match="at least one"):
            FewShotSolver(examples=[])

    def test_single_example(self):
        # SCENARIO: exactly one example
        # WHY: minimum valid case
        # EXPECTED: example appears, solver works
        solver = FewShotSolver(examples=[("hi", "bye")])
        prompts = []

        def capture(prompt: str) -> str:
            prompts.append(prompt)
            return "result"

        state = _make_state(inp="greet")
        solver.solve(state, capture)
        assert "hi" in prompts[0]
        assert "bye" in prompts[0]

    def test_many_examples(self):
        # SCENARIO: 10 examples
        # WHY: must handle arbitrary count
        # EXPECTED: all examples in prompt
        examples = [(f"q{i}", f"a{i}") for i in range(10)]
        solver = FewShotSolver(examples=examples)
        prompts = []

        def capture(prompt: str) -> str:
            prompts.append(prompt)
            return "final"

        state = _make_state(inp="q_new")
        solver.solve(state, capture)
        for i in range(10):
            assert f"q{i}" in prompts[0]
            assert f"a{i}" in prompts[0]

    def test_skips_when_completed(self):
        # SCENARIO: state.completed is True
        # WHY: completed flag = skip
        # EXPECTED: no model call
        solver = FewShotSolver(examples=[("a", "b")])
        called = []

        def track(prompt: str) -> str:
            called.append(1)
            return "x"

        state = _make_state(completed=True)
        solver.solve(state, track)
        assert len(called) == 0

    def test_stores_output(self):
        # SCENARIO: output populated after solve
        # WHY: core contract
        # EXPECTED: state.output has model response
        solver = FewShotSolver(examples=[("2+2", "4")])
        state = _make_state(inp="3+3")
        result = solver.solve(state, fixed_model)
        assert result.output == "4"

    def test_adds_messages(self):
        # SCENARIO: messages grow after solve
        # WHY: conversation history
        # EXPECTED: user + assistant messages added
        solver = FewShotSolver(examples=[("a", "b")])
        state = _make_state()
        result = solver.solve(state, fixed_model)
        roles = [m["role"] for m in result.messages]
        assert "user" in roles
        assert "assistant" in roles

    def test_name_property(self):
        # SCENARIO: check solver name
        # EXPECTED: "FewShotSolver"
        solver = FewShotSolver(examples=[("a", "b")])
        assert solver.name == "FewShotSolver"


# ===============================================================
# chain()
# ===============================================================


class TestChain:
    """chain(): composing solvers into pipelines."""

    def test_runs_solvers_in_sequence(self):
        # SCENARIO: chain two solvers
        # WHY: order must be preserved
        # EXPECTED: second solver sees first's output
        order = []

        class S1(Solver):
            def solve(self, state, generate):
                order.append("s1")
                state.metadata["s1"] = True
                return state

        class S2(Solver):
            def solve(self, state, generate):
                order.append("s2")
                state.metadata["s2"] = True
                return state

        pipeline = chain(S1(), S2())
        state = _make_state()
        result = pipeline.solve(state, fixed_model)
        assert order == ["s1", "s2"]
        assert result.metadata["s1"] is True
        assert result.metadata["s2"] is True

    def test_short_circuits_on_completed(self):
        # SCENARIO: first solver sets completed=True
        # WHY: short-circuit must prevent second solver
        # EXPECTED: second solver never called
        class Stopper(Solver):
            def solve(self, state, generate):
                state.completed = True
                return state

        class NeverCalled(Solver):
            def solve(self, state, generate):
                raise AssertionError("Should not run")

        pipeline = chain(Stopper(), NeverCalled())
        state = _make_state()
        result = pipeline.solve(state, fixed_model)
        assert result.completed is True

    def test_single_solver(self):
        # SCENARIO: chain with one solver
        # WHY: edge case — single solver pipeline
        # EXPECTED: works identically to calling solver
        pipeline = chain(GenerateSolver())
        state = _make_state()
        result = pipeline.solve(state, fixed_model)
        assert result.output == "4"

    def test_three_solvers(self):
        # SCENARIO: chain with three solvers
        # WHY: realistic pipeline depth
        # EXPECTED: all three run in order
        order = []

        class Numbered(Solver):
            def __init__(self, n):
                self._n = n

            def solve(self, state, generate):
                order.append(self._n)
                return state

        pipeline = chain(
            Numbered(1), Numbered(2), Numbered(3)
        )
        state = _make_state()
        pipeline.solve(state, fixed_model)
        assert order == [1, 2, 3]

    def test_empty_chain_raises(self):
        # SCENARIO: chain() with no arguments
        # WHY: must validate at least one solver
        # EXPECTED: ValueError
        with pytest.raises(ValueError, match="at least one"):
            chain()

    def test_pipeline_name(self):
        # SCENARIO: pipeline solver has descriptive name
        # WHY: debugging and logging
        # EXPECTED: name contains solver names
        pipeline = chain(
            GenerateSolver(), GenerateSolver()
        )
        assert "GenerateSolver" in pipeline.name
        assert "Pipeline" in pipeline.name


# ===============================================================
# Solver ABC and custom implementations
# ===============================================================


class TestSolverABC:
    """Solver base class and custom implementations."""

    def test_solver_name_returns_class_name(self):
        # SCENARIO: custom solver inherits name property
        # WHY: default name is class name
        # EXPECTED: name == "MySolver"
        class MySolver(Solver):
            def solve(self, state, generate):
                return state

        solver = MySolver()
        assert solver.name == "MySolver"

    def test_custom_solver_works(self):
        # SCENARIO: user implements Solver subclass
        # WHY: extensibility is part of API
        # EXPECTED: custom solve() called correctly
        class Upper(Solver):
            def solve(self, state, generate):
                state.output = state.output.upper()
                return state

        state = _make_state()
        state.output = "hello"
        result = Upper().solve(state, fixed_model)
        assert result.output == "HELLO"

    def test_solver_with_model_exception(self):
        # SCENARIO: model function raises exception
        # WHY: solver should propagate model errors
        # EXPECTED: exception propagates
        def bad_model(prompt: str) -> str:
            raise RuntimeError("model failed")

        solver = GenerateSolver()
        state = _make_state()
        with pytest.raises(RuntimeError, match="model"):
            solver.solve(state, bad_model)

    def test_state_mutation_preserved(self):
        # SCENARIO: chain preserves metadata mutations
        # WHY: state is mutable, mutations persist
        # EXPECTED: both solvers' metadata present
        class AddMeta(Solver):
            def __init__(self, key, val):
                self._key = key
                self._val = val

            def solve(self, state, generate):
                state.metadata[self._key] = self._val
                return state

        pipeline = chain(
            AddMeta("a", 1), AddMeta("b", 2)
        )
        state = _make_state()
        result = pipeline.solve(state, fixed_model)
        assert result.metadata == {"a": 1, "b": 2}

    def test_messages_accumulate(self):
        # SCENARIO: two GenerateSolvers in chain
        # WHY: each adds an assistant message
        # EXPECTED: 2 assistant messages
        pipeline = chain(
            GenerateSolver(), GenerateSolver()
        )
        state = _make_state()
        result = pipeline.solve(state, fixed_model)
        assistant_msgs = [
            m for m in result.messages
            if m["role"] == "assistant"
        ]
        assert len(assistant_msgs) == 2
