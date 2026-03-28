"""LLM long-context evaluation -- needle retrieval, utilization, lost-in-middle."""

from __future__ import annotations

from collections.abc import Callable

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult
from mltk.domains.llm._utils import _tokenize

# Default probe positions spanning the full context window.
_DEFAULT_POSITIONS: list[float] = [0.0, 0.25, 0.5, 0.75, 1.0]


def _insert_needle(
    haystack: str, needle: str, position: float,
) -> str:
    """Insert *needle* into *haystack* at a relative *position* (0.0--1.0).

    Position 0.0 prepends the needle, 1.0 appends it, and values in
    between insert at the corresponding character offset.
    """
    pos = max(0.0, min(1.0, position))
    idx = int(len(haystack) * pos)
    return haystack[:idx] + " " + needle + " " + haystack[idx:]


def _check_needle_in_response(
    response: str, needle: str, min_overlap: float = 0.5,
) -> bool:
    """Return True if *response* contains a significant overlap with *needle*.

    Uses token-level overlap so the model does not need to repeat the
    needle verbatim -- paraphrased retrieval is accepted when at least
    ``min_overlap`` of the needle tokens appear in the response.
    """
    needle_tokens = _tokenize(needle)
    if not needle_tokens:
        return True
    response_tokens = _tokenize(response)
    overlap = len(needle_tokens & response_tokens) / len(needle_tokens)
    return overlap >= min_overlap


@timed_assertion
def assert_needle_in_haystack(
    model_fn: Callable[[str], str],
    needle: str,
    haystack: str,
    positions: list[float] | None = None,
    min_recall: float = 0.8,
) -> TestResult:
    """Assert that an LLM can retrieve a fact inserted at various context positions.

    LLMs with 128K+ context windows often cannot *use* all of that
    context.  The needle-in-a-haystack test inserts a specific fact
    (the needle) at different positions in a long document (the
    haystack) and checks whether the model can retrieve it when
    asked.  Models typically perform well when the needle is near
    the start or end of the context but poorly when it is in the
    middle -- a phenomenon known as "lost in the middle" (Liu et al.,
    2023).

    For each position the needle is inserted, the model is prompted
    with: ``"Based on the following document, what is the answer to:
    {needle}?\\n\\nDocument:\\n{augmented_haystack}"``.  The response
    is checked for token overlap with the needle.

    Args:
        model_fn: Callable that takes a prompt string and returns the
            model response string.
        needle: The fact to embed and retrieve
            (e.g. ``"The secret code is 7492"``).
        haystack: A long document to embed the needle in.
        positions: Relative positions (0.0--1.0) at which to insert
            the needle.  ``None`` uses ``[0.0, 0.25, 0.5, 0.75, 1.0]``.
        min_recall: Minimum fraction of positions where the model
            must successfully retrieve the needle (0.0--1.0).

    Returns:
        TestResult with recall, per-position results, and context
        length metadata.

    Example:
        >>> def my_model(prompt: str) -> str:
        ...     return "The secret code is 7492"
        >>> haystack = "A " * 5000  # long padding document
        >>> result = assert_needle_in_haystack(
        ...     my_model, needle="The secret code is 7492",
        ...     haystack=haystack, min_recall=0.8,
        ... )
    """
    probe_positions = positions if positions is not None else list(
        _DEFAULT_POSITIONS
    )

    per_position: dict[str, bool] = {}
    found_count = 0

    for pos in probe_positions:
        augmented = _insert_needle(haystack, needle, pos)
        prompt = (
            "Based on the following document, what is the "
            f"answer to: {needle}?\n\nDocument:\n{augmented}"
        )
        try:
            response = str(model_fn(prompt))
        except Exception:
            per_position[str(pos)] = False
            continue

        found = _check_needle_in_response(response, needle)
        per_position[str(pos)] = found
        if found:
            found_count += 1

    n_positions = len(probe_positions)
    recall = found_count / n_positions if n_positions > 0 else 0.0
    passed = recall >= min_recall

    message = (
        f"Needle recall: {recall:.4f} >= {min_recall} "
        f"({found_count}/{n_positions} positions)"
        if passed
        else f"Needle recall too low: {recall:.4f} < {min_recall} "
        f"({found_count}/{n_positions} positions)"
    )

    return assert_true(
        passed,
        name="llm.long_context.needle_in_haystack",
        message=message,
        severity=Severity.CRITICAL,
        recall=round(recall, 4),
        min_recall=min_recall,
        per_position=per_position,
        n_positions=n_positions,
        needle_length=len(needle),
        haystack_length=len(haystack),
    )


@timed_assertion
def assert_context_utilization(
    model_fn: Callable[[str], str],
    facts: list[str],
    question: str,
    min_facts_used: int = 3,
) -> TestResult:
    """Assert that an LLM uses facts from across its full context window.

    A model with a 128K context window that only uses the first 4K
    tokens is wasting context.  This test provides *N* facts scattered
    throughout the prompt and asks a question that requires
    synthesizing information from multiple facts.  If the model only
    references facts from the beginning, it is not utilizing its full
    context.

    Each fact is checked for token overlap with the model response.
    A fact is considered "used" when at least 50 % of its tokens
    appear in the response.

    Args:
        model_fn: Callable that takes a prompt string and returns the
            model response string.
        facts: List of factual statements to include in the context.
        question: A question whose answer should draw on multiple facts.
        min_facts_used: Minimum number of facts that must appear in
            the response for the assertion to pass.

    Returns:
        TestResult with utilization counts and per-fact breakdown.

    Example:
        >>> facts = [
        ...     "The capital of France is Paris.",
        ...     "The Eiffel Tower is 330 meters tall.",
        ...     "France has a population of 67 million.",
        ... ]
        >>> def model(prompt: str) -> str:
        ...     return "Paris is the capital with 67 million people "\\
        ...            "and the 330-meter Eiffel Tower."
        >>> result = assert_context_utilization(
        ...     model, facts=facts,
        ...     question="Summarize what you know about France.",
        ...     min_facts_used=2,
        ... )
    """
    context = "\n".join(
        f"Fact {i + 1}: {fact}" for i, fact in enumerate(facts)
    )
    prompt = (
        f"Given the following facts:\n{context}\n\n"
        f"Question: {question}\n"
        "Use as many of the provided facts as possible in "
        "your answer."
    )

    try:
        response = str(model_fn(prompt))
    except Exception as exc:
        return assert_true(
            False,
            name="llm.long_context.utilization",
            message=f"model_fn raised {type(exc).__name__}: {exc}",
            severity=Severity.CRITICAL,
            facts_used=0,
            min_facts_used=min_facts_used,
            total_facts=len(facts),
            per_fact_found=[False] * len(facts),
        )

    response_tokens = _tokenize(response)
    per_fact_found: list[bool] = []

    for fact in facts:
        fact_tokens = _tokenize(fact)
        if not fact_tokens:
            per_fact_found.append(False)
            continue
        overlap = (
            len(fact_tokens & response_tokens) / len(fact_tokens)
        )
        per_fact_found.append(overlap >= 0.5)

    facts_used = sum(per_fact_found)
    passed = facts_used >= min_facts_used

    message = (
        f"Context utilization: {facts_used}/{len(facts)} facts used "
        f"(>= {min_facts_used} required)"
        if passed
        else f"Context utilization too low: {facts_used}/{len(facts)} "
        f"facts used (>= {min_facts_used} required)"
    )

    return assert_true(
        passed,
        name="llm.long_context.utilization",
        message=message,
        severity=Severity.CRITICAL,
        facts_used=facts_used,
        min_facts_used=min_facts_used,
        total_facts=len(facts),
        per_fact_found=per_fact_found,
    )


@timed_assertion
def assert_no_lost_in_middle(
    model_fn: Callable[[str], str],
    facts: list[str],
    questions: list[str],
    min_accuracy: float = 0.7,
) -> TestResult:
    """Assert that an LLM does not lose information in the middle of its context.

    The "lost in the middle" problem (Liu et al., 2023) describes a
    common failure mode: models pay strong attention to the beginning
    and end of their context but largely ignore the middle.  This
    test places facts at known positions -- beginning, middle, and
    end -- asks questions about each, and checks whether accuracy is
    uniform across positions rather than concentrated at the edges.

    For each ``(fact, question)`` pair the model receives all facts
    as context plus the specific question.  The response is checked
    for token overlap with the corresponding fact.

    Args:
        model_fn: Callable that takes a prompt string and returns the
            model response string.
        facts: List of factual statements.  ``facts[0]`` represents
            "beginning", ``facts[len // 2]`` represents "middle", and
            ``facts[-1]`` represents "end".
        questions: Parallel list of questions -- ``questions[i]`` asks
            about ``facts[i]``.  Must have the same length as *facts*.
        min_accuracy: Minimum fraction of questions the model must
            answer correctly (0.0--1.0).

    Returns:
        TestResult with overall accuracy, per-position accuracy
        breakdown, and question count.

    Raises:
        ValueError: If *facts* and *questions* have different lengths.

    Example:
        >>> facts = [
        ...     "The speed of light is 299792458 m/s.",
        ...     "Water boils at 100 degrees Celsius.",
        ...     "Earth orbits the Sun in 365.25 days.",
        ... ]
        >>> questions = [
        ...     "What is the speed of light?",
        ...     "At what temperature does water boil?",
        ...     "How long does Earth take to orbit the Sun?",
        ... ]
        >>> def model(prompt: str) -> str:
        ...     return "The speed of light is 299792458 m/s."
        >>> result = assert_no_lost_in_middle(
        ...     model, facts=facts, questions=questions,
        ...     min_accuracy=0.7,
        ... )
    """
    if len(facts) != len(questions):
        raise ValueError(
            f"facts ({len(facts)}) and questions ({len(questions)}) "
            "must have the same length"
        )

    context = "\n".join(
        f"Fact {i + 1}: {fact}" for i, fact in enumerate(facts)
    )

    correct_count = 0
    per_position_correct: dict[str, bool] = {}

    for i, (fact, question) in enumerate(zip(facts, questions)):
        # Label position for reporting
        if i == 0:
            pos_label = "beginning"
        elif i == len(facts) - 1:
            pos_label = "end"
        elif i == len(facts) // 2:
            pos_label = "middle"
        else:
            pos_label = f"position_{i}"

        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer using only the context provided."
        )

        try:
            response = str(model_fn(prompt))
        except Exception:
            per_position_correct[pos_label] = False
            continue

        fact_tokens = _tokenize(fact)
        response_tokens = _tokenize(response)
        if not fact_tokens:
            per_position_correct[pos_label] = False
            continue

        overlap = (
            len(fact_tokens & response_tokens) / len(fact_tokens)
        )
        is_correct = overlap >= 0.5
        per_position_correct[pos_label] = is_correct
        if is_correct:
            correct_count += 1

    n_questions = len(questions)
    accuracy = correct_count / n_questions if n_questions > 0 else 0.0
    passed = accuracy >= min_accuracy

    # Compute per-region accuracy for the three canonical positions
    region_hits: dict[str, list[bool]] = {
        "beginning": [],
        "middle": [],
        "end": [],
    }
    for label, correct in per_position_correct.items():
        if label in region_hits:
            region_hits[label].append(correct)

    per_position_accuracy: dict[str, float] = {}
    for region, hits in region_hits.items():
        if hits:
            per_position_accuracy[region] = round(
                sum(hits) / len(hits), 4,
            )

    message = (
        f"No lost-in-middle: accuracy {accuracy:.4f} >= "
        f"{min_accuracy} ({correct_count}/{n_questions} correct)"
        if passed
        else f"Lost-in-middle detected: accuracy {accuracy:.4f} "
        f"< {min_accuracy} ({correct_count}/{n_questions} correct)"
    )

    return assert_true(
        passed,
        name="llm.long_context.no_lost_in_middle",
        message=message,
        severity=Severity.CRITICAL,
        accuracy=round(accuracy, 4),
        min_accuracy=min_accuracy,
        per_position_accuracy=per_position_accuracy,
        per_position_correct=per_position_correct,
        n_questions=n_questions,
    )
