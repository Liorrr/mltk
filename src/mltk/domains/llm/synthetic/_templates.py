"""Prompt templates for each ``QuestionType``.

Each question type has:

- **system_prompt**: Sets the LLM's role for generation.
- **user_prompt_template**: Contains ``{context}`` (and
  ``{context_b}`` for multi-hop) placeholders.
- **parse_response**: Extracts question and answer strings
  from the raw LLM response.

Template mode fallback functions generate deterministic
QA pairs from text structure without any LLM call.
"""

from __future__ import annotations

import json
import re
from typing import Any

# ---------------------------------------------------------------
# Prompt templates — one dict per QuestionType
# ---------------------------------------------------------------

FACTUAL_TEMPLATE: dict[str, str] = {
    "system_prompt": (
        "You are generating a test QA pair from a "
        "document chunk. Return valid JSON only."
    ),
    "user_prompt": (
        "Context:\n{context}\n\n"
        "Generate ONE factual question whose answer is "
        "explicitly stated in the context above.\n"
        'Return JSON: {{"question": "...", "answer": "..."}}'
        "\n\nRules:\n"
        "- The answer must be directly found in the "
        "context (not inferred).\n"
        "- The question must make sense without reading "
        "the context.\n"
        "- Do NOT start the question with "
        '"According to the text..." or '
        '"Based on the passage...".'
    ),
}

REASONING_TEMPLATE: dict[str, str] = {
    "system_prompt": (
        "You are generating a reasoning QA pair that "
        "requires inference. Return valid JSON only."
    ),
    "user_prompt": (
        "Context:\n{context}\n\n"
        "Generate ONE reasoning question that requires "
        "inference beyond what is literally stated.\n"
        "The answer must still be derivable from the "
        "context, but requires a logical step.\n"
        'Return JSON: {{"question": "...", "answer": "..."}}'
    ),
}

MULTI_HOP_TEMPLATE: dict[str, str] = {
    "system_prompt": (
        "You are generating a multi-hop QA pair that "
        "requires information from multiple contexts. "
        "Return valid JSON only."
    ),
    "user_prompt": (
        "Context A:\n{context}\n\n"
        "Context B:\n{context_b}\n\n"
        "Generate ONE question that requires information "
        "from BOTH contexts to answer.\n"
        "The question should not be answerable from "
        "either context alone.\n"
        'Return JSON: {{"question": "...", "answer": "..."}}'
    ),
}

COUNTERFACTUAL_TEMPLATE: dict[str, str] = {
    "system_prompt": (
        "You are generating a counterfactual QA pair. "
        "Return valid JSON only."
    ),
    "user_prompt": (
        "Context:\n{context}\n\n"
        'Generate ONE counterfactual question that begins '
        'with "What if..." or "If ... were not true..."\n'
        "The question should be hypothetical but directly "
        "related to the context.\n"
        "The reference answer should explain the "
        "consequence, grounded in the context.\n"
        'Return JSON: {{"question": "...", "answer": "..."}}'
    ),
}

OUT_OF_SCOPE_TEMPLATE: dict[str, str] = {
    "system_prompt": (
        "You are generating an out-of-scope QA pair. "
        "Return valid JSON only."
    ),
    "user_prompt": (
        "Context:\n{context}\n\n"
        "Generate ONE question that is RELATED to the "
        "topic of the context but CANNOT be answered "
        "from the context alone.\n"
        'Return JSON: {{"question": "...", '
        '"answer": "This information is not available '
        'in the provided context."}}'
    ),
}

MULTI_HOP_ENHANCED_TEMPLATE: dict[str, str] = {
    "system_prompt": (
        "You are generating a multi-hop QA pair that "
        "requires combining information from multiple "
        "passages. Return valid JSON only."
    ),
    "user_prompt": (
        "Given these passages:\n{contexts}\n\n"
        "Generate a question that requires combining "
        "information from at least two of the passages "
        "above to answer correctly. The question "
        "should not be answerable from any single "
        "passage alone.\n"
        'Return JSON: {{"question": "...", '
        '"answer": "..."}}'
    ),
}

CONVERSATIONAL_TEMPLATE: dict[str, str] = {
    "system_prompt": (
        "You are generating a multi-turn conversation "
        "about a topic. Return valid JSON only."
    ),
    "user_prompt": (
        "Context:\n{context}\n\n"
        "Generate a {turns}-turn conversation about "
        "this topic. Each turn has a question and an "
        "answer. Follow-up questions should build on "
        "the previous answer.\n"
        "Return JSON: "
        '{{"turns": ['
        '{{"question": "...", "answer": "..."}}, '
        '{{"question": "...", "answer": "..."}} '
        "]}}"
    ),
}

DISTRACTING_TEMPLATE: dict[str, str] = {
    "system_prompt": (
        "You are generating a QA pair with a "
        "misleading element. Return valid JSON only."
    ),
    "user_prompt": (
        "Target context:\n{context}\n\n"
        "Distractor context:\n{distractor}\n\n"
        "Generate a question about the target context "
        "that includes a misleading detail from the "
        "distractor context. The correct answer should "
        "be based ONLY on the target context.\n"
        'Return JSON: {{"question": "...", '
        '"answer": "..."}}'
    ),
}


# Map from QuestionType.value -> template dict
TEMPLATES: dict[str, dict[str, str]] = {
    "factual": FACTUAL_TEMPLATE,
    "reasoning": REASONING_TEMPLATE,
    "multi_hop": MULTI_HOP_TEMPLATE,
    "counterfactual": COUNTERFACTUAL_TEMPLATE,
    "out_of_scope": OUT_OF_SCOPE_TEMPLATE,
    "conversational": CONVERSATIONAL_TEMPLATE,
    "distracting": DISTRACTING_TEMPLATE,
    "multi_hop_enhanced": MULTI_HOP_ENHANCED_TEMPLATE,
}


# ---------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------

def parse_response(
    raw: str,
) -> dict[str, str] | None:
    """Extract ``question`` and ``answer`` from LLM output.

    Tries JSON parsing first.  Falls back to regex extraction
    for responses that wrap JSON in markdown fences or contain
    extra text around the JSON object.

    Args:
        raw: The raw string returned by the LLM callable.

    Returns:
        Dict with ``"question"`` and ``"answer"`` keys, or
        ``None`` if parsing fails entirely.
    """
    if not raw or not raw.strip():
        return None

    # Strategy 1: direct JSON parse
    result = _try_json_parse(raw)
    if result is not None:
        return result

    # Strategy 2: extract JSON from markdown fences
    result = _try_fenced_json(raw)
    if result is not None:
        return result

    # Strategy 3: regex fallback
    return _try_regex_extract(raw)


def build_prompt(
    question_type_value: str,
    context: str,
    context_b: str | None = None,
    *,
    contexts: list[str] | None = None,
    distractor: str | None = None,
    turns: int = 2,
) -> str:
    """Build the full prompt for a given question type.

    Combines the system prompt and user prompt template,
    substituting ``{context}`` and optionally other
    placeholders depending on the question type.

    Args:
        question_type_value: The ``.value`` string of a
            ``QuestionType`` enum member.
        context: Primary context text.
        context_b: Secondary context for multi-hop
            questions.  Ignored for other types.
        contexts: List of context chunks for enhanced
            multi-hop questions.
        distractor: Distractor context for distracting
            questions.
        turns: Number of conversation turns for
            conversational questions.  Default 2.

    Returns:
        The complete prompt string ready to send to an LLM.
    """
    template = TEMPLATES.get(question_type_value)
    if template is None:
        template = FACTUAL_TEMPLATE

    system = template["system_prompt"]
    user = template["user_prompt"]

    if contexts is not None:
        formatted = "\n\n".join(
            f"Passage {i + 1}:\n{c}"
            for i, c in enumerate(contexts)
        )
        user = user.replace("{contexts}", formatted)
    user = user.replace("{context}", context)
    if context_b is not None:
        user = user.replace("{context_b}", context_b)
    if distractor is not None:
        user = user.replace("{distractor}", distractor)
    user = user.replace("{turns}", str(turns))

    return f"{system}\n\n{user}"


def parse_conversational_response(
    raw: str,
) -> list[dict[str, str]] | None:
    """Extract a list of QA turns from an LLM response.

    Expects a JSON object with a ``"turns"`` key containing
    a list of ``{"question": ..., "answer": ...}`` dicts.

    Args:
        raw: The raw string returned by the LLM callable.

    Returns:
        List of dicts with ``"question"`` and ``"answer"``
        keys, or ``None`` if parsing fails.
    """
    if not raw or not raw.strip():
        return None

    data = _try_json_parse_any(raw)
    if data is None:
        # Try fenced JSON
        match = re.search(
            r"```(?:json)?\s*\n?(.*?)\n?\s*```",
            raw,
            re.DOTALL,
        )
        if match:
            data = _try_json_parse_any(match.group(1))

    if data is None:
        return None

    return _validate_turns(data)


def _try_json_parse_any(raw: str) -> Any | None:
    """Attempt JSON parse, returning any valid result."""
    try:
        return json.loads(raw.strip())
    except (json.JSONDecodeError, ValueError):
        return None


def _validate_turns(
    data: Any,
) -> list[dict[str, str]] | None:
    """Validate a turns list from parsed JSON."""
    turns_list: list[Any] | None = None
    if isinstance(data, dict):
        turns_list = data.get("turns")
    elif isinstance(data, list):
        turns_list = data

    if not isinstance(turns_list, list):
        return None
    if not turns_list:
        return None

    result: list[dict[str, str]] = []
    for item in turns_list:
        validated = _validate_qa_dict(item)
        if validated is None:
            return None
        result.append(validated)
    return result


# ---------------------------------------------------------------
# Template-mode sentence extraction
# ---------------------------------------------------------------

def extract_key_sentences(
    text: str,
    max_sentences: int = 5,
) -> list[str]:
    """Extract the most informative sentences from *text*.

    Heuristic: longer sentences with nouns (capitalized words)
    are more likely to contain factual content worth asking
    about.  This is intentionally simple — template mode is a
    CI smoke-test tool, not a replacement for LLM generation.

    Args:
        text: Source chunk text.
        max_sentences: Maximum sentences to return.

    Returns:
        List of sentences sorted by estimated information
        content (most informative first).
    """
    sentences = _naive_sentence_split(text)
    if not sentences:
        return []

    scored: list[tuple[float, str]] = []
    for sent in sentences:
        words = sent.split()
        if len(words) < 4:
            continue
        # Heuristic score: word count + capitalized word bonus
        caps = sum(
            1 for w in words if w[0].isupper()
        )
        score = len(words) + caps * 2.0
        scored.append((score, sent))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:max_sentences]]


def declarative_to_interrogative(
    sentence: str,
) -> str | None:
    """Convert a declarative sentence to a question.

    Uses pattern matching on common sentence structures.
    Returns ``None`` if no pattern matches.

    Args:
        sentence: A declarative sentence.

    Returns:
        An interrogative form, or ``None``.

    Example::

        >>> declarative_to_interrogative(
        ...     "Python is a programming language."
        ... )
        'What is Python?'
    """
    sentence = sentence.strip().rstrip(".")

    # Pattern: "X is Y" -> "What is X?"
    match = re.match(
        r"^(.+?)\s+(is|are|was|were)\s+(.+)$",
        sentence,
        re.IGNORECASE,
    )
    if match:
        subject = match.group(1)
        verb = match.group(2)
        return f"What {verb} {subject}?"

    # Pattern: "X has Y" -> "What does X have?"
    match = re.match(
        r"^(.+?)\s+(?:has|have)\s+(.+)$",
        sentence,
        re.IGNORECASE,
    )
    if match:
        subject = match.group(1)
        return f"What does {subject} have?"

    # Pattern: "X verb Y" -> "What does X verb?"
    match = re.match(
        r"^(.+?)\s+(\w+(?:s|es|ed|ing)?)\s+(.+)$",
        sentence,
        re.IGNORECASE,
    )
    if match:
        subject = match.group(1)
        verb = match.group(2)
        return f"What does {subject} {verb}?"

    return None


# ---------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------

def _try_json_parse(
    raw: str,
) -> dict[str, str] | None:
    """Attempt direct JSON parse."""
    try:
        data = json.loads(raw.strip())
        return _validate_qa_dict(data)
    except (json.JSONDecodeError, ValueError):
        return None


def _try_fenced_json(
    raw: str,
) -> dict[str, str] | None:
    """Extract JSON from markdown code fences."""
    match = re.search(
        r"```(?:json)?\s*\n?(.*?)\n?\s*```",
        raw,
        re.DOTALL,
    )
    if match:
        return _try_json_parse(match.group(1))
    return None


def _try_regex_extract(
    raw: str,
) -> dict[str, str] | None:
    """Last-resort regex extraction of Q and A."""
    q_match = re.search(
        r'"question"\s*:\s*"([^"]+)"', raw,
    )
    a_match = re.search(
        r'"answer"\s*:\s*"([^"]+)"', raw,
    )
    if q_match and a_match:
        return {
            "question": q_match.group(1),
            "answer": a_match.group(1),
        }
    return None


def _validate_qa_dict(
    data: Any,
) -> dict[str, str] | None:
    """Ensure dict has non-empty question and answer."""
    if not isinstance(data, dict):
        return None
    q = data.get("question", "")
    a = data.get("answer", "")
    if q and a:
        return {"question": str(q), "answer": str(a)}
    return None


def _naive_sentence_split(text: str) -> list[str]:
    """Split text into sentences on period boundaries."""
    parts: list[str] = []
    current: list[str] = []
    for word in text.split():
        current.append(word)
        if word.endswith((".", "!", "?")):
            parts.append(" ".join(current))
            current = []
    if current:
        parts.append(" ".join(current))
    return [p.strip() for p in parts if p.strip()]
