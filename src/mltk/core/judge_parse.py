"""Shared, fail-closed parsing for numeric judge responses."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence

_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_FLOAT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)")
_EXPLICIT = re.compile(
    rf"(?:\bscore\s*:|(?<!\*)\*\*score\s*:\*\*(?!\*))"
    rf"\s*({_NUMBER})(?:\s*/\s*({_NUMBER}))?"
    rf"|(?<![\w.+/-])({_NUMBER})\s*/\s*({_NUMBER})",
    re.IGNORECASE,
)


def _finite_score(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    try:
        score = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return score if math.isfinite(score) else None


def _strip_reasoning(raw: str, delimiters: Sequence[tuple[str, str]]) -> str | None:
    pairs = [("<think>", "</think>"), *delimiters]
    tokens: dict[str, tuple[bool, str]] = {}
    for start, end in pairs:
        if not start or not end or start == end:
            raise ValueError("Reasoning delimiters must be distinct nonempty strings")
        for token, meaning in ((start, (True, end)), (end, (False, end))):
            if token in tokens and tokens[token] != meaning:
                raise ValueError("Reasoning delimiters must not conflict")
            tokens[token] = meaning
    pattern = re.compile("|".join(re.escape(t) for t in sorted(tokens, key=len, reverse=True)))
    stack: list[str] = []
    visible: list[str] = []
    pos = 0
    for match in pattern.finditer(raw):
        if not stack:
            visible.append(raw[pos:match.start()])
        opening, closing = tokens[match.group()]
        if opening:
            stack.append(closing)
        elif not stack or stack.pop() != closing:
            return None
        if not stack:
            visible.append(" ")
        pos = match.end()
    if stack:
        return None
    visible.append(raw[pos:])
    return "".join(visible)


def parse_judge_score(
    raw: str | float,
    *,
    reasoning_delimiters: Sequence[tuple[str, str]] = (),
    legacy: bool = False,
    expected_max_score: float | None = None,
) -> float | None:
    """Return a raw score, never a normalized fraction.

    Strip balanced <think> blocks and additional (start, end) delimiter pairs.
    The last score-bearing JSON object is authoritative, including invalid
    scores (returning None). Fences and prose lead-ins are supported. Otherwise
    accept a standalone number or one explicit score: N / N/M expression.
    Explicit scores may end a sentence; the label may be bold (**Score:**).
    Fractions return N; callers retain responsibility for their configured
    scale. When supplied, expected_max_score must match the denominator.
    Multiple expressions, malformed reasoning, booleans and nonfinite
    scores fail closed.

    legacy=True restores whole-response JSON followed by first unsigned number,
    including permissive nonfinite behavior; reasoning delimiters are ignored.
    """
    if isinstance(raw, (int, float)):
        return float(raw) if legacy else _finite_score(raw)
    if not isinstance(raw, str):
        return None
    if legacy:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            data = None
        if isinstance(data, dict) and "score" in data:
            try:
                return float(data["score"])
            except (ValueError, TypeError, OverflowError):
                return None
        match = _FLOAT_PATTERN.search(raw.strip())
        return float(match.group(1)) if match else None

    cleaned = _strip_reasoning(raw, reasoning_delimiters)
    if cleaned is None:
        return None
    decoder = json.JSONDecoder()
    found = False
    score = None
    pos = 0
    while pos < len(cleaned):
        if cleaned[pos] not in "{[":
            pos += 1
            continue
        try:
            data, end = decoder.raw_decode(cleaned, pos)
        except json.JSONDecodeError:
            # Never scrape numbers from a malformed authoritative score object.
            if re.search(r'"score"\s*:', cleaned[pos:]):
                return None
            pos += 1
            continue
        if isinstance(data, dict) and "score" in data:
            found = True
            score = _finite_score(data["score"])
        pos = end
    if found:
        return score
    if re.fullmatch(_NUMBER, cleaned.strip()):
        return _finite_score(cleaned.strip())
    matches = list(_EXPLICIT.finditer(cleaned))
    if len(matches) != 1:
        return None
    match = matches[0]
    tail = cleaned[match.end():]
    # A decimal's sentence period is separate from the numeric token. Only
    # accept it at end of text or before whitespace, never inside a run-on
    # number. Integers may already have consumed their period via _NUMBER.
    if tail.startswith("."):
        if cleaned[match.end() - 1] == "." or (
            len(tail) > 1 and not tail[1].isspace()
        ):
            return None
        tail = tail[1:]
    if re.match(r"[\w./+-]", tail) or tail.lstrip().startswith("/"):
        return None
    numerator = match[1] if match[1] is not None else match[3]
    denominator = match[2] if match[1] is not None else match[4]
    if denominator is not None:
        scale = _finite_score(denominator)
        if (
            scale is None or scale <= 0
            or (expected_max_score is not None and scale != expected_max_score)
        ):
            return None
    return _finite_score(numerator)
