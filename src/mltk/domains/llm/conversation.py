"""Multi-turn conversation evaluation — retention, relevancy, completeness."""

from __future__ import annotations

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult
from mltk.domains.llm._utils import _tokenize

_ON_EMPTY_OPTIONS = ("fail", "skip", "pass")


def _assistant_turns(turns: list[dict[str, str]]) -> list[str]:
    """Extract assistant message contents from a turn list."""
    return [t["content"] for t in turns if t.get("role") == "assistant"]


def _unknown_on_empty_result(name: str, on_empty: str) -> TestResult:
    """Return a failed result for an unsupported on_empty policy."""
    return assert_true(
        False,
        name=name,
        message=(
            f"Unknown on_empty: '{on_empty}'. "
            f"Supported: {', '.join(_ON_EMPTY_OPTIONS)}"
        ),
        severity=Severity.CRITICAL,
        on_empty=on_empty,
    )


def _empty_input_result(
    *,
    name: str,
    reason: str,
    on_empty: str,
    legacy_message: str,
    **legacy_details: object,
) -> TestResult:
    """Apply the configured empty-input policy."""
    if on_empty == "fail":
        return assert_true(
            False,
            name=name,
            message=f"{reason} -- empty input is not allowed",
            severity=Severity.CRITICAL,
        )
    if on_empty == "skip":
        return assert_true(
            True,
            name=name,
            message=f"Skipped: {reason}",
            severity=Severity.INFO,
            skipped=True,
            reason=reason,
        )
    return assert_true(
        True,
        name=name,
        message=legacy_message,
        severity=Severity.CRITICAL,
        **legacy_details,
    )


@timed_assertion
def assert_knowledge_retention(
    turns: list[dict[str, str]],
    min_score: float = 0.7,
    on_empty: str = "fail",
) -> TestResult:
    """Assert bot retains factual knowledge across conversation turns.

    For each pair of consecutive assistant responses, compute the token
    overlap between the earlier and later response.  A high score means
    the assistant keeps referencing the facts it introduced instead of
    contradicting itself or ignoring prior context.

    Score = mean Jaccard overlap across all consecutive assistant-turn pairs.
    If fewer than two assistant turns exist, ``on_empty`` controls the
    result; the default is to fail, and ``"pass"`` preserves the legacy
    score of 1.0.

    Args:
        turns: Conversation as [{"role": "user"|"assistant", "content": "..."}].
        min_score: Minimum mean overlap required (default 0.7).
        on_empty: Policy for no evaluable assistant-turn pairs:
            ``"fail"`` (default), ``"skip"``, or ``"pass"`` for
            legacy behavior.

    Returns:
        TestResult with retention score.

    Example:
        >>> turns = [
        ...     {"role": "user", "content": "My name is Alice."},
        ...     {"role": "assistant", "content": "Hello Alice, nice to meet you."},
        ...     {"role": "user", "content": "What is my name?"},
        ...     {"role": "assistant", "content": "Your name is Alice."},
        ... ]
        >>> assert_knowledge_retention(turns, min_score=0.3)
    """
    if on_empty not in _ON_EMPTY_OPTIONS:
        return _unknown_on_empty_result(
            "llm.conversation.knowledge_retention", on_empty
        )

    if not turns:
        return _empty_input_result(
            name="llm.conversation.knowledge_retention",
            reason="Empty turn list",
            on_empty=on_empty,
            legacy_message="Empty turn list — trivially retained (score=1.0)",
            score=1.0,
            min_score=min_score,
            assistant_turns=0,
        )

    assistant_contents = _assistant_turns(turns)

    if len(assistant_contents) < 2:
        return _empty_input_result(
            name="llm.conversation.knowledge_retention",
            reason="Fewer than 2 assistant turns",
            on_empty=on_empty,
            legacy_message=(
                "Fewer than 2 assistant turns — trivially retained (score=1.0)"
            ),
            score=1.0,
            min_score=min_score,
            assistant_turns=len(assistant_contents),
        )

    overlaps: list[float] = []
    for i in range(len(assistant_contents) - 1):
        tokens_a = _tokenize(assistant_contents[i])
        tokens_b = _tokenize(assistant_contents[i + 1])
        union = tokens_a | tokens_b
        if not union:
            overlaps.append(1.0)
        else:
            overlaps.append(len(tokens_a & tokens_b) / len(union))

    score = sum(overlaps) / len(overlaps)
    passed = score >= min_score

    message = (
        f"Knowledge retention: {score:.4f} >= {min_score} "
        f"(mean Jaccard over {len(overlaps)} consecutive assistant-turn pair(s))"
        if passed
        else f"Low knowledge retention: {score:.4f} < {min_score} "
        f"(mean Jaccard over {len(overlaps)} consecutive assistant-turn pair(s))"
    )

    return assert_true(
        passed,
        name="llm.conversation.knowledge_retention",
        message=message,
        severity=Severity.CRITICAL,
        score=score,
        min_score=min_score,
        assistant_turns=len(assistant_contents),
        pairs_evaluated=len(overlaps),
    )


@timed_assertion
def assert_turn_relevancy(
    turns: list[dict[str, str]],
    min_score: float = 0.5,
    on_empty: str = "fail",
) -> TestResult:
    """Assert each assistant turn is relevant to the preceding user turn.

    For every (user, assistant) adjacent pair in the conversation, compute
    the ratio of user-turn tokens that appear in the assistant response.
    Score = mean across all such pairs.

    Args:
        turns: Conversation as [{"role": "user"|"assistant", "content": "..."}].
        min_score: Minimum mean relevancy ratio required (default 0.5).
        on_empty: Policy for no evaluable user/assistant pairs:
            ``"fail"`` (default), ``"skip"``, or ``"pass"`` for
            legacy behavior.

    Returns:
        TestResult with turn relevancy score.

    Example:
        >>> turns = [
        ...     {"role": "user", "content": "What is Python?"},
        ...     {"role": "assistant", "content": "Python is a programming language."},
        ... ]
        >>> assert_turn_relevancy(turns, min_score=0.4)
    """
    if on_empty not in _ON_EMPTY_OPTIONS:
        return _unknown_on_empty_result(
            "llm.conversation.turn_relevancy", on_empty
        )

    if not turns:
        return _empty_input_result(
            name="llm.conversation.turn_relevancy",
            reason="Empty turn list",
            on_empty=on_empty,
            legacy_message="Empty turn list — trivially relevant (score=1.0)",
            score=1.0,
            min_score=min_score,
            pairs_evaluated=0,
        )

    # Build (user_msg, assistant_msg) pairs from adjacent turns
    pairs: list[tuple[str, str]] = []
    for i in range(len(turns) - 1):
        if turns[i].get("role") == "user" and turns[i + 1].get("role") == "assistant":
            pairs.append((turns[i]["content"], turns[i + 1]["content"]))

    if not pairs:
        return _empty_input_result(
            name="llm.conversation.turn_relevancy",
            reason="No (user, assistant) adjacent pairs found",
            on_empty=on_empty,
            legacy_message=(
                "No (user, assistant) adjacent pairs found — "
                "trivially relevant (score=1.0)"
            ),
            score=1.0,
            min_score=min_score,
            pairs_evaluated=0,
        )

    scores: list[float] = []
    for user_msg, assistant_msg in pairs:
        user_tokens = _tokenize(user_msg)
        assistant_tokens = _tokenize(assistant_msg)
        if not user_tokens:
            scores.append(1.0)
        elif not assistant_tokens:
            scores.append(0.0)
        else:
            overlap = len(user_tokens & assistant_tokens)
            scores.append(overlap / len(user_tokens))

    score = sum(scores) / len(scores)
    passed = score >= min_score

    message = (
        f"Turn relevancy: {score:.4f} >= {min_score} "
        f"(mean overlap over {len(pairs)} (user, assistant) pair(s))"
        if passed
        else f"Low turn relevancy: {score:.4f} < {min_score} "
        f"(mean overlap over {len(pairs)} (user, assistant) pair(s))"
    )

    return assert_true(
        passed,
        name="llm.conversation.turn_relevancy",
        message=message,
        severity=Severity.CRITICAL,
        score=score,
        min_score=min_score,
        pairs_evaluated=len(pairs),
    )


@timed_assertion
def assert_conversation_completeness(
    turns: list[dict[str, str]],
    expected_topics: list[str],
    min_coverage: float = 0.8,
    on_empty: str = "fail",
) -> TestResult:
    """Assert conversation covers all expected topics.

    Checks what fraction of ``expected_topics`` appears (as a substring or
    token) in the concatenated assistant responses.  Topic matching is
    case-insensitive and uses whole-word tokenization so that "python"
    matches the word "Python" in an answer.

    Score = topics_found / total_topics.

    Args:
        turns: Conversation as [{"role": "user"|"assistant", "content": "..."}].
        expected_topics: List of topic keywords the assistant should address.
        min_coverage: Minimum fraction of topics that must be covered (default 0.8).
        on_empty: Policy for empty expected topics: ``"fail"`` (default),
            ``"skip"``, or ``"pass"`` for legacy behavior.

    Returns:
        TestResult with coverage score and list of missing topics.

    Example:
        >>> turns = [
        ...     {"role": "user", "content": "Tell me about Python and Django."},
        ...     {"role": "assistant", "content": "Python is a language."},
        ... ]
        >>> assert_conversation_completeness(turns, ["python", "django"], min_coverage=1.0)
    """
    if on_empty not in _ON_EMPTY_OPTIONS:
        return _unknown_on_empty_result(
            "llm.conversation.completeness", on_empty
        )

    if not expected_topics:
        return _empty_input_result(
            name="llm.conversation.completeness",
            reason="No expected topics defined",
            on_empty=on_empty,
            legacy_message=(
                "No expected topics defined — trivially complete (score=1.0)"
            ),
            score=1.0,
            min_coverage=min_coverage,
            topics_found=0,
            topics_total=0,
            missing_topics=[],
        )

    assistant_text = " ".join(_assistant_turns(turns))
    assistant_tokens = _tokenize(assistant_text)

    found: list[str] = []
    missing: list[str] = []

    for topic in expected_topics:
        topic_tokens = _tokenize(topic)
        # A topic is covered if ALL of its tokens appear in the assistant text
        if topic_tokens and topic_tokens.issubset(assistant_tokens):
            found.append(topic)
        else:
            missing.append(topic)

    score = len(found) / len(expected_topics)
    passed = score >= min_coverage

    message = (
        f"Conversation completeness: {score:.4f} >= {min_coverage} "
        f"({len(found)}/{len(expected_topics)} topics covered)"
        if passed
        else f"Incomplete conversation: {score:.4f} < {min_coverage} "
        f"({len(found)}/{len(expected_topics)} topics covered); "
        f"missing: {missing}"
    )

    return assert_true(
        passed,
        name="llm.conversation.completeness",
        message=message,
        severity=Severity.CRITICAL,
        score=score,
        min_coverage=min_coverage,
        topics_found=len(found),
        topics_total=len(expected_topics),
        missing_topics=missing,
        covered_topics=found,
    )
