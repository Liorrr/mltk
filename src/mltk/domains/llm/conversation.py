"""Multi-turn conversation evaluation — retention, relevancy, completeness."""

from __future__ import annotations

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult
from mltk.domains.llm._utils import _tokenize


def _assistant_turns(turns: list[dict[str, str]]) -> list[str]:
    """Extract assistant message contents from a turn list."""
    return [t["content"] for t in turns if t.get("role") == "assistant"]


def _user_turns(turns: list[dict[str, str]]) -> list[str]:
    """Extract user message contents from a turn list."""
    return [t["content"] for t in turns if t.get("role") == "user"]


@timed_assertion
def assert_knowledge_retention(
    turns: list[dict[str, str]],
    min_score: float = 0.7,
) -> TestResult:
    """Assert bot retains factual knowledge across conversation turns.

    For each pair of consecutive assistant responses, compute the token
    overlap between the earlier and later response.  A high score means
    the assistant keeps referencing the facts it introduced instead of
    contradicting itself or ignoring prior context.

    Score = mean Jaccard overlap across all consecutive assistant-turn pairs.
    If fewer than two assistant turns exist the score is defined as 1.0
    (trivially retained).

    Args:
        turns: Conversation as [{"role": "user"|"assistant", "content": "..."}].
        min_score: Minimum mean overlap required (default 0.7).

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
    if not turns:
        return assert_true(
            True,
            name="llm.conversation.knowledge_retention",
            message="Empty turn list — trivially retained (score=1.0)",
            severity=Severity.CRITICAL,
            score=1.0,
            min_score=min_score,
            assistant_turns=0,
        )

    assistant_contents = _assistant_turns(turns)

    if len(assistant_contents) < 2:
        return assert_true(
            True,
            name="llm.conversation.knowledge_retention",
            message="Fewer than 2 assistant turns — trivially retained (score=1.0)",
            severity=Severity.CRITICAL,
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
) -> TestResult:
    """Assert each assistant turn is relevant to the preceding user turn.

    For every (user, assistant) adjacent pair in the conversation, compute
    the ratio of user-turn tokens that appear in the assistant response.
    Score = mean across all such pairs.

    Args:
        turns: Conversation as [{"role": "user"|"assistant", "content": "..."}].
        min_score: Minimum mean relevancy ratio required (default 0.5).

    Returns:
        TestResult with turn relevancy score.

    Example:
        >>> turns = [
        ...     {"role": "user", "content": "What is Python?"},
        ...     {"role": "assistant", "content": "Python is a programming language."},
        ... ]
        >>> assert_turn_relevancy(turns, min_score=0.4)
    """
    if not turns:
        return assert_true(
            True,
            name="llm.conversation.turn_relevancy",
            message="Empty turn list — trivially relevant (score=1.0)",
            severity=Severity.CRITICAL,
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
        return assert_true(
            True,
            name="llm.conversation.turn_relevancy",
            message="No (user, assistant) adjacent pairs found — trivially relevant (score=1.0)",
            severity=Severity.CRITICAL,
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

    Returns:
        TestResult with coverage score and list of missing topics.

    Example:
        >>> turns = [
        ...     {"role": "user", "content": "Tell me about Python and Django."},
        ...     {"role": "assistant", "content": "Python is a language."},
        ... ]
        >>> assert_conversation_completeness(turns, ["python", "django"], min_coverage=1.0)
    """
    if not expected_topics:
        return assert_true(
            True,
            name="llm.conversation.completeness",
            message="No expected topics defined — trivially complete (score=1.0)",
            severity=Severity.CRITICAL,
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
