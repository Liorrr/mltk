

"""LLM summarization evaluation -- coverage, compression, faithfulness."""

from __future__ import annotations

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult
from mltk.domains.llm._utils import _tokenize

__all__ = [
    "assert_summary_coverage",
    "assert_summary_compression",
    "assert_summary_faithfulness",
]


@timed_assertion
def assert_summary_coverage(
    source: str,
    summary: str,
    min_coverage: float = 0.3,
) -> TestResult:
    """Assert that a summary preserves key content from the source.

    Coverage measures what fraction of the source's important content
    appears in the summary, using token overlap:

        coverage = |source_tokens & summary_tokens| / |source_tokens|

    A coverage of 1.0 means every unique token in the source also
    appears in the summary.  A coverage of 0.0 means no overlap at
    all -- the summary discusses entirely different content.

    **When to use:** Coverage answers the question "did the summary
    miss important information?"  It is analogous to recall: high
    coverage means the summary retained the source's key points.

    Args:
        source: Original text being summarized.
        summary: Generated summary to evaluate.
        min_coverage: Minimum required coverage ratio (0.0--1.0).

    Returns:
        TestResult with coverage details including ``coverage``,
        ``min_coverage``, ``source_tokens``, ``summary_tokens``,
        and ``common_tokens``.

    Example:
        >>> source = "Machine learning uses data to train models."
        >>> summary = "ML trains models on data."
        >>> assert_summary_coverage(source, summary, min_coverage=0.3)
    """
    source_tokens = _tokenize(source)
    summary_tokens = _tokenize(summary)

    if not source_tokens:
        # Empty source -- nothing to cover; trivially passes.
        coverage = 1.0
    else:
        common = source_tokens & summary_tokens
        coverage = len(common) / len(source_tokens)

    common_count = len(source_tokens & summary_tokens)
    passed = coverage >= min_coverage

    message = (
        f"Summary coverage: {coverage:.4f} >= {min_coverage} "
        f"({common_count}/{len(source_tokens)} source tokens covered)"
        if passed
        else f"Summary coverage too low: {coverage:.4f} < {min_coverage} "
        f"({common_count}/{len(source_tokens)} source tokens covered)"
    )

    return assert_true(
        passed,
        name="llm.summarization.coverage",
        message=message,
        severity=Severity.CRITICAL,
        coverage=round(coverage, 4),
        min_coverage=min_coverage,
        source_tokens=len(source_tokens),
        summary_tokens=len(summary_tokens),
        common_tokens=common_count,
    )

@timed_assertion
def assert_summary_compression(
    source: str,
    summary: str,
    min_ratio: float = 0.1,
    max_ratio: float = 0.5,
) -> TestResult:
    """Assert that a summary has a reasonable compression ratio.

    Compression ratio measures how much shorter the summary is
    relative to the source:

        compression_ratio = len(summary) / len(source)

    A ratio of 0.25 means the summary is 25% the length of the
    source.  A ratio close to 1.0 means the "summary" is barely
    shorter than the original -- not a useful summary.  A ratio
    near 0.0 means the summary is extremely terse and may have
    lost critical information.

    **When to use:** Compression is a structural sanity check.
    It ensures the summary is actually shorter than the source
    but not so aggressively compressed that content is lost.

    Args:
        source: Original text being summarized.
        summary: Generated summary to evaluate.
        min_ratio: Minimum compression ratio (summary must be at
            least this fraction of the source length).
        max_ratio: Maximum compression ratio (summary must be at
            most this fraction of the source length).

    Returns:
        TestResult with compression details including
        ``compression_ratio``, ``min_ratio``, ``max_ratio``,
        ``source_length``, and ``summary_length``.

    Example:
        >>> source = "A " * 100  # 200 chars
        >>> summary = "A " * 30   # 60 chars
        >>> assert_summary_compression(source, summary)
    """
    source_length = len(source)
    summary_length = len(summary)

    if source_length == 0:
        # Empty source -- ratio is undefined; pass if summary is
        # also empty, fail otherwise.
        compression_ratio = 0.0 if summary_length == 0 else 1.0
    else:
        compression_ratio = summary_length / source_length

    passed = min_ratio <= compression_ratio <= max_ratio

    message = (
        f"Compression ratio: {compression_ratio:.4f} "
        f"in [{min_ratio}, {max_ratio}] "
        f"({summary_length}/{source_length} chars)"
        if passed
        else f"Compression ratio out of range: "
        f"{compression_ratio:.4f} not in "
        f"[{min_ratio}, {max_ratio}] "
        f"({summary_length}/{source_length} chars)"
    )

    return assert_true(
        passed,
        name="llm.summarization.compression",
        message=message,
        severity=Severity.CRITICAL,
        compression_ratio=round(compression_ratio, 4),
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        source_length=source_length,
        summary_length=summary_length,
    )

@timed_assertion
def assert_summary_faithfulness(
    source: str,
    summary: str,
    min_faithfulness: float = 0.5,
) -> TestResult:
    """Assert that a summary does not introduce content absent from the source.

    Faithfulness measures what fraction of the summary's content
    actually comes from the source:

        faithfulness = |summary_tokens & source_tokens| / |summary_tokens|

    A faithfulness of 1.0 means every token in the summary also
    exists in the source -- no hallucinated content.  A faithfulness
    of 0.3 means 70% of the summary's vocabulary is novel -- the
    summary likely introduces fabricated claims.

    **Coverage vs. faithfulness:** Coverage asks "did the summary
    capture the source?" (recall).  Faithfulness asks "did the
    summary stay faithful to the source?" (precision).  A perfect
    summary scores high on both.

    Args:
        source: Original text being summarized.
        summary: Generated summary to evaluate.
        min_faithfulness: Minimum required faithfulness (0.0--1.0).

    Returns:
        TestResult with faithfulness details including
        ``faithfulness``, ``min_faithfulness``, ``summary_tokens``,
        ``source_tokens``, and ``novel_tokens``.

    Example:
        >>> source = "Python is a popular programming language."
        >>> summary = "Python is popular."
        >>> assert_summary_faithfulness(source, summary)
    """
    source_tokens = _tokenize(source)
    summary_tokens = _tokenize(summary)

    if not summary_tokens:
        # Empty summary -- nothing to be unfaithful about.
        faithfulness = 1.0
        novel_count = 0
    else:
        common = summary_tokens & source_tokens
        faithfulness = len(common) / len(summary_tokens)
        novel_count = len(summary_tokens - source_tokens)

    passed = faithfulness >= min_faithfulness

    message = (
        f"Summary faithfulness: {faithfulness:.4f} "
        f">= {min_faithfulness} "
        f"({novel_count} novel tokens in summary)"
        if passed
        else f"Summary faithfulness too low: "
        f"{faithfulness:.4f} < {min_faithfulness} "
        f"({novel_count} novel tokens in summary)"
    )

    return assert_true(
        passed,
        name="llm.summarization.faithfulness",
        message=message,
        severity=Severity.CRITICAL,
        faithfulness=round(faithfulness, 4),
        min_faithfulness=min_faithfulness,
        summary_tokens=len(summary_tokens),
        source_tokens=len(source_tokens),
        novel_tokens=novel_count,
    )
