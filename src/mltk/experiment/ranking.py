"""Ranking strategies for experiment hypothesis results."""

from __future__ import annotations

from mltk.experiment.hypothesis import HypothesisResult

__all__ = ["rank_hypotheses"]

_CONFIDENCE_SCORES = {"high": 3, "medium": 2, "low": 1}


def _confidence_score(hr: HypothesisResult) -> int:
    """Convert confidence string to numeric score."""
    return _CONFIDENCE_SCORES.get(
        hr.hypothesis.fix.confidence, 0
    )


def rank_hypotheses(
    results: list[HypothesisResult],
    strategy: str = "passed",
) -> list[HypothesisResult]:
    """Rank hypothesis results and assign rank numbers.

    Mutates each result's .rank field (1=best, 2=second-best, etc).
    Returns the same list sorted by rank.

    Strategies:
        "passed": Binary -- fixes that pass rank highest. Tiebreak by
            confidence, then metric_delta.
        "delta": Metric improvement -- highest metric_delta ranks first.
            Tiebreak by passed, then confidence.
        "composite": Weighted score combining pass (0.5), confidence (0.3),
            speed (0.2). Tiebreak by metric_delta.

    Args:
        results: List of HypothesisResult objects to rank.
        strategy: One of "passed", "delta", "composite".

    Returns:
        The same list, sorted by rank, with .rank fields set.

    Raises:
        ValueError: If strategy is not recognized.
    """
    if strategy not in ("passed", "delta", "composite"):
        raise ValueError(
            f"Unknown ranking strategy {strategy!r}, "
            f"expected one of 'passed', 'delta', 'composite'"
        )

    if not results:
        return results

    if strategy == "passed":
        results.sort(key=_key_passed)
    elif strategy == "delta":
        results.sort(key=_key_delta)
    else:
        results.sort(key=_key_composite)

    for i, hr in enumerate(results, start=1):
        hr.rank = i

    return results


# -- sort key helpers (lower = better) ------------------------------------


def _key_passed(hr: HypothesisResult) -> tuple[int, int, float, float]:
    """Sort key for 'passed' strategy.

    (not passed, -confidence, -metric_delta, duration_ms)
    """
    return (
        0 if hr.fixed_result.passed else 1,
        -_confidence_score(hr),
        -hr.metric_delta,
        hr.fixed_result.duration_ms,
    )


def _key_delta(hr: HypothesisResult) -> tuple[float, int, int, float]:
    """Sort key for 'delta' strategy.

    (-metric_delta, not passed, -confidence, duration_ms)
    """
    return (
        -hr.metric_delta,
        0 if hr.fixed_result.passed else 1,
        -_confidence_score(hr),
        hr.fixed_result.duration_ms,
    )


def _key_composite(hr: HypothesisResult) -> tuple[float, float]:
    """Sort key for 'composite' strategy.

    (-composite_score, -metric_delta)
    """
    pass_score = 1.0 if hr.fixed_result.passed else 0.0
    conf_score = _confidence_score(hr) / 3.0
    speed_score = 1.0 - min(hr.fixed_result.duration_ms / 1000.0, 1.0)

    composite = pass_score * 0.5 + conf_score * 0.3 + speed_score * 0.2

    return (-composite, -hr.metric_delta)
