"""Recommendation system evaluation -- beyond-accuracy quality metrics.

Recommendation systems silently fail in ways that accuracy metrics miss.
A model can achieve high precision while showing every user the same ten
popular items, creating filter bubbles, suppressing long-tail content,
and delivering zero serendipity.  These failures are invisible in
standard A/B tests because users click on *familiar* items -- not
because the recommendations are *good*.

This module provides five assertions that evaluate recommendation quality
from complementary angles:

- **Hit Rate** -- basic health: are users getting *any* relevant items?
- **Diversity** -- category spread: are recommendations varied or repetitive?
- **Novelty** -- long-tail surfacing: are niche items being recommended?
- **Coverage** -- catalog utilization: how much of the catalog is visible?
- **Serendipity** -- unexpected relevance: the holy grail of recommendation.

All computations are pure Python with ``math`` (no external dependencies).
Each assertion follows the ``@timed_assertion`` pattern, returns a
``TestResult``, and integrates directly into pytest pipelines.
"""

from __future__ import annotations

import math

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult

# ------------------------------------------------------------------
# Public assertions
# ------------------------------------------------------------------


@timed_assertion
def assert_hit_rate(
    recommended: list[list],
    relevant: list[set],
    min_rate: float = 0.5,
) -> TestResult:
    """Assert that hit rate meets a minimum threshold.

    Hit rate is the fraction of users who received at least one
    relevant item in their recommendation list.  This is the most
    basic recommender health check: if 40% of users get zero
    relevant items, the system is failing nearly half the audience.

    A hit rate below threshold signals that the recommender is
    broken for a significant portion of users -- they see only
    irrelevant noise.

    Formula:
        hit_rate = count(users with >= 1 relevant item) / n_users

    Args:
        recommended: Recommendation lists per user.
            ``recommended[i]`` is the list of item IDs recommended
            to user *i*.
        relevant: Relevant item sets per user.  ``relevant[i]``
            is the set of item IDs that are relevant for user *i*.
        min_rate: Minimum acceptable hit rate (default 0.5).

    Returns:
        TestResult with ``hit_rate``, ``min_rate``, ``n_hits``,
        and ``n_users`` in details.

    Example:
        >>> recs = [["a", "b"], ["c", "d"], ["e", "f"]]
        >>> rels = [{"a", "x"}, {"y", "z"}, {"f"}]
        >>> assert_hit_rate(recs, rels, min_rate=0.5)
    """
    if not recommended:
        return assert_true(
            True,
            name="domains.recommendation.hit_rate",
            message="No users provided -- trivially passing "
            "(hit_rate=1.0)",
            severity=Severity.CRITICAL,
            hit_rate=1.0,
            min_rate=min_rate,
            n_hits=0,
            n_users=0,
        )

    n_users = len(recommended)
    n_hits = 0
    for recs, rels in zip(recommended, relevant):
        rec_set = set(recs)
        if rec_set & rels:
            n_hits += 1

    hit_rate = n_hits / n_users
    passed = hit_rate >= min_rate

    message = (
        f"Hit rate: {hit_rate:.4f} >= {min_rate} "
        f"({n_hits}/{n_users} users hit)"
        if passed
        else f"Low hit rate: {hit_rate:.4f} < {min_rate} "
        f"({n_hits}/{n_users} users hit)"
    )

    return assert_true(
        passed,
        name="domains.recommendation.hit_rate",
        message=message,
        severity=Severity.CRITICAL,
        hit_rate=hit_rate,
        min_rate=min_rate,
        n_hits=n_hits,
        n_users=n_users,
    )


@timed_assertion
def assert_diversity(
    recommended: list[list],
    item_categories: dict,
    min_diversity: float = 0.5,
) -> TestResult:
    """Assert that recommendation diversity meets a minimum threshold.

    A recommender that shows ten action movies to every user is
    unhelpful even if users like action.  Diversity measures how
    varied recommendations are within each user's list by computing
    category coverage: the fraction of all known categories
    represented in the recommendations.

    Low diversity indicates a filter bubble -- the system pigeonholes
    users into narrow content slices, reducing discovery and long-term
    engagement.

    Formula per user:
        diversity_i = |unique categories in recs_i| / |all categories|

    The final score is the mean diversity across all users.

    Args:
        recommended: Recommendation lists per user.
            ``recommended[i]`` is the list of item IDs recommended
            to user *i*.
        item_categories: Mapping from item ID to category string.
            Items not present in this dict are silently skipped.
        min_diversity: Minimum acceptable mean diversity (default 0.5).

    Returns:
        TestResult with ``avg_diversity``, ``min_diversity``, and
        ``per_user_diversity`` (list of floats) in details.

    Example:
        >>> recs = [["m1", "m2", "m3"], ["m4", "m5", "m6"]]
        >>> cats = {
        ...     "m1": "action", "m2": "comedy", "m3": "drama",
        ...     "m4": "action", "m5": "action", "m6": "action",
        ... }
        >>> assert_diversity(recs, cats, min_diversity=0.3)
    """
    if not recommended:
        return assert_true(
            True,
            name="domains.recommendation.diversity",
            message="No users provided -- trivially passing "
            "(avg_diversity=1.0)",
            severity=Severity.CRITICAL,
            avg_diversity=1.0,
            min_diversity=min_diversity,
            per_user_diversity=[],
        )

    all_categories = set(item_categories.values())
    n_categories = len(all_categories)

    if n_categories == 0:
        return assert_true(
            True,
            name="domains.recommendation.diversity",
            message="No categories defined -- trivially passing "
            "(avg_diversity=1.0)",
            severity=Severity.CRITICAL,
            avg_diversity=1.0,
            min_diversity=min_diversity,
            per_user_diversity=[1.0] * len(recommended),
        )

    per_user: list[float] = []
    for recs in recommended:
        user_cats = set()
        for item in recs:
            if item in item_categories:
                user_cats.add(item_categories[item])
        per_user.append(len(user_cats) / n_categories)

    avg_diversity = sum(per_user) / len(per_user)
    passed = avg_diversity >= min_diversity

    message = (
        f"Diversity: {avg_diversity:.4f} >= {min_diversity} "
        f"(averaged over {len(per_user)} users, "
        f"{n_categories} categories)"
        if passed
        else f"Low diversity: {avg_diversity:.4f} < {min_diversity} "
        f"(averaged over {len(per_user)} users, "
        f"{n_categories} categories)"
    )

    return assert_true(
        passed,
        name="domains.recommendation.diversity",
        message=message,
        severity=Severity.CRITICAL,
        avg_diversity=avg_diversity,
        min_diversity=min_diversity,
        per_user_diversity=per_user,
    )


@timed_assertion
def assert_novelty(
    recommended: list[list],
    popularity: dict,
    min_novelty: float = 0.3,
) -> TestResult:
    """Assert that recommendation novelty meets a minimum threshold.

    Recommending only popular items (Harry Potter, iPhone, Taylor
    Swift) is safe but useless -- users already know about them.
    Novelty rewards surfacing less-known items using inverse
    popularity: ``-log2(popularity)``.

    High novelty means the recommender is introducing users to
    items they are unlikely to have discovered on their own.  Low
    novelty means the system is a popularity echo chamber.

    Formula per user:
        novelty_i = mean(-log2(popularity[item]))
            for each item in the user's recommendation list

    The final score is the mean novelty across all users.  Items
    not present in the popularity dict are silently skipped.

    Args:
        recommended: Recommendation lists per user.
            ``recommended[i]`` is the list of item IDs recommended
            to user *i*.
        popularity: Mapping from item ID to popularity score in
            (0, 1].  The value is the fraction of all users who
            interacted with the item (e.g. 0.8 means 80% of users
            have seen it).
        min_novelty: Minimum acceptable mean novelty (default 0.3).

    Returns:
        TestResult with ``avg_novelty``, ``min_novelty``, and
        ``per_user_novelty`` (list of floats) in details.

    Example:
        >>> recs = [["niche1", "niche2"], ["popular1", "popular2"]]
        >>> pop = {
        ...     "niche1": 0.01, "niche2": 0.02,
        ...     "popular1": 0.9, "popular2": 0.8,
        ... }
        >>> assert_novelty(recs, pop, min_novelty=1.0)
    """
    if not recommended:
        return assert_true(
            True,
            name="domains.recommendation.novelty",
            message="No users provided -- trivially passing "
            "(avg_novelty=1.0)",
            severity=Severity.CRITICAL,
            avg_novelty=1.0,
            min_novelty=min_novelty,
            per_user_novelty=[],
        )

    per_user: list[float] = []
    for recs in recommended:
        item_novelties: list[float] = []
        for item in recs:
            if item in popularity and popularity[item] > 0:
                item_novelties.append(
                    -math.log2(popularity[item])
                )
        if item_novelties:
            per_user.append(
                sum(item_novelties) / len(item_novelties)
            )
        else:
            per_user.append(0.0)

    avg_novelty = sum(per_user) / len(per_user)
    passed = avg_novelty >= min_novelty

    message = (
        f"Novelty: {avg_novelty:.4f} >= {min_novelty} "
        f"(averaged over {len(per_user)} users)"
        if passed
        else f"Low novelty: {avg_novelty:.4f} < {min_novelty} "
        f"(averaged over {len(per_user)} users)"
    )

    return assert_true(
        passed,
        name="domains.recommendation.novelty",
        message=message,
        severity=Severity.CRITICAL,
        avg_novelty=avg_novelty,
        min_novelty=min_novelty,
        per_user_novelty=per_user,
    )


@timed_assertion
def assert_coverage(
    recommended: list[list],
    catalog_size: int,
    min_coverage: float = 0.1,
) -> TestResult:
    """Assert that catalog coverage meets a minimum threshold.

    If a catalog has 100K items but the recommender only ever
    suggests 500 of them, 99.5% of the catalog is invisible.  Low
    coverage means long-tail items never get recommended, hurting
    user discovery and business revenue from niche products.

    Coverage measures the fraction of the total catalog that appears
    in at least one user's recommendation list.

    Formula:
        coverage = |unique items across all rec lists| / catalog_size

    Args:
        recommended: Recommendation lists per user.
            ``recommended[i]`` is the list of item IDs recommended
            to user *i*.
        catalog_size: Total number of items in the catalog.
        min_coverage: Minimum acceptable coverage (default 0.1).

    Returns:
        TestResult with ``coverage``, ``min_coverage``,
        ``unique_items``, and ``catalog_size`` in details.

    Example:
        >>> recs = [["a", "b"], ["b", "c"], ["c", "d"]]
        >>> assert_coverage(recs, catalog_size=10, min_coverage=0.3)
    """
    if catalog_size <= 0:
        return assert_true(
            True,
            name="domains.recommendation.coverage",
            message="Empty catalog -- trivially passing "
            "(coverage=1.0)",
            severity=Severity.CRITICAL,
            coverage=1.0,
            min_coverage=min_coverage,
            unique_items=0,
            catalog_size=catalog_size,
        )

    all_items: set = set()
    for recs in recommended:
        all_items.update(recs)

    unique_count = len(all_items)
    coverage = unique_count / catalog_size
    passed = coverage >= min_coverage

    message = (
        f"Coverage: {coverage:.4f} >= {min_coverage} "
        f"({unique_count}/{catalog_size} items recommended)"
        if passed
        else f"Low coverage: {coverage:.4f} < {min_coverage} "
        f"({unique_count}/{catalog_size} items recommended)"
    )

    return assert_true(
        passed,
        name="domains.recommendation.coverage",
        message=message,
        severity=Severity.CRITICAL,
        coverage=coverage,
        min_coverage=min_coverage,
        unique_items=unique_count,
        catalog_size=catalog_size,
    )


@timed_assertion
def assert_serendipity(
    recommended: list[list],
    expected: list[list],
    relevant: list[set],
    min_serendipity: float = 0.1,
) -> TestResult:
    """Assert that recommendation serendipity meets a minimum threshold.

    Serendipity is the holy grail of recommendation: items that are
    UNEXPECTED but RELEVANT.  The ``expected`` list represents what
    a simple baseline (e.g. popularity ranking) would recommend.
    Serendipitous items are those that the user finds relevant but
    would NOT have received from the baseline.

    A system with zero serendipity is indistinguishable from a
    popularity chart.  High serendipity means the recommender adds
    genuine value by connecting users to content they would not have
    found on their own.

    Formula per user:
        serendipity_i = |relevant & recommended & ~expected|
                        / |recommended|

    Items that are relevant AND recommended but NOT in expected.

    Args:
        recommended: Recommendation lists per user.
            ``recommended[i]`` is the list of item IDs recommended
            to user *i*.
        expected: Baseline recommendation lists per user.
            ``expected[i]`` is the list of items a naive baseline
            (e.g. most-popular) would recommend to user *i*.
        relevant: Relevant item sets per user.  ``relevant[i]``
            is the set of item IDs that are relevant for user *i*.
        min_serendipity: Minimum acceptable mean serendipity
            (default 0.1).

    Returns:
        TestResult with ``avg_serendipity``, ``min_serendipity``,
        and ``per_user_serendipity`` (list of floats) in details.

    Example:
        >>> recs = [["a", "b", "c", "d"]]
        >>> baseline = [["a", "b"]]
        >>> rels = [{"a", "c", "d"}]
        >>> # "c" and "d" are relevant + recommended + NOT expected
        >>> assert_serendipity(recs, baseline, rels, min_serendipity=0.3)
    """
    if not recommended:
        return assert_true(
            True,
            name="domains.recommendation.serendipity",
            message="No users provided -- trivially passing "
            "(avg_serendipity=1.0)",
            severity=Severity.CRITICAL,
            avg_serendipity=1.0,
            min_serendipity=min_serendipity,
            per_user_serendipity=[],
        )

    per_user: list[float] = []
    for recs, exp, rels in zip(recommended, expected, relevant):
        if not recs:
            per_user.append(0.0)
            continue
        rec_set = set(recs)
        exp_set = set(exp)
        # Serendipitous = relevant AND recommended AND NOT expected
        serendipitous = rec_set & rels - exp_set
        per_user.append(len(serendipitous) / len(recs))

    avg_serendipity = sum(per_user) / len(per_user)
    passed = avg_serendipity >= min_serendipity

    message = (
        f"Serendipity: {avg_serendipity:.4f} >= {min_serendipity} "
        f"(averaged over {len(per_user)} users)"
        if passed
        else f"Low serendipity: "
        f"{avg_serendipity:.4f} < {min_serendipity} "
        f"(averaged over {len(per_user)} users)"
    )

    return assert_true(
        passed,
        name="domains.recommendation.serendipity",
        message=message,
        severity=Severity.CRITICAL,
        avg_serendipity=avg_serendipity,
        min_serendipity=min_serendipity,
        per_user_serendipity=per_user,
    )
