"""Tests for mltk.domains.recommendation -- beyond-accuracy metrics."""

import math

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.domains.recommendation import (
    assert_coverage,
    assert_diversity,
    assert_hit_rate,
    assert_novelty,
    assert_serendipity,
)


# ------------------------------------------------------------------
# Hit Rate
# ------------------------------------------------------------------


class TestHitRate:
    """Hit rate -- fraction of users with at least one relevant item."""

    def test_all_users_hit(self) -> None:
        # SCENARIO: Every user has at least one relevant item.
        # WHY: Hit rate should be 1.0 -- perfect health.
        # EXPECTED: Passes with any threshold <= 1.0.
        recs = [["a", "b"], ["c", "d"], ["e", "f"]]
        rels = [{"a"}, {"c"}, {"f"}]
        result = assert_hit_rate(recs, rels, min_rate=1.0)
        assert result.passed is True
        assert abs(result.details["hit_rate"] - 1.0) < 1e-9
        assert result.details["n_hits"] == 3
        assert result.details["n_users"] == 3

    def test_no_users_hit(self) -> None:
        # SCENARIO: No user gets any relevant item.
        # WHY: Hit rate = 0 -- system is completely broken.
        # EXPECTED: Fails when min_rate > 0.
        recs = [["a", "b"], ["c", "d"]]
        rels = [{"x"}, {"y"}]
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_hit_rate(recs, rels, min_rate=0.1)
        assert exc_info.value.result.details["hit_rate"] == 0.0
        assert exc_info.value.result.details["n_hits"] == 0

    def test_partial_hit(self) -> None:
        # SCENARIO: 2 out of 4 users get a relevant item.
        # WHY: Hit rate = 0.5.
        # EXPECTED: Passes with min_rate=0.5, fails with 0.6.
        recs = [["a"], ["b"], ["c"], ["d"]]
        rels = [{"a"}, {"x"}, {"c"}, {"y"}]
        result = assert_hit_rate(recs, rels, min_rate=0.5)
        assert result.passed is True
        assert abs(result.details["hit_rate"] - 0.5) < 1e-9

    def test_empty_recommendations(self) -> None:
        # SCENARIO: Users get empty recommendation lists.
        # WHY: Empty lists have no relevant items.
        # EXPECTED: Hit rate = 0.
        recs: list[list] = [[], []]
        rels = [{"a"}, {"b"}]
        with pytest.raises(MltkAssertionError):
            assert_hit_rate(recs, rels, min_rate=0.5)

    def test_empty_users_trivially_passes(self) -> None:
        # SCENARIO: No users at all.
        # WHY: Edge case -- nothing to evaluate.
        # EXPECTED: Trivially passes with hit_rate=1.0.
        result = assert_hit_rate([], [], min_rate=0.9)
        assert result.passed is True
        assert result.details["hit_rate"] == 1.0
        assert result.details["n_users"] == 0

    def test_single_user_hit(self) -> None:
        # SCENARIO: One user with a relevant item.
        # WHY: Single-user edge case.
        # EXPECTED: Hit rate = 1.0.
        recs = [["item1", "item2"]]
        rels = [{"item2"}]
        result = assert_hit_rate(recs, rels, min_rate=0.5)
        assert result.passed is True
        assert result.details["n_hits"] == 1

    def test_result_has_timing(self) -> None:
        # SCENARIO: @timed_assertion must populate duration_ms.
        # WHY: Performance tracking requirement.
        # EXPECTED: duration_ms > 0.
        result = assert_hit_rate(
            [["a"]], [{"a"}], min_rate=0.5,
        )
        assert result.duration_ms > 0


# ------------------------------------------------------------------
# Diversity
# ------------------------------------------------------------------


class TestDiversity:
    """Diversity -- category coverage in recommendations."""

    def test_all_same_category_low_diversity(self) -> None:
        # SCENARIO: All items in every user's list are same category.
        # WHY: Only 1 out of 3 categories represented = 0.333.
        # EXPECTED: Fails when min_diversity=0.5.
        recs = [["m1", "m2", "m3"]]
        cats = {
            "m1": "action", "m2": "action", "m3": "action",
            "m4": "comedy", "m5": "drama",
        }
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_diversity(recs, cats, min_diversity=0.5)
        d = exc_info.value.result.details["avg_diversity"]
        assert abs(d - 1.0 / 3.0) < 1e-9

    def test_diverse_recommendations_pass(self) -> None:
        # SCENARIO: User gets items from all 3 categories.
        # WHY: 3/3 = 1.0 diversity.
        # EXPECTED: Passes.
        recs = [["m1", "m2", "m3"]]
        cats = {
            "m1": "action", "m2": "comedy", "m3": "drama",
        }
        result = assert_diversity(recs, cats, min_diversity=0.9)
        assert result.passed is True
        assert abs(
            result.details["avg_diversity"] - 1.0
        ) < 1e-9

    def test_multiple_users_averaged(self) -> None:
        # SCENARIO: User A has 3/3 categories, user B has 1/3.
        # WHY: Mean = (1.0 + 0.333) / 2 = 0.667.
        # EXPECTED: Passes with min_diversity=0.6.
        recs = [["m1", "m2", "m3"], ["m4", "m5", "m6"]]
        cats = {
            "m1": "action", "m2": "comedy", "m3": "drama",
            "m4": "action", "m5": "action", "m6": "action",
        }
        result = assert_diversity(recs, cats, min_diversity=0.6)
        assert result.passed is True
        per_user = result.details["per_user_diversity"]
        assert abs(per_user[0] - 1.0) < 1e-9
        assert abs(per_user[1] - 1.0 / 3.0) < 1e-9

    def test_empty_users_trivially_passes(self) -> None:
        # SCENARIO: No users at all.
        # WHY: Edge case -- nothing to evaluate.
        # EXPECTED: Trivially passes.
        result = assert_diversity([], {}, min_diversity=0.5)
        assert result.passed is True
        assert result.details["avg_diversity"] == 1.0

    def test_missing_items_in_category_dict(self) -> None:
        # SCENARIO: Some recommended items not in category dict.
        # WHY: Items without categories are silently skipped.
        # EXPECTED: Diversity computed from known items only.
        recs = [["m1", "unknown1", "unknown2"]]
        cats = {"m1": "action", "m2": "comedy"}
        result = assert_diversity(recs, cats, min_diversity=0.1)
        assert result.passed is True
        # Only "action" found out of {"action", "comedy"}
        assert abs(
            result.details["per_user_diversity"][0] - 0.5
        ) < 1e-9


# ------------------------------------------------------------------
# Novelty
# ------------------------------------------------------------------


class TestNovelty:
    """Novelty -- inverse popularity of recommended items."""

    def test_popular_items_low_novelty(self) -> None:
        # SCENARIO: All items are extremely popular (pop=0.9).
        # WHY: -log2(0.9) ~= 0.152, which is low novelty.
        # EXPECTED: Fails when min_novelty=1.0.
        recs = [["a", "b"]]
        pop = {"a": 0.9, "b": 0.9}
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_novelty(recs, pop, min_novelty=1.0)
        n = exc_info.value.result.details["avg_novelty"]
        expected = -math.log2(0.9)
        assert abs(n - expected) < 1e-9

    def test_niche_items_high_novelty(self) -> None:
        # SCENARIO: All items are very niche (pop=0.01).
        # WHY: -log2(0.01) ~= 6.644.
        # EXPECTED: Passes with min_novelty=5.0.
        recs = [["a", "b"]]
        pop = {"a": 0.01, "b": 0.01}
        result = assert_novelty(recs, pop, min_novelty=5.0)
        assert result.passed is True
        expected = -math.log2(0.01)
        assert abs(
            result.details["avg_novelty"] - expected
        ) < 1e-9

    def test_mixed_popularity(self) -> None:
        # SCENARIO: One popular (0.8) and one niche (0.1) item.
        # WHY: Mean of -log2(0.8) and -log2(0.1).
        # EXPECTED: Intermediate novelty value.
        recs = [["pop", "niche"]]
        pop = {"pop": 0.8, "niche": 0.1}
        result = assert_novelty(recs, pop, min_novelty=0.1)
        assert result.passed is True
        expected = (
            (-math.log2(0.8) + -math.log2(0.1)) / 2.0
        )
        assert abs(
            result.details["avg_novelty"] - expected
        ) < 1e-9

    def test_empty_users_trivially_passes(self) -> None:
        # SCENARIO: No users at all.
        # WHY: Edge case -- nothing to evaluate.
        # EXPECTED: Trivially passes.
        result = assert_novelty([], {}, min_novelty=0.5)
        assert result.passed is True
        assert result.details["avg_novelty"] == 1.0

    def test_missing_items_in_popularity_dict(self) -> None:
        # SCENARIO: Some recommended items not in popularity dict.
        # WHY: Unknown items are silently skipped.
        # EXPECTED: Novelty computed from known items only.
        recs = [["known", "unknown"]]
        pop = {"known": 0.5}
        result = assert_novelty(recs, pop, min_novelty=0.1)
        assert result.passed is True
        expected = -math.log2(0.5)  # 1.0
        assert abs(
            result.details["per_user_novelty"][0] - expected
        ) < 1e-9

    def test_all_items_unknown_zero_novelty(self) -> None:
        # SCENARIO: No recommended items exist in popularity dict.
        # WHY: No items to compute novelty from = 0.0.
        # EXPECTED: Fails when min_novelty > 0.
        recs = [["x", "y"]]
        pop: dict = {}
        with pytest.raises(MltkAssertionError):
            assert_novelty(recs, pop, min_novelty=0.1)


# ------------------------------------------------------------------
# Coverage
# ------------------------------------------------------------------


class TestCoverage:
    """Coverage -- fraction of catalog recommended."""

    def test_full_catalog_covered(self) -> None:
        # SCENARIO: All 4 catalog items appear in recommendations.
        # WHY: Coverage = 4/4 = 1.0.
        # EXPECTED: Passes.
        recs = [["a", "b"], ["c", "d"]]
        result = assert_coverage(recs, catalog_size=4, min_coverage=1.0)
        assert result.passed is True
        assert abs(result.details["coverage"] - 1.0) < 1e-9
        assert result.details["unique_items"] == 4

    def test_small_fraction_covered(self) -> None:
        # SCENARIO: Only 2 out of 1000 items recommended.
        # WHY: Coverage = 2/1000 = 0.002.
        # EXPECTED: Fails when min_coverage=0.1.
        recs = [["a", "b"]]
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_coverage(
                recs, catalog_size=1000, min_coverage=0.1,
            )
        c = exc_info.value.result.details["coverage"]
        assert abs(c - 0.002) < 1e-9

    def test_duplicate_items_counted_once(self) -> None:
        # SCENARIO: Same item recommended to multiple users.
        # WHY: Unique count should not double-count duplicates.
        # EXPECTED: coverage = 2/10 = 0.2.
        recs = [["a", "b"], ["a", "b"], ["a", "b"]]
        result = assert_coverage(
            recs, catalog_size=10, min_coverage=0.1,
        )
        assert result.passed is True
        assert result.details["unique_items"] == 2
        assert abs(result.details["coverage"] - 0.2) < 1e-9

    def test_empty_recommendations(self) -> None:
        # SCENARIO: No recommendations given to any user.
        # WHY: Zero items recommended, coverage = 0.
        # EXPECTED: Fails when min_coverage > 0.
        recs: list[list] = [[], []]
        with pytest.raises(MltkAssertionError):
            assert_coverage(
                recs, catalog_size=100, min_coverage=0.01,
            )

    def test_zero_catalog_trivially_passes(self) -> None:
        # SCENARIO: Catalog size is zero.
        # WHY: Edge case -- no items to cover.
        # EXPECTED: Trivially passes.
        result = assert_coverage(
            [["a"]], catalog_size=0, min_coverage=0.5,
        )
        assert result.passed is True
        assert result.details["coverage"] == 1.0

    def test_single_user_single_item(self) -> None:
        # SCENARIO: One user gets one item from a 5-item catalog.
        # WHY: Coverage = 1/5 = 0.2.
        # EXPECTED: Passes with min_coverage=0.1.
        recs = [["item1"]]
        result = assert_coverage(
            recs, catalog_size=5, min_coverage=0.1,
        )
        assert result.passed is True
        assert abs(result.details["coverage"] - 0.2) < 1e-9


# ------------------------------------------------------------------
# Serendipity
# ------------------------------------------------------------------


class TestSerendipity:
    """Serendipity -- unexpected relevant items."""

    def test_unexpected_relevant_items(self) -> None:
        # SCENARIO: 2 out of 4 recommended items are relevant but
        #   not in the expected set.
        # WHY: Serendipity = 2/4 = 0.5.
        # EXPECTED: Passes with min_serendipity=0.3.
        recs = [["a", "b", "c", "d"]]
        expected = [["a", "b"]]
        rels = [{"a", "c", "d"}]
        # "c" and "d" are relevant + recommended + NOT expected
        result = assert_serendipity(
            recs, expected, rels, min_serendipity=0.3,
        )
        assert result.passed is True
        assert abs(
            result.details["avg_serendipity"] - 0.5
        ) < 1e-9

    def test_only_expected_items_zero_serendipity(self) -> None:
        # SCENARIO: All relevant recommended items were expected.
        # WHY: No surprises -- serendipity = 0.
        # EXPECTED: Fails when min_serendipity > 0.
        recs = [["a", "b", "c"]]
        expected = [["a", "b", "c"]]
        rels = [{"a", "b"}]
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_serendipity(
                recs, expected, rels, min_serendipity=0.1,
            )
        s = exc_info.value.result.details["avg_serendipity"]
        assert abs(s - 0.0) < 1e-9

    def test_no_relevant_items_zero_serendipity(self) -> None:
        # SCENARIO: Recommended items are unexpected but irrelevant.
        # WHY: Must be BOTH unexpected and relevant.
        # EXPECTED: Serendipity = 0.
        recs = [["a", "b"]]
        expected = [["c", "d"]]
        rels = [{"c", "d"}]  # relevant items are only expected ones
        with pytest.raises(MltkAssertionError):
            assert_serendipity(
                recs, expected, rels, min_serendipity=0.1,
            )

    def test_empty_users_trivially_passes(self) -> None:
        # SCENARIO: No users at all.
        # WHY: Edge case -- nothing to evaluate.
        # EXPECTED: Trivially passes.
        result = assert_serendipity(
            [], [], [], min_serendipity=0.5,
        )
        assert result.passed is True
        assert result.details["avg_serendipity"] == 1.0

    def test_empty_recommendation_list(self) -> None:
        # SCENARIO: User has an empty recommendation list.
        # WHY: Cannot divide by zero -- should return 0.
        # EXPECTED: Per-user serendipity = 0.0.
        recs: list[list] = [[]]
        expected = [["a"]]
        rels = [{"a"}]
        with pytest.raises(MltkAssertionError):
            assert_serendipity(
                recs, expected, rels, min_serendipity=0.1,
            )

    def test_multiple_users_averaged(self) -> None:
        # SCENARIO: User A has 0.5 serendipity, user B has 0.
        # WHY: Mean = 0.25.
        # EXPECTED: Passes with min_serendipity=0.2.
        recs = [["a", "b", "c", "d"], ["e", "f"]]
        expected = [["a", "b"], ["e", "f"]]
        rels = [{"a", "c", "d"}, {"e"}]
        # User A: serendipitous = {c, d}, seren = 2/4 = 0.5
        # User B: serendipitous = {}, seren = 0/2 = 0.0
        # Mean = 0.25
        result = assert_serendipity(
            recs, expected, rels, min_serendipity=0.2,
        )
        assert result.passed is True
        per_user = result.details["per_user_serendipity"]
        assert abs(per_user[0] - 0.5) < 1e-9
        assert abs(per_user[1] - 0.0) < 1e-9

    def test_result_has_timing(self) -> None:
        # SCENARIO: @timed_assertion must populate duration_ms.
        # WHY: Performance tracking requirement.
        # EXPECTED: duration_ms > 0.
        result = assert_serendipity(
            [["a", "b"]], [["a"]], [{"b"}],
            min_serendipity=0.1,
        )
        assert result.duration_ms > 0


# ------------------------------------------------------------------
# S63 Hardening -- parametrized & edge-case tests
# ------------------------------------------------------------------


class TestHitRateParametrized:
    """Parametrized hit rate edge cases."""

    @pytest.mark.parametrize("min_rate", [0.0, 0.3, 0.5, 0.8, 1.0])
    def test_hit_rate_thresholds(
        self, min_rate: float,
    ) -> None:
        # 2 of 4 users hit => hit_rate = 0.5.
        recs = [["a"], ["b"], ["c"], ["d"]]
        rels = [{"a"}, {"x"}, {"c"}, {"y"}]
        if min_rate <= 0.5:
            r = assert_hit_rate(recs, rels, min_rate=min_rate)
            assert r.passed is True
            assert abs(r.details["hit_rate"] - 0.5) < 1e-9
        else:
            with pytest.raises(MltkAssertionError) as exc:
                assert_hit_rate(
                    recs, rels, min_rate=min_rate,
                )
            hr = exc.value.result.details["hit_rate"]
            assert abs(hr - 0.5) < 1e-9

    def test_hit_rate_empty_relevant_sets(self) -> None:
        # Every user has an empty relevant set.
        recs = [["a", "b"], ["c"]]
        rels: list[set] = [set(), set()]
        with pytest.raises(MltkAssertionError) as exc:
            assert_hit_rate(recs, rels, min_rate=0.1)
        assert exc.value.result.details["n_hits"] == 0

    def test_hit_rate_single_item_lists(self) -> None:
        # Each user gets exactly one item.
        recs = [["x"], ["y"], ["z"]]
        rels = [{"x"}, {"y"}, {"z"}]
        r = assert_hit_rate(recs, rels, min_rate=1.0)
        assert r.passed is True
        assert r.details["n_hits"] == 3

    def test_hit_rate_details_field_types(self) -> None:
        # Verify types in details dict.
        recs = [["a"]]
        rels = [{"a"}]
        r = assert_hit_rate(recs, rels, min_rate=0.5)
        assert isinstance(r.details["hit_rate"], float)
        assert isinstance(r.details["min_rate"], float)
        assert isinstance(r.details["n_hits"], int)
        assert isinstance(r.details["n_users"], int)


class TestDiversityHardened:
    """Diversity edge cases from S63."""

    def test_one_category_all_same(self) -> None:
        # Only 1 category in catalog => diversity = 1.0.
        recs = [["a", "b", "c"]]
        cats = {"a": "cat", "b": "cat", "c": "cat"}
        r = assert_diversity(recs, cats, min_diversity=0.5)
        assert r.passed is True
        assert abs(
            r.details["per_user_diversity"][0] - 1.0
        ) < 1e-9

    def test_all_unique_categories(self) -> None:
        # Each item is a different category.
        recs = [["a", "b", "c"]]
        cats = {"a": "x", "b": "y", "c": "z"}
        r = assert_diversity(recs, cats, min_diversity=0.9)
        assert r.passed is True
        assert abs(
            r.details["avg_diversity"] - 1.0
        ) < 1e-9

    def test_diversity_details_field_types(self) -> None:
        # Verify types in details dict.
        recs = [["a"]]
        cats = {"a": "x"}
        r = assert_diversity(recs, cats, min_diversity=0.5)
        d = r.details
        assert isinstance(d["avg_diversity"], float)
        assert isinstance(d["min_diversity"], float)
        assert isinstance(d["per_user_diversity"], list)


class TestNoveltyHardened:
    """Novelty edge cases from S63."""

    def test_popularity_one_zero_novelty(self) -> None:
        # popularity = 1.0 => -log2(1.0) = 0.0.
        recs = [["a", "b"]]
        pop = {"a": 1.0, "b": 1.0}
        with pytest.raises(MltkAssertionError) as exc:
            assert_novelty(recs, pop, min_novelty=0.01)
        n = exc.value.result.details["avg_novelty"]
        assert abs(n - 0.0) < 1e-9

    def test_very_low_popularity_high_novelty(self) -> None:
        # popularity = 0.001 => -log2(0.001) ~ 9.97.
        recs = [["rare"]]
        pop = {"rare": 0.001}
        r = assert_novelty(recs, pop, min_novelty=5.0)
        assert r.passed is True
        expected = -math.log2(0.001)
        assert abs(
            r.details["avg_novelty"] - expected
        ) < 1e-6

    def test_novelty_details_field_types(self) -> None:
        # Verify types in details dict.
        recs = [["a"]]
        pop = {"a": 0.5}
        r = assert_novelty(recs, pop, min_novelty=0.1)
        d = r.details
        assert isinstance(d["avg_novelty"], float)
        assert isinstance(d["min_novelty"], float)
        assert isinstance(d["per_user_novelty"], list)


class TestCoverageHardened:
    """Coverage edge cases from S63."""

    def test_duplicates_across_users_deduped(self) -> None:
        # Same 3 items across 5 users => unique = 3.
        recs = [["a", "b", "c"]] * 5
        r = assert_coverage(
            recs, catalog_size=10, min_coverage=0.1,
        )
        assert r.details["unique_items"] == 3
        assert abs(r.details["coverage"] - 0.3) < 1e-9

    def test_coverage_details_field_types(self) -> None:
        # Verify types in details dict.
        recs = [["a"]]
        r = assert_coverage(
            recs, catalog_size=5, min_coverage=0.1,
        )
        d = r.details
        assert isinstance(d["coverage"], float)
        assert isinstance(d["min_coverage"], float)
        assert isinstance(d["unique_items"], int)
        assert isinstance(d["catalog_size"], int)


class TestSerendipityHardened:
    """Serendipity edge cases from S63."""

    def test_no_expected_items(self) -> None:
        # Empty expected => every relevant rec is serendipitous.
        recs = [["a", "b", "c", "d"]]
        expected: list[list] = [[]]
        rels = [{"a", "b", "c", "d"}]
        r = assert_serendipity(
            recs, expected, rels, min_serendipity=0.5,
        )
        assert r.passed is True
        assert abs(
            r.details["per_user_serendipity"][0] - 1.0
        ) < 1e-9

    def test_all_items_expected_zero(self) -> None:
        # Every rec is expected => serendipity = 0.
        recs = [["a", "b"]]
        expected = [["a", "b"]]
        rels = [{"a", "b"}]
        with pytest.raises(MltkAssertionError) as exc:
            assert_serendipity(
                recs, expected, rels,
                min_serendipity=0.1,
            )
        s = exc.value.result.details["avg_serendipity"]
        assert abs(s - 0.0) < 1e-9

    def test_serendipity_details_field_types(self) -> None:
        # Verify types in details dict.
        recs = [["a", "b"]]
        expected = [["a"]]
        rels = [{"b"}]
        r = assert_serendipity(
            recs, expected, rels, min_serendipity=0.1,
        )
        d = r.details
        assert isinstance(d["avg_serendipity"], float)
        assert isinstance(d["min_serendipity"], float)
        assert isinstance(d["per_user_serendipity"], list)


class TestRecommendationLargeScale:
    """Large-scale / performance tests from S63."""

    def test_large_dataset_1000_users(self) -> None:
        # 1000 users, 20 items each -- must not crash.
        import random
        rng = random.Random(42)
        catalog = [f"item_{i}" for i in range(500)]
        recs = [
            rng.sample(catalog, 20) for _ in range(1000)
        ]
        rels = [
            set(rng.sample(catalog, 5))
            for _ in range(1000)
        ]
        r = assert_hit_rate(recs, rels, min_rate=0.01)
        assert r.passed is True
        assert r.details["n_users"] == 1000

        pop = {
            item: rng.uniform(0.01, 1.0)
            for item in catalog
        }
        rn = assert_novelty(recs, pop, min_novelty=0.01)
        assert rn.passed is True

        rc = assert_coverage(
            recs, catalog_size=500, min_coverage=0.01,
        )
        assert rc.passed is True
