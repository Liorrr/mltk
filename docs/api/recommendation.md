# Recommendation System Evaluation

mltk is the **first ML testing toolkit** to offer recommendation system quality metrics as pytest-native assertions. No other testing framework provides these out of the box.

## Why Test Recommendations?

Recommendation systems fail silently in ways that accuracy metrics miss entirely:

| Failure Mode | What Happens | Which Metric Catches It |
|---|---|---|
| **Popularity bias** | System only recommends blockbusters everyone already knows | `assert_novelty` |
| **Filter bubble** | User sees only one genre/category forever | `assert_diversity` |
| **Long-tail blindness** | 99% of catalog items are never recommended to anyone | `assert_coverage` |
| **Broken relevance** | Half your users get zero useful recommendations | `assert_hit_rate` |
| **No added value** | System is indistinguishable from a "most popular" list | `assert_serendipity` |

A recommender can have 90% precision while exhibiting **all five failure modes simultaneously**. These assertions catch what accuracy cannot.

## The 5-Metric Evaluation Pipeline

Use all five together for a complete picture:

```
Hit Rate    --> Basic health: are users getting relevant items at all?
Diversity   --> User experience: are recommendations varied?
Novelty     --> Discovery: are niche items surfaced?
Coverage    --> Business: is the full catalog utilized?
Serendipity --> Delight: are there unexpected-but-relevant surprises?
```

---

## assert_hit_rate

Fraction of users who received at least one relevant recommendation.

**Formula:**

```
hit_rate = count(users with >= 1 relevant item) / total_users
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `recommended` | `list[list]` | required | Recommendation lists per user |
| `relevant` | `list[set]` | required | Relevant item sets per user |
| `min_rate` | `float` | `0.5` | Minimum acceptable hit rate |

**Returns:** TestResult with `hit_rate`, `min_rate`, `n_hits`, `n_users`

**Example:**

```python
from mltk.domains.recommendation import assert_hit_rate

# 3 users, each gets a list of recommended items
recommended = [
    ["movie_a", "movie_b", "movie_c"],
    ["movie_d", "movie_e", "movie_f"],
    ["movie_g", "movie_h", "movie_i"],
]

# Ground truth: items each user actually liked
relevant = [
    {"movie_a", "movie_z"},    # user 0: hit (movie_a)
    {"movie_x", "movie_y"},    # user 1: miss
    {"movie_i", "movie_j"},    # user 2: hit (movie_i)
]

# 2/3 users hit = 0.667
result = assert_hit_rate(recommended, relevant, min_rate=0.5)
assert result.details["hit_rate"] >= 0.5
```

**When to use:** As your first health check. If hit rate is low, fix relevance before measuring anything else.

---

## assert_diversity

Average category coverage across users' recommendation lists.

**Formula:**

```
diversity_per_user = |unique categories in user's recs| / |total categories|
diversity = mean(diversity_per_user) across all users
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `recommended` | `list[list]` | required | Recommendation lists per user |
| `item_categories` | `dict` | required | Mapping: item_id -> category string |
| `min_diversity` | `float` | `0.5` | Minimum acceptable mean diversity |

**Returns:** TestResult with `avg_diversity`, `min_diversity`, `per_user_diversity`

**Example:**

```python
from mltk.domains.recommendation import assert_diversity

recommended = [
    ["movie_1", "movie_2", "movie_3"],  # action, comedy, drama
    ["movie_4", "movie_5", "movie_6"],  # action, action, action
]

item_categories = {
    "movie_1": "action",  "movie_2": "comedy",  "movie_3": "drama",
    "movie_4": "action",  "movie_5": "action",  "movie_6": "action",
}

# User 0: 3/3 categories = 1.0
# User 1: 1/3 categories = 0.33
# Mean = 0.667
result = assert_diversity(recommended, item_categories, min_diversity=0.5)
```

**When to use:** To detect filter bubbles. If users see only one category, the system is pigeonholing them.

---

## assert_novelty

Average inverse-popularity of recommended items. Rewards surfacing niche content.

**Formula:**

```
novelty_per_item = -log2(popularity[item])
novelty_per_user = mean(novelty_per_item) for items in user's list
novelty = mean(novelty_per_user) across all users
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `recommended` | `list[list]` | required | Recommendation lists per user |
| `popularity` | `dict` | required | Mapping: item_id -> float in (0, 1] (fraction of users who interacted) |
| `min_novelty` | `float` | `0.3` | Minimum acceptable mean novelty |

**Returns:** TestResult with `avg_novelty`, `min_novelty`, `per_user_novelty`

**Example:**

```python
from mltk.domains.recommendation import assert_novelty

recommended = [
    ["niche_film", "indie_doc"],       # obscure items
    ["blockbuster_1", "blockbuster_2"],  # everyone knows these
]

popularity = {
    "niche_film": 0.01,       # 1% of users saw it -> high novelty
    "indie_doc": 0.02,        # 2% of users saw it -> high novelty
    "blockbuster_1": 0.90,    # 90% of users saw it -> low novelty
    "blockbuster_2": 0.85,    # 85% of users saw it -> low novelty
}

# -log2(0.01) = 6.64, -log2(0.90) = 0.15
result = assert_novelty(recommended, popularity, min_novelty=1.0)
```

**When to use:** To prevent popularity echo chambers. If novelty is near zero, the system is only recommending items users already know about.

---

## assert_coverage

Fraction of the total catalog that appears in at least one recommendation list.

**Formula:**

```
coverage = |unique items across all recommendation lists| / catalog_size
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `recommended` | `list[list]` | required | Recommendation lists per user |
| `catalog_size` | `int` | required | Total number of items in the catalog |
| `min_coverage` | `float` | `0.1` | Minimum acceptable coverage |

**Returns:** TestResult with `coverage`, `min_coverage`, `unique_items`, `catalog_size`

**Example:**

```python
from mltk.domains.recommendation import assert_coverage

recommended = [
    ["item_1", "item_2"],
    ["item_2", "item_3"],
    ["item_3", "item_4"],
]

# 4 unique items recommended out of 1000 in catalog
# coverage = 4/1000 = 0.004 -- very low!
result = assert_coverage(
    recommended,
    catalog_size=1000,
    min_coverage=0.01,  # require at least 1% coverage
)
```

**When to use:** To protect business revenue from long-tail items. If 99% of the catalog is invisible, niche products never get a chance.

---

## assert_serendipity

Fraction of recommended items that are relevant but unexpected.

**Formula:**

```
serendipity_per_user = |relevant & recommended & ~expected| / |recommended|
serendipity = mean(serendipity_per_user) across all users
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `recommended` | `list[list]` | required | Recommendation lists per user |
| `expected` | `list[list]` | required | Baseline (e.g., most-popular) recommendation lists per user |
| `relevant` | `list[set]` | required | Relevant item sets per user |
| `min_serendipity` | `float` | `0.1` | Minimum acceptable mean serendipity |

**Returns:** TestResult with `avg_serendipity`, `min_serendipity`, `per_user_serendipity`

**Example:**

```python
from mltk.domains.recommendation import assert_serendipity

# What the model recommends
recommended = [["a", "b", "c", "d"]]

# What a simple popularity baseline would recommend
expected = [["a", "b", "e", "f"]]

# What the user actually found relevant
relevant = [{"a", "c", "d"}]

# "a" is relevant but expected (not serendipitous)
# "c" and "d" are relevant AND unexpected (serendipitous!)
# serendipity = 2/4 = 0.5
result = assert_serendipity(
    recommended, expected, relevant, min_serendipity=0.3,
)
```

**When to use:** The ultimate test of recommender value-add. If serendipity is zero, your sophisticated model is no better than sorting by popularity.

---

## Complete Pipeline Example

Run all five metrics together in a pytest test suite:

```python
import pytest
from mltk.domains.recommendation import (
    assert_hit_rate,
    assert_diversity,
    assert_novelty,
    assert_coverage,
    assert_serendipity,
)


class TestRecommenderQuality:
    """Full beyond-accuracy evaluation of the recommendation model."""

    def test_basic_health(self, model_output, ground_truth):
        """At least 70% of users should get a relevant item."""
        assert_hit_rate(
            model_output.recommendations,
            ground_truth.relevant_items,
            min_rate=0.7,
        )

    def test_not_a_filter_bubble(self, model_output, catalog):
        """Recommendations should span multiple categories."""
        assert_diversity(
            model_output.recommendations,
            catalog.item_categories,
            min_diversity=0.4,
        )

    def test_not_a_popularity_contest(self, model_output, catalog):
        """Niche items should appear, not just blockbusters."""
        assert_novelty(
            model_output.recommendations,
            catalog.item_popularity,
            min_novelty=2.0,
        )

    def test_catalog_utilization(self, model_output, catalog):
        """At least 10% of the catalog should be recommended."""
        assert_coverage(
            model_output.recommendations,
            catalog_size=catalog.total_items,
            min_coverage=0.1,
        )

    def test_genuine_discovery(
        self, model_output, baseline_output, ground_truth,
    ):
        """Model should surface items a popularity sort would miss."""
        assert_serendipity(
            model_output.recommendations,
            baseline_output.recommendations,
            ground_truth.relevant_items,
            min_serendipity=0.05,
        )
```

## When to Use Which Metric

| Goal | Metric | Threshold Guidance |
|---|---|---|
| **Is the system working at all?** | `hit_rate` | >= 0.5 for most domains |
| **Are users getting varied content?** | `diversity` | >= 0.3 category coverage |
| **Are niche items surfaced?** | `novelty` | Depends on domain; higher = more long-tail |
| **Is the full catalog visible?** | `coverage` | >= 0.1 (10% minimum utilization) |
| **Does the model add real value?** | `serendipity` | >= 0.05 (even 5% surprise is valuable) |

## First-Mover Advantage

No other ML testing framework offers recommendation beyond-accuracy metrics as native assertions:

| Framework | Hit Rate | Diversity | Novelty | Coverage | Serendipity |
|---|---|---|---|---|---|
| **mltk** | Yes | Yes | Yes | Yes | Yes |
| Great Expectations | No | No | No | No | No |
| Deepchecks | No | No | No | No | No |
| Evidently | No | No | No | No | No |
| WhyLogs | No | No | No | No | No |

These metrics exist in academic papers but have never been packaged as pytest-native, CI-ready assertions until now.
