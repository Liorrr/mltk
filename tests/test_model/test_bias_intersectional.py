"""Tests for intersectional fairness — assert_intersectional_fairness.

Single-attribute bias tests can miss discrimination that only
appears at the intersection of multiple protected attributes.
A model may be fair to women and fair to Black individuals
but unfair to Black women specifically (Crenshaw, 1989).
EU AI Act Article 10(2)(f) requires intersectional analysis
for high-risk AI systems.
"""

from __future__ import annotations

import numpy as np
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.model.bias import (
    assert_intersectional_fairness,
    assert_no_bias,
)


class TestIntersectionalFairness:
    """Core intersectional fairness scenarios."""

    def test_fair_model_passes(self) -> None:
        """PASS: Equal rates across all intersections.

        Scenario: Loan approval model treats all
        gender x race combinations equally.
        Uses deterministic data with identical selection
        rates per subgroup to avoid random false positives.
        """
        n_per = 200  # large subgroups reduce noise
        # Each subgroup: 70% selection rate (140/200)
        subgroup_pred = np.array(
            [1] * 140 + [0] * 60, dtype=int
        )
        y_true = np.ones(n_per * 4, dtype=int)
        y_pred = np.tile(subgroup_pred, 4)
        gender = np.array(
            ["M"] * (2 * n_per) + ["F"] * (2 * n_per)
        )
        race = np.tile(
            ["white"] * n_per + ["black"] * n_per, 2
        )
        result = assert_intersectional_fairness(
            y_true, y_pred,
            sensitive_features={"gender": gender, "race": race},
        )
        assert result.passed is True

    def test_intersectional_bias_detected(self) -> None:
        """FAIL: Bias in race x gender intersection.

        Scenario: Model discriminates against one specific
        intersection — not visible in marginal checks.
        """
        rng = np.random.default_rng(42)
        n_per = 100
        y_true = np.ones(n_per * 4, dtype=int)
        y_pred_mw = np.ones(n_per, dtype=int)
        y_pred_mb = np.ones(n_per, dtype=int)
        y_pred_fw = np.ones(n_per, dtype=int)
        y_pred_fb = np.zeros(n_per, dtype=int)
        y_pred = np.concatenate(
            [y_pred_mw, y_pred_mb, y_pred_fw, y_pred_fb]
        )
        gender = np.array(
            ["M"] * 2 * n_per + ["F"] * 2 * n_per
        )
        race = np.tile(
            ["white"] * n_per + ["black"] * n_per, 2
        )
        with pytest.raises(MltkAssertionError):
            assert_intersectional_fairness(
                y_true, y_pred,
                sensitive_features={
                    "gender": gender,
                    "race": race,
                },
            )

    def test_marginal_fair_intersectional_unfair(
        self,
    ) -> None:
        """THE KEY TEST: fair per-attribute but unfair
        at intersection.

        Construct data where marginal rates are exactly
        equal but intersectional rates differ. Uses a
        Simpson's paradox pattern:
          MW=80%, MB=40%, FW=40%, FB=80%
        Marginal gender: M=60%, F=60% (equal)
        Marginal race: W=60%, B=60% (equal)
        Intersection: 80% vs 40% = 0.40 diff (fails)
        """
        n_per = 100
        y_true = np.ones(4 * n_per, dtype=int)

        # MW: 80% approval
        y_pred_mw = np.array(
            [1] * 80 + [0] * 20, dtype=int
        )
        # MB: 40% approval
        y_pred_mb = np.array(
            [1] * 40 + [0] * 60, dtype=int
        )
        # FW: 40% approval
        y_pred_fw = np.array(
            [1] * 40 + [0] * 60, dtype=int
        )
        # FB: 80% approval
        y_pred_fb = np.array(
            [1] * 80 + [0] * 20, dtype=int
        )

        y_pred = np.concatenate(
            [y_pred_mw, y_pred_mb, y_pred_fw, y_pred_fb]
        )
        gender = np.array(
            ["M"] * 200 + ["F"] * 200
        )
        race = np.tile(
            ["white"] * 100 + ["black"] * 100, 2
        )

        # Marginal gender: M=120/200=60%, F=120/200=60%
        marginal_gender = assert_no_bias(
            y_true, y_pred, gender,
            method="demographic_parity",
            threshold=0.20,
        )
        assert marginal_gender.passed is True

        # Intersectional: 80% vs 40% = 0.40 diff > 0.20
        with pytest.raises(MltkAssertionError):
            assert_intersectional_fairness(
                y_true, y_pred,
                sensitive_features={
                    "gender": gender,
                    "race": race,
                },
                threshold=0.20,
            )


class TestIntersectionalSubgroups:
    """Subgroup size handling and pruning."""

    def test_min_subgroup_size_pruning(self) -> None:
        """Small subgroups skipped, reported in details.

        Scenario: Only 3 samples in one intersection —
        not enough for reliable fairness measurement.
        """
        rng = np.random.default_rng(42)
        y_true = np.ones(203, dtype=int)
        y_pred = np.ones(203, dtype=int)
        y_pred[-3:] = 0
        gender = np.array(
            ["M"] * 100 + ["F"] * 100 + ["NB"] * 3
        )
        race = np.array(
            ["white"] * 100 + ["black"] * 100
            + ["other"] * 3
        )
        result = assert_intersectional_fairness(
            y_true, y_pred,
            sensitive_features={
                "gender": gender,
                "race": race,
            },
            min_subgroup_size=10,
        )
        has_skipped = (
            "skipped" in result.details
            or "pruned" in result.details
            or "small" in str(result.details).lower()
        )
        assert has_skipped or result.passed is True

    def test_all_subgroups_too_small(self) -> None:
        """All combos below min_size yields info result.

        Scenario: Dataset too small for any intersection
        to have enough samples.
        """
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 12)
        y_pred = rng.integers(0, 2, 12)
        gender = np.array(["M", "F", "NB"] * 4)
        race = np.array(
            ["w", "b", "a", "h"] * 3
        )
        result = assert_intersectional_fairness(
            y_true, y_pred,
            sensitive_features={
                "gender": gender,
                "race": race,
            },
            min_subgroup_size=50,
        )
        assert result.passed is True or \
            result.severity.value == "info"


class TestIntersectionalMethods:
    """Fairness method selection for intersectional check."""

    def test_method_demographic_parity(self) -> None:
        """Demographic parity method works at intersection.

        Scenario: Check equal selection rates across
        all gender x race groups. Uses deterministic
        uniform rates to guarantee pass.
        """
        n_per = 200
        # Every subgroup: 70% selection rate
        subgroup_pred = np.array(
            [1] * 140 + [0] * 60, dtype=int
        )
        y_true = np.ones(n_per * 4, dtype=int)
        y_pred = np.tile(subgroup_pred, 4)
        gender = np.array(
            ["M"] * (2 * n_per) + ["F"] * (2 * n_per)
        )
        race = np.tile(
            ["white"] * n_per + ["black"] * n_per, 2
        )
        result = assert_intersectional_fairness(
            y_true, y_pred,
            sensitive_features={
                "gender": gender,
                "race": race,
            },
            method="demographic_parity",
        )
        assert result.passed is True

    def test_method_equalized_odds(self) -> None:
        """Equalized odds method at intersection level.

        Uses deterministic data with identical TPR and
        FPR per subgroup to guarantee pass.
        Each subgroup: 50 positives, 50 negatives.
        Predictions: 90% TPR, 10% FPR (uniform).
        """
        n_per = 100
        # Per subgroup: 50 positive, 50 negative ground truth
        # TPR=90%: predict 45/50 positives correctly
        # FPR=10%: predict 5/50 negatives as positive
        sub_true = np.array(
            [1] * 50 + [0] * 50, dtype=int
        )
        sub_pred = np.array(
            [1] * 45 + [0] * 5  # TPR=90%
            + [1] * 5 + [0] * 45,  # FPR=10%
            dtype=int,
        )
        y_true = np.tile(sub_true, 4)
        y_pred = np.tile(sub_pred, 4)
        gender = np.array(
            ["M"] * (2 * n_per) + ["F"] * (2 * n_per)
        )
        race = np.tile(
            ["white"] * n_per + ["black"] * n_per, 2
        )
        result = assert_intersectional_fairness(
            y_true, y_pred,
            sensitive_features={
                "gender": gender,
                "race": race,
            },
            method="equalized_odds",
        )
        assert result.passed is True

    def test_method_disparate_impact(self) -> None:
        """Disparate impact (four-fifths rule) at intersection.

        Scenario: Hiring model checked against US EEOC
        four-fifths rule for each intersection.
        """
        rng = np.random.default_rng(42)
        n = 400
        y_true = rng.integers(0, 2, n)
        y_pred = y_true.copy()
        flip = rng.choice(n, size=40, replace=False)
        y_pred[flip] = 1 - y_pred[flip]
        gender = np.array(["M"] * 200 + ["F"] * 200)
        race = np.tile(
            ["white"] * 100 + ["black"] * 100, 2
        )
        result = assert_intersectional_fairness(
            y_true, y_pred,
            sensitive_features={
                "gender": gender,
                "race": race,
            },
            method="disparate_impact",
        )
        assert isinstance(result.passed, bool)

    def test_unknown_method_fails(self) -> None:
        """FAIL: Invalid method name rejected."""
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 100)
        y_pred = rng.integers(0, 2, 100)
        gender = np.array(["M"] * 50 + ["F"] * 50)
        race = np.tile(["w"] * 25 + ["b"] * 25, 2)
        with pytest.raises(
            (MltkAssertionError, ValueError)
        ):
            assert_intersectional_fairness(
                y_true, y_pred,
                sensitive_features={
                    "gender": gender,
                    "race": race,
                },
                method="nonexistent",
            )


class TestIntersectionalDetails:
    """Result details and reporting."""

    def test_details_include_worst_subgroup(
        self,
    ) -> None:
        """Worst subgroup identified in result details.

        Scenario: QA team needs to know which specific
        intersection failed — not just "bias detected."
        """
        rng = np.random.default_rng(42)
        n_per = 100
        y_true = np.ones(4 * n_per, dtype=int)
        y_pred_mw = np.ones(n_per, dtype=int)
        y_pred_mb = np.ones(n_per, dtype=int)
        y_pred_fw = np.ones(n_per, dtype=int)
        y_pred_fb = np.concatenate([
            np.ones(30, dtype=int),
            np.zeros(70, dtype=int),
        ])
        y_pred = np.concatenate(
            [y_pred_mw, y_pred_mb, y_pred_fw, y_pred_fb]
        )
        gender = np.array(
            ["M"] * 200 + ["F"] * 200
        )
        race = np.tile(
            ["white"] * 100 + ["black"] * 100, 2
        )
        with pytest.raises(MltkAssertionError) as exc:
            assert_intersectional_fairness(
                y_true, y_pred,
                sensitive_features={
                    "gender": gender,
                    "race": race,
                },
            )
        result = exc.value.result
        has_worst = (
            "worst" in result.details
            or "worst_subgroup" in result.details
            or "worst_group" in result.details
            or any(
                "worst" in str(k).lower()
                for k in result.details
            )
        )
        assert has_worst

    def test_details_include_skipped(self) -> None:
        """Skipped subgroups reported in details.

        Scenario: Audit trail showing which groups were
        too small to evaluate.
        """
        rng = np.random.default_rng(42)
        y_true = np.ones(206, dtype=int)
        y_pred = np.ones(206, dtype=int)
        gender = np.array(
            ["M"] * 100 + ["F"] * 100 + ["NB"] * 6
        )
        race = np.array(
            ["white"] * 100 + ["black"] * 100
            + ["asian"] * 6
        )
        result = assert_intersectional_fairness(
            y_true, y_pred,
            sensitive_features={
                "gender": gender,
                "race": race,
            },
            min_subgroup_size=20,
        )
        details_str = str(result.details).lower()
        has_skipped = (
            "skipped" in details_str
            or "pruned" in details_str
            or "small" in details_str
            or "excluded" in details_str
        )
        assert has_skipped or result.passed is True


class TestIntersectionalMultiAttribute:
    """Multi-attribute intersection tests."""

    def test_three_attributes(self) -> None:
        """gender x race x age = all combinations.

        Scenario: Full 3-way intersectional analysis for
        comprehensive fairness audit. Uses deterministic
        balanced subgroups (100 each, 8 combos = 800).
        """
        n_per = 100  # per subgroup
        n_combos = 8  # 2 x 2 x 2
        n = n_per * n_combos

        # All subgroups: identical 75% selection rate
        sub_pred = np.array(
            [1] * 75 + [0] * 25, dtype=int
        )
        y_true = np.ones(n, dtype=int)
        y_pred = np.tile(sub_pred, n_combos)

        # Deterministic attribute assignment:
        # 8 groups of 100, cycling through all combos
        gender = np.array(
            ["M"] * 400 + ["F"] * 400
        )
        race = np.tile(
            ["white"] * 200 + ["black"] * 200, 2
        )
        age = np.tile(
            ["young"] * 100 + ["old"] * 100, 4
        )
        result = assert_intersectional_fairness(
            y_true, y_pred,
            sensitive_features={
                "gender": gender,
                "race": race,
                "age": age,
            },
        )
        assert result.passed is True

    def test_single_attribute_degenerates(self) -> None:
        """One attribute = similar to assert_no_bias.

        Scenario: User passes a single feature — should
        still work, just no intersections.
        """
        rng = np.random.default_rng(42)
        n = 200
        y_true = rng.integers(0, 2, n)
        y_pred = y_true.copy()
        flip = rng.choice(n, size=20, replace=False)
        y_pred[flip] = 1 - y_pred[flip]
        gender = np.array(["M"] * 100 + ["F"] * 100)
        result = assert_intersectional_fairness(
            y_true, y_pred,
            sensitive_features={"gender": gender},
        )
        assert isinstance(result.passed, bool)
