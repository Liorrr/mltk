"""Tests for mltk.model.bias -- fairness testing across demographic groups.

Each test simulates a real-world bias scenario: fair models, biased models,
edge cases. Covers EU AI Act compliance and US four-fifths rule.
"""

import numpy as np
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.model.bias import assert_no_bias


class TestDemographicParity:
    """Demographic parity: equal selection rates across groups."""

    def test_fair_model(self) -> None:
        """PASS: Both groups have similar selection rates.

        Scenario: Hiring model selects candidates at ~50% rate
        for both male and female applicants.
        """
        # Both groups: 3 selected out of 5 = 60% each → diff=0
        y_true = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1])
        groups = np.array(["M", "M", "M", "M", "M", "F", "F", "F", "F", "F"])
        result = assert_no_bias(y_true, y_pred, groups, method="demographic_parity")
        assert result.passed is True

    def test_biased_model(self) -> None:
        """FAIL: One group selected at much higher rate.

        Scenario: Model selects 80% of group A but only 20% of group B.
        Clear demographic parity violation.
        """
        y_true = np.array([1, 1, 1, 1, 0, 1, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        groups = np.array(["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"])
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_bias(y_true, y_pred, groups, method="demographic_parity", threshold=0.10)
        assert "Bias detected" in str(exc.value)


class TestEqualizedOdds:
    """Equalized odds: equal TPR and FPR across groups."""

    def test_fair_model(self) -> None:
        """PASS: Similar TPR and FPR across groups."""
        rng = np.random.default_rng(42)
        n = 200
        y_true = rng.integers(0, 2, n)
        y_pred = y_true.copy()
        flip = rng.choice(n, size=20, replace=False)
        y_pred[flip] = 1 - y_pred[flip]
        groups = np.array(["A"] * 100 + ["B"] * 100)
        result = assert_no_bias(y_true, y_pred, groups, method="equalized_odds")
        assert result.passed is True

    def test_biased_model(self) -> None:
        """FAIL: Group B has much lower TPR than group A.

        Scenario: Face recognition works well for group A (90% TPR)
        but poorly for group B (30% TPR). Classic bias.
        """
        y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        y_pred_a = np.array([1, 1, 1, 1, 1])  # 100% TPR for A
        y_pred_b = np.array([0, 0, 0, 0, 1])  # 20% TPR for B
        y_pred = np.concatenate([y_pred_a, y_pred_b])
        groups = np.array(["A"] * 5 + ["B"] * 5)
        with pytest.raises(MltkAssertionError):
            assert_no_bias(y_true, y_pred, groups, method="equalized_odds", threshold=0.10)


class TestDisparateImpact:
    """Disparate impact: four-fifths rule (selection rate ratio >= 0.80)."""

    def test_passes_four_fifths(self) -> None:
        """PASS: Selection rate ratio meets four-fifths rule.

        Scenario: Group A selected at 50%, Group B at 45%.
        Ratio = 0.45/0.50 = 0.90 >= 0.80. Legal compliance met.
        """
        y_true = np.zeros(20, dtype=int)
        y_pred = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                           1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        groups = np.array(["A"] * 10 + ["B"] * 10)
        result = assert_no_bias(y_true, y_pred, groups, method="disparate_impact", threshold=0.80)
        assert result.passed is True

    def test_fails_four_fifths(self) -> None:
        """FAIL: Selection rate ratio below 0.80 -- legal risk.

        Scenario: Group A selected at 80%, Group B at 20%.
        Ratio = 0.20/0.80 = 0.25 < 0.80. Four-fifths rule violated.
        """
        y_true = np.zeros(10, dtype=int)
        y_pred = np.array([1, 1, 1, 1, 0, 1, 0, 0, 0, 0])
        groups = np.array(["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"])
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_bias(y_true, y_pred, groups, method="disparate_impact", threshold=0.80)
        assert "four-fifths" in str(exc.value)


class TestPredictiveParity:
    """Predictive parity: equal PPV (precision) across groups."""

    def test_equal_ppv(self) -> None:
        """PASS: Similar precision across groups."""
        y_true = np.array([1, 1, 0, 1, 1, 0])
        y_pred = np.array([1, 1, 0, 1, 1, 0])
        groups = np.array(["A", "A", "A", "B", "B", "B"])
        result = assert_no_bias(y_true, y_pred, groups, method="predictive_parity")
        assert result.passed is True


class TestEdgeCases:
    """Edge cases for bias testing."""

    def test_single_group(self) -> None:
        """PASS: Only one group -- bias check not applicable.

        Scenario: Dataset only contains one demographic.
        Can't measure between-group differences.
        """
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])
        groups = np.array(["A", "A", "A", "A"])
        result = assert_no_bias(y_true, y_pred, groups)
        assert result.passed is True

    def test_unknown_method(self) -> None:
        """FAIL: Invalid method name raises error."""
        with pytest.raises(MltkAssertionError):
            assert_no_bias([0, 1], [0, 1], ["A", "B"], method="invalid")

    def test_empty_arrays(self) -> None:
        """FAIL: Empty input handled gracefully."""
        with pytest.raises(MltkAssertionError):
            assert_no_bias([], [], [], method="demographic_parity")
