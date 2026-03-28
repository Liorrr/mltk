"""Tests for mltk.domains.healthcare -- clinical diagnostic metrics."""

import math

import numpy as np
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.domains.healthcare import (
    assert_clinical_agreement,
    assert_npv,
    assert_ppv,
    assert_sensitivity,
    assert_specificity,
)


# ------------------------------------------------------------------
# Sensitivity (True Positive Rate)
# ------------------------------------------------------------------


class TestSensitivity:
    """Sensitivity -- fraction of actual positives detected."""

    def test_perfect_sensitivity(self) -> None:
        # SCENARIO: Model catches every positive case.
        # WHY: TP=5, FN=0 => sensitivity=1.0.
        # EXPECTED: Passes with any threshold <= 1.0.
        y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1, 1, 0, 0, 0])
        result = assert_sensitivity(
            y_true, y_pred, min_sensitivity=1.0,
        )
        assert result.passed is True
        assert abs(result.details["sensitivity"] - 1.0) < 1e-9
        assert result.details["tp"] == 5
        assert result.details["fn"] == 0
        assert result.details["n_positive"] == 5

    def test_zero_sensitivity(self) -> None:
        # SCENARIO: Model misses every positive case.
        # WHY: TP=0, FN=4 => sensitivity=0.0.
        # EXPECTED: Fails when min_sensitivity > 0.
        y_true = np.array([1, 1, 1, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 0, 0, 0])
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_sensitivity(
                y_true, y_pred, min_sensitivity=0.1,
            )
        r = exc_info.value.result
        assert abs(r.details["sensitivity"] - 0.0) < 1e-9
        assert r.details["fn"] == 4

    def test_partial_sensitivity(self) -> None:
        # SCENARIO: Model catches 3 out of 4 positives.
        # WHY: sensitivity = 3/4 = 0.75.
        # EXPECTED: Passes with 0.7, fails with 0.8.
        y_true = np.array([1, 1, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0, 0])
        result = assert_sensitivity(
            y_true, y_pred, min_sensitivity=0.7,
        )
        assert result.passed is True
        assert abs(result.details["sensitivity"] - 0.75) < 1e-9

        with pytest.raises(MltkAssertionError):
            assert_sensitivity(
                y_true, y_pred, min_sensitivity=0.8,
            )

    def test_no_positive_cases_undefined(self) -> None:
        # SCENARIO: y_true has no positive cases.
        # WHY: TP + FN = 0 => sensitivity undefined.
        # EXPECTED: Fails with descriptive message.
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 1, 0])
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_sensitivity(y_true, y_pred)
        assert "undefined" in exc_info.value.result.message
        assert math.isnan(
            exc_info.value.result.details["sensitivity"]
        )

    def test_empty_arrays(self) -> None:
        # SCENARIO: Both arrays are empty.
        # WHY: Edge case -- nothing to evaluate.
        # EXPECTED: Fails with descriptive message.
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_sensitivity(
                np.array([]), np.array([]),
            )
        assert "empty" in exc_info.value.result.message

    def test_timing_populated(self) -> None:
        # SCENARIO: @timed_assertion must populate duration_ms.
        # WHY: Performance tracking requirement.
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])
        result = assert_sensitivity(
            y_true, y_pred, min_sensitivity=0.5,
        )
        assert result.duration_ms > 0

    def test_assertion_name(self) -> None:
        # SCENARIO: Verify the assertion name follows convention.
        y_true = np.array([1, 0])
        y_pred = np.array([1, 0])
        result = assert_sensitivity(
            y_true, y_pred, min_sensitivity=0.5,
        )
        assert result.name == "healthcare.sensitivity"


# ------------------------------------------------------------------
# Specificity (True Negative Rate)
# ------------------------------------------------------------------


class TestSpecificity:
    """Specificity -- fraction of actual negatives cleared."""

    def test_perfect_specificity(self) -> None:
        # SCENARIO: Model correctly identifies all negatives.
        # WHY: TN=4, FP=0 => specificity=1.0.
        # EXPECTED: Passes.
        y_true = np.array([0, 0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 1, 1])
        result = assert_specificity(
            y_true, y_pred, min_specificity=1.0,
        )
        assert result.passed is True
        assert abs(
            result.details["specificity"] - 1.0
        ) < 1e-9
        assert result.details["tn"] == 4
        assert result.details["fp"] == 0

    def test_low_specificity_false_alarms(self) -> None:
        # SCENARIO: Model flags 3 out of 5 negatives as positive.
        # WHY: specificity = 2/5 = 0.4.
        # EXPECTED: Fails with min_specificity=0.5.
        y_true = np.array([0, 0, 0, 0, 0, 1, 1])
        y_pred = np.array([1, 1, 1, 0, 0, 1, 1])
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_specificity(
                y_true, y_pred, min_specificity=0.5,
            )
        r = exc_info.value.result
        assert abs(r.details["specificity"] - 0.4) < 1e-9
        assert r.details["fp"] == 3

    def test_no_negative_cases_undefined(self) -> None:
        # SCENARIO: y_true has no negative cases.
        # WHY: TN + FP = 0 => specificity undefined.
        # EXPECTED: Fails with descriptive message.
        y_true = np.array([1, 1, 1])
        y_pred = np.array([1, 1, 0])
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_specificity(y_true, y_pred)
        assert "undefined" in exc_info.value.result.message

    def test_assertion_name(self) -> None:
        # SCENARIO: Verify the assertion name follows convention.
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        result = assert_specificity(
            y_true, y_pred, min_specificity=0.5,
        )
        assert result.name == "healthcare.specificity"


# ------------------------------------------------------------------
# PPV (Positive Predictive Value)
# ------------------------------------------------------------------


class TestPPV:
    """PPV -- precision of positive predictions."""

    def test_perfect_ppv(self) -> None:
        # SCENARIO: Every positive prediction is correct.
        # WHY: TP=3, FP=0 => PPV=1.0.
        # EXPECTED: Passes.
        y_true = np.array([1, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0])
        result = assert_ppv(y_true, y_pred, min_ppv=1.0)
        assert result.passed is True
        assert abs(result.details["ppv"] - 1.0) < 1e-9

    def test_half_ppv(self) -> None:
        # SCENARIO: Half of positive predictions are wrong.
        # WHY: TP=2, FP=2 => PPV=0.5.
        # EXPECTED: Passes at 0.5, fails at 0.6.
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 1, 1, 1])
        result = assert_ppv(y_true, y_pred, min_ppv=0.5)
        assert result.passed is True
        assert abs(result.details["ppv"] - 0.5) < 1e-9

        with pytest.raises(MltkAssertionError):
            assert_ppv(y_true, y_pred, min_ppv=0.6)

    def test_no_positive_predictions_undefined(self) -> None:
        # SCENARIO: Model never predicts positive.
        # WHY: TP + FP = 0 => PPV undefined.
        # EXPECTED: Fails.
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_ppv(y_true, y_pred)
        assert "undefined" in exc_info.value.result.message

    def test_assertion_name(self) -> None:
        # SCENARIO: Verify the assertion name follows convention.
        y_true = np.array([1, 0])
        y_pred = np.array([1, 0])
        result = assert_ppv(y_true, y_pred, min_ppv=0.5)
        assert result.name == "healthcare.ppv"


# ------------------------------------------------------------------
# NPV (Negative Predictive Value)
# ------------------------------------------------------------------


class TestNPV:
    """NPV -- precision of negative predictions."""

    def test_perfect_npv(self) -> None:
        # SCENARIO: Every negative prediction is correct.
        # WHY: TN=3, FN=0 => NPV=1.0.
        # EXPECTED: Passes.
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1])
        result = assert_npv(y_true, y_pred, min_npv=1.0)
        assert result.passed is True
        assert abs(result.details["npv"] - 1.0) < 1e-9

    def test_missed_positive_lowers_npv(self) -> None:
        # SCENARIO: Model says "negative" but patient is positive.
        # WHY: TN=3, FN=1 => NPV=3/4=0.75.
        # EXPECTED: Passes at 0.7, fails at 0.8.
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 1])
        result = assert_npv(y_true, y_pred, min_npv=0.7)
        assert result.passed is True
        assert abs(result.details["npv"] - 0.75) < 1e-9

        with pytest.raises(MltkAssertionError):
            assert_npv(y_true, y_pred, min_npv=0.8)

    def test_no_negative_predictions_undefined(self) -> None:
        # SCENARIO: Model predicts positive for everything.
        # WHY: TN + FN = 0 => NPV undefined.
        # EXPECTED: Fails.
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 1, 1])
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_npv(y_true, y_pred)
        assert "undefined" in exc_info.value.result.message

    def test_assertion_name(self) -> None:
        # SCENARIO: Verify the assertion name follows convention.
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        result = assert_npv(y_true, y_pred, min_npv=0.5)
        assert result.name == "healthcare.npv"


# ------------------------------------------------------------------
# Clinical Agreement (Cohen's Kappa)
# ------------------------------------------------------------------


class TestClinicalAgreement:
    """Cohen's Kappa -- agreement beyond chance."""

    def test_perfect_agreement(self) -> None:
        # SCENARIO: Model matches ground truth perfectly.
        # WHY: p_observed=1.0, kappa=1.0.
        # EXPECTED: Passes.
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        result = assert_clinical_agreement(
            y_true, y_pred, min_kappa=1.0,
        )
        assert result.passed is True
        assert abs(result.details["kappa"] - 1.0) < 1e-9
        assert abs(
            result.details["p_observed"] - 1.0
        ) < 1e-9

    def test_random_agreement_low_kappa(self) -> None:
        # SCENARIO: Imbalanced dataset where always-predict-0
        #   achieves high accuracy but low kappa.
        # WHY: 95% negative base rate. Model predicts all 0.
        #   accuracy=0.95 but kappa~=0 (no better than chance).
        # EXPECTED: Fails with min_kappa=0.1.
        rng = np.random.RandomState(42)
        n = 200
        y_true = np.zeros(n, dtype=int)
        y_true[:10] = 1  # 5% positive rate
        rng.shuffle(y_true)
        y_pred = np.zeros(n, dtype=int)  # always predict 0

        with pytest.raises(MltkAssertionError) as exc_info:
            assert_clinical_agreement(
                y_true, y_pred, min_kappa=0.1,
            )
        r = exc_info.value.result
        # Kappa should be near 0 (no better than chance)
        assert r.details["kappa"] < 0.1

    def test_moderate_agreement(self) -> None:
        # SCENARIO: Model agrees on most but not all cases.
        # WHY: Verify intermediate kappa values.
        # EXPECTED: Kappa between 0 and 1.
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 0])
        y_pred = np.array([1, 1, 0, 1, 0, 0, 1, 0, 1, 0])
        # Manual: TP=4, TN=4, FP=1, FN=1, N=10
        # p_o = 8/10 = 0.8
        # p_yes = (5/10)*(5/10) = 0.25
        # p_no  = (5/10)*(5/10) = 0.25
        # p_e = 0.5
        # kappa = (0.8 - 0.5) / (1 - 0.5) = 0.6
        result = assert_clinical_agreement(
            y_true, y_pred, min_kappa=0.6,
        )
        assert result.passed is True
        assert abs(result.details["kappa"] - 0.6) < 1e-9

    def test_degenerate_all_same_class(self) -> None:
        # SCENARIO: All samples are class 0, model predicts 0.
        # WHY: p_expected=1.0, kappa undefined.
        # EXPECTED: Fails with descriptive message.
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_clinical_agreement(y_true, y_pred)
        assert "undefined" in exc_info.value.result.message

    def test_assertion_name(self) -> None:
        # SCENARIO: Verify the assertion name follows convention.
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])
        result = assert_clinical_agreement(
            y_true, y_pred, min_kappa=0.5,
        )
        assert (
            result.name
            == "healthcare.clinical_agreement"
        )


# ------------------------------------------------------------------
# Cross-cutting: validation & edge cases
# ------------------------------------------------------------------


class TestValidation:
    """Input validation shared across all assertions."""

    def test_length_mismatch_sensitivity(self) -> None:
        # SCENARIO: y_true and y_pred have different lengths.
        # EXPECTED: Fails with descriptive message.
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_sensitivity(
                np.array([1, 0, 1]),
                np.array([1, 0]),
            )
        assert "mismatch" in exc_info.value.result.message

    def test_non_binary_values_specificity(self) -> None:
        # SCENARIO: Labels contain values other than 0/1.
        # EXPECTED: Fails with descriptive message.
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_specificity(
                np.array([0, 1, 2]),
                np.array([0, 1, 1]),
            )
        assert "Non-binary" in exc_info.value.result.message

    def test_non_binary_values_ppv(self) -> None:
        # SCENARIO: Predictions contain values other than 0/1.
        # EXPECTED: Fails with descriptive message.
        with pytest.raises(MltkAssertionError) as exc_info:
            assert_ppv(
                np.array([0, 1, 0]),
                np.array([0, 3, 0]),
            )
        assert "Non-binary" in exc_info.value.result.message

    def test_warning_severity_does_not_raise(self) -> None:
        # SCENARIO: Failing assertion with WARNING severity.
        # WHY: Only CRITICAL raises MltkAssertionError.
        # EXPECTED: Returns result with passed=False, no raise.
        from mltk.core.result import Severity
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        result = assert_sensitivity(
            y_true, y_pred,
            min_sensitivity=0.9,
            severity=Severity.WARNING,
        )
        assert result.passed is False
        assert result.details["sensitivity"] == 0.0


# ------------------------------------------------------------------
# Large-scale deterministic test
# ------------------------------------------------------------------


class TestLargeScale:
    """Large dataset with fixed seed -- no randomness."""

    def test_1000_samples_deterministic(self) -> None:
        # SCENARIO: 1000 samples, known seed, verify all 5 metrics.
        # WHY: Ensures assertions scale and produce stable results.
        rng = np.random.RandomState(123)
        n = 1000
        y_true = rng.randint(0, 2, size=n)
        # Create predictions with ~80% accuracy
        y_pred = y_true.copy()
        flip_idx = rng.choice(n, size=200, replace=False)
        y_pred[flip_idx] = 1 - y_pred[flip_idx]

        # All should run without error at reasonable thresholds
        r_sens = assert_sensitivity(
            y_true, y_pred, min_sensitivity=0.5,
        )
        assert r_sens.passed is True
        assert r_sens.details["n_positive"] > 0

        r_spec = assert_specificity(
            y_true, y_pred, min_specificity=0.5,
        )
        assert r_spec.passed is True
        assert r_spec.details["n_negative"] > 0

        r_ppv = assert_ppv(
            y_true, y_pred, min_ppv=0.5,
        )
        assert r_ppv.passed is True

        r_npv = assert_npv(
            y_true, y_pred, min_npv=0.5,
        )
        assert r_npv.passed is True

        r_kappa = assert_clinical_agreement(
            y_true, y_pred, min_kappa=0.3,
        )
        assert r_kappa.passed is True
        assert r_kappa.details["n_samples"] == 1000
