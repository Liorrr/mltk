"""Tests for assert_no_concept_drift.

Verifies detection of concept drift (P(Y|X)) -- when the relationship between
inputs and outputs changes. This is distinct from input drift (P(X)) and output
drift (P(Y-hat)): a model can receive identically distributed features and
produce identically distributed scores while getting the answers wrong in a
completely different pattern.
"""

from __future__ import annotations

import numpy as np
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.monitor.concept_drift import assert_no_concept_drift


class TestAssertNoConceptDrift:
    """Tests for assert_no_concept_drift."""

    # ── Same error rate (should pass) ────────────────────────────────────

    def test_same_error_rate_chi2(self) -> None:
        """SCENARIO: Both windows have the same error rate.
        WHY: Happy path -- model accuracy is stable across windows. Chi-squared
             test on identical error rates should yield a high p-value.
        EXPECTED: pass=True, drift_detected == False.
        """
        rng = np.random.default_rng(42)
        n = 500
        y_true = rng.integers(0, 2, size=n)
        # Introduce ~20% error rate in both windows
        y_pred_ref = y_true.copy()
        flip_ref = rng.choice(n, size=100, replace=False)
        y_pred_ref[flip_ref] = 1 - y_pred_ref[flip_ref]

        y_pred_cur = y_true.copy()
        flip_cur = rng.choice(n, size=100, replace=False)
        y_pred_cur[flip_cur] = 1 - y_pred_cur[flip_cur]

        result = assert_no_concept_drift(
            y_true, y_pred_ref, y_true, y_pred_cur, method="chi2",
        )
        assert result.passed is True
        assert result.details["drift_detected"] is False
        assert result.details["method"] == "chi2"

    def test_same_error_rate_proportion(self) -> None:
        """SCENARIO: Both windows have the same error rate, tested with proportion method.
        WHY: Validates the Z-test for proportions with stable error rates.
        EXPECTED: pass=True, drift_detected == False.
        """
        rng = np.random.default_rng(43)
        n = 500
        y_true = rng.integers(0, 2, size=n)
        y_pred_ref = y_true.copy()
        flip_ref = rng.choice(n, size=100, replace=False)
        y_pred_ref[flip_ref] = 1 - y_pred_ref[flip_ref]

        y_pred_cur = y_true.copy()
        flip_cur = rng.choice(n, size=100, replace=False)
        y_pred_cur[flip_cur] = 1 - y_pred_cur[flip_cur]

        result = assert_no_concept_drift(
            y_true, y_pred_ref, y_true, y_pred_cur, method="proportion",
        )
        assert result.passed is True
        assert result.details["drift_detected"] is False
        assert result.details["method"] == "proportion"

    # ── Different error rate (should fail) ───────────────────────────────

    def test_different_error_rate_chi2(self) -> None:
        """SCENARIO: Reference window has 5% errors, current window has 50% errors.
        WHY: A dramatic increase in error rate signals concept drift. The chi2
             test should detect this significant difference.
        EXPECTED: MltkAssertionError raised, drift_detected == True.
        """
        rng = np.random.default_rng(44)
        n = 500
        y_true = rng.integers(0, 2, size=n)

        # Reference: ~5% error rate
        y_pred_ref = y_true.copy()
        flip_ref = rng.choice(n, size=25, replace=False)
        y_pred_ref[flip_ref] = 1 - y_pred_ref[flip_ref]

        # Current: ~50% error rate
        y_pred_cur = y_true.copy()
        flip_cur = rng.choice(n, size=250, replace=False)
        y_pred_cur[flip_cur] = 1 - y_pred_cur[flip_cur]

        with pytest.raises(MltkAssertionError) as exc:
            assert_no_concept_drift(
                y_true, y_pred_ref, y_true, y_pred_cur, method="chi2",
            )
        result = exc.value.result
        assert result.details["drift_detected"] is True
        assert result.details["p_value"] < 0.05

    def test_different_error_rate_proportion(self) -> None:
        """SCENARIO: Clear error rate difference tested with proportion method.
        WHY: The Z-test for proportions should also detect a large shift in
             error rates between windows.
        EXPECTED: MltkAssertionError raised, drift_detected == True.
        """
        rng = np.random.default_rng(45)
        n = 500
        y_true = rng.integers(0, 2, size=n)

        # Reference: ~5% errors
        y_pred_ref = y_true.copy()
        flip_ref = rng.choice(n, size=25, replace=False)
        y_pred_ref[flip_ref] = 1 - y_pred_ref[flip_ref]

        # Current: ~50% errors
        y_pred_cur = y_true.copy()
        flip_cur = rng.choice(n, size=250, replace=False)
        y_pred_cur[flip_cur] = 1 - y_pred_cur[flip_cur]

        with pytest.raises(MltkAssertionError) as exc:
            assert_no_concept_drift(
                y_true, y_pred_ref, y_true, y_pred_cur, method="proportion",
            )
        result = exc.value.result
        assert result.details["drift_detected"] is True
        assert result.details["method"] == "proportion"

    # ── Fisher method ────────────────────────────────────────────────────

    def test_fisher_method_no_drift(self) -> None:
        """SCENARIO: Fisher's exact test with same error rate.
        WHY: Fisher is preferred for small sample sizes. Validates that the
             method works and correctly identifies stable error rates.
        EXPECTED: pass=True, method is 'fisher' (or 'chi2' if scipy unavailable).
        """
        rng = np.random.default_rng(46)
        n = 50
        y_true = rng.integers(0, 2, size=n)
        y_pred_ref = y_true.copy()
        flip_ref = rng.choice(n, size=10, replace=False)
        y_pred_ref[flip_ref] = 1 - y_pred_ref[flip_ref]

        y_pred_cur = y_true.copy()
        flip_cur = rng.choice(n, size=10, replace=False)
        y_pred_cur[flip_cur] = 1 - y_pred_cur[flip_cur]

        result = assert_no_concept_drift(
            y_true, y_pred_ref, y_true, y_pred_cur, method="fisher",
        )
        assert result.passed is True
        assert result.details["method"] in ("fisher", "chi2")

    def test_fisher_method_with_drift(self) -> None:
        """SCENARIO: Fisher's exact test detects concept drift.
        WHY: Even with small samples, Fisher should detect a dramatic
             accuracy difference (0% vs 80% error rate).
        EXPECTED: MltkAssertionError raised, drift_detected == True.
        """
        # Reference: perfect predictions
        y_true_ref = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 3
        y_pred_ref = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 3

        # Current: mostly wrong
        y_true_cur = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 3
        y_pred_cur = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 3

        with pytest.raises(MltkAssertionError) as exc:
            assert_no_concept_drift(
                y_true_ref, y_pred_ref, y_true_cur, y_pred_cur, method="fisher",
            )
        result = exc.value.result
        assert result.details["drift_detected"] is True

    # ── Edge cases ───────────────────────────────────────────────────────

    def test_perfect_predictions_both_windows(self) -> None:
        """SCENARIO: Both windows have 100% accuracy.
        WHY: When both windows are perfect (0% error rate), there is nothing
             to compare -- the test should pass with p=1.0.
        EXPECTED: pass=True, error rates are both 0.0.
        """
        y_true = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        y_pred = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

        result = assert_no_concept_drift(y_true, y_pred, y_true, y_pred)
        assert result.passed is True
        assert result.details["error_rate_ref"] == 0.0
        assert result.details["error_rate_cur"] == 0.0
        assert result.details["p_value"] == 1.0

    def test_all_wrong_both_windows(self) -> None:
        """SCENARIO: Both windows have 0% accuracy (all predictions wrong).
        WHY: When both windows are equally terrible, there is no concept drift
             -- the model is consistently wrong. Error rates are identical.
        EXPECTED: pass=True, error rates are both 1.0.
        """
        y_true = [0, 1, 0, 1, 0, 1, 0, 1]
        y_pred = [1, 0, 1, 0, 1, 0, 1, 0]

        result = assert_no_concept_drift(y_true, y_pred, y_true, y_pred)
        assert result.passed is True
        assert result.details["error_rate_ref"] == 1.0
        assert result.details["error_rate_cur"] == 1.0
        assert result.details["p_value"] == 1.0

    def test_very_small_sample_size(self) -> None:
        """SCENARIO: Only 3 samples per window.
        WHY: With very small samples, even large error rate differences may not
             be statistically significant. The test should handle gracefully
             without crashing.
        EXPECTED: Result returned (pass or fail based on math), no crash.
        """
        y_true = [0, 1, 0]
        y_pred_ref = [0, 1, 0]  # 0% error
        y_pred_cur = [1, 0, 1]  # 100% error

        # With only 3 samples, even 0% vs 100% might not be significant
        # at alpha=0.05. The key test is that it doesn't crash.
        try:
            result = assert_no_concept_drift(
                y_true, y_pred_ref, y_true, y_pred_cur, method="proportion",
            )
            assert isinstance(result.passed, bool)
        except MltkAssertionError:
            pass  # Drift detected is acceptable for small samples

        # Key: didn't crash

    # ── Details verification ─────────────────────────────────────────────

    def test_details_contain_all_fields(self) -> None:
        """SCENARIO: Caller inspects TestResult details for audit logging.
        WHY: Downstream dashboards, alerts, and audit systems need specific
             fields (error rates, p-value, sample sizes, method, statistic)
             to display monitoring context without re-running the test.
        EXPECTED: All documented detail fields are present and correctly typed.
        """
        rng = np.random.default_rng(47)
        n = 200
        y_true = rng.integers(0, 2, size=n)
        y_pred = y_true.copy()
        flip = rng.choice(n, size=40, replace=False)
        y_pred[flip] = 1 - y_pred[flip]

        result = assert_no_concept_drift(
            y_true, y_pred, y_true, y_pred, method="chi2", alpha=0.05,
        )

        # Verify all documented fields exist
        assert "error_rate_ref" in result.details
        assert "error_rate_cur" in result.details
        assert "error_rate_diff" in result.details
        assert "p_value" in result.details
        assert "alpha" in result.details
        assert "method" in result.details
        assert "n_ref" in result.details
        assert "n_cur" in result.details
        assert "statistic" in result.details
        assert "drift_detected" in result.details

        # Verify types
        assert isinstance(result.details["error_rate_ref"], float)
        assert isinstance(result.details["error_rate_cur"], float)
        assert isinstance(result.details["p_value"], float)
        assert isinstance(result.details["n_ref"], int)
        assert isinstance(result.details["n_cur"], int)
        assert isinstance(result.details["statistic"], float)
        assert isinstance(result.details["drift_detected"], bool)

        # Verify values make sense
        assert 0.0 <= result.details["error_rate_ref"] <= 1.0
        assert 0.0 <= result.details["error_rate_cur"] <= 1.0
        assert 0.0 <= result.details["p_value"] <= 1.0
        assert result.details["n_ref"] == n
        assert result.details["n_cur"] == n
        assert result.details["alpha"] == 0.05

    def test_empty_arrays(self) -> None:
        """SCENARIO: Empty arrays passed (no data in window).
        WHY: A monitoring window may have no samples. The assertion must not
             crash -- it should pass gracefully.
        EXPECTED: pass=True, drift_detected == False.
        """
        result = assert_no_concept_drift([], [], [], [])
        assert result.passed is True
        assert result.details["drift_detected"] is False

    def test_unknown_method(self) -> None:
        """SCENARIO: Caller passes an invalid method name.
        WHY: Defensive check -- typos happen. Should fail with a clear message.
        EXPECTED: MltkAssertionError raised with helpful message.
        """
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_concept_drift([0], [0], [0], [0], method="invalid")
        assert "Unknown method" in exc.value.result.message

    def test_list_inputs(self) -> None:
        """SCENARIO: All inputs are plain Python lists (not numpy arrays).
        WHY: Many callers will pass plain lists from database queries or API
             responses. The function must accept and convert them correctly.
        EXPECTED: pass=True, no type errors.
        """
        y_true_ref = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        y_pred_ref = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        y_true_cur = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        y_pred_cur = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

        result = assert_no_concept_drift(
            y_true_ref, y_pred_ref, y_true_cur, y_pred_cur,
        )
        assert result.passed is True
        assert result.name == "monitor.concept_drift"

    def test_custom_alpha(self) -> None:
        """SCENARIO: Moderate error rate difference passes at default alpha but fails
        at a stricter (higher) alpha.
        WHY: alpha is the significance threshold -- p >= alpha means 'no drift'.
             A higher alpha is stricter: it requires a higher p-value to pass.
             With p~0.13, alpha=0.05 passes (0.13 >= 0.05) but alpha=0.20 fails
             (0.13 < 0.20).
        EXPECTED: pass=True at alpha=0.05, fail at alpha=0.20.
        """
        rng = np.random.default_rng(48)
        n = 200
        y_true = rng.integers(0, 2, size=n)

        # Reference: ~10% error
        y_pred_ref = y_true.copy()
        flip_ref = rng.choice(n, size=20, replace=False)
        y_pred_ref[flip_ref] = 1 - y_pred_ref[flip_ref]

        # Current: ~15% error (slight difference, p~0.13)
        y_pred_cur = y_true.copy()
        flip_cur = rng.choice(n, size=30, replace=False)
        y_pred_cur[flip_cur] = 1 - y_pred_cur[flip_cur]

        # Passes at default alpha=0.05 (p=0.13 >= 0.05)
        result = assert_no_concept_drift(
            y_true, y_pred_ref, y_true, y_pred_cur, alpha=0.05,
        )
        assert result.passed is True
        assert result.details["alpha"] == 0.05

        # Fails at stricter alpha=0.20 (p=0.13 < 0.20)
        with pytest.raises(MltkAssertionError) as exc:
            assert_no_concept_drift(
                y_true, y_pred_ref, y_true, y_pred_cur, alpha=0.20,
            )
        assert exc.value.result.details["alpha"] == 0.20
