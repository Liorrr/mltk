"""Tests for mltk.model.counterfactual -- counterfactual fairness testing.

Each test simulates a realistic ML scenario: fair models that ignore protected
attributes, biased models that rely on them, custom perturbation strategies,
threshold boundaries, and multi-class sensitive features.
"""

import numpy as np
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.model.counterfactual import assert_counterfactual_fairness


class TestBinaryNoFlips:
    """Binary sensitive feature with a model that ignores it entirely."""

    def test_fair_model_no_flips(self) -> None:
        """PASS: Model only uses non-sensitive features -- zero predictions flip.

        Scenario: Credit scoring model uses only income (column 1) to decide
        loan approval. Gender (column 0) is present but unused. Flipping
        gender changes nothing.
        """
        rng = np.random.default_rng(42)
        n = 200
        gender = rng.integers(0, 2, n)
        income = rng.uniform(20_000, 120_000, n)
        X = np.column_stack([gender, income])

        # Model only looks at income > 50K. Gender is irrelevant.
        def fair_model(X: np.ndarray) -> np.ndarray:
            return (X[:, 1] > 50_000).astype(int)

        result = assert_counterfactual_fairness(
            fair_model, X, sensitive_col=0, max_flip_rate=0.05
        )
        assert result.passed is True
        assert result.details["flip_rate"] == 0.0
        assert result.details["n_flipped"] == 0


class TestBinaryManyFlips:
    """Binary sensitive feature with a model that directly uses it."""

    def test_biased_model_many_flips(self) -> None:
        """FAIL: Model uses the sensitive feature -- many predictions flip.

        Scenario: Hiring model literally checks gender. Flipping gender
        changes the outcome for everyone. This is textbook discrimination.
        """
        rng = np.random.default_rng(42)
        n = 100
        gender = rng.integers(0, 2, n)
        experience = rng.uniform(0, 20, n)
        X = np.column_stack([gender, experience])

        # Model approves only if male (gender == 1) AND experienced.
        def biased_model(X: np.ndarray) -> np.ndarray:
            return ((X[:, 0] == 1) & (X[:, 1] > 5)).astype(int)

        with pytest.raises(MltkAssertionError) as exc:
            assert_counterfactual_fairness(
                biased_model, X, sensitive_col=0, max_flip_rate=0.05
            )
        assert "violated" in str(exc.value)


class TestCustomPerturbation:
    """Custom perturbation function -- user controls how the attribute changes."""

    def test_custom_perturbation_fn(self) -> None:
        """PASS: Custom perturbation adds Gaussian noise to age, model is robust.

        Scenario: Instead of flipping a category, the user wants to test if
        the model is robust to small changes in a continuous sensitive attribute
        (age). A custom perturbation adds +/- 5 years of noise. A well-
        calibrated model should not change predictions for small age shifts.
        """
        rng = np.random.default_rng(42)
        n = 100
        age = rng.uniform(25, 65, n)
        income = rng.uniform(30_000, 150_000, n)
        X = np.column_stack([age, income])

        # Model only uses income. Age is irrelevant.
        def income_model(X: np.ndarray) -> np.ndarray:
            return (X[:, 1] > 60_000).astype(int)

        # Custom perturbation: add small noise to age column.
        def age_perturbation(X: np.ndarray) -> np.ndarray:
            X_p = X.copy()
            noise = np.random.default_rng(123).normal(0, 5, len(X_p))
            X_p[:, 0] = X_p[:, 0] + noise
            return X_p

        result = assert_counterfactual_fairness(
            income_model, X, sensitive_col=0,
            perturbation_fn=age_perturbation, max_flip_rate=0.05,
        )
        assert result.passed is True
        assert result.details["flip_rate"] == 0.0


class TestThresholdBoundary:
    """Boundary behavior of the max_flip_rate threshold."""

    def test_exact_threshold(self) -> None:
        """PASS: flip_rate exactly equals max_flip_rate -- should pass (<=).

        Scenario: Exactly 1 out of 20 predictions flips = 5% flip rate.
        With max_flip_rate=0.05, this is on the boundary and should pass.
        """
        n = 20
        X = np.column_stack([
            np.array([0] * 10 + [1] * 10),  # Sensitive feature
            np.arange(n, dtype=float),        # Non-sensitive feature
        ])

        # Model: predicts 1 only for sample index 0 when col 0 is 0.
        # After flip, sample 0 gets col 0 = 1, and predicts differently.
        # Exactly 1/20 = 0.05 flips.
        def boundary_model(X: np.ndarray) -> np.ndarray:
            preds = (X[:, 1] > 9.5).astype(int)
            # Make exactly one prediction depend on sensitive feature.
            mask = (X[:, 1] == 0.0)
            preds[mask] = X[mask, 0].astype(int)
            return preds

        result = assert_counterfactual_fairness(
            boundary_model, X, sensitive_col=0, max_flip_rate=0.05
        )
        assert result.passed is True
        assert abs(result.details["flip_rate"] - 0.05) < 1e-9

    def test_just_over_threshold(self) -> None:
        """FAIL: flip_rate slightly exceeds max_flip_rate -- should fail.

        Scenario: 2 out of 20 predictions flip = 10% > 5% threshold.
        """
        n = 20
        X = np.column_stack([
            np.array([0] * 10 + [1] * 10),
            np.arange(n, dtype=float),
        ])

        # Two predictions depend on sensitive feature.
        def over_boundary_model(X: np.ndarray) -> np.ndarray:
            preds = (X[:, 1] > 9.5).astype(int)
            for idx in [0.0, 1.0]:
                mask = X[:, 1] == idx
                preds[mask] = X[mask, 0].astype(int)
            return preds

        with pytest.raises(MltkAssertionError):
            assert_counterfactual_fairness(
                over_boundary_model, X, sensitive_col=0, max_flip_rate=0.05
            )


class TestMultiClassSensitive:
    """Multi-class (>2 categories) sensitive features."""

    def test_three_category_fair_model(self) -> None:
        """PASS: Three-category sensitive feature, model ignores it.

        Scenario: Race encoded as 0/1/2. Model only uses test scores.
        The default perturbation cycles: 0->1, 1->2, 2->0.
        Since the model ignores race, no predictions change.
        """
        rng = np.random.default_rng(42)
        n = 150
        race = np.array([0] * 50 + [1] * 50 + [2] * 50)
        score = rng.uniform(0, 100, n)
        X = np.column_stack([race, score])

        def score_model(X: np.ndarray) -> np.ndarray:
            return (X[:, 1] > 50).astype(int)

        result = assert_counterfactual_fairness(
            score_model, X, sensitive_col=0, max_flip_rate=0.05
        )
        assert result.passed is True


class TestModelIgnoresSensitive:
    """Model architecturally cannot use the sensitive feature."""

    def test_model_drops_sensitive_column(self) -> None:
        """PASS: Model explicitly drops the sensitive column before predicting.

        Scenario: A diligent ML engineer removes the sensitive feature from
        the feature vector before feeding it to the model. Counterfactual
        perturbation has zero effect. This is the gold standard.
        """
        rng = np.random.default_rng(42)
        n = 100
        gender = rng.integers(0, 2, n)
        feature_a = rng.standard_normal(n)
        feature_b = rng.standard_normal(n)
        X = np.column_stack([gender, feature_a, feature_b])

        # Model uses only columns 1 and 2, never column 0.
        def clean_model(X: np.ndarray) -> np.ndarray:
            return (X[:, 1] + X[:, 2] > 0).astype(int)

        result = assert_counterfactual_fairness(
            clean_model, X, sensitive_col=0, max_flip_rate=0.05
        )
        assert result.passed is True
        assert result.details["n_flipped"] == 0


class TestModelUsesSensitiveDirect:
    """Model directly uses the sensitive feature as a decision boundary."""

    def test_direct_discrimination(self) -> None:
        """FAIL: Model prediction equals the sensitive feature value.

        Scenario: The model literally returns gender as its prediction.
        Flipping gender flips 100% of predictions. Maximum possible violation.
        """
        n = 50
        gender = np.array([0] * 25 + [1] * 25)
        filler = np.zeros(n)
        X = np.column_stack([gender, filler])

        # Model = identity on sensitive column. Worst-case scenario.
        def identity_model(X: np.ndarray) -> np.ndarray:
            return X[:, 0].astype(int)

        with pytest.raises(MltkAssertionError) as exc:
            assert_counterfactual_fairness(
                identity_model, X, sensitive_col=0, max_flip_rate=0.05
            )
        result = exc.value.result
        assert result.details["flip_rate"] == 1.0
        assert result.details["n_flipped"] == n


class TestReturnDetails:
    """Validate the structure and content of returned TestResult details."""

    def test_details_structure(self) -> None:
        """TestResult contains all expected detail fields with correct types.

        This test verifies the API contract: downstream consumers can rely
        on these fields being present and correctly typed.
        """
        rng = np.random.default_rng(42)
        n = 50
        X = np.column_stack([
            rng.integers(0, 2, n),
            rng.standard_normal(n),
        ])

        def model(X: np.ndarray) -> np.ndarray:
            return (X[:, 1] > 0).astype(int)

        result = assert_counterfactual_fairness(
            model, X, sensitive_col=0, max_flip_rate=0.10
        )

        # All required fields present.
        assert "flip_rate" in result.details
        assert "max_flip_rate" in result.details
        assert "n_flipped" in result.details
        assert "n_total" in result.details
        assert "sensitive_col" in result.details

        # Types.
        assert isinstance(result.details["flip_rate"], float)
        assert isinstance(result.details["max_flip_rate"], float)
        assert isinstance(result.details["n_flipped"], int)
        assert isinstance(result.details["n_total"], int)
        assert isinstance(result.details["sensitive_col"], int)

        # Values make sense.
        assert 0.0 <= result.details["flip_rate"] <= 1.0
        assert result.details["n_total"] == n
        assert result.details["n_flipped"] <= n
        assert result.details["sensitive_col"] == 0

        # Timing was recorded.
        assert result.duration_ms >= 0.0
        assert result.name == "model.counterfactual_fairness"
