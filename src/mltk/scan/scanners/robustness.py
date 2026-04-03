"""Robustness scanner -- detect prediction instability.

A model that flips predictions when inputs change by 0.01 is
dangerous in production: sensor noise, rounding errors, or
minor data pipeline changes will cause random failures.
RobustnessScanner adds small gaussian noise to numeric features
and checks how many predictions change.

How it works:
  1. Select only numeric columns from the dataset.
  2. Call ``assert_robust`` with gaussian noise at the
     configured epsilon (default 0.01).
  3. Each failure becomes a ScanFinding with the exact
     assertion call for reproduction.

Only runs on numeric columns -- categorical features are
skipped since adding gaussian noise to them is meaningless.
"""

from __future__ import annotations

import numpy as np

from mltk.core.assertion import MltkAssertionError
from mltk.core.result import Severity, TestResult
from mltk.model.adversarial import assert_robust
from mltk.scan.config import ScanContext
from mltk.scan.finding import FixSuggestion, ScanFinding
from mltk.scan.scanners.base import Scanner

__all__ = ["RobustnessScanner"]


class RobustnessScanner(Scanner):
    """Detect prediction instability under small noise.

    Adds gaussian noise to numeric features and checks
    whether predictions remain stable. Categorical columns
    are excluded since noise on them is meaningless.

    Requires:
        model_fn, X, y
    """

    name = "robustness"
    category = "model_quality"
    requires: set[str] = {"model_fn", "X", "y"}

    _DEFAULT_EPSILON: float = 0.01
    _DEFAULT_STABILITY: float = 0.95

    # ----------------------------------------------------------
    # public API
    # ----------------------------------------------------------

    def scan(
        self,
        ctx: ScanContext,
    ) -> list[ScanFinding]:
        """Run robustness check and return findings.

        Args:
            ctx: Scan context with model_fn, X, y.

        Returns:
            List of ScanFinding if model is fragile.
        """
        findings: list[ScanFinding] = []
        epsilon = self._get_epsilon(ctx)
        stability = self._get_stability(ctx)

        # Only use numeric columns for noise injection.
        numeric_cols = [
            c for c in ctx.numeric_columns
            if c in ctx.X.columns
        ]
        if not numeric_cols:
            return findings

        X_numeric = ctx.X[numeric_cols].values.astype(
            np.float64,
        )

        try:
            assert_robust(
                ctx.model_fn,
                X_numeric,
                perturbation="gaussian",
                epsilon=epsilon,
                stability=stability,
            )
        except MltkAssertionError as exc:
            result = TestResult(
                name="scan.robustness",
                passed=False,
                severity=Severity.WARNING,
                message=(
                    f"Model is fragile under noise: "
                    f"{exc.result.message}"
                ),
                details={
                    "epsilon": epsilon,
                    "stability_threshold": stability,
                    "numeric_columns": numeric_cols,
                    **exc.result.details,
                },
            )
            findings.append(ScanFinding(
                result=result,
                assertion_fn=assert_robust,
                assertion_args=(
                    ctx.model_fn,
                    X_numeric.copy(),
                ),
                assertion_kwargs={
                    "perturbation": "gaussian",
                    "epsilon": epsilon,
                    "stability": stability,
                },
                suggested_test=self._gen_test(
                    epsilon, stability,
                ),
                scanner_name=self.name,
                suggested_fixes=self._gen_fix(),
            ))
        return findings

    # ----------------------------------------------------------
    # helpers
    # ----------------------------------------------------------

    def _get_epsilon(
        self, ctx: ScanContext,
    ) -> float:
        """Read epsilon from scanner config or default."""
        scanner_cfg = ctx.config.scanner_config.get(
            self.name, {},
        )
        return float(
            scanner_cfg.get(
                "epsilon",
                self._DEFAULT_EPSILON,
            ),
        )

    def _get_stability(
        self, ctx: ScanContext,
    ) -> float:
        """Read stability from scanner config or default."""
        scanner_cfg = ctx.config.scanner_config.get(
            self.name, {},
        )
        return float(
            scanner_cfg.get(
                "stability",
                self._DEFAULT_STABILITY,
            ),
        )

    @staticmethod
    def _gen_fix() -> list[FixSuggestion]:
        """Generate fix suggestions for robustness failure.

        Returns:
            2 FixSuggestions ranked by confidence.
        """
        return [
            FixSuggestion(
                category="code",
                title=(
                    "Add noise augmentation to "
                    "training"
                ),
                description=(
                    "Model predictions change "
                    "significantly under small "
                    "perturbations. Add Gaussian "
                    "noise augmentation during "
                    "training to improve "
                    "robustness."
                ),
                confidence="high",
                code_snippet=(
                    "from sklearn.utils import "
                    "check_random_state\n"
                    "noise = check_random_state(42)"
                    ".normal(0, 0.01, "
                    "X_train.shape)\n"
                    "X_aug = np.vstack("
                    "[X_train, X_train + noise])"
                ),
            ),
            FixSuggestion(
                category="config",
                title=(
                    "Increase model capacity or "
                    "ensemble"
                ),
                description=(
                    "A more expressive model or "
                    "ensemble may be more robust "
                    "to input perturbations."
                ),
                confidence="medium",
            ),
        ]

    @staticmethod
    def _gen_test(
        epsilon: float, stability: float,
    ) -> str:
        """Generate a pytest snippet for robustness."""
        return (
            "def test_robustness():\n"
            "    \"\"\"Model must be stable under"
            " small noise.\"\"\"\n"
            "    assert_robust(\n"
            "        model_fn,\n"
            "        X_test.select_dtypes("
            "'number').values,\n"
            f"        epsilon={epsilon},\n"
            f"        stability={stability},\n"
            "    )\n"
        )
