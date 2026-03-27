"""Model quality — metrics, regression, slicing, bias, adversarial, A/B, conformal, attribution."""

from mltk.model.ab_test import assert_ab_significance
from mltk.model.adversarial import assert_robust
from mltk.model.attribution import assert_attribution_cosine_stability, assert_top_k_stable
from mltk.model.bias import assert_no_bias
from mltk.model.causal import assert_ate_significant, assert_no_confounding
from mltk.model.conformal import (
    assert_conditional_coverage,
    assert_conformal_calibration,
    assert_interval_coverage,
    assert_prediction_set_size,
)
from mltk.model.counterfactual import assert_counterfactual_fairness
from mltk.model.metrics import assert_metric
from mltk.model.overfitting import assert_label_drift, assert_no_overfitting
from mltk.model.regression import assert_no_regression, save_baseline
from mltk.model.slicing import assert_calibration, assert_slice_performance

__all__ = [
    "assert_metric",
    "assert_no_regression",
    "save_baseline",
    "assert_slice_performance",
    "assert_calibration",
    "assert_no_bias",
    "assert_robust",
    "assert_no_overfitting",
    "assert_label_drift",
    "assert_ab_significance",
    "assert_interval_coverage",
    "assert_prediction_set_size",
    "assert_conformal_calibration",
    "assert_conditional_coverage",
    "assert_top_k_stable",
    "assert_attribution_cosine_stability",
    # counterfactual + causal
    "assert_counterfactual_fairness",
    "assert_ate_significant",
    "assert_no_confounding",
]
