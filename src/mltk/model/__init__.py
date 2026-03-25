"""Model quality testing -- metrics, regression, slicing, bias, adversarial."""

from mltk.model.adversarial import assert_robust
from mltk.model.bias import assert_no_bias
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
]
