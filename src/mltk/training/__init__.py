"""Training bug detection — gradient health, numerical stability, and data leakage."""

from mltk.training.gradient import (
    assert_gradient_flow,
    assert_loss_finite,
    assert_no_exploding_gradient,
    assert_no_vanishing_gradient,
)
from mltk.training.leakage import (
    assert_no_target_leakage,
    assert_no_train_test_overlap,
    assert_temporal_split,
)
from mltk.training.numerical import (
    assert_loss_decreasing,
    assert_no_loss_divergence,
    assert_no_nan_inf,
    assert_softmax_valid,
)

__all__ = [
    # gradient.py
    "assert_gradient_flow",
    "assert_no_vanishing_gradient",
    "assert_no_exploding_gradient",
    "assert_loss_finite",
    # numerical.py
    "assert_no_nan_inf",
    "assert_loss_decreasing",
    "assert_no_loss_divergence",
    "assert_softmax_valid",
    # leakage.py
    "assert_no_train_test_overlap",
    "assert_temporal_split",
    "assert_no_target_leakage",
]
