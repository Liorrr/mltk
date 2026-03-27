"""Training bug detection — gradient health, numerical stability, data leakage, and serving skew."""

from mltk.training.augmentation import (
    assert_augmentation_preserves_signal,
    assert_no_augmentation_on_test,
)
from mltk.training.checkpoint import (
    assert_checkpoint_complete,
    assert_resume_loss_continuous,
)
from mltk.training.distributed import (
    assert_effective_batch_size,
    assert_gradient_alignment,
    assert_gradient_clipped,
    assert_gradient_sync,
    assert_n_rank_gradient_sync,
    assert_weight_divergence,
)
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
from mltk.training.memory import (
    assert_loss_is_detached,
    assert_no_memory_leak,
)
from mltk.training.numerical import (
    assert_loss_decreasing,
    assert_no_loss_divergence,
    assert_no_nan_inf,
    assert_softmax_valid,
)
from mltk.training.skew import assert_no_training_serving_skew

__all__ = [
    # augmentation.py
    "assert_no_augmentation_on_test",
    "assert_augmentation_preserves_signal",
    # checkpoint.py
    "assert_checkpoint_complete",
    "assert_resume_loss_continuous",
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
    # distributed.py
    "assert_effective_batch_size",
    "assert_gradient_alignment",
    "assert_gradient_clipped",
    "assert_gradient_sync",
    "assert_n_rank_gradient_sync",
    "assert_weight_divergence",
    # memory.py
    "assert_no_memory_leak",
    "assert_loss_is_detached",
    # skew.py
    "assert_no_training_serving_skew",
]
