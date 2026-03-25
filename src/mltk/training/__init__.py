"""Training bug detection — data leakage, feature leakage, split validation."""

from mltk.training.leakage import (
    assert_no_target_leakage,
    assert_no_train_test_overlap,
    assert_temporal_split,
)

__all__ = [
    "assert_no_train_test_overlap",
    "assert_temporal_split",
    "assert_no_target_leakage",
]
