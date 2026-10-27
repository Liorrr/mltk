# mltk.training

## Overview
Training-bug assertions: gradient health, numerical stability, data leakage, checkpoint/resume, distributed correctness, augmentation, memory leaks and training-serving skew.

## Interface
- gradient.py: `assert_gradient_flow`, `assert_no_vanishing_gradient`, `assert_no_exploding_gradient`, `assert_loss_finite`
- numerical.py: `assert_no_nan_inf`, `assert_loss_decreasing`, `assert_no_loss_divergence`, `assert_softmax_valid`
- leakage.py: `assert_no_train_test_overlap`, `assert_temporal_split`, `assert_no_target_leakage`
- checkpoint.py: `assert_checkpoint_complete`, `assert_resume_loss_continuous`
- distributed.py: `assert_effective_batch_size`, `assert_gradient_sync`, `assert_n_rank_gradient_sync`, `assert_gradient_alignment`, `assert_weight_divergence`, `assert_gradient_clipped`
- augmentation.py: `assert_no_augmentation_on_test`, `assert_augmentation_preserves_signal`
- memory.py: `assert_no_memory_leak`, `assert_loss_is_detached`
- skew.py: `assert_no_training_serving_skew`

## Invariants
- Assertions return `TestResult`; a failed check at `Severity.CRITICAL` (the default) raises `MltkAssertionError` via `mltk.core.assert_true`. They are plain functions usable inside pytest.

## Side Effects
- checkpoint.py reads checkpoint files; `assert_no_memory_leak` only analyses caller-supplied `memory_readings_mb` and samples nothing itself.

## Verification
- Run: `python -m pytest tests/test_training -q`
