# mltk.model

## Overview
Model-quality assertions: metrics, regression vs baseline, slicing, calibration, bias/fairness, adversarial robustness, A/B tests, conformal prediction, attribution stability, causal inference.

## Interface
- `assert_metric` (metrics.py)
- `save_baseline`, `assert_no_regression` (regression.py)
- `assert_slice_performance`, `assert_calibration` (slicing.py)
- `assert_no_bias`, `assert_intersectional_fairness` (bias.py)
- `assert_counterfactual_fairness` (counterfactual.py)
- `assert_robust` (adversarial.py)
- `assert_ab_significance` (ab_test.py)
- `assert_interval_coverage`, `assert_prediction_set_size`, `assert_conformal_calibration`, `assert_conditional_coverage` (conformal.py)
- `assert_top_k_stable`, `assert_attribution_cosine_stability` (attribution.py)
- `assert_ate_significant`, `assert_no_confounding` (causal.py)
- `assert_no_overfitting`, `assert_label_drift` (overfitting.py)

## Invariants
- Assertions return `TestResult`; a failed check at `Severity.CRITICAL` (the default) raises `MltkAssertionError` via `mltk.core.assert_true`. They are plain functions usable inside pytest.
- Assertions do not mutate their inputs.

## Side Effects
- `save_baseline` writes a baseline file to disk; `assert_no_regression` reads it.

## Verification
- Run: `python -m pytest tests/test_model -q`
