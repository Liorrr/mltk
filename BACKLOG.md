# MLTK Backlog

Tracked items for the ML Test Kit project. Updated after each sprint.

## Status Legend
- **DONE** -- shipped and tested
- **IN PROGRESS** -- currently being built
- **PLANNED** -- scheduled for a specific sprint
- **BACKLOG** -- accepted, not yet scheduled
- **IDEA** -- needs evaluation

---

## DONE

### Sprint 0 -- Project Skeleton
- [x] Repo structure, pyproject.toml, Cargo.toml
- [x] Core types: MltkConfig, TestResult, TestSuite, Severity
- [x] Base assertion framework: assert_true, MltkAssertionError
- [x] Rust crate skeleton (PyO3 0.28 + Maturin)
- [x] pytest plugin with ML markers
- [x] CLI skeleton (Typer)
- [x] CI/CD workflows (lint, test, release)
- [x] 12 tests passing

### Sprint 1 -- Core + Data Quality + Docs
- [x] Config loading from pyproject.toml and mltk.yaml
- [x] assert_schema, assert_no_nulls, assert_dtypes
- [x] assert_range, assert_unique, assert_no_outliers (IQR)
- [x] assert_freshness, assert_row_count
- [x] MkDocs documentation site (8 pages, API reference)
- [x] 47 tests passing

---

## PLANNED

### Sprint 2 -- Data Drift + PII + Labels + Rust
- [x] assert_no_drift (KS, PSI, KL, chi-squared) -- 4 methods with default thresholds
- [x] assert_no_pii, scan_pii (11 regex patterns from ShrimPK)
- [x] assert_label_balance, assert_label_coverage
- [x] Rust acceleration: real KS test + PSI (not stubs)
- [x] 78 Python tests + 5 Rust tests

### Sprint 3 -- Model Quality
- [ ] assert_metric (accuracy, F1, AUC, precision, recall, MSE, R2)
- [ ] assert_no_regression (compare against saved baseline)
- [ ] assert_slice_performance (subgroup evaluation)
- [ ] save_baseline utility

### Sprint 4 -- Model Bias + pytest Plugin
- [ ] assert_no_bias (demographic parity, equalized odds, predictive parity)
- [ ] assert_robust (adversarial perturbation testing)
- [ ] pytest plugin: fixtures (ml_config, ml_baseline, ml_report)
- [ ] pytest plugin: HTML report hook via --mltk-report

### Sprint 5 -- Inference + CLI
- [ ] assert_latency (P50, P95, P99 percentiles)
- [ ] assert_throughput (requests per second)
- [ ] assert_api_contract (input/output JSON schema)
- [ ] CLI: mltk init, mltk scan, mltk drift, mltk version

### Sprint 6 -- Reports + Pipeline + ML Test Score
- [ ] HTML report generator (Plotly + Jinja2)
- [ ] assert_reproducible, assert_pipeline
- [ ] Google ML Test Score rubric (28-test scoring)
- [ ] CLI: mltk score, mltk report

### Sprint 7 -- Domain Kit: Computer Vision
- [ ] assert_map, assert_iou (object detection metrics)
- [ ] assert_frame_accuracy, assert_temporal_consistency (video)
- [ ] Example: mycompany_cv_test.py

### Sprint 8 -- Domain Kits: NLP + Speech
- [ ] NLP: assert_ner_f1, assert_bleu, assert_rouge, assert_no_prompt_injection
- [ ] Speech: assert_wer, assert_rtf, assert_accent_coverage

### Sprint 9 -- Monitoring + Tabular + Full Docs
- [ ] drift_monitor (continuous), assert_no_degradation, assert_sla
- [ ] Tabular: assert_feature_drift, assert_calibration, assert_class_balance
- [ ] Complete MkDocs documentation site

### Sprint 10 -- v0.1.0 Release
- [ ] PyPI publish via trusted publishing
- [ ] Cross-platform wheels (maturin-action)
- [ ] README badges, benchmarks, contributor guide

---

## BACKLOG (not yet scheduled)

### Core Enhancements
- [ ] YAML-driven test definitions (run tests from config, no code)
- [ ] Test registry with @register_test decorator for custom assertions
- [ ] Config from environment variables (MLTK_DRIFT_METHOD, etc.)

### Training Bug Detection (from Sprint 2 research -- 51 assertions)
- [ ] **P0 Data Leakage:** assert_no_train_test_overlap, assert_temporal_split, assert_group_split, assert_no_future_leakage, assert_preprocessing_after_split
- [ ] **P0 Feature Leakage:** assert_no_feature_target_leakage, assert_feature_available_at_inference, assert_no_id_features, assert_target_encoding_is_oof
- [ ] **P1 Gradient Pathologies:** assert_gradient_flow, assert_weight_update_ratio, assert_gradient_bounded, assert_loss_finite, assert_neuron_utilization, assert_gradient_snr
- [ ] **P1 Learning Rate:** assert_loss_decreasing, assert_no_loss_divergence, assert_lr_schedule_matches, assert_lr_bounded, assert_warmup_fraction
- [ ] **P1 Batch Normalization:** assert_eval_mode_set, assert_bn_statistics_valid, assert_bn_batch_size, assert_train_eval_consistency
- [ ] **P1 Numerical Stability:** assert_no_nan_inf, assert_loss_scale_effective, assert_numerical_stability, assert_softmax_valid, assert_variance_positive, assert_mixed_precision_config
- [ ] **P1 Data Augmentation:** assert_no_augmentation_on_test, assert_label_augmentation_consistency, assert_augmentation_preserves_signal, assert_augmentation_class_independent, assert_mixup_alpha_valid
- [ ] **P2 Checkpoint/Resume:** assert_checkpoint_complete, assert_resume_lr_matches, assert_resume_loss_continuous, assert_optimizer_state_loaded, assert_rng_state_restored
- [ ] **P2 Distributed Training:** assert_distributed_sampler_used, assert_effective_batch_size, assert_gradient_sync, assert_all_parameters_used, assert_reproducible_across_runs
- [ ] **P2 Memory Leaks:** assert_no_memory_leak, assert_loss_is_detached, assert_grad_accumulation_correct, assert_gpu_utilization_stable, assert_scaler_scale_bounded

### Testing Patterns (from research)
- [ ] Statistical assertion primitives (tolerance bands, semantic similarity)
- [ ] LLM-as-judge evaluation patterns for generative AI
- [ ] Smart test selection (only re-run tests affected by specific changes)
- [ ] Golden test set management (versioned baselines)
- [ ] Non-deterministic test retry with confidence intervals

### Regulatory / Compliance
- [ ] EU AI Act compliance report template
- [ ] FDA AI device audit trail export
- [ ] Bias/fairness report with demographic breakdowns

### Integrations
- [ ] GitHub Actions action (`uses: liorrr/mltk-action@v1`)
- [ ] VS Code extension (inline test results)
- [ ] Jupyter notebook integration (rich output for assertions)
- [ ] MLflow integration (log mltk results as MLflow metrics)

### Performance
- [ ] Rust: KL divergence, chi-squared, Wasserstein distance
- [ ] Rust: fast PII scanning with regex
- [ ] Rust: SIMD-accelerated distribution comparisons
- [ ] Benchmarks vs Great Expectations, Deepchecks, Evidently

### Monetization (Pro tier -- future)
- [ ] Cloud dashboard: hosted report aggregation
- [ ] CI/CD GitHub App: auto-run mltk on PRs
- [ ] Compliance PDF exports (EU AI Act, FDA)
- [ ] Custom domain kits (healthcare, fintech, autonomous)

---

## IDEAS (needs evaluation)

- [ ] `mltk doctor` -- diagnose common ML pipeline issues automatically
- [ ] Visual diff for model predictions (before/after comparison)
- [ ] Slack/Teams notifications for drift alerts
- [ ] Data contract specification format (like OpenAPI but for datasets)
- [ ] Plugin system for third-party assertion libraries

---

## Research Findings (Sprint 1)

### Competitor Intelligence (March 2026)

| Competitor | Stars | Status | mltk Opportunity |
|---|---|---|---|
| **Cleanlab** | 11.4K | **ACQUIRED by Handshake (Jan 28)** — OSS in maintenance mode, no releases since | Capture displaced users. Add label quality checks. |
| **Giskard** | 5.2K | v2 deprecated, v3 unreleased — users in limbo | Stable alternative for bias/fairness testing |
| **Deepchecks** | 4K | 1.7/4 review score, pivoting to LLM-only | Beat on DX and traditional ML coverage |
| **Evidently** | 7.3K | $500/mo entry, limited OSS support | Free, full-featured alternative |
| **Great Expectations** | 11.3K | Data-only (no model testing), v1.15.1 | Complementary, not competitive |
| **DeepEval** | 14.3K | Fastest-growing, LLM-only ("pytest for LLMs") | Own "pytest for ALL ML" (not just LLMs) |
| **MLflow 3.x** | Massive | Adding evaluation features, absorbing Giskard as plugin | Stay independent, integrate as output target |
| **Inspect AI** | New | UK govt-backed, adopted by Anthropic/DeepMind | Safety niche, not general ML testing |

### Strategic Priorities (from research)
1. **Capture Cleanlab's orphaned users** — add label quality checks (Sprint 2)
2. **Fill the Giskard v2→v3 gap** — stable bias/fairness testing (Sprint 4)
3. **Don't pivot to LLM-only** — everyone else is. Own full-spectrum ML testing.
4. **Beat Evidently on price** — fully free OSS, no $500/mo gate
5. **Beat Deepchecks on DX** — 1.7/4 review score = low bar to clear

### QA Pain Points (ranked by frequency)
1. Non-deterministic outputs break traditional assertions
2. Silent drift in production with no alerts
3. No unified framework across ML lifecycle (OUR VALUE PROP)
4. GPU costs block CI/CD testing
5. Regulatory audit trails missing from testing tools

### Key Stats
- 87% of ML models never reach production
- 46% of devs distrust AI accuracy (up from 31% in 2024)
- 89% piloting GenAI in QA, only 15% have enterprise implementations
- Cleanlab acquired = 11.4K star user base looking for alternatives
- Every major competitor pivoting to LLM-only = traditional ML testing gap widening

---

## Workflow Rules

1. **Docs first**: Write documentation before building. Docs define the contract.
2. **Build to match docs**: Implementation must match documented API signatures and behavior.
3. **Verify alignment**: After building, compare docs vs implementation. If they differ, convene team (ml-test-engineer + qa-lead + ml-engineer) to reach consensus on the better design.
4. **Research every sprint**: Dispatch 2-3 background researchers per sprint (competitor watch, gap analysis, user pain points).
5. **Backlog in repo**: This file (BACKLOG.md) is the single source of truth. Update after every sprint.
6. **All agents get write permission**: Every subagent dispatched with write access from the start.

---

## Research Findings (Sprint 2)

### Drift Detection Methods — State of the Art
**Add to Sprint 2 (quick wins):**
- **Jensen-Shannon divergence** — symmetric, bounded [0,1], handles zero bins. Evidently's default for categorical on large datasets. Implementation: `0.5 * KL(p||m) + 0.5 * KL(q||m)`. Threshold: 0.1.
- **Wasserstein distance** — Evidently's default for numerical when n>1000. Proportional to mean shift. `scipy.stats.wasserstein_distance`. Threshold: 0.1.

**Defer to Sprint 9 (Monitoring):**
- MMD (multivariate), domain classifier (embedding drift), ADWIN/Page-Hinkley (streaming)

**Key insight:** For n>1000, KS test fires too aggressively (catches noise). Evidently switches to Wasserstein for numerical, JS for categorical. Consider `method="auto"` that selects based on sample size + dtype.

### PII Detection Gaps — Patterns to Add
**Tier 1 — High frequency, add next (10 patterns):**
1. IPv4/IPv6 addresses (most frequently leaked in training data)
2. URL with auth tokens (`?token=...`)
3. JWT tokens (`eyJ...`)
4. PEM private keys (`-----BEGIN RSA PRIVATE KEY-----`)
5. Database connection strings (`postgres://user:pass@host`)
6. Stripe live/test keys (`sk_live_...`)
7. Bearer tokens (`Bearer ...`)
8. Google API keys (`AIza...`)
9. IBAN (with MOD-97 checksum validation)

**Tier 2 — Regional / GDPR (add Sprint 3-4):**
- Israel Teudat Zehut (9 digits + Luhn + keyword anchor)
- UK NHS, UK NINO, Germany Steuer-ID, France NIR, Italy Codice Fiscale, Spain DNI
- India Aadhaar (Verhoeff checksum), India PAN
- International phone numbers (`+country code`)

**False positive mitigation:**
- Checksum validation (Luhn, MOD-97, MOD-11) eliminates ~100% FP on structured IDs
- Keyword anchoring for ambiguous patterns (DOB, passport, driver's license)
- Confidence scoring (0.0-1.0) per match
- Allowlists for known-safe patterns (test card numbers, example.com)

---

## Research Findings (Sprint 2 -- Training Bug Catalog)

### Comprehensive Catalog of ML Training Cycle Bugs

Research mission: catalog every known class of training-specific bug, with detection
strategies and assertions that mltk could provide. Organized into 10 categories.

---

#### 1. Data Leakage (train/test contamination)

**What goes wrong:**
Training data contaminates the test or validation set, producing artificially inflated
metrics that do not generalize. The model appears to perform well but fails in production.

**Common causes:**
- **Splitting after preprocessing.** Fitting a scaler, encoder, or imputer on the full
  dataset before splitting leaks test statistics into training.
- **Duplicate/near-duplicate rows** spanning train and test sets. Especially common with
  scraped data, augmented copies, or time-series windows that overlap.
- **Temporal leakage.** Random splitting of time-series data means the model trains on
  future data to predict the past. Must use chronological splits.
- **Group leakage.** Multiple rows from the same entity (patient, user, device) appear
  in both splits. The model memorizes entity-specific patterns.
- **Target leakage via proxy.** A column that is a downstream consequence of the target
  (e.g., "treatment outcome" used to predict "diagnosis") leaks the answer.

**How to detect:**
- Suspiciously high validation accuracy (>99%) on a non-trivial task.
- Model performance drops drastically when deployed on genuinely unseen data.
- Feature importance shows a feature that should not logically predict the target.
- Train and test distributions are identical (zero drift) on features derived from labels.

**Assertions for mltk:**
```python
assert_no_train_test_overlap(train_df, test_df, key_cols)
# Verify zero row overlap on key columns (exact + fuzzy near-duplicate)

assert_temporal_split(train_df, test_df, time_col)
# Verify max(train[time_col]) < min(test[time_col])

assert_group_split(train_df, test_df, group_col)
# Verify set(train[group_col]) & set(test[group_col]) == empty

assert_no_future_leakage(df, time_col, feature_cols)
# Verify no feature is derived from future timestamps relative to each row

assert_preprocessing_after_split(pipeline)
# Static analysis: verify fit() is called only on train partition
```

---

#### 2. Feature Leakage (target encoding in features)

**What goes wrong:**
A feature inadvertently encodes the target variable, making the model trivially
accurate during training but useless in production where that feature is unavailable
or differently distributed.

**Common causes:**
- **Direct target proxy.** A column that is a deterministic function of the target
  (e.g., "loan_approved" used to predict "default_risk" -- approval implies low risk).
- **Leaky aggregates.** Computing group-level statistics (mean target per category)
  and joining them back without proper out-of-fold encoding.
- **ID columns as features.** Sequential IDs that correlate with label ordering.
- **Timestamp features.** When the target is time-dependent and timestamp is included
  raw, the model learns temporal shortcuts instead of causal patterns.
- **Label-dependent preprocessing.** Feature selection using mutual information with the
  target computed on the full dataset, then applied to train/test splits.

**How to detect:**
- Single feature achieves near-perfect accuracy alone.
- Feature correlation with target is >0.95.
- Removing one feature causes a disproportionate performance collapse.
- Feature is unavailable at inference time (not in production schema).

**Assertions for mltk:**
```python
assert_no_feature_target_leakage(df, features, target, threshold=0.95)
# Flag any feature with Pearson/Spearman/mutual info > threshold with target

assert_feature_available_at_inference(train_schema, inference_schema)
# Verify all training features exist in the production input schema

assert_no_id_features(df, feature_cols)
# Detect monotonic integer columns or UUID columns used as features

assert_target_encoding_is_oof(encoder, cv_folds)
# Verify target-encoded features used out-of-fold encoding
```

---

#### 3. Gradient Pathologies

**What goes wrong:**
Gradients become too small (vanishing), too large (exploding), or stuck at zero
(dead neurons), preventing the network from learning or causing numerical instability.

**3a. Vanishing Gradients**

*Cause:* Deep networks with saturating activations (sigmoid, tanh) compress gradients
toward zero through repeated chain-rule multiplication. Each layer multiplies by values
<1, so by layer 50+ the gradient is effectively zero.

*Detection:* Per-layer gradient norms decrease exponentially from output to input.
Early layers show near-zero weight updates. Loss plateaus despite non-converged value.

*Assertions:*
```python
assert_gradient_flow(model, min_grad_norm=1e-7)
# After one backward pass, verify every layer's gradient norm > min_grad_norm

assert_weight_update_ratio(model, min_ratio=1e-6)
# Verify |delta_w| / |w| > min_ratio for each parameter group
```

**3b. Exploding Gradients**

*Cause:* Large weight initializations, high learning rates, or recurrent architectures
(unrolled over many timesteps) produce gradient norms that grow exponentially.

*Detection:* Loss becomes NaN or Inf. Gradient norm spikes orders of magnitude above
baseline. Weights diverge to extreme values.

*Assertions:*
```python
assert_gradient_bounded(model, max_grad_norm=1e3)
# Verify no parameter gradient exceeds max_grad_norm

assert_loss_finite(loss_history)
# Verify no NaN or Inf values in loss tensor at every step

assert_weight_bounded(model, max_weight=1e6)
# Verify no parameter exceeds max_weight absolute value
```

**3c. Dead ReLU / Dead Neurons**

*Cause:* Neurons with ReLU activation that receive only negative inputs output zero
permanently. Once dead, gradient is zero so they never recover. Caused by large
negative bias initialization, high learning rates pushing weights negative, or
batch normalization bugs.

*Detection:* Fraction of neurons with zero activation across an entire validation batch.
Typically >10% dead neurons indicates a problem; >50% is critical.

*Assertions:*
```python
assert_neuron_utilization(model, dataloader, min_alive_ratio=0.90)
# Run forward pass, count fraction of neurons with non-zero activation
# Fail if alive ratio < min_alive_ratio

assert_no_dead_layers(model, dataloader)
# Verify no layer has 100% dead neurons (entire layer produces zeros)
```

**3d. Gradient Noise / Instability**

*Cause:* Very small batch sizes produce noisy gradient estimates. Gradient variance
across mini-batches is so high that training oscillates without converging.

*Detection:* Loss curve is extremely noisy (high variance between steps). Gradient
signal-to-noise ratio (mean/std across batches) is below 1.0.

*Assertions:*
```python
assert_gradient_snr(model, dataloader, min_snr=1.0, n_batches=10)
# Compute gradient mean and std across n_batches
# Verify mean / std > min_snr for each parameter group
```

---

#### 4. Learning Rate Issues

**What goes wrong:**
Incorrect learning rate causes divergence (too high), premature stagnation (too low),
or schedule bugs that reset or corrupt the rate mid-training.

**4a. Learning Rate Too High (Divergence)**

*Cause:* Steps overshoot minima. Loss oscillates wildly or increases monotonically.
In extreme cases, loss becomes NaN within the first few batches.

*Detection:* Loss increases over first 100 steps. Loss variance is higher than loss mean.
Gradient norms spike repeatedly.

**4b. Learning Rate Too Low (Stagnation)**

*Cause:* Steps are too small to escape saddle points or make meaningful progress.
Training runs for many epochs with negligible loss decrease.

*Detection:* Loss decreases less than 0.1% per epoch over 10+ consecutive epochs.
Training takes 10x longer than comparable architectures.

**4c. Schedule Bugs**

*Cause:* Warmup period is too long (delays useful training). Step-based scheduler
used with epoch-based training loop (fires N times too fast). Cosine schedule
restarts at wrong period. OneCycleLR total_steps mismatches actual training steps.

*Detection:* Learning rate at any step is outside expected bounds. Schedule produces
rates that don't match the documented curve.

**Assertions for mltk:**
```python
assert_loss_decreasing(loss_history, window=100, min_decrease=0.001)
# Over a rolling window, verify loss trend is negative

assert_no_loss_divergence(loss_history, max_increase_ratio=2.0)
# Verify loss never increases more than max_increase_ratio vs initial loss

assert_lr_schedule_matches(scheduler, expected_lrs, tolerance=1e-6)
# Record LR at each step, compare against expected schedule curve

assert_lr_bounded(scheduler, total_steps, min_lr=1e-8, max_lr=1.0)
# Verify LR stays within [min_lr, max_lr] for all steps

assert_warmup_fraction(scheduler, total_steps, max_warmup_ratio=0.1)
# Verify warmup phase is < 10% of total training
```

---

#### 5. Batch Normalization Bugs

**What goes wrong:**
Batch normalization layers behave differently in training vs evaluation mode.
Incorrect mode setting causes silent accuracy degradation.

**5a. Train/Eval Mode Mismatch**

*Cause:* Forgetting to call `model.eval()` before validation/inference. In training
mode, BN uses per-batch statistics instead of running averages, causing inconsistent
predictions (especially with small batches).

*Detection:* Validation accuracy fluctuates wildly between batches. Same input produces
different outputs on repeated inference calls. Accuracy differs between
`model.eval()` and `model.train()` by >5%.

**5b. Running Statistics Corruption**

*Cause:* Running mean/var accumulate incorrectly when `momentum` is misconfigured or
when the model is trained with very small batches. After training, the running
statistics don't represent the true data distribution.

*Detection:* Compare BN running_mean/running_var against statistics computed from a
full pass over the training set. Large discrepancy indicates corruption.

**5c. BN + Dropout Interaction**

*Cause:* Dropout changes the scale of activations at training time. BN running
statistics are calibrated for dropout-scaled activations. At inference (dropout off),
activations have different scale, causing BN mismatch.

*Detection:* Accuracy in eval mode is notably worse than in training mode (inverted
from expected pattern).

**5d. Small Batch BN**

*Cause:* Batch sizes of 1-4 produce unreliable batch statistics. The per-batch mean
and variance are poor estimates, adding noise to training. Common in fine-tuning
large models where GPU memory limits batch size.

*Detection:* Training loss is much noisier than expected. Switching to GroupNorm or
LayerNorm fixes the issue.

**Assertions for mltk:**
```python
assert_eval_mode_set(model)
# Before any inference call, verify model.training == False

assert_bn_statistics_valid(model, dataloader, tolerance=0.1)
# Compare BN running stats vs full-dataset computed stats
# Fail if relative difference > tolerance

assert_bn_batch_size(dataloader, min_batch_size=8)
# Warn if batch size < min_batch_size and model contains BN layers

assert_train_eval_consistency(model, test_input, max_diff=0.05)
# Run same input in train vs eval mode, verify outputs differ by < max_diff
# (Large diff indicates BN or Dropout misconfiguration)
```

---

#### 6. Distributed Training Bugs

**What goes wrong:**
Multi-GPU / multi-node training introduces synchronization, aggregation, and
data-sharding bugs that produce silently wrong models.

**6a. Gradient Aggregation Errors**

*Cause:* AllReduce averaging vs summing mismatch. If gradients are summed but
learning rate is not divided by world_size, the effective LR is N times too high.
Conversely, averaging + dividing by world_size = effective LR 1/N too low.

*Detection:* Distributed training diverges or stagnates when single-GPU works fine.
Effective batch size does not match expected (batch_per_gpu * world_size).

**6b. Non-Deterministic Reduction Order**

*Cause:* Floating-point addition is not associative. Different reduction orders
across nodes produce slightly different gradient sums. Over thousands of steps,
this causes reproducibility failures.

*Detection:* Same hyperparameters + seed produce different final metrics across runs.
Delta grows with training length.

**6c. Data Sharding Bugs**

*Cause:* Without DistributedSampler, each GPU sees the full dataset (N times more
data per epoch than intended). With incorrect sampler configuration, some GPUs see
duplicate data while others miss samples entirely.

*Detection:* Training is N times slower or faster than expected. Validation metrics
differ between GPU ranks.

**6d. Unused Parameter Errors (DDP)**

*Cause:* In PyTorch DDP, any parameter that does not receive a gradient in the
backward pass causes a hang (DDP waits for AllReduce on all parameters). Common
with conditional computation or multi-task models.

*Detection:* Training hangs at backward pass. Setting `find_unused_parameters=True`
adds overhead but avoids the hang (at a performance cost).

**6e. Batch Normalization in DDP**

*Cause:* Standard BN computes statistics per-GPU. With small per-GPU batch sizes,
statistics are noisy. SyncBatchNorm synchronizes across GPUs but adds communication
overhead and can deadlock if not all GPUs reach the BN layer.

*Detection:* Model accuracy differs between single-GPU and multi-GPU training.
Per-GPU BN statistics diverge across ranks.

**Assertions for mltk:**
```python
assert_distributed_sampler_used(dataloader)
# Verify dataloader.sampler is DistributedSampler when world_size > 1

assert_effective_batch_size(batch_per_gpu, world_size, expected_total)
# Verify batch_per_gpu * world_size == expected_total

assert_gradient_sync(model, tolerance=1e-5)
# After backward, compare gradients across ranks; fail if diff > tolerance

assert_all_parameters_used(model, loss)
# Verify every parameter received a gradient after backward pass

assert_reproducible_across_runs(train_fn, seed, max_metric_diff=0.01, n_runs=2)
# Run training twice with same seed, verify final metric diff < threshold
```

---

#### 7. Data Augmentation Bugs

**What goes wrong:**
Augmentation is applied incorrectly -- to test data, inconsistently with labels,
or in ways that create impossible inputs the model cannot learn from.

**7a. Augmentation Applied to Test Data**

*Cause:* Same transform pipeline is used for train and test without conditional
logic. Random crops, flips, color jitter applied to validation/test data makes
evaluation non-deterministic and biased.

*Detection:* Validation accuracy changes between evaluation runs (same model, same
data). Test pipeline includes random transforms.

**7b. Label-Inconsistent Augmentation**

*Cause:* Geometric transforms (rotation, flip, crop) applied to inputs but not to
corresponding labels. In object detection, bounding boxes are not transformed with
the image. In segmentation, masks are not flipped/rotated with the input.

*Detection:* IoU between augmented labels and visual ground truth is <1.0.
Model converges but produces spatially offset predictions.

**7c. Augmentation Too Aggressive**

*Cause:* Extreme augmentation (heavy noise, large rotations, aggressive cutout)
destroys the signal in training data. The model cannot learn meaningful patterns
from heavily distorted inputs.

*Detection:* Training loss plateaus at a high value (model cannot fit even training
data). Reducing augmentation immediately improves training loss.

**7d. Augmentation Leaks Information**

*Cause:* Augmentation that is class-dependent (only applied to minority class for
balancing) creates distributional artifacts. The model learns to detect augmentation
artifacts rather than genuine class features.

*Detection:* Model performs well on augmented validation data but poorly on real
(non-augmented) data. Feature attribution shows augmentation-artifact regions.

**7e. Mixing Augmentations (MixUp/CutMix) Bugs**

*Cause:* MixUp creates interpolated labels (0.7 cat, 0.3 dog) but the loss function
expects hard labels. Or lambda is drawn from Beta(alpha, alpha) with alpha=0 (no
mixing) or alpha>>1 (everything becomes uniform mush).

*Detection:* Loss curve with MixUp is much higher than expected baseline. MixUp
lambda distribution is degenerate.

**Assertions for mltk:**
```python
assert_no_augmentation_on_test(test_transforms)
# Verify test pipeline contains no random/stochastic transforms

assert_label_augmentation_consistency(image, label, transform)
# Apply transform, verify label geometry matches transformed image

assert_augmentation_preserves_signal(dataset, transform, snr_threshold=0.5)
# Verify augmented samples retain enough signal to be classifiable

assert_augmentation_class_independent(dataset, transform)
# Verify augmentation distribution is identical across classes

assert_mixup_alpha_valid(alpha, min_val=0.1, max_val=2.0)
# Verify MixUp/CutMix alpha parameter is in reasonable range
```

---

#### 8. Checkpoint / Resume Bugs

**What goes wrong:**
Training is interrupted and resumed, but the saved state is incomplete or
incorrectly restored, causing training to regress or produce a worse model.

**8a. Optimizer State Not Saved**

*Cause:* Only model weights are checkpointed, not the optimizer state (momentum
buffers, adaptive learning rate accumulators). Adam's first and second moment
estimates are reset to zero, causing a spike in effective learning rate on resume.

*Detection:* Loss spikes immediately after resume. Training dynamics differ from
continuous run. Adam effectively restarts warmup.

**8b. Learning Rate Schedule Reset**

*Cause:* Scheduler state_dict is not saved. On resume, the scheduler starts from
step 0, repeating warmup or applying the wrong LR for the current epoch.

*Detection:* LR at resume step does not match LR that would have been used in
continuous training. Second warmup is visible in LR plot.

**8c. RNG State Not Saved**

*Cause:* Random number generator states (Python random, NumPy, PyTorch, CUDA) are
not saved. On resume, data augmentation and dropout produce different random
sequences than a continuous run, breaking reproducibility.

*Detection:* Resumed training produces different results than continuous training
with the same total epochs. Data ordering changes.

**8d. Epoch / Step Counter Off-by-One**

*Cause:* Resume starts at epoch N+1 when it should start at epoch N (the interrupted
epoch was incomplete). Or step counter is not saved, so per-step schedulers receive
the wrong step count.

*Detection:* Total training steps differ between resumed and continuous runs.
LR schedule is shifted by one epoch.

**8e. DataLoader State Not Saved**

*Cause:* Shuffled data order is not reproducible on resume. The model may re-train
on some samples and never see others from the interrupted epoch. For distributed
training, sampler state determines which GPU sees which data.

*Detection:* Sample coverage in the resume epoch is uneven. Some samples appear
twice while others are skipped.

**Assertions for mltk:**
```python
assert_checkpoint_complete(checkpoint, required_keys)
# Verify checkpoint dict contains: model_state_dict, optimizer_state_dict,
# scheduler_state_dict, epoch, step, rng_states

assert_resume_lr_matches(scheduler, expected_lr_at_step, tolerance=1e-6)
# After loading scheduler state, verify current LR matches expected

assert_resume_loss_continuous(pre_interrupt_loss, post_resume_loss, max_spike=0.1)
# Verify loss after resume is within max_spike of pre-interrupt loss

assert_optimizer_state_loaded(optimizer, expected_state_keys)
# Verify optimizer state contains momentum buffers / moment estimates

assert_rng_state_restored(checkpoint)
# Verify Python, NumPy, torch, and CUDA RNG states are present and loaded
```

---

#### 9. Memory Leaks During Training

**What goes wrong:**
GPU memory grows unboundedly during training, eventually causing OOM (Out of
Memory) crashes. Or CPU RAM grows due to accumulated Python objects.

**9a. Tensor Accumulation (Retaining Computation Graph)**

*Cause:* Storing `loss` (a tensor with grad_fn) in a Python list for logging
retains the entire computation graph. Each epoch adds the full graph to memory.
Fix: store `loss.item()` (a Python float) instead.

*Detection:* GPU memory grows linearly with training steps. `torch.cuda.memory_allocated()`
increases monotonically.

**9b. Hidden References in Closures / Callbacks**

*Cause:* Logging callbacks, metric trackers, or visualization hooks hold references
to tensors. TensorBoard SummaryWriter with `add_histogram` on large tensors.
Callbacks that capture `model.parameters()` in a closure.

*Detection:* Memory growth correlates with callback frequency. Disabling callbacks
stops the leak. `gc.get_referrers()` on suspect tensors shows unexpected references.

**9c. DataLoader Worker Leaks**

*Cause:* `num_workers > 0` with `persistent_workers=True` and a dataset that grows
(e.g., appending to a list in `__getitem__`). Worker processes share memory via
fork() and accumulate state.

*Detection:* Worker process RSS grows over time. Reducing `num_workers` reduces leak
rate. Occurs specifically with persistent workers.

**9d. Gradient Accumulation Without Clearing**

*Cause:* Forgetting `optimizer.zero_grad()` between accumulation steps. Gradients
accumulate indefinitely, growing linearly. Or calling zero_grad at the wrong point
in the accumulation loop.

*Detection:* Gradient magnitudes grow linearly with step count. Memory does not
directly grow (gradients are fixed-size) but values overflow.

**9e. Mixed Precision Scaler State**

*Cause:* `GradScaler` with `growth_interval` too short causes frequent scale
increases. Scale overflows to Inf, all gradients become NaN, optimizer steps are
skipped indefinitely.

*Detection:* `scaler.get_scale()` grows without bound. Optimizer step count stops
increasing. Loss appears stuck.

**Assertions for mltk:**
```python
assert_no_memory_leak(train_step_fn, dataloader, max_growth_mb=100, n_steps=100)
# Run n_steps, measure GPU memory at start and end
# Fail if growth > max_growth_mb

assert_loss_is_detached(loss_history)
# Verify loss_history contains Python floats, not tensors with grad_fn

assert_grad_accumulation_correct(optimizer, accumulation_steps, step_count)
# Verify zero_grad is called every accumulation_steps

assert_gpu_utilization_stable(training_loop, max_memory_ratio=0.95)
# Monitor peak GPU memory; warn if approaching OOM threshold

assert_scaler_scale_bounded(scaler, max_scale=2**24)
# Verify GradScaler scale factor stays within reasonable bounds
```

---

#### 10. Numerical Stability / Mixed Precision Issues

**What goes wrong:**
Floating-point arithmetic limitations cause silent precision loss, overflow, or
NaN propagation, especially in float16 / bfloat16 mixed-precision training.

**10a. Float16 Overflow**

*Cause:* float16 max value is 65504. Activations, logits, or intermediate values
exceeding this become Inf. Common in large batch training, pre-LayerNorm
transformers, and models without proper initialization.

*Detection:* NaN/Inf in activations or loss. Occurs only in float16, not float32.
`torch.isinf()` on intermediate tensors.

**10b. Float16 Underflow (Loss of Precision)**

*Cause:* float16 minimum subnormal is ~5.96e-8. Small gradients or weight updates
round to zero ("swamped" by larger values). Model appears to train but converges to
a suboptimal solution because small-but-important updates are lost.

*Detection:* Comparing float16 vs float32 training shows different final metrics.
Gradient histograms show clumping at zero in float16.

**10c. Loss Scaling Misconfiguration**

*Cause:* Dynamic loss scaling (GradScaler) multiplies loss by a large factor before
backward to shift gradients into float16's representable range. If initial scale is
too high, all gradients overflow to Inf and every step is skipped. If too low,
underflow persists.

*Detection:* GradScaler skips many consecutive optimizer steps. Scale oscillates
rapidly (growing then immediately shrinking). Training makes no progress.

**10d. Softmax / Log-Sum-Exp Instability**

*Cause:* Computing `softmax(x)` directly causes overflow when `max(x)` is large.
Must use the numerically stable form: `softmax(x - max(x))`. Most frameworks
handle this internally, but custom implementations often miss it.

*Detection:* NaN in softmax output. Probability distribution does not sum to 1.0.
Occurs with large logit values.

**10e. Cross-Entropy with Hard Labels**

*Cause:* `log(0)` produces -Inf when a predicted probability is exactly zero for the
true class. Happens with confident wrong predictions when no label smoothing is
used. `-Inf * 0` in the gradient produces NaN.

*Detection:* NaN loss appearing sporadically (only on specific samples). Adding
label smoothing (epsilon=0.1) eliminates the NaN.

**10f. Catastrophic Cancellation**

*Cause:* Subtracting two nearly-equal large numbers loses most significant digits.
Common in variance computation: `E[x^2] - (E[x])^2` can produce negative values
due to cancellation, yielding NaN in `sqrt(var)`. Welford's online algorithm avoids
this.

*Detection:* Negative variance values. NaN in batch normalization or layer
normalization. Numerical issues that appear only with specific data distributions.

**10g. BFloat16 Accumulation Errors**

*Cause:* bfloat16 has same exponent range as float32 (no overflow) but only 8 bits
of mantissa (vs 23). Long summations (e.g., loss over many tokens) accumulate
rounding errors. Dot products of long vectors lose precision.

*Detection:* bfloat16 and float32 training produce notably different losses by
mid-training. Accumulation of >1000 values shows visible drift.

**Assertions for mltk:**
```python
assert_no_nan_inf(tensor, name="")
# Verify tensor contains no NaN or Inf values

assert_loss_scale_effective(scaler, min_finite_ratio=0.9, window=100)
# Over last `window` steps, verify >= min_finite_ratio had finite gradients
# (i.e., scaler is not skipping most steps)

assert_numerical_stability(model, input_batch, dtype=torch.float16)
# Forward pass in float16, compare against float32 reference
# Fail if max relative error > threshold (1e-2 for fp16, 1e-3 for bf16)

assert_softmax_valid(probs)
# Verify all values in [0,1] and each row sums to 1.0 within tolerance

assert_variance_positive(bn_layer)
# Verify running_var contains no negative values (cancellation check)

assert_mixed_precision_config(model, scaler)
# Verify: loss computed in float32, model params in float16/bf16,
# optimizer master weights in float32, scaler is enabled
```

---

### Summary: Training Bug Categories and Priority for mltk

| # | Category | Assertions | Priority | Target Sprint |
|---|---|---|---|---|
| 1 | Data Leakage | 5 | **P0 -- Critical** | Sprint 3 or 4 |
| 2 | Feature Leakage | 4 | **P0 -- Critical** | Sprint 3 or 4 |
| 3 | Gradient Pathologies | 6 | P1 -- High | Sprint 6 |
| 4 | Learning Rate Issues | 5 | P1 -- High | Sprint 6 |
| 5 | Batch Normalization | 4 | P1 -- High | Sprint 6 |
| 6 | Distributed Training | 5 | P2 -- Medium | Sprint 9+ |
| 7 | Data Augmentation | 5 | P1 -- High | Sprint 7 (CV kit) |
| 8 | Checkpoint/Resume | 5 | P2 -- Medium | Sprint 6 |
| 9 | Memory Leaks | 5 | P2 -- Medium | Sprint 9+ |
| 10 | Numerical Stability | 7 | P1 -- High | Sprint 6 |
| | **TOTAL** | **51** | | |

### Implementation Notes

**Framework-agnostic approach:**
Most assertions above use PyTorch semantics but mltk should provide framework-agnostic
wrappers. Strategy:
- Core assertions work on NumPy arrays and Python objects (DataFrames, lists, dicts)
- PyTorch-specific assertions in `mltk.training.torch` submodule
- TensorFlow-specific assertions in `mltk.training.tf` submodule (stretch goal)
- JAX-specific assertions in `mltk.training.jax` submodule (stretch goal)

**What to build first (P0):**
Data leakage and feature leakage are the highest-impact, lowest-cost assertions.
They work on DataFrames (no GPU needed), catch the most common production failures,
and differentiate mltk from competitors (none of whom offer these as assertions).

**What competitors offer:**
- Deepchecks: train/test overlap, feature-label correlation (deprecated API)
- Evidently: drift only (no leakage detection)
- Great Expectations: schema/data quality only (no ML-specific assertions)
- None offer gradient, checkpoint, or distributed training assertions.
The training-specific assertions (categories 3-10) would be **unique to mltk**.

---

*Last updated: Sprint 2 completion (March 25, 2026)*
