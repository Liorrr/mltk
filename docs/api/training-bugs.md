# Training Bug Detection

Catch the most common ML training bugs before they reach production. Covers data leakage (P0) and gradient/numerical pathologies (P1).

**Module:** `mltk.training`

---

## Data Leakage (P0)

### assert_no_train_test_overlap

Verify zero row overlap between train and test DataFrames on key columns.

```python
from mltk.training import assert_no_train_test_overlap

assert_no_train_test_overlap(train_df, test_df, key_cols=["user_id"])
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `train_df` | `pd.DataFrame` | *(required)* | Training DataFrame |
| `test_df` | `pd.DataFrame` | *(required)* | Test DataFrame |
| `key_cols` | `list[str]` | *(required)* | Columns to check for overlap |

---

### assert_temporal_split

Verify train data is strictly before test data (no temporal leakage).

```python
from mltk.training import assert_temporal_split

assert_temporal_split(train_df, test_df, time_col="created_at")
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `train_df` | `pd.DataFrame` | *(required)* | Training DataFrame |
| `test_df` | `pd.DataFrame` | *(required)* | Test DataFrame |
| `time_col` | `str` | *(required)* | Name of datetime/timestamp column |

---

### assert_no_target_leakage

Detect features too correlated with the target variable (proxy leakage).

```python
from mltk.training import assert_no_target_leakage

assert_no_target_leakage(df, target_col="label", corr_threshold=0.95)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `df` | `pd.DataFrame` | *(required)* | DataFrame with features and target |
| `target_col` | `str` | *(required)* | Target column name |
| `feature_cols` | `list[str] | None` | `None` | Feature columns (None = all numeric) |
| `corr_threshold` | `float` | `0.95` | Max allowed absolute correlation |

---

## Gradient Pathologies (P1)

### assert_gradient_flow

Check that gradients are flowing through all layers (not dead/zero).

```python
from mltk.training import assert_gradient_flow

grads = [layer.grad.numpy() for layer in model.parameters()]
assert_gradient_flow(grads, min_mean_grad=1e-7)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `gradients` | `list[np.ndarray]` | *(required)* | Gradient arrays, one per layer |
| `min_mean_grad` | `float` | `1e-7` | Minimum allowed mean |gradient| per layer |

#### Returns

`TestResult` with details:
- `per_layer_means` -- mean |gradient| for each layer
- `dead_layers` -- indices of layers below threshold

---

### assert_no_vanishing_gradient

Check that no layer has vanishing gradients (norm too small).

```python
from mltk.training import assert_no_vanishing_gradient

assert_no_vanishing_gradient(grads, min_grad_norm=1e-8)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `gradients` | `list[np.ndarray]` | *(required)* | Gradient arrays, one per layer |
| `min_grad_norm` | `float` | `1e-8` | Minimum allowed L2 norm per layer |

---

### assert_no_exploding_gradient

Check that no layer has exploding gradients (norm too large).

```python
from mltk.training import assert_no_exploding_gradient

assert_no_exploding_gradient(grads, max_grad_norm=1000.0)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `gradients` | `list[np.ndarray]` | *(required)* | Gradient arrays, one per layer |
| `max_grad_norm` | `float` | `1000.0` | Maximum allowed L2 norm per layer |

---

### assert_loss_finite

Check that all loss values are finite (no NaN or Inf).

```python
from mltk.training import assert_loss_finite

assert_loss_finite(loss_history)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `losses` | `array-like` | *(required)* | 1D array of loss values over training steps |

---

## Numerical Stability (P1)

### assert_no_nan_inf

Check arrays for NaN or Inf values. Works on weights, activations, outputs.

```python
from mltk.training import assert_no_nan_inf

assert_no_nan_inf([weights, activations], names=["weights", "activations"])
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `arrays` | `list[np.ndarray]` | *(required)* | Arrays to check |
| `names` | `list[str] | None` | `None` | Optional names for each array |

---

### assert_loss_decreasing

Check that training loss is generally decreasing over a sliding window.

```python
from mltk.training import assert_loss_decreasing

assert_loss_decreasing(loss_history, window=10, min_decrease=0.0)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `losses` | `array-like` | *(required)* | 1D array of loss values |
| `window` | `int` | `10` | Window size for comparison |
| `min_decrease` | `float` | `0.0` | Minimum required decrease (end - start) |

---

### assert_no_loss_divergence

Check that loss hasn't diverged (spiked dramatically).

```python
from mltk.training import assert_no_loss_divergence

assert_no_loss_divergence(loss_history, max_increase_ratio=10.0)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `losses` | `array-like` | *(required)* | 1D array of loss values |
| `max_increase_ratio` | `float` | `10.0` | Maximum allowed max/min ratio |

---

### assert_softmax_valid

Check that softmax outputs are valid probabilities (sum to ~1, in [0,1]).

```python
from mltk.training import assert_softmax_valid

assert_softmax_valid(model_output_probs)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `probabilities` | `array-like` | *(required)* | 2D array (samples x classes) |

---

## Augmentation Safety (P2)

### assert_no_augmentation_on_test

Verify that no data augmentation is applied to the test/validation set.

```python
from mltk.training import assert_no_augmentation_on_test

assert_no_augmentation_on_test(original_samples, augmented_samples)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `original_samples` | `list[np.ndarray]` | *(required)* | Original (un-augmented) samples |
| `augmented_samples` | `list[np.ndarray]` | *(required)* | Samples as processed by the pipeline |

---

### assert_augmentation_preserves_signal

Verify that augmented samples remain similar enough to the originals (augmentation isn't destroying the input signal).

```python
from mltk.training import assert_augmentation_preserves_signal

assert_augmentation_preserves_signal(original_samples, augmented_samples, max_mean_diff=0.5)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `original_samples` | `list[np.ndarray]` | *(required)* | Original samples |
| `augmented_samples` | `list[np.ndarray]` | *(required)* | Augmented samples |
| `max_mean_diff` | `float` | `0.5` | Maximum allowed mean absolute difference per sample |

---

## Checkpoint Integrity (P2)

### assert_checkpoint_complete

Verify that a checkpoint dictionary contains all required keys.

```python
from mltk.training import assert_checkpoint_complete

assert_checkpoint_complete(checkpoint, required_keys=["model_state", "optimizer_state", "epoch"])
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `checkpoint` | `dict` | *(required)* | Loaded checkpoint dictionary |
| `required_keys` | `list[str]` | *(required)* | Keys that must be present |

---

### assert_resume_loss_continuous

Verify that the loss after resuming from a checkpoint is close to the loss at the checkpoint boundary (no unexpected jump).

```python
from mltk.training import assert_resume_loss_continuous

assert_resume_loss_continuous(loss_before_save, loss_after_resume, max_delta=0.5)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `loss_before_save` | `float` | *(required)* | Last loss value recorded before saving the checkpoint |
| `loss_after_resume` | `float` | *(required)* | First loss value recorded after loading the checkpoint |
| `max_delta` | `float` | `0.5` | Maximum allowed absolute difference |

---

## Distributed Training (P2)

### assert_effective_batch_size

Verify that `local_batch_size * world_size` equals the expected effective batch size.

```python
from mltk.training import assert_effective_batch_size

assert_effective_batch_size(local_batch_size=32, world_size=4, expected_batch_size=128)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `local_batch_size` | `int` | *(required)* | Batch size per GPU/process |
| `world_size` | `int` | *(required)* | Total number of GPUs/processes |
| `expected_batch_size` | `int` | *(required)* | Batch size the training recipe was designed for |

#### Returns

`TestResult` with details:
- `effective_batch_size` — computed `local_batch_size * world_size`
- `local_batch_size`, `world_size`, `expected_batch_size` — inputs for diagnostics

---

### assert_gradient_sync

Verify that gradient arrays from two ranks are equal within floating-point tolerance after an all-reduce.

```python
from mltk.training import assert_gradient_sync

grads0 = [layer.grad.numpy() for layer in model.parameters()]
# ... collect grads1 from rank 1 via your test harness ...
assert_gradient_sync(grads0, grads1, tolerance=1e-5)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `grads_rank0` | `list[np.ndarray]` | *(required)* | Gradient arrays from rank 0, one per layer |
| `grads_rank1` | `list[np.ndarray]` | *(required)* | Gradient arrays from rank 1, one per layer |
| `tolerance` | `float` | `1e-5` | Maximum allowed element-wise absolute difference |

#### Returns

`TestResult` with details:
- `max_diff` — largest element-wise difference found across all layers
- `diverged_layers` — indices of layers exceeding tolerance
- `num_layers` — total layer count compared
- `tolerance` — threshold used

---

## Memory Leak Detection (P2)

### assert_no_memory_leak

Detect unbounded memory growth during training by comparing the mean of early vs late memory readings.

```python
from mltk.training import assert_no_memory_leak

# Record RSS or GPU memory (MB) once per step
readings = [get_memory_mb() for _ in training_loop]
assert_no_memory_leak(readings, max_growth_mb=100.0, window=10)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `memory_readings_mb` | `list[float]` | *(required)* | Memory measurements in MB, one per step or epoch |
| `max_growth_mb` | `float` | `100.0` | Maximum allowed growth from start to end window |
| `window` | `int` | `10` | Number of readings to average at start and end |

#### Returns

`TestResult` with details:
- `growth_mb` — `end_mean - start_mean`
- `start_mean_mb`, `end_mean_mb` — windowed averages
- `window` — effective window used (clamped for short sequences)
- `num_readings` — total readings provided

---

### assert_loss_is_detached

Detect computation graph accumulation by checking that memory growth per training step is bounded.

```python
from mltk.training import assert_loss_is_detached

memory_per_step = [get_memory_mb() for _ in training_loop]
assert_loss_is_detached(memory_per_step, max_growth_per_step_mb=1.0)
```

If loss is stored as a tensor (not `.item()` or `.detach()`), the entire computation graph is retained and memory grows linearly. This assertion fits a linear slope to the readings and fails if it exceeds the threshold.

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `memory_per_step_mb` | `list[float]` | *(required)* | Memory usage in MB after each training step |
| `max_growth_per_step_mb` | `float` | `1.0` | Maximum allowed memory growth per step in MB |

#### Returns

`TestResult` with details:
- `slope_mb_per_step` — linear regression slope of memory vs step
- `max_growth_per_step_mb` — threshold used
- `num_steps` — number of steps analysed

---
