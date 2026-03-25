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
