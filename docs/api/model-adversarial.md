# Model Adversarial Robustness

Adversarial robustness testing checks if small input perturbations cause the model to change its predictions. A robust model should be stable: adding tiny noise to inputs shouldn't flip the output. Fragile models are dangerous in production -- they fail unpredictably on slightly unusual inputs.

**Module:** `mltk.model.adversarial`

**ML Lifecycle Stage:** Post-training evaluation / Security gate

---

## assert_robust

Assert model predictions are stable under input perturbations.

```python
from mltk.model import assert_robust

assert_robust(model.predict, X_test, perturbation="gaussian", epsilon=0.01, stability=0.95)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `model_fn` | `Callable` | *(required)* | Function that takes inputs and returns predictions |
| `inputs` | `array-like` | *(required)* | Test inputs to perturb |
| `perturbation` | `str` | `"gaussian"` | Noise type: `"gaussian"` or `"uniform"` |
| `epsilon` | `float` | `0.01` | Noise magnitude (std for gaussian, range for uniform) |
| `stability` | `float` | `0.95` | Min fraction of inputs that must keep same prediction |

### How it works

```
For each input x in inputs:
    x_noisy = x + noise(epsilon)
    if predict(x) != predict(x_noisy):
        count as unstable

stability_score = stable_count / total_count
PASS if stability_score >= stability threshold
```

### Example

```python
@pytest.mark.ml_model
def test_model_robust_to_noise(model, X_test):
    assert_robust(model.predict, X_test, epsilon=0.01, stability=0.95)
```

### Related Tests

| Test | What it validates |
|------|-------------------|
| `test_stable_model` | Robust model passes (>95% stable) |
| `test_fragile_model` | Unstable model fails |
| `test_gaussian_perturbation` | Gaussian noise applied correctly |
| `test_uniform_perturbation` | Uniform noise applied correctly |
| `test_custom_epsilon` | Different noise magnitudes |
| `test_empty_inputs` | Empty input handled gracefully |

---
