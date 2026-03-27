# Streaming Drift Detection

Detect distribution shifts in real-time data streams using per-element stateful detectors.

**Module:** `mltk.monitor.streaming_drift`

**ML Lifecycle Stage:** Production monitoring

**When to use:**
- Real-time inference pipelines where data arrives one observation at a time
- Production monitoring where batch windows are too slow to catch urgent shifts
- Streaming feature stores that need immediate drift alerts
- Any system where you cannot afford to wait for a full batch before detecting change

---

## Overview

Unlike batch drift detection (`assert_no_drift`), streaming drift detects shifts as they happen -- one observation at a time. Batch methods require collecting a window of data, computing a statistic, and comparing it against a reference. Streaming methods maintain internal state and update incrementally, raising an alarm as soon as sufficient evidence of a shift accumulates.

This matters because production ML systems often process data continuously. A credit scoring model receiving loan applications one at a time cannot wait for a nightly batch job to discover that the income distribution shifted three hours ago. Streaming drift detection bridges this gap.

mltk provides two streaming drift detectors:

| Detector | Best For | Complexity | Parameters |
|----------|----------|------------|------------|
| **ADWIN** | Adaptive window, auto-tunes sensitivity | O(log W) per element | `delta` (confidence) |
| **CUSUM** | Fixed threshold, simple, fast | O(1) per element | `threshold`, `drift_level` |

Both detectors share the same assertion interface (`assert_no_streaming_drift`) and can also be used directly for lower-level control.

---

## Quick Start

```python
import numpy as np
from mltk.monitor.streaming_drift import assert_no_streaming_drift

# Simulate a stable stream that shifts at element 500
stable = np.random.normal(0.0, 1.0, 500)
shifted = np.random.normal(2.0, 1.0, 500)
stream = np.concatenate([stable, shifted])

# ADWIN (default) — adaptive, auto-tunes window size
result = assert_no_streaming_drift(stream, method="adwin", delta=0.002)
# result.details["drift_detected"] -> True
# result.details["drift_point"]    -> ~500 (where the shift was detected)

# CUSUM — fixed threshold, lightweight
result = assert_no_streaming_drift(stream, method="cusum", threshold=5.0)
```

### In a pytest test

```python
import pytest
from mltk.monitor.streaming_drift import assert_no_streaming_drift

@pytest.mark.ml_drift
def test_feature_stream_stable():
    """Incoming feature values have not drifted from expected distribution."""
    stream = load_recent_feature_values("income")
    assert_no_streaming_drift(stream, method="adwin", delta=0.002)
```

---

## Detectors

### ADWIN (Adaptive Windowing)

ADWIN (ADaptive WINdowing) maintains a variable-length window of recent observations and automatically detects when the statistical properties of the window change. It works by comparing sub-windows: when the difference between two sub-windows exceeds a confidence bound derived from the Hoeffding inequality, it declares drift and shrinks the window.

**Why ADWIN:** You do not need to choose a fixed window size. ADWIN grows the window when data is stable (increasing statistical power) and shrinks it when drift occurs (adapting quickly). This makes it robust across different drift speeds -- gradual shifts and sudden jumps alike.

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `delta` | `float` | `0.002` | Confidence parameter. Lower values make the detector less sensitive (fewer false alarms but slower to react). Range: (0, 1). |
| `min_window` | `int` | `30` | Minimum number of observations before drift can be declared. Prevents false alarms on insufficient data. |

#### Algorithm

```
For each new element x:
    1. Add x to the window
    2. Compress the window into logarithmic buckets
    3. For each possible split of the window into [old | recent]:
        If |mean(old) - mean(recent)| >= epsilon_cut(delta, n):
            Declare DRIFT at this point
            Drop the old sub-window
```

The bound `epsilon_cut` is derived from the Hoeffding inequality and depends on `delta` and the sub-window sizes. Smaller `delta` means a tighter bound (harder to trigger).

#### Direct usage

```python
from mltk.monitor.streaming_drift import ADWINDetector

detector = ADWINDetector(delta=0.002, min_window=30)

for value in incoming_stream:
    detector.update(value)
    if detector.drift_detected:
        print(f"Drift at observation {detector.total_count}")
        print(f"Window mean: {detector.window_mean:.4f}")
        detector.reset()  # optional: reset after handling drift
```

#### Choosing delta

| delta | Sensitivity | Use case |
|-------|-------------|----------|
| 0.01 | Low | Noisy data, many features, want few false alarms |
| 0.002 | Medium (default) | General purpose |
| 0.0001 | High | Safety-critical systems, must catch subtle shifts |

---

### CUSUM (Cumulative Sum)

CUSUM tracks the cumulative deviation of observations from a target mean. When the cumulative sum exceeds a threshold, drift is declared. It is one of the oldest and simplest change-detection methods (Page, 1954).

**Why CUSUM:** When you know the expected mean of your stream and want a simple, fast detector with O(1) per-element cost. CUSUM is deterministic and easy to reason about -- the threshold directly controls how much cumulative deviation you tolerate.

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `threshold` | `float` | `5.0` | Cumulative sum threshold. Drift is declared when the cumulative sum exceeds this value. |
| `drift_level` | `float` | `0.5` | Allowance parameter (slack). Subtracted from each deviation to prevent small fluctuations from accumulating. |
| `target_mean` | `float \| None` | `None` | Expected mean of the stream. If `None`, estimated from the first `min_window` observations. |

#### Algorithm

```
Initialize: S_pos = 0, S_neg = 0
For each new element x:
    S_pos = max(0, S_pos + (x - target_mean) - drift_level)
    S_neg = max(0, S_neg - (x - target_mean) - drift_level)
    If S_pos > threshold OR S_neg > threshold:
        Declare DRIFT
```

CUSUM tracks both positive shifts (mean increase) and negative shifts (mean decrease) using two accumulators.

#### Direct usage

```python
from mltk.monitor.streaming_drift import CUSUMDetector

detector = CUSUMDetector(threshold=5.0, drift_level=0.5, target_mean=0.0)

for value in incoming_stream:
    detector.update(value)
    if detector.drift_detected:
        print(f"Drift at observation {detector.total_count}")
        print(f"CUSUM statistic: {detector.cusum_value:.4f}")
        detector.reset()
```

#### Choosing threshold

| threshold | Sensitivity | Use case |
|-----------|-------------|----------|
| 3.0 | High | Small shifts matter, low-noise data |
| 5.0 | Medium (default) | General purpose |
| 10.0 | Low | High-noise data, only care about large shifts |

---

## assert_no_streaming_drift

The unified assertion interface for streaming drift detection.

```python
from mltk.monitor.streaming_drift import assert_no_streaming_drift

result = assert_no_streaming_drift(
    stream,
    method="adwin",
    delta=0.002,
    min_window=30,
)
```

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `stream` | `list[float] \| np.ndarray` | *(required)* | Time-ordered sequence of observations. |
| `method` | `str` | `"adwin"` | Detector: `"adwin"` or `"cusum"`. |
| `delta` | `float` | `0.002` | ADWIN confidence parameter. Ignored if `method="cusum"`. |
| `min_window` | `int` | `30` | Minimum observations before drift can fire. Used by both methods. |
| `threshold` | `float` | `5.0` | CUSUM cumulative sum threshold. Ignored if `method="adwin"`. |
| `drift_level` | `float` | `0.5` | CUSUM allowance/slack parameter. Ignored if `method="adwin"`. |
| `target_mean` | `float \| None` | `None` | CUSUM expected mean. If `None`, estimated from initial observations. Ignored if `method="adwin"`. |

### Returns

`TestResult` with:

- `name`: `"monitor.streaming_drift"`
- `passed`: `True` if no drift detected, `False` if drift detected
- `severity`: `CRITICAL`
- `details.drift_detected`: boolean flag
- `details.drift_point`: index in the stream where drift was first detected (or `None` if no drift)
- `details.method`: detector method used (`"adwin"` or `"cusum"`)
- `details.window_size`: effective window size at detection (ADWIN) or total observations processed (CUSUM)
- `details.stream_length`: total number of elements in the input stream

### Example

```python
import numpy as np
import pytest
from mltk.monitor.streaming_drift import assert_no_streaming_drift

@pytest.mark.ml_drift
def test_prediction_stream_stable():
    """Model prediction scores have not drifted in the last hour."""
    recent_scores = load_prediction_scores(last_n_hours=1)
    assert_no_streaming_drift(recent_scores, method="adwin", delta=0.002)

@pytest.mark.ml_drift
def test_feature_stream_no_sudden_shift():
    """Feature pipeline has not experienced a sudden distribution shift."""
    feature_values = load_feature_stream("transaction_amount")
    assert_no_streaming_drift(
        feature_values,
        method="cusum",
        threshold=5.0,
        target_mean=150.0,  # expected mean transaction amount
    )
```

---

## ADWIN vs. CUSUM: When to Use Which

| Criterion | ADWIN | CUSUM |
|-----------|-------|-------|
| **Prior knowledge of mean** | Not needed | Helps (optional) |
| **Window size tuning** | Automatic | N/A (cumulative) |
| **Gradual drift** | Good (adapts window) | Good (accumulates deviation) |
| **Sudden drift** | Good (shrinks window fast) | Good (threshold crossed quickly) |
| **Memory** | O(log W) | O(1) |
| **Interpretability** | Moderate (adaptive window internals) | High (single cumulative sum) |
| **False alarm control** | `delta` parameter | `threshold` + `drift_level` |

**Rule of thumb:** Use ADWIN when you do not know what to expect. Use CUSUM when you have a clear target mean and want a simple, interpretable detector.

---

## Edge Cases

- **Empty stream**: Returns a passing result with `drift_detected=False` and `drift_point=None`.
- **Stream shorter than min_window**: Returns a passing result (insufficient data to judge).
- **Constant stream**: No drift detected (zero variance).
- **Unknown method**: Fails with a descriptive error listing supported methods.
- **NaN values**: Observations containing NaN are skipped (not counted toward the window).

---

## Drift Story: Complete Coverage

mltk provides four layers of drift detection, each answering a different question:

| Type | What Changed | Question Answered | mltk Assertion |
|------|-------------|-------------------|----------------|
| Input drift P(X) | Feature distributions | Has the data the model sees changed? | `assert_no_drift` |
| Output drift P(Y-hat) | Prediction distributions | Have the model's outputs shifted? | `assert_no_output_drift` |
| Streaming drift | Real-time distribution shift | Is the data changing right now? | `assert_no_streaming_drift` |
| Concept drift P(Y\|X) | Input-output relationship | Has the meaning of the data changed? | `assert_no_concept_drift` |

Use them together for defense in depth:

```python
# Batch: check daily
assert_no_drift(train_features, today_features)
assert_no_output_drift(baseline_scores, today_scores)

# Streaming: check continuously
assert_no_streaming_drift(live_feature_stream, method="adwin")

# Concept: check when labels arrive
assert_no_concept_drift(ref_labels, ref_preds, cur_labels, cur_preds)
```

---

## References

- Bifet, A. and Gavalda, R. (2007). "Learning from Time-Changing Data with Adaptive Windowing." *SIAM International Conference on Data Mining.*
- Page, E. S. (1954). "Continuous Inspection Schemes." *Biometrika*, 41(1/2), 100-115.

---
