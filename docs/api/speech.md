# Speech Testing

Speech-specific assertions for recognition accuracy (WER, CER), real-time performance (RTF), and accent fairness.

**Module:** `mltk.domains.speech`

**Install:** `pip install mltk[speech]`

---

## Recognition Accuracy

### assert_wer

Assert Word Error Rate is below threshold. Uses `jiwer` library.

```python
from mltk.domains.speech import assert_wer

assert_wer(references, hypotheses, max_wer=0.1)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `references` | `list[str]` | *(required)* | Ground truth transcriptions |
| `hypotheses` | `list[str]` | *(required)* | Model transcriptions |
| `max_wer` | `float` | `0.1` | Maximum allowed WER (0.1 = 10% error rate) |

#### Returns

`TestResult` with details:
- `wer` -- computed Word Error Rate
- `max_wer` -- configured threshold
- `num_samples` -- number of samples evaluated

---

### assert_cer

Assert Character Error Rate is below threshold. Uses `jiwer` library.

```python
from mltk.domains.speech import assert_cer

assert_cer(references, hypotheses, max_cer=0.05)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `references` | `list[str]` | *(required)* | Ground truth transcriptions |
| `hypotheses` | `list[str]` | *(required)* | Model transcriptions |
| `max_cer` | `float` | `0.05` | Maximum allowed CER (0.05 = 5% error rate) |

#### Returns

`TestResult` with details:
- `cer` -- computed Character Error Rate
- `max_cer` -- configured threshold
- `num_samples` -- number of samples evaluated

---

## Performance

### assert_rtf

Assert Real-Time Factor is below threshold. RTF < 1.0 means real-time capable.

```python
from mltk.domains.speech import assert_rtf

assert_rtf(process_fn, audio_durations, max_rtf=1.0)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `process_fn` | `Callable` | *(required)* | Function that processes audio (called once per duration value) |
| `audio_durations` | `list[float]` | *(required)* | Audio durations in seconds |
| `max_rtf` | `float` | `1.0` | Maximum allowed RTF |

#### Returns

`TestResult` with details:
- `rtf` -- computed Real-Time Factor (processing_time / audio_duration)
- `max_rtf` -- configured threshold
- `total_processing_sec` -- total processing time in seconds
- `total_audio_sec` -- total audio duration in seconds

---

## Fairness

### assert_accent_coverage

Detect accent bias by checking WER gap between best and worst accents.

```python
from mltk.domains.speech import assert_accent_coverage

assert_accent_coverage(wer_by_accent, max_gap=0.05)
```

#### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `wer_by_accent` | `dict[str, float]` | *(required)* | Dict mapping accent name to WER value |
| `max_gap` | `float` | `0.05` | Maximum allowed WER gap between best and worst accent |

#### Returns

`TestResult` with details:
- `gap` -- actual WER gap (worst - best)
- `max_gap` -- configured threshold
- `best_accent` -- accent with lowest WER
- `best_wer` -- lowest WER value
- `worst_accent` -- accent with highest WER
- `worst_wer` -- highest WER value
- `wer_by_accent` -- full dict of per-accent WER

#### Edge Cases

- **Fewer than 2 accents**: Returns a passing result with `INFO` severity (need at least 2 accents for comparison).

---
