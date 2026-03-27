# Multimodal & Reinforcement Learning Testing

Validate multimodal alignment and RL reward functions.

**Modules:** `mltk.domains.multimodal`, `mltk.domains.rl`

---

## Why Multimodal Testing

Multimodal AI models (CLIP, BLIP, LLaVA, GPT-4V) learn to map different modalities -- images, text, audio -- into a shared embedding space. When working correctly, semantically related inputs from different modalities land close together: a photo of a dog and the sentence "a photo of a dog" should have high cosine similarity.

Multimodal failures are subtle and dangerous:

- **After fine-tuning**: Alignment degrades because the model over-fits to one modality
- **After quantization**: Reduced precision shifts embeddings enough to break retrieval
- **After data updates**: New training data introduces modality-specific biases
- **In production**: Cross-modal inconsistency causes contradictory outputs

### What mltk Tests

mltk does NOT compute embeddings -- that is the model's job. These assertions take **pre-computed** embeddings or predictions and validate their quality:

1. **Image-text alignment**: Do image and text embeddings agree in shared space?
2. **Cross-modal consistency**: When the same content is processed through different modalities, do predictions agree?

---

## `assert_image_text_alignment`

Assert that paired image and text embeddings are aligned in a shared embedding space.

**Module:** `mltk.domains.multimodal`

### Why This Matters

CLIP-style models encode images and text into the same vector space. Aligned pairs (e.g., a photo and its caption) should have high cosine similarity. This assertion computes per-pair cosine similarity and checks that the average meets a minimum threshold.

Use this after fine-tuning, quantization, or data updates to catch alignment degradation before it reaches production.

### Example with CLIP Embeddings

```python
import numpy as np
from mltk.domains.multimodal import assert_image_text_alignment

# In practice, these come from your CLIP model:
#   image_emb = clip_model.encode_image(images)
#   text_emb = clip_model.encode_text(captions)

# Simulated aligned embeddings (high similarity)
image_embeddings = np.array([
    [0.9, 0.1, 0.0, 0.1],
    [0.1, 0.8, 0.2, 0.0],
    [0.0, 0.1, 0.9, 0.1],
])
text_embeddings = np.array([
    [0.85, 0.15, 0.05, 0.1],
    [0.15, 0.75, 0.25, 0.05],
    [0.05, 0.15, 0.85, 0.15],
])

result = assert_image_text_alignment(
    image_embeddings=image_embeddings,
    text_embeddings=text_embeddings,
    min_cosine=0.5,
)
assert result.passed
print(f"Average cosine similarity: {result.details['avg_cosine']:.4f}")
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_embeddings` | `np.ndarray` | *(required)* | Image embeddings, shape `(n_pairs, dim)` or `(dim,)` for a single pair |
| `text_embeddings` | `np.ndarray` | *(required)* | Text embeddings, same shape. Each row matches the corresponding image. |
| `min_cosine` | `float` | `0.5` | Minimum average cosine similarity required |

Returns `TestResult` (name: `domains.multimodal.image_text_alignment`) with details:

| Detail Key | Type | Description |
|------------|------|-------------|
| `avg_cosine` | `float` | Average cosine similarity across all pairs |
| `min_cosine` | `float` | The threshold that was required |
| `min_pair_cosine` | `float` | Lowest per-pair cosine similarity |
| `max_pair_cosine` | `float` | Highest per-pair cosine similarity |
| `n_pairs` | `int` | Number of image-text pairs evaluated |

### Cosine Similarity Reference Values

| Scenario | Typical Cosine Similarity |
|----------|:------------------------:|
| CLIP matched pairs | 0.70 - 0.85 |
| CLIP random pairs | 0.20 - 0.35 |
| Post-quantization (healthy) | 0.60 - 0.80 |
| Post-quantization (degraded) | 0.30 - 0.50 |
| Broken alignment | < 0.20 |

---

## `assert_cross_modal_consistency`

Assert that predictions from two modalities agree on the same content.

**Module:** `mltk.domains.multimodal`

### Why This Matters

When a model processes the same input through different modalities (text description vs. image of the same scene), the predictions should be consistent. Disagreement reveals modality-specific biases or failures in multimodal fusion.

Real-world examples where this catches bugs:

- **Medical**: Text report says "benign" but image classifier says "malignant"
- **Autonomous driving**: LIDAR detects obstacle but camera classifier says "clear"
- **Content moderation**: Text is safe but image contains violations

### Example

```python
from mltk.domains.multimodal import assert_cross_modal_consistency

# Predictions from text modality
preds_text = ["cat", "dog", "cat", "bird", "cat"]

# Predictions from image modality (same content)
preds_image = ["cat", "dog", "dog", "bird", "cat"]

result = assert_cross_modal_consistency(
    predictions_a=preds_text,
    predictions_b=preds_image,
    min_agreement=0.7,
)
assert result.passed
print(f"Agreement: {result.details['agreement_rate']:.2%}")  # 80%
print(f"Disagreements at indices: {result.details['disagreements']}")  # [2]
```

### With Numeric Predictions

```python
import numpy as np

# Probabilities from two modalities
scores_audio = np.array([0.9, 0.1, 0.8, 0.7])
scores_text = np.array([0.9, 0.1, 0.8, 0.7])

# Convert to labels for comparison
labels_audio = (scores_audio > 0.5).astype(int)
labels_text = (scores_text > 0.5).astype(int)

result = assert_cross_modal_consistency(
    predictions_a=labels_audio,
    predictions_b=labels_text,
    min_agreement=0.9,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `predictions_a` | `np.ndarray \| list` | *(required)* | Predictions from modality A |
| `predictions_b` | `np.ndarray \| list` | *(required)* | Predictions from modality B (same length) |
| `min_agreement` | `float` | `0.8` | Minimum fraction of samples that must agree |

Returns `TestResult` (name: `domains.multimodal.cross_modal_consistency`) with details:

| Detail Key | Type | Description |
|------------|------|-------------|
| `agreement_rate` | `float` | Fraction of samples where both modalities agree |
| `min_agreement` | `float` | The threshold that was required |
| `n_agreed` | `int` | Number of agreeing predictions |
| `n_total` | `int` | Total number of predictions compared |
| `disagreements` | `list[int]` | Indices where the modalities disagree |

---

## Why RL Testing

Reinforcement learning trains agents by trial-and-error using reward signals. Unlike supervised learning where labels are fixed, RL reward functions are hand-designed and notoriously buggy. A broken reward function wastes GPU-days of training on an agent that learns the wrong behavior (reward hacking) or learns nothing at all.

### The Two Most Common RL Bugs

1. **Unbounded rewards**: A reward that shoots to infinity (or negative infinity) destabilizes training. Gradient updates become enormous, weights explode, and the agent diverges. This is the RL equivalent of a NaN loss in supervised learning.

2. **Low cumulative reward**: If the agent finishes episodes with low total reward, it is not solving the task. This is the RL equivalent of low accuracy -- the agent has not learned useful behavior. Possible causes:
   - The reward function is too sparse (agent never discovers reward)
   - The environment is broken (agent cannot reach goal states)
   - The policy has collapsed to a single action

Both assertions work on raw reward arrays from episodes, requiring no special RL framework. They integrate into CI pipelines to gate model promotions.

---

## `assert_reward_bounded`

Assert that all reward values fall within specified bounds.

**Module:** `mltk.domains.rl`

### Example

```python
import numpy as np
from mltk.domains.rl import assert_reward_bounded

# Rewards from a training episode
rewards = np.array([0.1, 0.5, -0.2, 0.8, 0.3, -0.1, 0.6])

result = assert_reward_bounded(
    rewards=rewards,
    min_reward=-1.0,
    max_reward=1.0,
)
assert result.passed
print(f"Actual range: [{result.details['actual_min']:.2f}, {result.details['actual_max']:.2f}]")
```

### Catching Unbounded Rewards

```python
# Bug: reward function produces extreme values
buggy_rewards = np.array([0.1, 0.5, 1e8, -0.2, 0.3])

result = assert_reward_bounded(
    rewards=buggy_rewards,
    min_reward=-10.0,
    max_reward=10.0,
)
assert not result.passed
print(f"Violations: {result.details['n_violations']}")  # 1
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rewards` | `np.ndarray \| list` | *(required)* | Array of reward values from one or more episodes |
| `min_reward` | `float \| None` | `None` | Minimum allowed reward (inclusive). `None` to skip lower bound. |
| `max_reward` | `float \| None` | `None` | Maximum allowed reward (inclusive). `None` to skip upper bound. |

At least one of `min_reward` or `max_reward` must be provided.

Returns `TestResult` (name: `domains.rl.reward_bounded`) with details:

| Detail Key | Type | Description |
|------------|------|-------------|
| `actual_min` | `float` | Minimum reward observed |
| `actual_max` | `float` | Maximum reward observed |
| `n_violations` | `int` | Number of rewards outside bounds |
| `n_total` | `int` | Total number of reward values |

---

## `assert_cumulative_reward`

Assert that the total reward in an episode meets a minimum threshold.

**Module:** `mltk.domains.rl`

### Example

```python
import numpy as np
from mltk.domains.rl import assert_cumulative_reward

# Agent earns rewards over 100 steps
rewards = np.array([1.0, 2.0, 1.5, 0.5, 3.0, 1.0, 2.0, 0.5, 1.5, 2.0])

result = assert_cumulative_reward(
    rewards=rewards,
    min_cumulative=10.0,
)
assert result.passed
print(f"Total reward: {result.details['cumulative_reward']:.1f}")  # 15.0
print(f"Mean per step: {result.details['mean_reward']:.2f}")       # 1.50
```

### Detecting a Non-Learning Agent

```python
# Agent barely earns any reward -- not solving the task
weak_rewards = np.array([0.01, 0.0, 0.02, 0.0, 0.01, 0.0])

result = assert_cumulative_reward(
    rewards=weak_rewards,
    min_cumulative=1.0,
)
assert not result.passed
# cumulative_reward = 0.04, far below the 1.0 threshold
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rewards` | `np.ndarray \| list` | *(required)* | Per-step reward values from an episode |
| `min_cumulative` | `float` | *(required)* | Minimum required sum of rewards |

Returns `TestResult` (name: `domains.rl.cumulative_reward`) with details:

| Detail Key | Type | Description |
|------------|------|-------------|
| `cumulative_reward` | `float` | Sum of all rewards in the episode |
| `min_cumulative` | `float` | The threshold that was required |
| `n_steps` | `int` | Number of steps in the episode |
| `mean_reward` | `float` | Average reward per step |

---

## pytest Integration

### Multimodal Testing in CI

```python
import pytest
import numpy as np
from mltk.domains.multimodal import (
    assert_image_text_alignment,
    assert_cross_modal_consistency,
)

def test_clip_alignment_after_finetuning(finetuned_clip, eval_pairs):
    """CLIP alignment must survive fine-tuning."""
    image_embs = finetuned_clip.encode_image(eval_pairs.images)
    text_embs = finetuned_clip.encode_text(eval_pairs.captions)

    result = assert_image_text_alignment(
        image_embs, text_embs, min_cosine=0.6
    )
    assert result.passed, (
        f"Alignment degraded after fine-tuning: "
        f"avg_cosine={result.details['avg_cosine']:.4f}"
    )

def test_medical_cross_modal_consistency(text_model, image_model, cases):
    """Text and image diagnoses must agree."""
    text_preds = text_model.predict(cases.reports)
    image_preds = image_model.predict(cases.scans)

    result = assert_cross_modal_consistency(
        text_preds, image_preds, min_agreement=0.95
    )
    assert result.passed, (
        f"Modality disagreement on {len(result.details['disagreements'])} cases"
    )
```

### RL Testing in CI

```python
from mltk.domains.rl import assert_reward_bounded, assert_cumulative_reward

def test_reward_function_sanity(env, trained_agent):
    """Reward function must be bounded and agent must solve the task."""
    obs = env.reset()
    rewards = []
    for _ in range(1000):
        action = trained_agent.predict(obs)
        obs, reward, done, _ = env.step(action)
        rewards.append(reward)
        if done:
            break

    # Check bounds
    bounded = assert_reward_bounded(rewards, min_reward=-10, max_reward=10)
    assert bounded.passed, f"Unbounded reward: [{bounded.details['actual_min']}, {bounded.details['actual_max']}]"

    # Check cumulative performance
    cumulative = assert_cumulative_reward(rewards, min_cumulative=50.0)
    assert cumulative.passed, f"Low reward: {cumulative.details['cumulative_reward']}"
```
