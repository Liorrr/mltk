# Multimodal LLM Evaluation

A vision-language model that confidently describes objects
not present in an image is broken. A VQA system that
misreads the text on a sign answers factual questions with
invented data. An image-generating pipeline that produces
outputs semantically unrelated to the prompt has a
regression that no text-only test can catch.

These failures are qualitatively different from text
hallucination. A text model hallucinates facts from
training data; a VLM hallucinates objects it never saw
in the input image. POPE (Li et al., NeurIPS 2023) proved
this is systematic -- models hallucinate statistically
frequent objects even when those objects are absent from
the scene.

mltk closes the multimodal evaluation gap with
pytest-native assertions for the LLM-as-Judge path
(v0.8.0) and a numerical/embedding path (v0.9.0). The
LLM-as-Judge assertions cover image-text alignment,
document coherence, image utility, and VQA accuracy.
v0.9.0 adds four numerical assertions: CLIPScore,
object hallucination probing (POPE), SSIM edit
preservation, and OCR accuracy. No competitor provides
any of these as pytest-native CI assertions.

**Module:** `mltk.domains.multimodal`

**ML Lifecycle Stage:** VLM evaluation / image generation
QA / document AI testing / CI gate

**Bugs caught:**

- VLM responses describing objects or attributes not
  present in the input image (hallucination)
- Generated captions semantically inconsistent with the
  source image (alignment failure)
- Multi-image documents where the images do not support
  the surrounding text (coherence failure)
- Illustrations that add no information value to the
  document (helpfulness failure)
- VQA systems answering factual questions incorrectly
  (accuracy regression)

---

## Why Multimodal Evaluation

Text-only evaluation misses an entire class of production
failures in systems that process or generate images.

**The VLM hallucination problem.** Hallucination in
vision-language models is not the same as text
hallucination. A text LLM hallucinates facts it saw
during training. A VLM hallucinates objects based on
statistical co-occurrence in training data -- if "dog"
appears near "park bench" frequently, the model predicts
"dog" even when no dog is in the image. POPE (Li et al.,
NeurIPS 2023) demonstrated this systematically using
binary yes/no probes across three sampling strategies.
Under adversarial probing, state-of-the-art VLMs
hallucinate at 20-40% rates on common object categories.

**Image-text alignment as a CI regression gate.** Whether
a model generates images from prompts (text-to-image) or
generates text from images (image captioning), alignment
between the image and text is the core quality metric.
CLIPScore (Hessel et al., EMNLP 2021) is the standard
reference-free metric: fast, numerically comparable
across model versions, and deployable as a CI gate without
human review. v0.8.0 provides the LLM-as-Judge proxy
for this; v0.9.0 adds the numerical CLIPScore path
(`assert_clip_score`) with a zero-dependency embeddings
mode and an optional live CLIP model path.

**The competitor gap.** DeepEval provides 7 multimodal
metrics, all using LLM-as-Judge (GPT-4V required). No
open-source tool provides a pytest-native VQA accuracy
assertion. No competitor provides CLIPScore,
edit-preservation SSIM, or POPE-style object hallucination
probing as CI assertions. mltk v0.8.0 establishes the
LLM-judge path; v0.9.0 delivers the differentiated
numerical path that DeepEval entirely lacks.

---

## Image Input

All assertions accept an `ImageInput` type defined in
`mltk.domains.multimodal`:

```python
ImageInput = str | Path | bytes
```

Supported variants:

| Variant | Example | Notes |
|---------|---------|-------|
| `str` file path | `"/data/img.png"` | Read as binary |
| `pathlib.Path` | `Path("img.png")` | Same as str path |
| `bytes` | `response.content` | In-memory image data |

**PIL.Image objects are NOT accepted directly** to avoid
coupling the public API to Pillow. Convert PIL images to
bytes first with `image.tobytes()` or by saving to a
`BytesIO` buffer.

**Base64 strings are not accepted directly.** Decode to
`bytes` first: `base64.b64decode(b64_str)`. This avoids
ambiguity between base64 strings and file paths that would
lead to silent misdetection bugs.

### `image_description` Escape Hatch

Every assertion that accepts an image also accepts an
optional `image_description: str | None` parameter. When provided, the assertion uses this text
description in the judge prompt instead of loading and
encoding the image. Pillow is not imported.

This enables two usage patterns:

**Pattern 1 -- VLM judge (raw image):** Pass the image
directly. The assertion base64-encodes it into the
evaluation prompt for a VLM-capable `judge_fn`.

```python
assert_prompt_faithfulness(
    prompt="a red bicycle in a park",
    image="/data/bicycle.jpg",
    judge_fn=my_vlm_judge,  # GPT-4o, Claude, Gemini
)
```

**Pattern 2 -- text judge (zero-dep):** Pre-describe
the image using any VLM, then pass the description with
a text-only judge. No Pillow required.

```python
description = my_vlm.describe("/data/bicycle.jpg")
assert_prompt_faithfulness(
    prompt="a red bicycle in a park",
    image="/data/bicycle.jpg",  # not loaded if description given
    judge_fn=my_text_judge,
    image_description=description,
)
```

Pattern 2 is the zero-dependency path. It works in
environments where Pillow is unavailable and with judges
that do not accept image inputs.

### Lazy Import

Pillow is an optional dependency. It is imported lazily
inside each assertion when an image must be loaded. A
clear `ImportError` with the install command is raised if
Pillow is not available and `image_description` is not
provided.

```
ImportError: Pillow is required for multimodal assertions.
Install with: pip install mltk[multimodal]
```

---

## Assertions

### assert_prompt_faithfulness

Verifies semantic consistency between a text prompt and
the image it produced or is paired with. Catches the
most common text-to-image failure: the model generates
an image that does not match the prompt.

```python
from mltk.domains.multimodal import (
    assert_prompt_faithfulness,
)

assert_prompt_faithfulness(
    prompt: str,
    image: ImageInput | None,
    judge_fn: Callable[[str], str],
    min_score: float = 0.7,
    image_description: str | None = None,
) -> TestResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | required | The text prompt the image should represent. |
| `image` | `ImageInput \| None` | required | The image to evaluate, or `None` when using `image_description`. |
| `judge_fn` | `Callable[[str], str]` | required | LLM judge. Receives the full eval prompt, returns a response containing a score. |
| `min_score` | `float` | `0.7` | Minimum faithfulness score (0.0-1.0). |
| `image_description` | `str \| None` | `None` | Pre-computed text description. If provided, the `image` parameter is ignored and Pillow is not imported. |

**What it catches:** Mismatched colors, wrong subject
matter, missing objects described in the prompt, scene
context violations.

**Research basis:** FaithScore (Jing et al., EMNLP
Findings 2024, arXiv:2311.01477) -- claim-level
faithfulness verification for VLM responses. TextToImage
scoring component from MLLM-Eval (DeepEval, 2024).

```python
import os

def openai_judge(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content

def test_image_faithfulness():
    assert_prompt_faithfulness(
        prompt="A golden retriever running on a beach",
        image="outputs/generated_00042.png",
        judge_fn=openai_judge,
        min_score=0.7,
    )
```

---

### assert_image_coherence

Tests whether images in a document or multi-modal
response are coherent with the surrounding text. Catches
images that contradict or are unrelated to the document
content.

```python
from mltk.domains.multimodal import assert_image_coherence

assert_image_coherence(
    text: str,
    image: ImageInput | None,
    judge_fn: Callable[[str], str],
    min_score: float = 0.7,
    image_description: str | None = None,
) -> TestResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | required | The text context that the image accompanies. |
| `image` | `ImageInput \| None` | required | Image source (file path, bytes) or `None` when using `image_description`. |
| `judge_fn` | `Callable[[str], str]` | required | LLM judge callable. |
| `min_score` | `float` | `0.7` | Minimum coherence score. |
| `image_description` | `str \| None` | `None` | Pre-computed text description of the image. If provided, the `image` parameter is ignored. |

**What it catches:** Images inserted from wrong documents,
charts that describe different data than the surrounding
text, stock photos semantically disconnected from content.

**Research basis:** ImageCoherence metric (DeepEval
MLLM-Eval framework, 2024). Multimodal document
understanding literature (MMMU, Yue et al., CVPR 2024,
arXiv:2311.16502).

```python
def test_report_coherence():
    assert_image_coherence(
        text=(
            "Q3 revenue grew 23% YoY driven by "
            "enterprise sales expansion."
        ),
        image="charts/q3_revenue.png",
        judge_fn=openai_judge,
        min_score=0.75,
    )
```

---

### assert_image_helpfulness

Tests whether images improve comprehension of the
document. An image is helpful if a reader would
understand the document better with it than without it.
Catches decorative filler images and stock photos that
add no informational value.

```python
from mltk.domains.multimodal import (
    assert_image_helpfulness,
)

assert_image_helpfulness(
    question: str,
    image: ImageInput | None,
    answer: str,
    judge_fn: Callable[[str], str],
    min_score: float = 0.7,
    image_description: str | None = None,
) -> TestResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `question` | `str` | required | The question being asked about or with the image. |
| `image` | `ImageInput \| None` | required | Image source (path, bytes) or `None` if using `image_description`. |
| `answer` | `str` | required | The answer produced (by a VLM or human). |
| `judge_fn` | `Callable[[str], str]` | required | LLM judge callable. |
| `min_score` | `float` | `0.7` | Minimum helpfulness score. |
| `image_description` | `str \| None` | `None` | Pre-computed text description of the image. If provided, the `image` parameter is ignored. |

**What it catches:** Generic stock photos serving as
visual filler, diagrams that duplicate text without
adding structure, images that do not match the
document's topic.

**Research basis:** ImageHelpfulness metric (DeepEval
MLLM-Eval framework, 2024). MM-Vet v2 (2024) -- 6
integrated VL skill dimensions including information
utility.

```python
def test_tutorial_images():
    assert_image_helpfulness(
        question="How do I install the package?",
        image="screenshots/install_terminal.png",
        answer="Run: pip install mltk",
        judge_fn=openai_judge,
        min_score=0.8,
    )
```

---

### assert_vqa_accuracy

Evaluates visual question answering (VQA) accuracy by
comparing an expected answer against the model's actual
answer for a single question. Supports two modes:
normalized exact match (no dependencies) and LLM-judge
semantic scoring. This is the only pytest-native VQA
accuracy assertion -- no competitor provides this.

```python
from mltk.domains.multimodal import assert_vqa_accuracy

assert_vqa_accuracy(
    question: str,
    image: ImageInput | None,
    expected_answer: str,
    actual_answer: str,
    judge_fn: Callable[[str], str] | None = None,
    min_score: float = 0.7,
    image_description: str | None = None,
) -> TestResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `question` | `str` | required | The question asked about the image. |
| `image` | `ImageInput \| None` | required | Image source (path, bytes) or `None` if using `image_description` or text-only matching. |
| `expected_answer` | `str` | required | The ground-truth correct answer. |
| `actual_answer` | `str` | required | The model's answer to evaluate. |
| `judge_fn` | `Callable[[str], str] \| None` | `None` | Optional callable for semantic comparison. If `None`, uses normalized exact match. |
| `min_score` | `float` | `0.7` | Minimum score to pass. For exact match: 1.0 if match, 0.0 if not. |
| `image_description` | `str \| None` | `None` | Pre-computed text description of the image. |

**Evaluation modes:**

| Mode | Selected by | Description |
|------|------------|-------------|
| **Exact match** | `judge_fn=None` (default) | Case-insensitive, whitespace-stripped comparison. Fast but brittle. Use for short, unambiguous answers (colors, counts, yes/no). |
| **LLM judge** | `judge_fn` provided | LLM scores semantic equivalence between expected and actual answers. Handles paraphrasing and different wording for the same meaning. |

**What it catches:** VQA accuracy regressions across
model versions, systematic failures on specific question
types (color, count, spatial relation), object naming
inconsistencies.

**Note:** This assertion evaluates a single question-
answer pair. Run your VLM separately against the image
and pass its output here. mltk evaluates the output,
not the VLM call itself. For batch evaluation, call
this assertion in a loop or use `@pytest.mark.parametrize`.

**Research basis:** VQA v2 evaluation protocol (Goyal
et al., CVPR 2017). VQAScore (Lin et al., CMU/Meta,
ECCV 2024, arXiv:2404.01291) for the judge variant.

```python
def test_vqa_exact():
    """Exact match mode -- no LLM needed."""
    assert_vqa_accuracy(
        question="How many dogs?",
        image=None,
        expected_answer="2",
        actual_answer="2",
    )

def test_vqa_system():
    """LLM-judge mode for semantic comparison."""
    assert_vqa_accuracy(
        question="What color is the bicycle?",
        image="test_image.jpg",
        expected_answer="red",
        actual_answer="The bicycle is red.",
        judge_fn=openai_judge,
        min_score=0.8,
    )
```

---

## judge_fn Pattern

All LLM-judge assertions use the same `Callable[[str],
str]` signature established across the mltk codebase.
The assertion constructs the full evaluation prompt
(including image content or description) and passes it
as a single string. The callable sends it to an LLM and
returns the response. mltk parses the score from the
response.

This is the same signature used in `assert_llm_judge_score`,
`assert_faithfulness`, `assert_context_relevancy`, and all
other judge-based assertions. One callable works across
the entire mltk library.

**With OpenAI:**

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def openai_judge(prompt: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
    )
    return resp.choices[0].message.content
```

**With Ollama (local, no API key):**

```python
import urllib.request, json

def ollama_judge(prompt: str) -> str:
    payload = json.dumps({
        "model": "llava:13b",
        "prompt": prompt,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())["response"]
```

**With any OpenAI-compatible endpoint:**

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-...",
    base_url="https://api.together.xyz/v1",
)

def together_judge(prompt: str) -> str:
    resp = client.chat.completions.create(
        model="meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content
```

---

## Installation

The multimodal assertions are part of the mltk core.
`assert_vqa_accuracy` with the default exact match mode
requires no additional dependencies.

For image-loading support (Pillow) and SSIM-based
edit preservation (scikit-image):

```bash
pip install mltk[multimodal]
```

This installs `Pillow>=9.0` and `scikit-image>=0.20`.
Pillow is required when passing images directly without
an `image_description`. scikit-image is required by
`assert_edit_preservation` when using `method="ssim"`.
The `method="pixel_diff"` fallback requires only Pillow.

For the live CLIPScore path (downloads ~340MB model):

```bash
pip install mltk[clip]
```

This installs `open-clip-torch>=2.24` and `Pillow>=9.0`.
The CLIP ViT-B/32 checkpoint downloads on first use and
is cached via the HuggingFace hub. The zero-dependency
embeddings path (`assert_clip_score` with pre-computed
embeddings) requires no extra install.

**Summary of extras:**

| Extra | Installs | Enables |
|-------|----------|---------|
| `mltk[multimodal]` | Pillow, scikit-image | All v1+v2 assertions with raw image input |
| `mltk[clip]` | open-clip-torch, Pillow | `assert_clip_score` live CLIP model path |
| `mltk[classifier]` | transformers, torch | Planned `assert_no_nsfw` |

All assertions work without extras when `image_description`
is provided or pre-computed embeddings are passed.

---

## Competitive Comparison

| Capability | mltk | DeepEval | Promptfoo | Inspect AI | Giskard |
|-----------|------|----------|-----------|------------|---------|
| Pytest-native multimodal | **Yes** | No (custom runner) | No | No | No |
| LLM-judge assertions | **4** | 7 (GPT-4V only) | 0 | 0 | 0 |
| VQA accuracy assertion | **Yes (first-mover)** | No | No | No | No |
| Numerical CLIPScore | **Yes (v0.9.0)** | No | No | No | No |
| SSIM edit preservation | **Yes (v0.9.0)** | No | No | No | No |
| POPE hallucination probe | **Yes (v0.9.0)** | No | No | No | No |
| OCR accuracy (CER/WER) | **Yes (v0.9.0)** | No | No | No | No |
| NSFW safety gate | Planned | No | No | No | No |
| Zero-dep path | **Yes** | No | No | No | No |
| Offline / air-gapped | **Yes (text judge path)** | No | No | No | No |

**The gap DeepEval cannot close:** All 7 DeepEval
multimodal metrics require GPT-4V -- they are expensive,
require an OpenAI API key in CI, and cannot run in
air-gapped or cost-constrained environments. DeepEval has
no numerical metrics: no CLIPScore, no SSIM, no POPE. The
numerical path is mltk's first-mover differentiator.

**The gap no other tool has:** `assert_vqa_accuracy` is
the first pytest-native VQA accuracy assertion. No
existing open-source testing tool exposes VQA pass/fail
as a CI assertion.

---

## v2 Assertions — Numerical Path

v0.9.0 ships four numerical assertions that require no
LLM judge. These live in two new submodules:

- `mltk.domains.multimodal.metrics` —
  `assert_clip_score`, `assert_edit_preservation`
- `mltk.domains.multimodal.hallucination` —
  `assert_object_hallucination`
- `mltk.domains.multimodal.vlm` (extended) —
  `assert_ocr_accuracy`

All four are re-exported from `mltk.domains.multimodal`.

| Assertion | Method | Extra needed |
|-----------|--------|-------------|
| `assert_clip_score` | CLIP cosine similarity | `mltk[clip]` (live path only) |
| `assert_object_hallucination` | POPE binary probing | None |
| `assert_edit_preservation` | SSIM or pixel diff | `mltk[multimodal]` (ssim path) |
| `assert_ocr_accuracy` | CER / WER | None (pure Python) |

---

### assert_clip_score

Measures image-text semantic similarity using CLIP cosine
distance. Catches alignment regressions between model
versions without an LLM judge.

```python
from mltk.domains.multimodal import assert_clip_score
import numpy as np

assert_clip_score(
    image: ImageInput | None = None,
    text: str | None = None,
    image_embedding: np.ndarray | None = None,
    text_embedding: np.ndarray | None = None,
    min_score: float = 0.25,
    model_name: str = "ViT-B-32",
) -> TestResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | `ImageInput \| None` | `None` | Image for live CLIP encoding. |
| `text` | `str \| None` | `None` | Text for live CLIP encoding. |
| `image_embedding` | `np.ndarray \| None` | `None` | Pre-computed image embedding (zero-dep path). |
| `text_embedding` | `np.ndarray \| None` | `None` | Pre-computed text embedding (zero-dep path). |
| `min_score` | `float` | `0.25` | Minimum cosine similarity to pass. |
| `model_name` | `str` | `"ViT-B-32"` | CLIP model name for live path. |

**Two paths:**

- **Embeddings path** (`image_embedding` + `text_embedding`
  provided): Computes cosine similarity with pure numpy.
  Zero dependencies beyond mltk itself. Use when you
  already have embeddings from your own CLIP pipeline.

- **Live path** (`image` + `text` provided): Lazy-imports
  `open-clip-torch`, encodes both inputs with ViT-B/32,
  and computes cosine similarity. Requires `mltk[clip]`.

Raise `ValueError` if neither pair is complete.

**Threshold guidance:** ViT-B/32 cosine similarities are
lower than typical embedding spaces. Random pairs score
near 0.0-0.15; matched pairs 0.25-0.35; strong semantic
matches 0.35+. The default `min_score` of 0.25 is the
recommended CI gate for ViT-B/32.

**CLIPScore limitations:** ViT-B/32 is compositionally
blind — it fails on attribute binding ("red cube left of
blue sphere"), spatial relations, and negation. For
compositional prompts, use `assert_vqa_accuracy` with a
VQAScore-style judge instead. CLIPScore is a fast
regression gate, not a compositional correctness metric.

**Citation:** CLIPScore (Hessel et al., EMNLP 2021,
arXiv:2104.08718)

```python
# Zero-dep path: pre-computed embeddings
import numpy as np

image_emb = my_clip_pipeline.encode_image("photo.jpg")
text_emb = my_clip_pipeline.encode_text("a dog on grass")

assert_clip_score(
    image_embedding=image_emb,
    text_embedding=text_emb,
    min_score=0.25,
)

# Live path: raw inputs (requires mltk[clip])
assert_clip_score(
    image="outputs/generated.png",
    text="a golden retriever on a beach",
    min_score=0.28,
)
```

---

### assert_object_hallucination

POPE-style binary probing for object hallucination in
VLMs. Asks yes/no questions about object presence and
measures the rate at which the model falsely claims
absent objects exist.

```python
from mltk.domains.multimodal import (
    assert_object_hallucination,
)
from collections.abc import Callable

assert_object_hallucination(
    vqa_fn: Callable[[str, ImageInput | None, str | None], str],
    image: ImageInput | None,
    objects_present: list[str],
    objects_absent: list[str],
    threshold: float = 0.8,
    image_description: str | None = None,
) -> TestResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vqa_fn` | `Callable[[str, ImageInput \| None, str \| None], str]` | required | Wraps your VLM. Receives `(question, image, description)`, returns string answer. |
| `image` | `ImageInput \| None` | required | The image to probe, or `None` when using `image_description`. |
| `objects_present` | `list[str]` | required | Objects known to be in the image. Expected answer: "yes". |
| `objects_absent` | `list[str]` | required | Objects known to NOT be in the image. Expected answer: "no". |
| `threshold` | `float` | `0.8` | Minimum overall accuracy to pass (accuracy = correct answers / total questions). |
| `image_description` | `str \| None` | `None` | Optional text description of the image. Passed to `vqa_fn` as the third argument. |

**How it works:** For each object in `objects_present`,
the assertion calls `vqa_fn("Is there a [object] in the
image?", image, image_description)` and expects "yes".
For each object in `objects_absent`, it expects "no".
Ambiguous answers (neither "yes" nor "no" detected) are
treated as incorrect -- this is the conservative choice
for safety.

`accuracy = correct_answers / total_questions`

**TestResult fields:** `passed`, `score` (overall POPE
accuracy = correct answers / total probes),
`hallucination_rate`, `false_positives`,
`false_negatives`, `total_present`, `total_absent`,
`per_object` (list of per-object results).

**mltk does not bundle COCO object lists.** Users supply
`objects_absent` — for adversarial probing, choose objects
that statistically co-occur with `objects_present`.

**Citation:** POPE (Li et al., NeurIPS 2023,
arXiv:2305.10355)

```python
def ask_vlm(question: str, image: str, description: str | None) -> str:
    response = my_vlm_client.chat(
        image=image, prompt=question
    )
    return response.text

assert_object_hallucination(
    vqa_fn=ask_vlm,
    image="living_room.jpg",
    objects_present=["sofa", "television"],
    # adversarial: co-occur with sofa/tv in training data
    objects_absent=["dog", "laptop", "book"],
    threshold=0.8,
)
```

---

### assert_edit_preservation

Verifies structural similarity between an original image
and an edited version. Catches image editing pipelines
that alter unintended regions or degrade overall
structure.

```python
from mltk.domains.multimodal import (
    assert_edit_preservation,
)

assert_edit_preservation(
    original: ImageInput,
    edited: ImageInput,
    method: str = "ssim",
    threshold: float = 0.8,
    max_image_size: int = 512,
) -> TestResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `original` | `ImageInput` | required | The unedited reference image. |
| `edited` | `ImageInput` | required | The edited image to evaluate. |
| `method` | `str` | `"ssim"` | `"ssim"` (scikit-image, spec-compliant) or `"pixel_diff"` (pure numpy, lighter). |
| `threshold` | `float` | `0.8` | Minimum similarity score to pass. SSIM range is [-1, 1] (typically [0, 1]); 0.85+ is typical for well-preserved edits. |
| `max_image_size` | `int` | `512` | Resize images to this maximum dimension before comparison to control memory use. |

**Two methods:**

- **`method="ssim"`** (default): Uses
  `skimage.metrics.structural_similarity` — 11x11
  Gaussian windowing, correct boundary conditions,
  multichannel RGB support. Requires `mltk[multimodal]`
  (scikit-image). Spec-compliant per Wang et al. (2004).

- **`method="pixel_diff"`**: Pure numpy. Computes
  `1.0 - mean(abs(a - b)) / 255.0`. Weaker signal than
  SSIM but zero extra dependencies beyond Pillow.
  Use in environments where scikit-image cannot be
  installed.

Pillow is required for both paths (image loading and
resizing). The `max_image_size` parameter prevents
memory errors on large images — images are resized to
fit within this dimension before comparison.

**Citation:** SSIM (Wang et al., IEEE Trans. Image
Processing, 2004)

```python
assert_edit_preservation(
    original="before_inpaint.png",
    edited="after_inpaint.png",
    method="ssim",
    threshold=0.85,
    max_image_size=512,
)

# Lightweight fallback (no scikit-image needed)
assert_edit_preservation(
    original="before.png",
    edited="after.png",
    method="pixel_diff",
    threshold=0.90,
)
```

---

### assert_ocr_accuracy

Measures OCR output quality using Character Error Rate
(CER) or Word Error Rate (WER). Zero external
dependencies — uses a pure Python Levenshtein
implementation.

```python
from mltk.domains.multimodal import assert_ocr_accuracy

assert_ocr_accuracy(
    expected_text: str,
    actual_text: str,
    method: str = "cer",
    threshold: float = 0.1,
) -> TestResult
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `expected_text` | `str` | required | Ground-truth text (the correct text). |
| `actual_text` | `str` | required | OCR-extracted text to evaluate. |
| `method` | `str` | `"cer"` | `"cer"` (Character Error Rate) or `"wer"` (Word Error Rate). |
| `threshold` | `float` | `0.1` | Maximum error rate allowed before failing (default 10%). |

**Metrics:**

- **CER** = `edit_distance(chars) / len(reference)` —
  sensitive to individual character errors. Correct for
  OCR evaluation (a single transposed character in a
  number is a critical error).

- **WER** = `edit_distance(words) / len(ref_words)` —
  forgiving of minor character errors but sensitive to
  word splits and insertions. Useful for transcription.

Both metrics can exceed 1.0 if the actual text is much
longer than the expected text — this is correct behavior.

**No external dependencies.** The Levenshtein distance
is computed in pure Python. This assertion works in any
environment, including offline CI and air-gapped systems.

```python
# Test a VLM-based invoice reader
assert_ocr_accuracy(
    expected_text="Invoice #1234",     # known correct
    actual_text="Iuvoice #1234",       # VLM output
    method="cer",
    threshold=0.1,
)

# Word-level transcription check
assert_ocr_accuracy(
    expected_text="the quick brown fox",
    actual_text="the quick bron fox",
    method="wer",
    threshold=0.10,
)
```

---

## Research Citations

| Paper | Authors | Venue | Link |
|-------|---------|-------|------|
| CLIPScore | Hessel et al. | EMNLP 2021 | arXiv:2104.08718 |
| POPE | Li et al. | NeurIPS 2023 | arXiv:2305.10355 |
| H-POPE | Cui et al. | NeurIPS 2024 | arXiv:2411.04077 |
| SSIM | Wang et al. | IEEE Trans. Image Processing, 2004 | doi:10.1109/TIP.2003.819861 |
| FaithScore | Jing et al. | EMNLP Findings 2024 | arXiv:2311.01477 |
| VQAScore | Lin et al. (CMU/Meta) | ECCV 2024 | arXiv:2404.01291 |
| T2I-CompBench++ | Huang et al. | TPAMI 2024 | arXiv:2307.06350 |
| MMMU | Yue et al. | CVPR 2024 | arXiv:2311.16502 |
| THRONE | Kaul et al. | CVPR 2024 | CVPR 2024 proceedings |

Full research brief and rejected alternatives:
`docs/research/multimodal-evaluation-research.md`.
Feasibility verification (CLIP model sizes, Pillow
status, SSIM implementation options):
`docs/research/multimodal-feasibility-check.md`.
v2 API verification (exact code patterns, formula
constants, POPE protocol):
`docs/research/multimodal-v2-verification.md`.
Architecture decisions (subpackage layout, ImageInput
type, judge_fn signature, namespace, sprint scope):
`audit/s78-multimodal-architect-review.md`.
v2 architect quick check (file placement, CLIP API,
SSIM API, scikit-image version):
`audit/s79-architect-quick-check.md`.
