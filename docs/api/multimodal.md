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
(v0.8.0) and a numerical/embedding path coming in a
future release. The LLM-as-Judge assertions cover
image-text alignment, document coherence, image utility,
and VQA accuracy. No competitor provides VQA accuracy as
a pytest-native assertion.

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
for this; a future release adds the numerical CLIPScore
path.

**The competitor gap.** DeepEval provides 7 multimodal
metrics, all using LLM-as-Judge (GPT-4V required). No
open-source tool provides a pytest-native VQA accuracy
assertion. No competitor provides CLIPScore,
edit-preservation SSIM, POPE-style object hallucination
probing, or NSFW safety gating as CI assertions. mltk
v0.8.0 establishes the LLM-judge path; a future release
delivers the differentiated numerical path that DeepEval
entirely lacks.

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

For image-loading support (Pillow):

```bash
pip install mltk[multimodal]
```

This installs `Pillow>=9.0` and `scikit-image>=0.21`.
Pillow is required when passing images directly without
an `image_description`. scikit-image is used by
`assert_edit_preservation` (future release).

For the numerical CLIPScore path (future release):

```bash
pip install mltk[clip]
```

This installs `open-clip-torch>=2.24` and `Pillow>=9.0`.
The CLIP ViT-B/32 checkpoint (~340MB) downloads on first
use and is cached via the HuggingFace hub.

**Summary of extras:**

| Extra | Installs | Enables |
|-------|----------|---------|
| `mltk[multimodal]` | Pillow, scikit-image | All assertions with raw image input; future SSIM |
| `mltk[clip]` | open-clip-torch, Pillow | Future `assert_clip_score` live path |
| `mltk[classifier]` | transformers, torch | Future `assert_no_nsfw` |

All assertions work without extras when `image_description`
is provided or pre-computed embeddings are passed.

---

## Competitive Comparison

| Capability | mltk | DeepEval | Promptfoo | Inspect AI | Giskard |
|-----------|------|----------|-----------|------------|---------|
| Pytest-native multimodal | **Yes** | No (custom runner) | No | No | No |
| LLM-judge assertions | **4** | 7 (GPT-4V only) | 0 | 0 | 0 |
| VQA accuracy assertion | **Yes (first-mover)** | No | No | No | No |
| Numerical CLIPScore | **Planned** | No | No | No | No |
| SSIM edit preservation | **Planned** | No | No | No | No |
| POPE hallucination probe | **Planned** | No | No | No | No |
| NSFW safety gate | **Planned** | No | No | No | No |
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

## Planned: Numerical Path

A future release delivers the numerical/embedding path
that DeepEval entirely lacks. All planned assertions are
first-mover in the open-source pytest-native space.

| Assertion | Method | New dep |
|-----------|--------|---------|
| `assert_clip_score` | CLIP cosine similarity | `open-clip-torch` (optional) |
| `assert_object_hallucination` | POPE-style binary probe | None (user-provided `vqa_fn`) |
| `assert_edit_preservation` | SSIM structural similarity | `scikit-image` |
| `assert_ocr_accuracy` | CER / WER edit distance | None (pure Python) |
| `assert_no_nsfw` | ViT classifier | `transformers` + `torch` |
| `assert_image_editing_score` | LLM-judge over image pair | Pillow |

`assert_clip_score` supports a zero-dependency
`method="embedding"` path when users supply pre-computed
CLIP embeddings. The `method="clip"` live path downloads
the ViT-B/32 checkpoint (~340MB) on first use.

`assert_object_hallucination` implements the POPE
protocol (Li et al., NeurIPS 2023, arXiv:2305.10355)
over a user-provided `vqa_fn`. mltk does not bundle the
COCO dataset; users supply their own probe objects. Three
sampling strategies are supported: `"random"`,
`"popular"`, and `"adversarial"`.

Note on CLIPScore limitations: ViT-B/32 CLIPScore is
fast and reference-free but compositionally blind --
it fails on attribute binding ("red cube left of blue
sphere"), spatial relationships, and negation. For
compositional prompts, VQAScore (Lin et al., ECCV 2024)
outperforms CLIPScore by 15%+ and is the recommended
metric. CLIPScore is the regression gate; VQAScore is
the composition test.

---

## Research Citations

| Paper | Authors | Venue | Link |
|-------|---------|-------|------|
| CLIPScore | Hessel et al. | EMNLP 2021 | arXiv:2104.08718 |
| POPE | Li et al. | NeurIPS 2023 | arXiv:2305.10355 |
| H-POPE | Cui et al. | NeurIPS 2024 | arXiv:2411.04077 |
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
Architecture decisions (subpackage layout, ImageInput
type, judge_fn signature, namespace, sprint scope):
`audit/s78-multimodal-architect-review.md`.
