"""Multimodal evaluation -- alignment, consistency, VLM, and VQA.

This subpackage provides assertions for evaluating multimodal AI
systems -- models that process images, text, and other modalities.

Organized by evaluation concern:

- **alignment**: Embedding-based alignment between image and text
  representations (CLIPScore-style) and cross-modal consistency.
- **metrics**: Numerical metrics -- CLIPScore (cosine similarity)
  and edit preservation (SSIM / pixel diff).
- **hallucination**: POPE-style object hallucination detection.
- **vlm**: Vision-Language Model evaluation -- image helpfulness,
  Visual Question Answering (VQA), and OCR accuracy.

All LLM-as-Judge assertions follow the same pattern as
``mltk.domains.llm.judge``: the user provides a ``judge_fn``
callable, mltk builds the prompt and parses the score.
"""

from mltk.domains.multimodal._image import (
    ImageInput,
    image_to_base64,
    load_image,
)
from mltk.domains.multimodal.alignment import (
    assert_cross_modal_consistency,
    assert_image_coherence,
    assert_image_text_alignment,
    assert_prompt_faithfulness,
)
from mltk.domains.multimodal.hallucination import (
    assert_object_hallucination,
)
from mltk.domains.multimodal.metrics import (
    assert_clip_score,
    assert_edit_preservation,
)
from mltk.domains.multimodal.vlm import (
    assert_image_helpfulness,
    assert_ocr_accuracy,
    assert_vqa_accuracy,
)

__all__ = [
    # types
    "ImageInput",
    # image utilities
    "load_image",
    "image_to_base64",
    # alignment (existing)
    "assert_image_text_alignment",
    "assert_cross_modal_consistency",
    "assert_prompt_faithfulness",
    "assert_image_coherence",
    # metrics (v2)
    "assert_clip_score",
    "assert_edit_preservation",
    # hallucination (v2)
    "assert_object_hallucination",
    # VLM evaluation (existing + v2)
    "assert_image_helpfulness",
    "assert_vqa_accuracy",
    "assert_ocr_accuracy",
]
