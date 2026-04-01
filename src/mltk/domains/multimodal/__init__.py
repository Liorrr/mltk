"""Multimodal evaluation -- alignment, consistency, VLM, and VQA.

This subpackage provides assertions for evaluating multimodal AI
systems -- models that process images, text, and other modalities.

Organized by evaluation concern:

- **alignment**: Embedding-based alignment between image and text
  representations (CLIPScore-style) and cross-modal consistency.
- **vlm**: Vision-Language Model evaluation -- image helpfulness
  and Visual Question Answering (VQA) accuracy.

All LLM-as-Judge assertions follow the same pattern as
``mltk.domains.llm.judge``: the user provides a ``judge_fn``
callable, mltk builds the prompt and parses the score.
"""

from mltk.domains.multimodal._image import ImageInput, image_to_base64, load_image
from mltk.domains.multimodal.alignment import (
    assert_cross_modal_consistency,
    assert_image_coherence,
    assert_image_text_alignment,
    assert_prompt_faithfulness,
)
from mltk.domains.multimodal.vlm import (
    assert_image_helpfulness,
    assert_vqa_accuracy,
)

__all__ = [
    # types
    "ImageInput",
    # image utilities
    "load_image",
    "image_to_base64",
    # alignment (existing + new)
    "assert_image_text_alignment",
    "assert_cross_modal_consistency",
    "assert_prompt_faithfulness",
    "assert_image_coherence",
    # VLM evaluation (new)
    "assert_image_helpfulness",
    "assert_vqa_accuracy",
]
