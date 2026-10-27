# multimodal

## Overview
Assertions for image-text systems: embedding alignment, CLIPScore, edit preservation, VLM/VQA/OCR, and object-hallucination probing. Judge-based assertions take a user-supplied `judge_fn`, as in `mltk.domains.llm.judge`.

## Interface
- `alignment.py`: `assert_image_text_alignment`, `assert_cross_modal_consistency`, `assert_prompt_faithfulness`, `assert_image_coherence`
- `metrics.py`: `assert_clip_score`, `assert_edit_preservation`
- `hallucination.py`: `assert_object_hallucination`
- `vlm.py`: `assert_image_helpfulness`, `assert_vqa_accuracy`, `assert_ocr_accuracy`
- `_image.py`: `ImageInput` (`str | Path | bytes`), `load_image`, `image_to_base64`
- `_scoring.py`: private judge-response parsing and vector math.

## Invariants
- Image arguments use the `ImageInput` type.
- Heavy dependencies (e.g. torch) are imported lazily inside functions.

## Side Effects
- `load_image` may read image files from disk.
- Calls user-supplied judge/model callables.

## Verification
- Run: `python -m pytest tests/test_domains -q`
