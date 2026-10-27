# cv

## Overview
Computer-vision assertions: detection, classification, tracking, video, and face-recognition metrics. Each `assert_*` returns a `TestResult` (built on `mltk.core.assertion`).

## Interface
- `classification.py`: `assert_topk_accuracy`
- `detection.py`: `compute_iou`, `assert_iou`, `assert_map`
- `face.py`: `assert_face_far`
- `tracking.py`: `assert_mota`, `assert_motp`, `assert_idf1`
- `video.py`: `assert_frame_accuracy`, `assert_temporal_consistency`
- `__init__.py` re-exports all of the above via `__all__`.

## Invariants
- Public assertions are named `assert_*` and return `TestResult`; a failed check at `Severity.CRITICAL` (the default) raises `MltkAssertionError` via `mltk.core.assert_true`.
- Inputs are validated (via `mltk.core.validation`) before scoring.

## Side Effects
- None; pure computation on in-memory inputs.

## Verification
- Run: `python -m pytest tests/test_domains -q`
