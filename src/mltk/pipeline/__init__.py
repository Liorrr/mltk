"""Pipeline testing — E2E validation, reproducibility, ONNX export."""

from mltk.pipeline.compatibility import StageSpec, assert_pipeline_stages_compatible
from mltk.pipeline.e2e import assert_pipeline
from mltk.pipeline.onnx import assert_onnx_valid
from mltk.pipeline.reproducibility import assert_checksum, assert_reproducible
from mltk.pipeline.resilience import (
    DEFAULT_FAULTS,
    apply_fault,
    assert_pipeline_resilient,
)

__all__ = [
    "assert_reproducible",
    "assert_checksum",
    "assert_pipeline",
    "assert_onnx_valid",
    # stage compatibility
    "StageSpec",
    "assert_pipeline_stages_compatible",
    # resilience / chaos engineering
    "DEFAULT_FAULTS",
    "apply_fault",
    "assert_pipeline_resilient",
]
