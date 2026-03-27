"""ONNX model validation -- verify exported models load and produce correct outputs.

Validates that an ONNX model file loads into an inference session, accepts
the provided input tensor, and (optionally) produces output within tolerance
of expected values. Catches export corruption, shape mismatches, and
numerical divergence from the source framework.

Functions:
    assert_onnx_valid -- ONNX model loads, runs inference, output matches expected
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


@timed_assertion
def assert_onnx_valid(
    model_path: str | Path,
    test_input: np.ndarray,
    expected_output: np.ndarray | None = None,
    tolerance: float = 0.01,
    severity: Severity = Severity.CRITICAL,
) -> TestResult:
    """Assert ONNX model loads, accepts input, and produces expected output.

    Lazily imports ``onnxruntime`` so the assertion is available without
    requiring onnxruntime at install time. If onnxruntime is missing, the
    test fails with an ImportError message.

    Args:
        model_path: Path to the ``.onnx`` model file.
        test_input: Input tensor for inference (must match model's expected shape/dtype).
        expected_output: Expected output tensor. ``None`` = skip output comparison.
        tolerance: Maximum allowed absolute difference per element (default 0.01).
        severity: Severity level (default CRITICAL).

    Returns:
        TestResult with inference details (output shape, max difference).

    Example:
        >>> import numpy as np
        >>> assert_onnx_valid("model.onnx", np.zeros((1, 10), dtype=np.float32))
    """
    model_path = Path(model_path)

    # Check file exists
    if not model_path.exists():
        return assert_true(
            False,
            name="pipeline.onnx_valid",
            message=f"ONNX model not found: {model_path}",
            severity=severity,
        )

    # Lazy import onnxruntime
    try:
        import onnxruntime as ort
    except ImportError:
        return assert_true(
            False,
            name="pipeline.onnx_valid",
            message="onnxruntime not installed -- pip install onnxruntime",
            severity=Severity.WARNING,
        )

    # Load model and create session
    try:
        session = ort.InferenceSession(str(model_path))
    except Exception as exc:
        return assert_true(
            False,
            name="pipeline.onnx_valid",
            message=f"Failed to load ONNX model: {type(exc).__name__}: {exc}",
            severity=severity,
        )

    # Run inference
    input_name = session.get_inputs()[0].name
    try:
        outputs = session.run(None, {input_name: test_input})
    except Exception as exc:
        return assert_true(
            False,
            name="pipeline.onnx_valid",
            message=f"ONNX inference failed: {type(exc).__name__}: {exc}",
            severity=severity,
            input_shape=list(test_input.shape),
            input_dtype=str(test_input.dtype),
        )

    output = outputs[0]

    # Compare output to expected if provided
    if expected_output is not None:
        max_diff = float(np.max(np.abs(output - expected_output)))
        passed = max_diff <= tolerance
        message = (
            f"ONNX output within tolerance: max_diff={max_diff:.6f} <= {tolerance}"
            if passed
            else f"ONNX output diverged: max_diff={max_diff:.6f} > {tolerance}"
        )
        return assert_true(
            passed,
            name="pipeline.onnx_valid",
            message=message,
            severity=severity,
            output_shape=list(output.shape),
            max_diff=max_diff,
            tolerance=tolerance,
        )

    # No expected output -- just verify inference ran
    return assert_true(
        True,
        name="pipeline.onnx_valid",
        message=f"ONNX model loaded and inference OK -- output shape {list(output.shape)}",
        severity=severity,
        output_shape=list(output.shape),
    )
