"""Tests for mltk.pipeline.onnx -- ONNX export validation.

Verifies that assert_onnx_valid correctly validates ONNX model files:
loading, inference, and output comparison. Uses pytest.importorskip
to skip when onnxruntime is not installed.
"""

import numpy as np
import pytest

ort = pytest.importorskip("onnxruntime", reason="onnxruntime required for ONNX tests")

from mltk.core.assertion import MltkAssertionError  # noqa: E402
from mltk.pipeline.onnx import assert_onnx_valid  # noqa: E402


def _create_simple_onnx_model(path, input_dim: int = 10):
    """Create a minimal ONNX model (identity + add bias) for testing."""
    onnx = pytest.importorskip("onnx", reason="onnx required to create test models")
    from onnx import TensorProto, helper

    # Identity-like model: output = input (via a single MatMul with identity matrix)
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, input_dim])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, input_dim])

    # Identity weight matrix
    weight = helper.make_tensor(
        "W",
        TensorProto.FLOAT,
        [input_dim, input_dim],
        np.eye(input_dim, dtype=np.float32).flatten().tolist(),
    )

    matmul = helper.make_node("MatMul", ["X", "W"], ["Y"])
    graph = helper.make_graph([matmul], "test_model", [X], [Y], [weight])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, str(path))


class TestAssertOnnxValid:
    """ONNX model validation tests."""

    def test_valid_model_inference(self, tmp_path) -> None:
        """PASS: ONNX model loads and produces correct output.

        WHY: The most common case -- model exported correctly, inference
        produces results matching the source framework within tolerance.
        """
        model_path = tmp_path / "identity.onnx"
        _create_simple_onnx_model(model_path, input_dim=5)

        test_input = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
        expected = test_input.copy()

        result = assert_onnx_valid(
            model_path, test_input, expected_output=expected, tolerance=0.001
        )
        assert result.passed is True
        assert result.details["output_shape"] == [1, 5]

    def test_model_not_found(self, tmp_path) -> None:
        """FAIL: Model file does not exist.

        WHY: Catches broken CI/CD pipelines where the export step was
        skipped or wrote to the wrong path.
        """
        missing = tmp_path / "nonexistent.onnx"
        test_input = np.zeros((1, 10), dtype=np.float32)

        with pytest.raises(MltkAssertionError) as exc:
            assert_onnx_valid(missing, test_input)
        assert "not found" in str(exc.value)

    def test_output_diverged(self, tmp_path) -> None:
        """FAIL: Model output differs from expected beyond tolerance.

        WHY: Detects numerical divergence between source framework and
        ONNX export (e.g., quantization artifacts, op-level differences).
        """
        model_path = tmp_path / "identity.onnx"
        _create_simple_onnx_model(model_path, input_dim=3)

        test_input = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        # Expected output is way off from identity
        wrong_expected = np.array([[99.0, 99.0, 99.0]], dtype=np.float32)

        with pytest.raises(MltkAssertionError) as exc:
            assert_onnx_valid(
                model_path, test_input, expected_output=wrong_expected, tolerance=0.01
            )
        assert "diverged" in str(exc.value)
