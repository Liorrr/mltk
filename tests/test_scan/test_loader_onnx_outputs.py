"""Tests for ONNX output normalization in the scan model loader."""

from __future__ import annotations

import numpy as np
import pytest

from mltk.scan.loader import _to_labels, load_model


def _make_model(path, *, output_count: int = 1, output_shape=None) -> None:
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    output_shape = output_shape or [None, 3]
    class_count = output_shape[-1] if isinstance(output_shape[-1], int) else 3
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 2])
    logits = helper.make_tensor_value_info("logits", TensorProto.FLOAT, output_shape)
    weights = numpy_helper.from_array(
        np.eye(2, class_count, dtype=np.float32),
        name="W",
    )
    nodes = [helper.make_node("MatMul", ["X", "W"], ["logits"])]
    outputs = [logits]
    if output_count == 2:
        labels = helper.make_tensor_value_info("labels", TensorProto.INT64, [None])
        probabilities = helper.make_tensor_value_info(
            "probabilities", TensorProto.FLOAT, [None, 3]
        )
        nodes.extend(
            [
                helper.make_node("ArgMax", ["logits"], ["labels"], axis=1),
                helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1),
            ]
        )
        outputs = [labels, probabilities]
    else:
        outputs = [logits]

    graph = helper.make_graph(nodes, "loader_test", [X], outputs, [weights])
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        # Pin the IR version: newer onnx releases write IR versions that older
        # onnxruntime builds refuse. IR 8 pairs with opset 13 and loads everywhere.
        ir_version=8,
    )
    onnx.save(model, str(path))


def _load(path, *, output_type=None):
    pytest.importorskip("onnxruntime")
    return load_model(path, output_type=output_type)


def _expected_probabilities(X: np.ndarray) -> np.ndarray:
    logits = np.column_stack((X, np.zeros(len(X), dtype=np.float32)))
    return np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)


def test_to_labels_matches_torch_and_keras_rule() -> None:
    np.testing.assert_array_equal(_to_labels(np.array([[1.0, 4.0], [5.0, 2.0]])), [1, 0])
    np.testing.assert_array_equal(_to_labels(np.array([[1.0], [2.0]])), [1.0, 2.0])
    np.testing.assert_array_equal(_to_labels(np.array([1.0, 2.0])), [1.0, 2.0])


def test_single_output_logits_returns_labels_and_probabilities(tmp_path) -> None:
    path = tmp_path / "logits.onnx"
    _make_model(path)
    loaded = _load(path, output_type="logits")
    X = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32)

    np.testing.assert_array_equal(loaded.predict_fn(X), [1, 0])
    expected = _expected_probabilities(X)
    np.testing.assert_allclose(loaded.predict_proba_fn(X), expected, rtol=1e-6)


def test_single_output_one_column_returns_flat_labels(tmp_path) -> None:
    path = tmp_path / "single_column.onnx"
    _make_model(path, output_shape=[None, 1])
    loaded = _load(path)

    assert loaded.predict_fn(np.array([[4.0, 7.0]], dtype=np.float32)).shape == (1,)
    assert loaded.predict_proba_fn is None


def test_two_outputs_keep_labels_and_second_output_probabilities(tmp_path) -> None:
    path = tmp_path / "two_outputs.onnx"
    _make_model(path, output_count=2)
    loaded = _load(path)
    X = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32)

    np.testing.assert_array_equal(loaded.predict_fn(X), [1, 0])
    expected = _expected_probabilities(X)
    np.testing.assert_allclose(loaded.predict_proba_fn(X), expected, rtol=1e-6)


def _make_identity_model(path) -> None:
    """Single output whose width stays symbolic: Identity over a [N, C] input."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, "C"])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, "C"])
    graph = helper.make_graph([helper.make_node("Identity", ["X"], ["Y"])], "identity", [X], [Y])
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8
    )
    onnx.save(model, str(path))


def test_single_output_symbolic_last_dimension_has_no_probabilities(tmp_path) -> None:
    path = tmp_path / "symbolic.onnx"
    _make_identity_model(path)
    loaded = _load(path)

    # predict_fn still applies the label rule at call time.
    np.testing.assert_array_equal(
        loaded.predict_fn(np.array([[1.0, 3.0, 2.0]], dtype=np.float32)), [1]
    )

    assert loaded.predict_proba_fn is None
