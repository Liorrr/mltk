"""Tests for explicit opt-in when loading code-executing model formats."""

from __future__ import annotations

import ast
import importlib.util
import sys

import numpy as np
import pytest

from mltk.scan import PickleNotAllowedError
from mltk.scan.codegen import generate_test_file
from mltk.scan.loader import load_model


@pytest.mark.parametrize("suffix", [".pkl", ".pickle", ".joblib", ".pt", ".pth"])
def test_pickle_formats_refused_by_default(tmp_path, monkeypatch, suffix) -> None:
    path = tmp_path / f"model{suffix}"
    path.write_bytes(b"not loaded")
    monkeypatch.delenv("MLTK_ALLOW_PICKLE", raising=False)
    if importlib.util.find_spec("joblib") is not None:
        monkeypatch.setattr("joblib.load", pytest.fail)
    with pytest.raises(PickleNotAllowedError) as exc_info:
        load_model(path)
    assert isinstance(exc_info.value, ValueError)


def test_refusal_happens_before_library_import(tmp_path, monkeypatch) -> None:
    path = tmp_path / "model.pt"
    path.write_bytes(b"not loaded")
    monkeypatch.delenv("MLTK_ALLOW_PICKLE", raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(PickleNotAllowedError):
        load_model(path)


def test_explicit_fmt_pickle_refused(tmp_path, monkeypatch) -> None:
    path = tmp_path / "model.bin"
    path.write_bytes(b"not loaded")
    monkeypatch.delenv("MLTK_ALLOW_PICKLE", raising=False)
    with pytest.raises(PickleNotAllowedError):
        load_model(path, fmt="pickle")


def test_allow_pickle_true_loads_model(tmp_path) -> None:
    pytest.importorskip("sklearn")
    joblib = pytest.importorskip("joblib")
    from sklearn.linear_model import LogisticRegression

    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    model = LogisticRegression(random_state=1).fit(X, y)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)

    loaded = load_model(path, allow_pickle=True)
    np.testing.assert_array_equal(loaded.predict_fn(X), model.predict(X))


def test_env_var_opt_in_loads_model(tmp_path, monkeypatch) -> None:
    pytest.importorskip("sklearn")
    joblib = pytest.importorskip("joblib")
    from sklearn.linear_model import LogisticRegression

    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    path = tmp_path / "model.joblib"
    joblib.dump(LogisticRegression(random_state=1).fit(X, y), path)
    monkeypatch.setenv("MLTK_ALLOW_PICKLE", "TrUe")

    loaded = load_model(path)
    assert loaded.predict_fn(X).shape == (4,)


def test_error_message_names_flag_and_env_var(tmp_path, monkeypatch) -> None:
    path = tmp_path / "model.pkl"
    path.write_bytes(b"not loaded")
    monkeypatch.delenv("MLTK_ALLOW_PICKLE", raising=False)
    with pytest.raises(PickleNotAllowedError) as exc_info:
        load_model(path)
    message = str(exc_info.value)
    assert "allow_pickle=True" in message
    assert "MLTK_ALLOW_PICKLE=1" in message


def test_onnx_path_not_affected(tmp_path, monkeypatch) -> None:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"not an ONNX model")
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(ImportError, match="onnxruntime"):
        load_model(path)


def test_generated_harness_carries_opt_in() -> None:
    default_source = generate_test_file([])
    opted_in_source = generate_test_file([], allow_pickle=True)
    assert "load_model(path, allow_pickle=False)" in default_source
    assert "load_model(path, allow_pickle=True)" in opted_in_source
    ast.parse(default_source)
    ast.parse(opted_in_source)
