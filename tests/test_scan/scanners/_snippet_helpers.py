"""Shared checks for scanner-generated test bodies."""

from __future__ import annotations

import ast
import builtins
from typing import Any

import numpy as np
import pandas as pd
import pytest

import mltk
from mltk.core.assertion import MltkAssertionError
from mltk.scan import codegen


class TinyModel:
    """Small sklearn-like model used by generated-body tests."""

    def __init__(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray | None = None,
    ):
        self.predictions = np.asarray(predictions)
        self.probabilities = probabilities

    def predict(self, values: Any) -> np.ndarray:
        return self.predictions[: len(values)]

    def predict_proba(self, values: Any) -> np.ndarray:
        if self.probabilities is not None:
            return self.probabilities[: len(values)]
        predictions = self.predict(values).astype(float)
        return np.column_stack([1.0 - predictions, predictions])


class FunctionModel:
    """Adapt a prediction function to the tiny model protocol."""

    def __init__(self, predict_fn: Any):
        self.predict_fn = predict_fn

    def predict(self, values: Any) -> np.ndarray:
        return np.asarray(self.predict_fn(values))

    def predict_proba(self, values: Any) -> np.ndarray:
        predictions = self.predict(values).astype(float)
        return np.column_stack([1.0 - predictions, predictions])


def execute_finding_snippet(
    finding: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    model: TinyModel | None = None,
    X_train: pd.DataFrame | None = None,
    y_train: np.ndarray | None = None,
) -> None:
    """Validate and execute a finding body, expecting its real failure."""
    body = finding.suggested_test
    codegen.check_snippet(body)
    assert "def " not in body

    predictions = np.asarray(y).copy()
    model = model or TinyModel(predictions)
    namespace: dict[str, Any] = {
        "model": model,
        "data": (X, y),
        "X": X,
        "y": y,
        "y_pred": model.predict(X),
        "model_fn": model.predict,
        "X_train": X if X_train is None else X_train,
        "y_train": y if y_train is None else y_train,
        "np": np,
        "pd": pd,
        "pytest": pytest,
    }
    tree = ast.parse(body)
    loaded = {
        node.id for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    assigned = {
        node.id for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    for name in loaded - assigned - codegen.SNIPPET_PROVIDED_NAMES - set(dir(builtins)):
        namespace[name] = getattr(mltk, name)

    with pytest.raises(MltkAssertionError, match=r".+"):
        exec(compile(body, "<suggested_test>", "exec"), namespace)
