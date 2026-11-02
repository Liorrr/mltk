"""Generated snippets must test what the scanner tested.

Covers probability-shape handling (calibration), the exact numeric
column set (robustness), and a feature literally named ``target``
(leakage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import mltk
from mltk.core.assertion import MltkAssertionError
from mltk.scan.codegen import check_snippet
from mltk.scan.scanners.calibration import CalibrationScanner
from mltk.scan.scanners.leakage import LeakageScanner
from mltk.scan.scanners.robustness import RobustnessScanner


def _exec(body: str, **ns: object) -> None:
    check_snippet(body)
    namespace = {"np": np, "pd": pd, **ns}
    for name in dir(mltk):
        if name.startswith("assert_"):
            namespace[name] = getattr(mltk, name)
    try:
        exec(body, namespace)  # noqa: S102 -- generated test body under test
    except MltkAssertionError:
        pass


class _Model:
    def __init__(self, proba: np.ndarray) -> None:
        self._proba = proba

    def predict(self, X):  # noqa: N803
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):  # noqa: N803
        return self._proba


def test_calibration_snippet_handles_single_column_proba() -> None:
    body = CalibrationScanner._gen_test(0.1, 10)
    y = np.array([0, 1, 0, 1])
    proba = np.array([[0.1], [0.9], [0.2], [0.8]])
    X = pd.DataFrame({"a": [1, 2, 3, 4]})
    _exec(body, model=_Model(proba), X=X, y=y)  # must not IndexError


def test_calibration_snippet_handles_1d_proba() -> None:
    body = CalibrationScanner._gen_test(0.1, 10)
    y = np.array([0, 1, 0, 1])
    X = pd.DataFrame({"a": [1, 2, 3, 4]})
    _exec(body, model=_Model(np.array([0.1, 0.9, 0.2, 0.8])), X=X, y=y)


def test_robustness_snippet_uses_scanned_columns_only() -> None:
    body = RobustnessScanner._gen_test(0.01, 0.9, ["f1"])
    assert "['f1']" in body
    assert "select_dtypes" not in body
    seen: list[tuple[int, ...]] = []

    def model_fn(arr):
        seen.append(np.asarray(arr).shape)
        return np.zeros(len(arr))

    X = pd.DataFrame({"f1": [0.1, 0.2, 0.3], "code": [1, 2, 1]})
    _exec(body, model_fn=model_fn, X=X, y=np.zeros(3))
    assert seen
    assert all(shape[1] == 1 for shape in seen)


def test_leakage_snippet_keeps_feature_named_target() -> None:
    body = LeakageScanner._gen_test("target", 0.9)
    assert "__mltk_target__" in body
    X = pd.DataFrame({"target": [1.0, 2.0, 3.0, 4.0]})
    y = np.array([0, 1, 0, 1])
    _exec(body, X=X, y=y)
