"""CLI coverage for model loading opt-in."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Run the mltk CLI in a child interpreter (same pattern as test_cli)."""
    code = (
        "import sys; "
        f"sys.argv = ['mltk'] + {list(args)!r}; "
        "from mltk.cli.app import main; main()"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_scan_model_refuses_joblib_without_flag(tmp_path) -> None:
    pytest.importorskip("sklearn")
    joblib = pytest.importorskip("joblib")
    from sklearn.linear_model import LogisticRegression

    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    model_path = tmp_path / "model.joblib"
    joblib.dump(LogisticRegression(random_state=1).fit(X, y), model_path)

    result = _run_cli(
        "scan-model", "--model", str(model_path),
        "--data", str(tmp_path / "missing.csv"), "--target", "label",
    )
    assert result.returncode == 1
    assert "--allow-pickle" in result.stdout + result.stderr


def test_scan_model_accepts_joblib_with_flag(tmp_path) -> None:
    pytest.importorskip("sklearn")
    pytest.importorskip("pandas")
    joblib = pytest.importorskip("joblib")
    from sklearn.linear_model import LogisticRegression

    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    model_path = tmp_path / "model.joblib"
    data_path = tmp_path / "data.csv"
    joblib.dump(LogisticRegression(random_state=1).fit(X, y), model_path)
    pd.DataFrame({"feature": X.ravel(), "label": y}).to_csv(
        data_path, index=False,
    )

    result = _run_cli(
        "scan-model", "--model", str(model_path),
        "--data", str(data_path), "--target", "label",
        "--allow-pickle",
    )
    assert result.returncode in {0, 1}
    assert "Refusing to load" not in result.stdout + result.stderr
