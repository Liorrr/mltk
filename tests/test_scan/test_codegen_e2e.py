"""End-to-end verification of generated scan tests.

Generates a test file from a real scan of a toy model and runs it with
pytest in a subprocess.  Every finding must become a collected test that
passes, fails (the finding reproduces) or skips -- never errors.
"""

from __future__ import annotations

import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

from mltk.scan import scan


def test_generated_scan_file_runs_as_collected_tests(tmp_path) -> None:
    pytest.importorskip("sklearn")
    joblib = pytest.importorskip("joblib")
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(17)
    n = 160
    gender = rng.integers(0, 2, n)
    signal = rng.normal(size=n)
    # Binary label that depends on the sensitive column -> bias finding.
    label = ((gender + (signal > 0.8)) >= 1).astype(int)
    data = pd.DataFrame({"signal": signal, "gender": gender, "label": label})

    features = data[["signal", "gender"]]
    model = LogisticRegression(random_state=17, max_iter=500).fit(
        features, label,
    )
    model_path = tmp_path / "model.joblib"
    data_path = tmp_path / "data.csv"
    joblib.dump(model, model_path)
    data.to_csv(data_path, index=False)

    report = scan(
        model.predict,
        features,
        label,
        sensitive_columns=["gender"],
    )
    assert report.findings

    generated = tmp_path / "test_generated_scan.py"
    report.to_test_file(
        str(generated),
        model_path=str(model_path),
        data_path=str(data_path),
        target_col="label",
    )

    junit = tmp_path / "junit.xml"
    result = subprocess.run(
        [
            sys.executable, "-m", "pytest", str(generated), "-q",
            "-p", "no:cacheprovider", "--rootdir", str(tmp_path),
            f"--junitxml={junit}",
        ],
        cwd=tmp_path,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    assert junit.exists(), output

    suite = ET.parse(junit).getroot()
    if suite.tag == "testsuites":
        suite = suite[0]
    assert int(suite.get("tests", "0")) == len(report.findings), output
    assert int(suite.get("errors", "0")) == 0, output
    assert "NameError" not in output
    # The findings are real, so at least one generated test reproduces one.
    assert int(suite.get("failures", "0")) >= 1, output
