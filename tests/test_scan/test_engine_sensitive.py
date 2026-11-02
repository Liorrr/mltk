"""Engine-level sensitive-column auto-detection regression tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mltk.scan import ScanConfig, ScanEngine, scan


def _bias_config() -> ScanConfig:
    return ScanConfig(enabled_scanners=["bias"])


def test_engine_skips_high_cardinality_auto_detected_age() -> None:
    n = 100
    X = pd.DataFrame({"applicant_age": np.arange(n, dtype=float)})
    y = np.ones(n, dtype=int)
    preds = np.array([1, 0] * (n // 2))

    report = scan(
        lambda frame: preds[frame.index], X, y, config=_bias_config(),
    )

    assert not report.findings


def test_engine_keeps_binary_gender_auto_detected_with_existing_severity() -> None:
    n = 200
    X = pd.DataFrame({"gender": ["M"] * 100 + ["F"] * 100})
    y = np.ones(n, dtype=int)
    preds = np.array([1] * 100 + [0] * 100)

    report = ScanEngine(_bias_config()).scan(
        lambda frame: preds[frame.index], X, y,
    )

    gender_findings = [
        finding for finding in report.findings
        if finding.result.details.get("column") == "gender"
    ]
    assert gender_findings
    assert all(f.result.severity.value == "critical" for f in gender_findings)
