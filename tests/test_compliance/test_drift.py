"""Tests for compliance coverage-drift gate (mltk.compliance.drift)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mltk.compliance.drift import (
    FRAMEWORK_VERSIONS,
    assert_no_compliance_drift,
    snapshot_coverage,
    write_coverage_baseline,
)
from mltk.core.assertion import MltkAssertionError

NIST_VERSION = FRAMEWORK_VERSIONS["nist_ai_rmf"]


def _nist_results() -> list[dict]:
    """One assertion per NIST function, no overlapping prefixes."""
    return [
        {"name": "model.bias.demographic_parity", "passed": True},  # GV
        {"name": "model.slice.performance", "passed": True},  # MP
        {"name": "model.calibration", "passed": True},  # MS
        {"name": "monitor.degradation", "passed": True},  # MN
    ]


def test_unknown_framework_is_usage_error() -> None:
    with pytest.raises(ValueError, match="unknown framework"):
        snapshot_coverage("not-a-framework", [])


def test_snapshot_includes_every_nist_function() -> None:
    controls = snapshot_coverage("nist_ai_rmf", _nist_results())
    assert set(controls) == {"GV", "MP", "MS", "MN"}
    assert controls["GV"] == ["model.bias.demographic_parity"]
    assert controls["MP"] == ["model.slice.performance"]
    assert controls["MS"] == ["model.calibration"]
    assert controls["MN"] == ["monitor.degradation"]


def test_no_drift_when_coverage_stable(tmp_path: Path) -> None:
    controls = snapshot_coverage("nist_ai_rmf", _nist_results())
    path = tmp_path / "base.json"
    write_coverage_baseline(
        path,
        framework="nist_ai_rmf",
        framework_version=NIST_VERSION,
        controls=controls,
    )
    result = assert_no_compliance_drift(
        _nist_results(), path, framework="nist_ai_rmf",
    )
    assert result.passed
    assert result.details["internal_drift"] == []
    assert result.details["external_drift"] == []


def test_internal_drift_when_control_loses_all_assertions(tmp_path: Path) -> None:
    controls = snapshot_coverage("nist_ai_rmf", _nist_results())
    path = tmp_path / "base.json"
    write_coverage_baseline(
        path,
        framework="nist_ai_rmf",
        framework_version=NIST_VERSION,
        controls=controls,
    )
    stripped = [r for r in _nist_results() if not r["name"].startswith("model.bias")]
    with pytest.raises(MltkAssertionError) as exc:
        assert_no_compliance_drift(stripped, path, framework="nist_ai_rmf")
    details = exc.value.result.details
    assert details["internal_drift"]
    assert details["external_drift"] == []
    lost = {item["control"] for item in details["internal_drift"]}
    assert "GV" in lost


def test_adding_coverage_is_not_drift(tmp_path: Path) -> None:
    controls = snapshot_coverage("nist_ai_rmf", _nist_results())
    path = tmp_path / "base.json"
    write_coverage_baseline(
        path,
        framework="nist_ai_rmf",
        framework_version=NIST_VERSION,
        controls=controls,
    )
    extra = [*_nist_results(), {"name": "data.pii.scan", "passed": True}]
    result = assert_no_compliance_drift(extra, path, framework="nist_ai_rmf")
    assert result.passed


def test_external_drift_requires_human_confirm(tmp_path: Path) -> None:
    controls = snapshot_coverage("nist_ai_rmf", _nist_results())
    path = tmp_path / "base.json"
    write_coverage_baseline(
        path,
        framework="nist_ai_rmf",
        framework_version="AI-100-1-OLD",
        controls=controls,
    )
    with pytest.raises(MltkAssertionError) as exc:
        assert_no_compliance_drift(
            _nist_results(), path, framework="nist_ai_rmf",
        )
    details = exc.value.result.details
    assert details["external_drift"]
    assert details["internal_drift"] == []


def test_external_confirm_does_not_rewrite_baseline(tmp_path: Path) -> None:
    controls = snapshot_coverage("nist_ai_rmf", _nist_results())
    path = tmp_path / "base.json"
    write_coverage_baseline(
        path,
        framework="nist_ai_rmf",
        framework_version="AI-100-1-OLD",
        controls=controls,
    )
    raw_before = path.read_text(encoding="utf-8")
    result = assert_no_compliance_drift(
        _nist_results(),
        path,
        framework="nist_ai_rmf",
        confirmed_framework_version=NIST_VERSION,
    )
    assert result.passed
    assert path.read_text(encoding="utf-8") == raw_before


def test_dropped_control_is_internal_drift(tmp_path: Path) -> None:
    controls = snapshot_coverage("nist_ai_rmf", _nist_results())
    controls["GONE"] = ["some.old.assertion"]
    path = tmp_path / "base.json"
    write_coverage_baseline(
        path,
        framework="nist_ai_rmf",
        framework_version=NIST_VERSION,
        controls=controls,
    )
    with pytest.raises(MltkAssertionError) as exc:
        assert_no_compliance_drift(
            _nist_results(), path, framework="nist_ai_rmf",
        )
    kinds = {item["kind"] for item in exc.value.result.details["internal_drift"]}
    assert "dropped_control" in kinds


def test_baseline_for_another_framework_is_refused(tmp_path: Path) -> None:
    # SCENARIO: a NIST baseline checked with framework="iso_42001".
    # WHY: the two sides then describe different control universes, so
    #   every baseline id looks absent from the live mapper. Before the
    #   guard this reported four phantom `dropped_control` entries —
    #   a coverage regression that never happened, written into the
    #   audit trail — when the real fault was the path argument.
    # EXPECTED: ValueError naming both frameworks, and no drift verdict.
    results = _nist_results()
    path = tmp_path / "nist.json"
    write_coverage_baseline(
        path,
        framework="nist_ai_rmf",
        framework_version=NIST_VERSION,
        controls=snapshot_coverage("nist_ai_rmf", results),
    )

    with pytest.raises(ValueError, match="nist_ai_rmf") as exc:
        assert_no_compliance_drift(
            results, path, framework="iso_42001",
        )

    message = str(exc.value)
    assert "nist_ai_rmf" in message
    assert "iso_42001" in message


def test_matching_framework_baseline_still_passes(tmp_path: Path) -> None:
    # SCENARIO: the guard's happy path.
    # WHY: refusing a mismatch must not refuse the correct pairing.
    # EXPECTED: no drift, no raise.
    results = _nist_results()
    path = tmp_path / "nist.json"
    write_coverage_baseline(
        path,
        framework="nist_ai_rmf",
        framework_version=NIST_VERSION,
        controls=snapshot_coverage("nist_ai_rmf", results),
    )

    result = assert_no_compliance_drift(
        results, path, framework="nist_ai_rmf",
    )
    assert result.passed
