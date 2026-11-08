"""Canonical artifacts and mltk assertion recording."""
from __future__ import annotations

import csv
import json

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.parity.assertions import assert_parity
from mltk.parity.models import ParityReport
from mltk.parity.output import write_report


def test_deterministic_reordered_reports_and_gap_export(tmp_path):
    rows = [{"item_id": str(i), "layer": "ok", "stratum": "en", "value": {"b", "a"},
             "classification": "EXCLUDED", "group": "Unconcluded", "reason": "missing"}
            for i in range(15)]
    a = ParityReport("product", "NOT_VERIFIED", rows, refs={"recording": "hash"})
    b = ParityReport("product", "NOT_VERIFIED", list(reversed(rows)), refs={"recording": "hash"})
    first, second = write_report(a, tmp_path / "a"), write_report(b, tmp_path / "b")
    assert first.hashes == second.hashes
    assert first.rates.read_bytes() == second.rates.read_bytes()
    assert len(json.loads(first.gaps.read_text())["items"]) == 15
    with first.items.open(encoding="utf-8", newline="") as f:
        records = list(csv.DictReader(f))
    assert records[0]["engine_version"] == "1.0"
    assert json.loads(records[0]["refs"]) == {"recording": "hash"}
    with pytest.raises(MltkAssertionError) as caught:
        assert_parity(a, artifacts=first)
    assert len(caught.value.result.details["preview"]) == 10
    assert caught.value.result.details["offender_count"] == 15


def test_nonfinite_rejected_before_writing(tmp_path):
    report = ParityReport("pair", "PASS", rates=[{"rate": float("nan")}])
    with pytest.raises(ValueError, match="Out of range"):
        write_report(report, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_assertion_strict_flip_and_failure_gates():
    report = ParityReport("product", "PASS", rates=[{"negative_flip_count": 1}])
    assert assert_parity(report).passed
    with pytest.raises(MltkAssertionError):
        assert_parity(report, fail_on_negative_flip=True)
    report.rates = [{"from_failure_count": 1}]
    with pytest.raises(MltkAssertionError):
        assert_parity(report)


def test_diagnostic_counts_do_not_change_assertion_gate():
    report = ParityReport("product", "PASS", rates=[
        {"role": "verdict", "gate_passed": True},
        {"role": "diagnostic", "from_failure_count": 1, "execution_regression_count": 1,
         "negative_flip_count": 1},
    ])
    assert assert_parity(report, fail_on_negative_flip=True).passed
