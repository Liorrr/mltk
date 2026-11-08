"""Review regressions for distinct schemas and evidence that survives previews."""

from __future__ import annotations

import csv
import json
from importlib.metadata import PackageNotFoundError, version

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.parity.assertions import assert_parity
from mltk.parity.models import REPORT_SCHEMA_ID, REPORT_SCHEMA_VERSION, ParityReport
from mltk.parity.output import report_summary, write_report
from mltk.parity.presentation import finalize_report


def test_twelfth_rate_failure_precedes_passing_rates_and_is_compact(tmp_path):
    layers = [{"name": f"layer{i:02d}", "order": i} for i in range(12)]
    rates = [
        {
            "layer": layer["name"],
            "stratum": "en",
            "gate_passed": i != 11,
            "verdict": "UNCHANGED" if i != 11 else "DETECTED",
            "reason": None if i != 11 else "CHANGE_ABOVE_NOISE_FLOOR",
            "changed_rate": 0.5 if i == 11 else 0,
            "floor": 0.2,
            "mdc": 5,
            "b": 10,
            "c": 0,
            "n_compared": 20,
            "n_state_compared": 20,
            "changed_count": 10,
            "large_evidence": list(range(500)),
        }
        for i, layer in enumerate(layers)
    ]
    rows = [
        {
            "item_id": f"i{i}",
            "stratum": "en",
            "layer": "layer11",
            "classification": "NEG_FLIP",
            "large_evidence": list(range(500)),
        }
        for i in range(12)
    ]
    report = finalize_report(
        ParityReport("product", "FAIL", rows, rates, config={"layers": layers})
    )
    paths = write_report(report, tmp_path / "output")
    with pytest.raises(MltkAssertionError) as caught:
        assert_parity(report, artifacts=paths)
    result = caught.value.result
    assert result.message == report.message
    assert "layer11@en: DETECTED (CHANGE_ABOVE_NOISE_FLOOR)" in result.message.splitlines()[0]
    assert "changed 10/20 vs floor=0.2, MDC=5" in result.message
    assert "items.jsonl" in result.message
    assert str(tmp_path) not in result.message
    detail = result.details
    assert detail["failing"] == ["layer11@en"]
    assert detail["rates"][0]["layer"] == "layer11"
    assert len(detail["rates"]) == 10
    assert set(detail["rates"][0]) == {
        "layer",
        "stratum",
        "verdict",
        "reason",
        "changed_rate",
        "floor",
        "mdc",
        "b",
        "c",
        "n_compared",
    }
    assert len(detail["preview"]) == 10
    assert set(detail["preview"][0]) == {"item_id", "stratum", "layer", "classification"}
    assert detail["artifacts"]["rates"] == "rates.json"
    assert detail["artifacts"]["items_jsonl"] == "items.jsonl"


def test_jsonl_canonical_rows_and_envelopes_are_separate(tmp_path):
    rows = [
        {
            "item_id": "b",
            "layer": "z",
            "stratum": "en",
            "value": {"b", "a"},
            "classification": "VALUE_CHANGED",
        },
        {"item_id": "a", "layer": "z", "stratum": "he", "classification": "UNCHANGED"},
        {"item_id": "a", "layer": "z", "stratum": "en", "classification": "NEG_FLIP"},
        {
            "item_id": "a",
            "layer": "a",
            "stratum": "en",
            "classification": "EXCLUDED",
            "group": "Unconcluded",
        },
    ]
    config = {"layers": [{"name": "z", "order": 0}, {"name": "a", "order": 1}]}
    report = ParityReport("pair", "FAIL", rows, config=config, refs={"base": "run-a"})
    paths = write_report(report, tmp_path)
    assert set(paths.hashes) == {"rates.json", "items.jsonl", "items.csv", "gaps.json"}
    rates = json.loads(paths.rates.read_text(encoding="utf-8"))
    assert "items" not in rates
    assert rates["schema_id"] == REPORT_SCHEMA_ID == "mltk.parity.report"
    assert rates["schema_version"] == REPORT_SCHEMA_VERSION == "1.0"
    try:
        package_version = version("mlspec")
    except PackageNotFoundError:
        package_version = "unknown"
    assert rates["mlspec_version"] == package_version
    records = [
        json.loads(line) for line in paths.items_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["type"] == "header"
    assert records[0]["refs"] == rates["refs"]
    assert "rates" not in records[0]
    assert all(record["type"] == "item" for record in records[1:])
    items = [record["item"] for record in records[1:]]
    assert [(r["layer"], r["stratum"], r["item_id"]) for r in items] == [
        ("z", "en", "a"),
        ("z", "en", "b"),
        ("z", "he", "a"),
        ("a", "en", "a"),
    ]
    assert items[1]["value"] == {"type": "id_set", "values": ["a", "b"]}
    with paths.items.open(encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == len(items)
    assert not {"schema_id", "schema_version", "engine_version", "refs"} & csv_rows[0].keys()
    assert json.loads(paths.gaps.read_text(encoding="utf-8"))["items"] == [items[-1]]
    # Content changes must not move a row away from its stable identity.
    report.items[0]["classification"] = "AAA"
    again = write_report(report, tmp_path / "changed")
    changed = [
        json.loads(line)["item"]
        for line in again.items_jsonl.read_text(encoding="utf-8").splitlines()[1:]
    ]
    assert [(r["layer"], r["stratum"], r["item_id"]) for r in changed] == [
        (r["layer"], r["stratum"], r["item_id"]) for r in items
    ]


def test_message_for_missing_coverage_and_harness_ledger():
    product = ParityReport(
        "product",
        "NOT_VERIFIED",
        rates=[
            {
                "layer": "answer",
                "stratum": "en",
                "gate_passed": False,
                "verdict": "DETECTION_NOT_DEMONSTRATED",
                "reason": "CANDIDATE_NOT_ESTABLISHED",
                "n_state_compared": 1,
                "required_established": 16,
                "n_valid_aa": 20,
            }
        ],
    )
    assert "candidate n=1 vs required=16; A/A n=20" in product.message
    assert "supply passing control evidence" in product.message
    harness = ParityReport(
        "harness",
        "FAIL",
        rates=[
            {
                "layer": "score",
                "stratum": "en",
                "status": "FAIL",
                "required": True,
                "role": "diagnostic",
                "unlisted_count": 2,
                "stale_count": 1,
                "mismatch_count": 3,
                "reason": "expected_diff_stale",
                "n_compared": 20,
            }
        ],
        config={"expected_diff_audit": [{"status": "stale", "item_id": "stale-item"}]},
    )
    assert (
        "score@en: FAIL (expected_diff_stale); unlisted=2, stale=1, mismatch=3, n=20"
        in harness.message
    )
    assert "stale-item" in harness.message
    assert "remove stale entries" in harness.message
    assert report_summary(harness)["failing"] == ["score@en"]


def test_not_applicable_and_diagnostic_rates_do_not_enter_failing_list():
    report = ParityReport(
        "pair",
        "NOT_VERIFIED",
        rates=[
            {"layer": "absent", "stratum": "en", "verdict": "NOT_APPLICABLE", "gate_passed": False},
            {"layer": "diagnostic", "stratum": "en", "role": "diagnostic", "gate_passed": False},
            {"layer": "required", "stratum": "en", "status": "NOT_VERIFIED", "required": True},
        ],
    )
    assert report_summary(report)["failing"] == ["required@en"]
    assert "absent@en" not in report.message
    assert "diagnostic@en" not in report.message


@pytest.mark.parametrize("status", ["PASS", "FAIL", "NOT_RUN", "NOT_VERIFIED"])
def test_assertion_maps_status_only(status):
    report = ParityReport(
        "product",
        status,
        rates=[
            {"from_failure_count": 3, "execution_regression_count": 2, "negative_flip_count": 1}
        ],
    )
    if status == "PASS":
        assert assert_parity(report).passed
    else:
        with pytest.raises(MltkAssertionError) as caught:
            assert_parity(report)
        assert caught.value.result.message == report.message


def test_assertion_legacy_option_cannot_change_engine_decision():
    report = ParityReport("product", "PASS", config={"fail_on_negative_flip": True})
    with pytest.warns(DeprecationWarning, match="deprecated"):
        assert assert_parity(report, fail_on_negative_flip=True).passed
    with (
        pytest.warns(DeprecationWarning, match="deprecated"),
        pytest.raises(ValueError, match="report.config"),
    ):
        assert_parity(report, fail_on_negative_flip=False)


def test_rate_order_uses_stratum_before_layer_name_when_orders_tie(tmp_path):
    rates = [
        {"layer": "a", "stratum": "he", "gate_passed": True},
        {"layer": "z", "stratum": "en", "gate_passed": True},
        {"layer": "a", "stratum": "en", "gate_passed": True},
        {"layer": "first", "stratum": "he", "gate_passed": True},
    ]
    report = ParityReport(
        "pair",
        "PASS",
        rates=rates,
        config={
            "layers": [
                {"name": "a", "order": 1},
                {"name": "z", "order": 1},
                {"name": "first", "order": 0},
            ]
        },
    )
    paths = write_report(report, tmp_path)
    stored = json.loads(paths.rates.read_text(encoding="utf-8"))["rates"]
    expected = [("first", "he"), ("a", "en"), ("z", "en"), ("a", "he")]
    assert [(r["layer"], r["stratum"]) for r in stored] == expected
    assert [(r["layer"], r["stratum"]) for r in report_summary(report)["rates"]] == expected


def test_missing_package_metadata_is_explicitly_unknown(monkeypatch):
    def missing(name):
        assert name == "mlspec"
        raise PackageNotFoundError(name)

    monkeypatch.setattr("mltk.parity.models.version", missing)
    assert ParityReport("pair", "NOT_VERIFIED").mlspec_version == "unknown"
