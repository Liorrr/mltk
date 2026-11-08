"""Review regressions for harness eligibility, exception boundaries and evidence."""

from __future__ import annotations

import pytest
from tests.test_parity.test_harness import change, compare, entry, synthetic_runs

from mltk.parity import harness
from mltk.parity.models import LayerSpec


@pytest.mark.parametrize("role", ["verdict", "precondition"])
def test_only_eligible_stratum_is_required(role):
    fixture = synthetic_runs(strata=("en", "he"))
    fixture[2].eligible_ids = {"result": ("synthetic-0",)}
    layer = fixture[3][0]
    layer.role, layer.order = role, 7
    layer.oracle, layer.oracle_id = lambda value: value, "synthetic-identity-v1"
    # An ineligible cell has no score at all, and requires no oracle evidence.
    for run in fixture[:2]:
        for trial in run.items[1].trials:
            trial.values.clear()
    report = compare(fixture)
    assert report.status == "PASS"
    rates = {rate["stratum"]: rate for rate in report.rates}
    assert rates["en"]["required"] is True
    assert rates["en"]["status"] == "PASS"
    assert rates["he"]["status"] == rates["he"]["verdict"] == "NOT_APPLICABLE"
    assert rates["he"]["required"] is False
    assert rates["he"]["eligible"] == rates["he"]["n_compared"] == 0
    assert rates["he"]["order"] == 7
    assert rates["he"]["role"] == role


def test_ineligible_mean_cell_does_not_require_margin_or_statistics(monkeypatch):
    fixture = synthetic_runs(kind="number", reducer="mean", values=[1.0] * 3,
                             strata=("en", "en", "he"))
    fixture[2].eligible_ids = {"result": ("synthetic-0", "synthetic-1")}
    calls = []
    original = harness._tost

    def checked_tost(differences, margin, alpha):
        calls.append(differences)
        assert len(differences) == 2
        return original(differences, margin, alpha)

    monkeypatch.setattr(harness, "_tost", checked_tost)
    report = compare(fixture, margins={("result", "en"): 0.1})
    assert report.status == "PASS"
    assert len(calls) == 1
    assert report.rates[1]["status"] == "NOT_APPLICABLE"
    assert report.rates[1]["reason"] == "NO_ELIGIBLE_ITEMS"


@pytest.mark.parametrize("reducer", ["majority", "mean"])
def test_all_cells_ineligible_cannot_verify(reducer):
    fixture = synthetic_runs(kind="number", reducer=reducer, values=[1.0] * 3,
                             strata=("en", "he"))
    fixture[2].eligible_ids = {"result": ()}
    report = compare(fixture)
    assert report.status == "NOT_VERIFIED"
    assert report.reasons == ["NO_COMPARABLE_ITEMS"]
    assert all(rate["status"] == "NOT_APPLICABLE" for rate in report.rates)
    assert not any(rate["required"] for rate in report.rates)


def test_eligible_but_indeterminate_stratum_still_blocks():
    fixture = synthetic_runs(kind="class", values=["a"] * 3, strata=("en", "he"))
    for run in fixture[:2]:
        change(run, ["a", "b", "c"], index=1)
    report = compare(fixture)
    assert report.status == "NOT_VERIFIED"
    rate = next(rate for rate in report.rates if rate["stratum"] == "he")
    assert rate["required"] is True
    assert rate["eligible"] == 1
    assert rate["n_compared"] == 0
    assert rate["status"] == "NOT_VERIFIED"
    assert rate["reason"] == "unresolved_item_state"


@pytest.mark.parametrize("error_type", [RuntimeError, AssertionError, KeyError])
@pytest.mark.parametrize("stage", ["_integrity", "_compare"])
def test_programming_errors_propagate(monkeypatch, error_type, stage):
    def broken(*args, **kwargs):
        raise error_type("synthetic programming defect")

    monkeypatch.setattr(harness, stage, broken)
    with pytest.raises(error_type, match="synthetic programming defect"):
        compare(synthetic_runs())


@pytest.mark.parametrize("error_type", [ValueError, TypeError])
def test_validation_errors_remain_not_run_and_are_finalized(monkeypatch, error_type):
    seen = []

    def invalid(*args, **kwargs):
        raise error_type("synthetic invalid evidence")

    def finalized(report):
        seen.append(report)
        return report

    monkeypatch.setattr(harness, "_integrity", invalid)
    monkeypatch.setattr(harness, "finalize_report", finalized)
    report = compare(synthetic_runs())
    assert report.status == "NOT_RUN"
    assert "synthetic invalid evidence" in report.reasons[0]
    assert seen == [report]


def test_successful_report_is_finalized_once(monkeypatch):
    seen = []

    def finalized(report):
        seen.append(report)
        return report

    monkeypatch.setattr(harness, "finalize_report", finalized)
    report = compare(synthetic_runs())
    assert report.passed
    assert seen == [report]


def test_unlisted_count_counts_items_without_double_counting_trials():
    fixture = synthetic_runs(strata=("en", "he"))
    change(fixture[1], [False] * 3)
    report = compare(fixture)
    rates = {rate["stratum"]: rate for rate in report.rates}
    assert rates["en"]["unlisted_count"] == 1
    assert rates["en"]["stale_count"] == rates["en"]["mismatch_count"] == 0
    assert rates["he"]["unlisted_count"] == 0


@pytest.mark.parametrize("role", ["verdict", "diagnostic"])
def test_stale_ledger_marks_its_cell_failed_and_supplies_offender(role):
    fixture = synthetic_runs(strata=("en", "he"))
    fixture[3][0].role = role
    report = compare(fixture, expected_diff=[entry()])
    assert report.status == "FAIL"
    rate = next(rate for rate in report.rates if rate["stratum"] == "en")
    assert rate["status"] == rate["verdict"] == "FAIL"
    assert rate["required"] is True  # Ledger audit remains mandatory for diagnostics.
    assert rate["reason"] == "expected_diff_stale"
    assert rate["stale_count"] == 1
    assert rate["mismatch_count"] == rate["unlisted_count"] == 0
    row = next(row for row in report.items if row["item_id"] == "synthetic-0")
    assert row["status"] == "FAIL"
    assert row["classification"] == "STALE_EXPECTED_DIFF"


def test_mismatched_ledger_count_is_separate_from_stale_count():
    fixture = synthetic_runs(kind="class", values=["a"] * 3)
    change(fixture[1], ["b"] * 3)
    report = compare(fixture, expected_diff=[entry(**{"from": "a", "to": "c"})])
    assert report.status == "FAIL"
    assert report.rates[0]["mismatch_count"] == 1
    assert report.rates[0]["stale_count"] == 0
    assert report.rates[0]["unlisted_count"] == 1


def test_two_layers_with_disjoint_eligibility_do_not_block_each_other():
    fixture = synthetic_runs(strata=("en", "he"))
    fixture[3].append(LayerSpec("secondary", role="precondition", oracle=lambda value: value,
                                oracle_id="synthetic-identity-v1"))
    fixture[2].eligible_ids = {"result": ("synthetic-0",), "secondary": ("synthetic-1",)}
    for run in fixture[:2]:
        for trial in run.items[1].trials:
            trial.values = {"secondary": True}
    report = compare(fixture)
    assert report.status == "PASS"
    assert sum(rate["status"] == "NOT_APPLICABLE" for rate in report.rates) == 2
    assert sum(rate["required"] for rate in report.rates) == 2
