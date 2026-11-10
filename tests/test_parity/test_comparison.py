"""Synthetic state-pair precedence tests."""

from __future__ import annotations

import pytest

from mltk.parity.comparison import classify_pair


def state(status="VALID", value=True, oracle=True):
    return {"status": status, "value": value, "oracle": oracle}


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (state("NOT_SCORABLE"), state("FAILED"), "EXCLUDED"),
        (state("FAILED"), state("NOT_SCORABLE"), "EXCLUDED"),
        (state("FAILED"), state("FAILED"), "FAILED_BOTH"),
        (state("FAILED"), state("INDETERMINATE"), "FAILED_BASE"),
        (state("INDETERMINATE", oracle=None), state("FAILED"), "FAILED_CAND"),
        (state("INDETERMINATE"), state(), "INDETERMINATE"),
        (state(), state(), "UNCHANGED"),
        (state(), state(value=False, oracle=None), "CHANGED_NO_ORACLE"),
        (state(), state(value=False, oracle=False), "NEG_FLIP"),
        (state(value=False, oracle=False), state(), "POS_FLIP"),
        (state(value="a"), state(value="b"), "CHANGED_SAME_VERDICT"),
    ],
)
def test_precedence(a, b, expected):
    assert classify_pair(a, b)["classification"] == expected


def test_source_changed_precedence_and_unchanged_flag():
    result = classify_pair(
        state(), state(value=False, oracle=False), base_source="old", candidate_source="new"
    )
    assert result["classification"] == "CHANGED_SOURCE"
    assert not result["source_unknown"]
    result = classify_pair(state(), state(), base_source="old", candidate_source="new")
    assert result["classification"] == "UNCHANGED"
    assert result["source_changed"]


def test_unknown_source_is_comparable_and_failure_is_separate():
    assert classify_pair(state(), state(value=False, oracle=False))["classification"] == "NEG_FLIP"
    failed = classify_pair(state(), state("FAILED", oracle=None))
    assert failed["from_failure"]
    assert failed["group"] == "Failure"
    assert not classify_pair(state(oracle=False), state("FAILED"))["from_failure"]


def test_unconcluded_primary_and_secondary_reasons():
    result = classify_pair(
        state("NOT_SCORABLE"), state("NOT_SCORABLE"), base_source="old", candidate_source="new"
    )
    assert result["group"] == "Unconcluded"
    assert result["reason"] == "BASE_NOT_SCORABLE"
    assert result["reasons"] == ["BASE_NOT_SCORABLE", "CANDIDATE_NOT_SCORABLE", "SOURCE_CHANGED"]
