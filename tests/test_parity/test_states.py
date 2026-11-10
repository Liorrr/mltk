"""Synthetic trial reduction fixtures; no adopter measurements."""

from __future__ import annotations

import pytest

from mltk.parity.models import Item, LayerSpec, Manifest, Trial
from mltk.parity.states import reduce_item


def synthetic(values, failed=()):
    return Item(
        "i",
        "en",
        [
            Trial(
                str(n + 1),
                "failed" if n in failed else "ok",
                {"v": value},
                "synthetic" if n in failed else None,
            )
            for n, value in enumerate(values)
        ],
    )


@pytest.mark.parametrize(
    ("values", "failed", "status", "value", "agreement"),
    [
        ([True, True, False], (), "VALID", True, 2 / 3),
        ([True, False, None], (), "INDETERMINATE", None, None),
        ([None, None, True], (0, 1), "FAILED", None, None),
        ([None, None, None], (0,), "NOT_SCORABLE", None, None),
    ],
)
def test_t3(values, failed, status, value, agreement):
    layer = LayerSpec("v", oracle=lambda v: v, oracle_id="synthetic-v1")
    state = reduce_item(synthetic(values, failed), layer, Manifest({"i": "en"}))
    assert (state["status"], state["value"], state["agreement"]) == (status, value, agreement)
    assert state["n_failed"] == len(failed)
    assert state["k"] == 3


def test_missing_is_not_failed_and_agreement_uses_schedule():
    manifest = Manifest({"i": "en"})
    layer = LayerSpec("v", oracle=lambda v: v, oracle_id="synthetic-v1")
    state = reduce_item(synthetic([True, True]), layer, manifest)
    assert state["agreement"] == 2 / 3
    assert state["n_missing"] == 1
    assert state["pass_all_trials"] is None
    absent = reduce_item(None, layer, manifest)
    assert absent["reason"] == "MISSING_ITEM"
    assert (absent["n_missing"], absent["n_failed"]) == (3, 0)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([True, True, True], True),
        ([True, True, False], False),
        ([True, True, None], None),
        ([False, None, None], False),
    ],
)
def test_all_trials_oracle(values, expected):
    state = reduce_item(
        synthetic(values), LayerSpec("v", oracle=lambda v: v), Manifest({"i": "en"})
    )
    assert state["pass_all_trials"] is expected


def test_numeric_categorical_exact_and_mean_explicit():
    item = synthetic([1, 1.000000001, 1.000000002])
    manifest = Manifest({"i": "en"})
    assert reduce_item(item, LayerSpec("v", kind="number"), manifest)["status"] == "INDETERMINATE"
    mean = reduce_item(item, LayerSpec("v", kind="number", reducer="mean"), manifest)
    assert mean["value"] == pytest.approx(1.000000001)
    assert mean["agreement"] == 1


def test_sets_unordered_and_no_fuzzy_majority():
    state = reduce_item(
        synthetic([{"a", "b"}, {"b", "a"}, {"c"}]),
        LayerSpec("v", kind="id_set"),
        Manifest({"i": "en"}),
    )
    assert state["value"] == {"a", "b"}
    assert state["x"] == 2


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "1"])
def test_wrong_numeric_types_rejected(value):
    with pytest.raises(ValueError, match="wrong type|finite"):
        reduce_item(synthetic([value] * 3), LayerSpec("v", kind="number"), Manifest({"i": "en"}))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_valid": 0},
        {"failed_threshold": 4},
        {"min_valid": True},
        {"majority_fraction": 1},
        {"majority_fraction": 0.4},
        {"min_valid": 1, "failed_threshold": 2},
    ],
)
def test_impossible_policy(kwargs):
    with pytest.raises(ValueError, match="thresholds|majority_fraction"):
        reduce_item(None, LayerSpec("v", **kwargs), Manifest({"i": "en"}))


def test_general_k_explicit_and_errors_retained():
    manifest = Manifest({"i": "en"}, trial_ids=("1", "2", "3", "4", "5"))
    with pytest.raises(ValueError, match="General-k"):
        reduce_item(None, LayerSpec("v"), manifest)
    state = reduce_item(
        synthetic([True] * 5, (0, 1, 2)), LayerSpec("v", min_valid=3, failed_threshold=3), manifest
    )
    assert state["status"] == "FAILED"
    assert state["trial_errors"] == {"1": "synthetic", "2": "synthetic", "3": "synthetic"}


def test_oracle_requires_actual_bool():
    with pytest.raises(ValueError, match="Oracle"):
        reduce_item(
            synthetic([True] * 3), LayerSpec("v", oracle=lambda _: 1), Manifest({"i": "en"})
        )
