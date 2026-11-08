"""Regression tests for cross-modal prediction length validation."""

from __future__ import annotations

import numpy as np
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.domains.multimodal import assert_cross_modal_consistency


@pytest.mark.parametrize(
    ("predictions_a", "predictions_b"),
    [
        (["cat", "dog"], ["cat"]),
        (["cat"], ["cat", "dog"]),
        ([], ["cat"]),
    ],
    ids=["shorter_b", "longer_b", "empty_a_nonempty_b"],
)
def test_mismatched_prediction_lengths_raise_value_error(
    predictions_a: list[str], predictions_b: list[str]
) -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        assert_cross_modal_consistency(predictions_a, predictions_b)


def test_docstring_example_preserves_agreement_and_disagreements() -> None:
    result = assert_cross_modal_consistency(
        ["cat", "dog", "cat", "bird"],
        ["cat", "dog", "dog", "bird"],
        min_agreement=0.7,
    )

    assert result.details["agreement_rate"] == 0.75
    assert result.details["disagreements"] == [2]


@pytest.mark.parametrize(
    ("predictions_a", "predictions_b"),
    [
        (["cat", "dog", "cat", "bird"], ["cat", "dog", "dog", "bird"]),
        (
            np.array(["cat", "dog", "cat", "bird"]),
            np.array(["cat", "dog", "dog", "bird"]),
        ),
        (
            ["cat", "dog", "cat", "bird"],
            np.array(["cat", "dog", "dog", "bird"]),
        ),
    ],
    ids=["list_list", "ndarray_ndarray", "list_ndarray"],
)
def test_equal_length_inputs_match_list_list_result(
    predictions_a: list[str] | np.ndarray,
    predictions_b: list[str] | np.ndarray,
) -> None:
    result = assert_cross_modal_consistency(
        predictions_a, predictions_b, min_agreement=0.7
    )

    assert result.details["agreement_rate"] == 0.75
    assert result.details["disagreements"] == [2]


def test_both_inputs_empty_raise_existing_empty_assertion() -> None:
    with pytest.raises(MltkAssertionError, match="empty"):
        assert_cross_modal_consistency([], [])
