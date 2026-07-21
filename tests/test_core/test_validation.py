"""Tests for the shared require_same_length input contract (S102)."""
from __future__ import annotations

import pytest

from mltk.core.validation import require_same_length


class TestRequireSameLength:
    def test_equal_lengths_pass(self):
        require_same_length("assert_x", a=[1, 2, 3], b=[4, 5, 6])

    def test_equal_empty_sequences_pass(self):
        require_same_length("assert_x", a=[], b=[])

    def test_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            require_same_length("assert_x", a=[1, 2, 3], b=[4])

    def test_message_names_context_and_both_lengths(self):
        with pytest.raises(ValueError, match="length mismatch") as exc:
            require_same_length("assert_map", predictions=[1, 2, 3], ground_truth=[1])
        msg = str(exc.value)
        assert "assert_map" in msg
        assert "predictions (3)" in msg
        assert "ground_truth (1)" in msg

    def test_three_way_mismatch_reports_all(self):
        with pytest.raises(ValueError, match="length mismatch") as exc:
            require_same_length("assert_y", a=[1], b=[1, 2], c=[1, 2, 3])
        msg = str(exc.value)
        assert "a (1)" in msg
        assert "b (2)" in msg
        assert "c (3)" in msg

    def test_three_way_equal_passes(self):
        require_same_length("assert_y", a=[1], b=[2], c=[3])

    def test_single_sequence_is_noop(self):
        require_same_length("assert_x", only=[1, 2, 3])

    def test_no_sequences_is_noop(self):
        require_same_length("assert_x")

    def test_works_with_any_sized(self):
        # str/tuple/dict are Sized; the helper must not assume list.
        require_same_length("assert_x", a="ab", b=(1, 2), c={"x": 1, "y": 2})
        with pytest.raises(ValueError, match="length mismatch"):
            require_same_length("assert_x", a="abc", b=(1, 2))
