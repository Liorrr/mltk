"""Tests for mltk.pipeline.e2e -- end-to-end pipeline validation."""

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.pipeline.e2e import assert_pipeline


def _double(x: int) -> int:
    return x * 2


def _add_one(x: int) -> int:
    return x + 1


def _broken_step(x: int) -> int:
    raise ValueError("Data format error")


class TestAssertPipeline:
    """E2E pipeline tests."""

    def test_pipeline_runs(self) -> None:
        """PASS: All pipeline steps complete successfully."""
        result = assert_pipeline([_double, _add_one, _double], input_data=5)
        assert result.passed is True
        assert result.details["completed_steps"] == 3

    def test_pipeline_fails_at_step(self) -> None:
        """FAIL: Pipeline fails at broken step."""
        with pytest.raises(MltkAssertionError) as exc:
            assert_pipeline([_double, _broken_step, _add_one], input_data=5)
        assert "step 1" in str(exc.value)

    def test_pipeline_type_check(self) -> None:
        """PASS: Pipeline output matches expected type."""
        result = assert_pipeline(
            [_double, _add_one], input_data=5, expected_output_type=int
        )
        assert result.passed is True

    def test_pipeline_type_mismatch(self) -> None:
        """FAIL: Pipeline output doesn't match expected type."""
        with pytest.raises(MltkAssertionError):
            assert_pipeline(
                [_double], input_data=5, expected_output_type=str
            )
