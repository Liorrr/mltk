"""Tests for mltk.pipeline.e2e -- end-to-end pipeline validation.

E2E pipeline tests verify that a chain of processing steps (data loading,
feature engineering, model inference) runs without crashing and produces
the expected output type. These catch integration bugs that unit tests miss:
a step that works alone may fail when wired to the next step's output.
"""

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.pipeline.e2e import assert_pipeline


def _double(x: int) -> int:
    """Simulates a feature scaling step (multiply by 2)."""
    return x * 2


def _add_one(x: int) -> int:
    """Simulates a feature offset step (add 1)."""
    return x + 1


def _broken_step(x: int) -> int:
    """Simulates a pipeline step that crashes due to data format error."""
    raise ValueError("Data format error")


class TestAssertPipeline:
    """E2E pipeline tests.

    Validates that assert_pipeline correctly chains steps, reports which
    step failed, verifies output types, and counts completed steps.
    """

    def test_pipeline_runs(self) -> None:
        """PASS: All 3 pipeline steps complete successfully.

        WHY: The simplest pipeline scenario -- data flows through double,
        add_one, double without error. Verifying completed_steps=3 ensures
        every step actually executed (not just the first one).
        Expected: result.passed is True, completed_steps=3.
        """
        result = assert_pipeline([_double, _add_one, _double], input_data=5)
        assert result.passed is True
        assert result.details["completed_steps"] == 3

    def test_pipeline_fails_at_step(self) -> None:
        """FAIL: Pipeline fails at step 1 (the broken middle step).

        WHY: When a pipeline crashes, you need to know WHICH step failed.
        The error message must include the step index so engineers can
        debug the specific transformation that broke.
        Expected: MltkAssertionError with "step 1" in message.
        """
        with pytest.raises(MltkAssertionError) as exc:
            assert_pipeline([_double, _broken_step, _add_one], input_data=5)
        assert "step 1" in str(exc.value)

    def test_pipeline_type_check(self) -> None:
        """PASS: Pipeline output is an int, matching expected_output_type.

        WHY: ML pipelines often produce wrong types silently (e.g., numpy
        array instead of pandas Series). Type checking at the end catches
        interface mismatches before downstream consumers break.
        Expected: result.passed is True.
        """
        result = assert_pipeline(
            [_double, _add_one], input_data=5, expected_output_type=int
        )
        assert result.passed is True

    def test_pipeline_type_mismatch(self) -> None:
        """FAIL: Pipeline outputs int but str was expected.

        WHY: If a consumer expects a string prediction ("positive"/"negative")
        but the pipeline returns an integer (0/1), the integration silently
        breaks. This type gate catches the mismatch.
        Expected: MltkAssertionError raised.
        """
        with pytest.raises(MltkAssertionError):
            assert_pipeline(
                [_double], input_data=5, expected_output_type=str
            )
