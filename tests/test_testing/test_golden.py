"""Tests for mltk.testing.golden — versioned golden baseline management."""
from __future__ import annotations

import numpy as np
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.testing.golden import assert_matches_golden, load_golden, save_golden

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path):
    # SCENARIO: save a dict and load it back
    # WHY: basic persistence must be lossless for simple scalar data
    # EXPECTED: loaded data matches saved data exactly
    data = {"accuracy": 0.95, "f1": 0.88}
    path = tmp_path / "baseline.json"

    save_golden(data, path)
    envelope = load_golden(path)

    assert envelope["data"] == data


def test_save_with_version(tmp_path):
    # SCENARIO: save with an explicit version string
    # WHY: version metadata must round-trip correctly for provenance tracking
    # EXPECTED: loaded envelope has the correct version field
    path = tmp_path / "v_baseline.json"
    save_golden({"loss": 0.05}, path, version="2.1.0")

    envelope = load_golden(path)

    assert envelope["version"] == "2.1.0"
    assert "timestamp" in envelope


def test_assert_matches_golden_pass(tmp_path):
    # SCENARIO: current data is within tolerance of the golden baseline
    # WHY: assertion should pass without raising
    # EXPECTED: TestResult.passed == True
    path = tmp_path / "golden.json"
    save_golden({"metric": 0.90}, path)

    result = assert_matches_golden({"metric": 0.905}, path, tolerance=0.01)

    assert result.passed is True
    assert result.details["max_diff"] <= 0.01


def test_assert_matches_golden_fail(tmp_path):
    # SCENARIO: current data deviates beyond tolerance
    # WHY: assertion must raise MltkAssertionError on CRITICAL mismatch
    # EXPECTED: MltkAssertionError raised, result.passed == False
    path = tmp_path / "golden.json"
    save_golden({"metric": 0.90}, path)

    with pytest.raises(MltkAssertionError) as exc_info:
        assert_matches_golden({"metric": 0.80}, path, tolerance=0.01)

    assert exc_info.value.result.passed is False


def test_numpy_array_golden(tmp_path):
    # SCENARIO: save a numpy array and compare against a close array
    # WHY: numpy arrays must be serialised to JSON and compared numerically
    # EXPECTED: save succeeds, comparison within tolerance passes
    arr = np.array([0.1, 0.2, 0.3])
    path = tmp_path / "arr_golden.json"

    save_golden(arr, path)

    # Slightly perturbed — within tolerance
    result = assert_matches_golden(arr + 0.001, path, tolerance=0.01)
    assert result.passed is True


def test_numpy_array_golden_fail(tmp_path):
    # SCENARIO: numpy array deviates beyond tolerance
    # WHY: numeric comparison must detect large deviations in arrays
    # EXPECTED: MltkAssertionError raised
    arr = np.array([1.0, 2.0, 3.0])
    path = tmp_path / "arr_fail.json"
    save_golden(arr, path)

    with pytest.raises(MltkAssertionError):
        assert_matches_golden(arr + 1.0, path, tolerance=0.01)


def test_default_version(tmp_path):
    # SCENARIO: save without specifying a version
    # WHY: default version should be "1.0.0"
    # EXPECTED: envelope["version"] == "1.0.0"
    path = tmp_path / "default_version.json"
    save_golden([1, 2, 3], path)

    envelope = load_golden(path)
    assert envelope["version"] == "1.0.0"


def test_load_nonexistent_file(tmp_path):
    # SCENARIO: attempt to load a file that does not exist
    # WHY: should raise FileNotFoundError, not a cryptic JSON error
    # EXPECTED: FileNotFoundError raised
    with pytest.raises(FileNotFoundError):
        load_golden(tmp_path / "missing.json")


def test_result_has_duration(tmp_path):
    # SCENARIO: assert_matches_golden uses @timed_assertion decorator
    # WHY: duration_ms should be populated (>= 0)
    # EXPECTED: result.duration_ms >= 0
    path = tmp_path / "timed.json"
    save_golden({"v": 0.5}, path)

    result = assert_matches_golden({"v": 0.5}, path)
    assert result.duration_ms >= 0.0
