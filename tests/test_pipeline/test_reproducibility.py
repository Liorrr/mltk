"""Tests for mltk.pipeline.reproducibility -- deterministic training + artifact integrity."""

from pathlib import Path

import numpy as np
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.pipeline.reproducibility import assert_checksum, assert_reproducible


def _deterministic_func(x: int = 5) -> float:
    """Always returns the same value for same seed."""
    rng = np.random.default_rng(42)
    return float(rng.random() * x)


def _nondeterministic_func(x: int = 5) -> float:
    """Returns different values each call (ignores seed)."""
    import time

    return float(hash(time.time_ns()) % 1000) / 1000.0


class TestAssertReproducible:
    """Reproducibility tests."""

    def test_deterministic_passes(self) -> None:
        """PASS: Deterministic function produces same output every run."""
        result = assert_reproducible(_deterministic_func, 10, seed=42, runs=3)
        assert result.passed is True

    def test_nondeterministic_fails(self) -> None:
        """FAIL: Non-deterministic function produces different outputs."""
        with pytest.raises(MltkAssertionError):
            assert_reproducible(
                _nondeterministic_func, 10, seed=42, runs=3, tolerance=0.0001
            )

    def test_tolerance_boundary(self) -> None:
        """Tolerance allows small differences."""
        result = assert_reproducible(
            _deterministic_func, 10, seed=42, runs=3, tolerance=1.0
        )
        assert result.passed is True


class TestAssertChecksum:
    """File checksum validation tests."""

    def test_correct_checksum(self, tmp_path: Path) -> None:
        """PASS: File hash matches expected."""
        test_file = tmp_path / "model.bin"
        test_file.write_bytes(b"model data here")

        import hashlib

        expected = hashlib.sha256(b"model data here").hexdigest()
        result = assert_checksum(test_file, expected)
        assert result.passed is True

    def test_wrong_checksum(self, tmp_path: Path) -> None:
        """FAIL: File hash doesn't match — corruption detected."""
        test_file = tmp_path / "model.bin"
        test_file.write_bytes(b"model data here")

        with pytest.raises(MltkAssertionError):
            assert_checksum(test_file, "0000000000000000")

    def test_missing_file(self) -> None:
        """FAIL: File doesn't exist."""
        with pytest.raises(MltkAssertionError):
            assert_checksum("/nonexistent/model.bin", "abc123")
