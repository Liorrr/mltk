"""Tests for mltk.pipeline.reproducibility -- deterministic training + artifact integrity."""

import itertools
from pathlib import Path

import numpy as np
import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.pipeline.reproducibility import (
    _resolve_default_seed,
    assert_checksum,
    assert_reproducible,
)


class TestResolveDefaultSeed:
    """S102: config.seed was dead — MLTK_SEED/yaml never reached any consumer.

    Wiring follows the S101 explicit_fields provenance rule: an explicit arg
    wins, config is honoured only when the user explicitly set seed, and the
    legacy built-in default (42) is preserved otherwise.
    """

    def test_explicit_seed_wins_over_config(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MLTK_SEED", "7")
        assert _resolve_default_seed(99) == 99

    def test_explicit_zero_is_honoured_not_treated_as_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # 0 is falsy — the guard must test `is not None`, not truthiness.
        monkeypatch.setenv("MLTK_SEED", "7")
        assert _resolve_default_seed(0) == 0

    def test_explicitly_configured_seed_is_used(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MLTK_SEED", "7")
        assert _resolve_default_seed(None) == 7

    def test_unconfigured_seed_falls_back_to_legacy_default(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("MLTK_SEED", raising=False)
        assert _resolve_default_seed(None) == 42

    def test_config_load_failure_falls_back_to_legacy_default(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        def _boom(*_args, **_kwargs):
            raise RuntimeError("config exploded")

        monkeypatch.setattr("mltk.core.config.MltkConfig.load", _boom)
        assert _resolve_default_seed(None) == 42


def _deterministic_func(x: int = 5) -> float:
    """Always returns the same value for same seed."""
    rng = np.random.default_rng(42)
    return float(rng.random() * x)


_nondeterministic_calls = itertools.count()


def _nondeterministic_func(x: int = 5) -> float:
    """Returns a different value on every call (ignores seed).

    Uses a monotonic counter rather than a clock: on fast runners all
    of assert_reproducible's runs can land inside one timer-resolution
    tick, making clock-derived values collide and the test flake.
    """
    return float(next(_nondeterministic_calls))


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
