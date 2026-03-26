"""Smoke test for Rust+Python bridge integration."""
import pytest


def test_rust_bridge_importable():
    """Verify the Rust extension can be imported."""
    try:
        from mltk._rust import RUST_AVAILABLE
        # RUST_AVAILABLE may be False if not compiled, that's OK
        assert isinstance(RUST_AVAILABLE, bool)
    except ImportError:
        pytest.skip("Rust extension not compiled")


def test_bridge_functions_exist():
    """Verify all bridge functions are defined."""
    from mltk._rust import ks_test
    # These should work even without Rust (numpy fallback)
    stat, p = ks_test([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert isinstance(stat, float)
    assert isinstance(p, float)
