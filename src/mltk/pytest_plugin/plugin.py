"""pytest plugin entry point for mltk. Auto-registered via pyproject.toml entry-points."""

from __future__ import annotations


def pytest_configure(config):  # type: ignore[no-untyped-def]
    """Register mltk markers."""
    config.addinivalue_line("markers", "ml_data: data quality tests")
    config.addinivalue_line("markers", "ml_model: model quality tests")
    config.addinivalue_line("markers", "ml_drift: drift detection tests")
    config.addinivalue_line("markers", "ml_inference: inference performance tests")
    config.addinivalue_line("markers", "ml_slow: tests that take >30 seconds")
    config.addinivalue_line("markers", "ml_nondeterministic: tests with inherent randomness")
