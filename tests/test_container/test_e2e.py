"""End-to-end smoke test -- runs only when ``MLTK_CONTAINER_E2E=1``.

This test requires a working Trivy binary and network access to pull
the target image. Skipped in normal CI to keep the suite fast and
hermetic.
"""

from __future__ import annotations

import os

import pytest


@pytest.mark.ml_slow
@pytest.mark.skipif(
    not os.environ.get("MLTK_CONTAINER_E2E"),
    reason="Set MLTK_CONTAINER_E2E=1 to run real Trivy scans",
)
def test_real_scan_alpine() -> None:
    """Scan ``alpine:3.18`` against very permissive thresholds."""
    from mltk.container import assert_container_vulnerabilities  # noqa: PLC0415

    result = assert_container_vulnerabilities(
        "alpine:3.18", max_critical=999, max_high=999,
    )
    assert result.passed
    assert result.details["image"] == "alpine:3.18"
