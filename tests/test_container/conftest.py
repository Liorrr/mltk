"""Shared fixtures for mltk.container tests.

The fixtures here all operate against a canned Trivy JSON report
(``fixtures/trivy_report.json``) so no real Trivy invocation, no
container runtime, and no network access are required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from mltk.container.trivy_adapter import TrivyAdapter, TrivyReport

if TYPE_CHECKING:
    from collections.abc import Callable


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "trivy_report.json"


@pytest.fixture
def trivy_report_payload() -> dict:
    """Raw Trivy JSON payload loaded from the fixture file."""
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def trivy_report(trivy_report_payload: dict) -> TrivyReport:
    """Parsed :class:`TrivyReport` built from the fixture payload."""
    return TrivyReport.from_json(trivy_report_payload)


@pytest.fixture
def mock_trivy_adapter(
    monkeypatch: pytest.MonkeyPatch,
    trivy_report: TrivyReport,
) -> TrivyAdapter:
    """Return a :class:`TrivyAdapter` whose scan methods return the fixture.

    Both :meth:`TrivyAdapter.scan_image` and :meth:`TrivyAdapter.scan_fs`
    are stubbed to return the same report without shelling out.
    """
    monkeypatch.setattr(
        "mltk.container.trivy_adapter.find_trivy_binary",
        lambda: "/usr/bin/mock-trivy",
    )
    adapter = TrivyAdapter(binary="/usr/bin/mock-trivy")

    def _scan_image(
        image_ref: str,
        **_kwargs: object,
    ) -> TrivyReport:
        return trivy_report

    def _scan_fs(path: str, **_kwargs: object) -> TrivyReport:
        return trivy_report

    monkeypatch.setattr(adapter, "scan_image", _scan_image)
    monkeypatch.setattr(adapter, "scan_fs", _scan_fs)
    return adapter


@pytest.fixture
def make_empty_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[], TrivyAdapter]:
    """Factory for adapters that return a Trivy report with no findings."""
    monkeypatch.setattr(
        "mltk.container.trivy_adapter.find_trivy_binary",
        lambda: "/usr/bin/mock-trivy",
    )

    def _factory() -> TrivyAdapter:
        adapter = TrivyAdapter(binary="/usr/bin/mock-trivy")
        empty_report = TrivyReport.from_json(
            {
                "SchemaVersion": 2,
                "ArtifactName": "empty:latest",
                "Results": [],
            }
        )
        monkeypatch.setattr(adapter, "scan_image", lambda *a, **k: empty_report)
        monkeypatch.setattr(adapter, "scan_fs", lambda *a, **k: empty_report)
        return adapter

    return _factory
