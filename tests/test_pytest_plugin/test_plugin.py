"""Tests for mltk.pytest_plugin -- markers, fixtures, and --mltk-report.

These tests verify the pytest plugin works correctly: markers are registered,
fixtures load config, and --mltk-report produces output.
"""

import pytest

from mltk.core.config import MltkConfig
from mltk.pytest_plugin.plugin import MltkReportCollector


class TestMarkers:
    """Verify ML markers are registered."""

    def test_ml_data_marker_registered(self, request: pytest.FixtureRequest) -> None:
        """ml_data marker exists in pytest configuration."""
        marker_lines = request.config.getini("markers")
        marker_names = [str(line).split(":")[0].strip() for line in marker_lines]
        assert "ml_data" in marker_names

    def test_ml_model_marker_registered(self, request: pytest.FixtureRequest) -> None:
        """ml_model marker exists in pytest configuration."""
        marker_lines = request.config.getini("markers")
        marker_names = [line.split(":")[0].strip() for line in marker_lines]
        assert "ml_model" in marker_names

    def test_ml_smoke_marker_registered(self, request: pytest.FixtureRequest) -> None:
        """ml_smoke marker (new in Sprint 4) exists."""
        marker_lines = request.config.getini("markers")
        marker_names = [line.split(":")[0].strip() for line in marker_lines]
        assert "ml_smoke" in marker_names

    def test_ml_gpu_marker_registered(self, request: pytest.FixtureRequest) -> None:
        """ml_gpu marker (new in Sprint 4) exists."""
        marker_lines = request.config.getini("markers")
        marker_names = [line.split(":")[0].strip() for line in marker_lines]
        assert "ml_gpu" in marker_names


class TestFixtures:
    """Verify plugin fixtures work."""

    def test_ml_config_fixture(self, ml_config: MltkConfig) -> None:
        """ml_config fixture loads MltkConfig."""
        assert isinstance(ml_config, MltkConfig)
        assert ml_config.drift_method == "ks"


class TestReportCollector:
    """Verify report collector accumulates results."""

    def test_collector_add_and_count(self) -> None:
        """Collector tracks pass/fail counts."""
        collector = MltkReportCollector()
        collector.add("test_a", "passed", 0.1)
        collector.add("test_b", "failed", 0.2)
        collector.add("test_c", "passed", 0.1)

        assert collector.total == 3
        assert collector.passed_count == 2
        assert collector.failed_count == 1
