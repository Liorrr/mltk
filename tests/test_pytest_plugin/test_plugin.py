"""Tests for mltk.pytest_plugin -- markers, fixtures, and --mltk-report.

The pytest plugin is the primary user interface for mltk. It registers
custom markers (ml_data, ml_model, ml_smoke, ml_gpu) for test selection,
provides the ml_config fixture for accessing MltkConfig, and collects
results for HTML report generation. These tests verify:
1. All ML markers are registered (so -m "ml_data" works)
2. The ml_config fixture provides a valid config object
3. The report collector correctly tracks pass/fail counts
"""

import pytest

from mltk.core.config import MltkConfig
from mltk.pytest_plugin.plugin import MltkReportCollector


class TestMarkers:
    """Verify ML markers are registered in pytest.

    Markers allow users to run subsets of ML tests (e.g., `pytest -m ml_data`
    to run only data validation tests). If markers are not registered, pytest
    warns about unknown markers and -m filtering silently skips tests.
    """

    def test_ml_data_marker_registered(self, request: pytest.FixtureRequest) -> None:
        """PASS: ml_data marker is registered in pytest configuration.

        WHY: Users run `pytest -m ml_data` to execute only data quality tests
        (schema, drift, freshness, PII). If this marker is not registered,
        pytest shows "Unknown marker" warnings and the filter may not work.
        Expected: "ml_data" found in configured marker names.
        """
        marker_lines = request.config.getini("markers")
        marker_names = [str(line).split(":")[0].strip() for line in marker_lines]
        assert "ml_data" in marker_names

    def test_ml_model_marker_registered(self, request: pytest.FixtureRequest) -> None:
        """PASS: ml_model marker is registered in pytest configuration.

        WHY: Users run `pytest -m ml_model` to execute only model quality tests
        (metrics, bias, regression, slicing). Marker must be registered for
        clean pytest output and correct test selection.
        Expected: "ml_model" found in configured marker names.
        """
        marker_lines = request.config.getini("markers")
        marker_names = [line.split(":")[0].strip() for line in marker_lines]
        assert "ml_model" in marker_names

    def test_ml_smoke_marker_registered(self, request: pytest.FixtureRequest) -> None:
        """PASS: ml_smoke marker is registered (added in Sprint 4).

        WHY: Smoke tests are a fast subset (<30s) used in pre-commit hooks
        and PR checks. They must be selectable via `-m ml_smoke`.
        Expected: "ml_smoke" found in configured marker names.
        """
        marker_lines = request.config.getini("markers")
        marker_names = [line.split(":")[0].strip() for line in marker_lines]
        assert "ml_smoke" in marker_names

    def test_ml_gpu_marker_registered(self, request: pytest.FixtureRequest) -> None:
        """PASS: ml_gpu marker is registered (added in Sprint 4).

        WHY: GPU-only tests (e.g., CUDA inference latency) should be skippable
        on CPU-only CI runners. The ml_gpu marker enables `-m "not ml_gpu"`
        to exclude these tests.
        Expected: "ml_gpu" found in configured marker names.
        """
        marker_lines = request.config.getini("markers")
        marker_names = [line.split(":")[0].strip() for line in marker_lines]
        assert "ml_gpu" in marker_names


class TestFixtures:
    """Verify plugin fixtures provide correct objects.

    The ml_config fixture is the standard way to access MltkConfig in tests.
    It respects the config cascade (YAML > TOML > defaults) and is shared
    across the test session.
    """

    def test_ml_config_fixture(self, ml_config: MltkConfig) -> None:
        """PASS: ml_config fixture returns a valid MltkConfig with defaults.

        WHY: Every test that uses mltk assertions accesses config via this
        fixture (drift thresholds, report format, seed). If the fixture
        returns None or wrong type, all downstream assertions will crash.
        Expected: Instance of MltkConfig with drift_method="ks" (default).
        """
        assert isinstance(ml_config, MltkConfig)
        assert ml_config.drift_method == "ks"


class TestReportCollector:
    """Verify the report collector accumulates test results.

    MltkReportCollector is used by the pytest plugin to gather results
    during the test run. After collection, it feeds into the HTML report
    generator and ML Test Score calculator.
    """

    def test_collector_add_and_count(self) -> None:
        """PASS: Collector correctly counts passed and failed tests.

        WHY: The collector feeds into the HTML report summary ("2 passed,
        1 failed"). If counts are wrong, the report shows incorrect totals
        and the CI exit code may be wrong (exit 0 when tests actually failed).
        Expected: total=3, passed=2, failed=1.
        """
        collector = MltkReportCollector()
        collector.add("test_a", "passed", 0.1)
        collector.add("test_b", "failed", 0.2)
        collector.add("test_c", "passed", 0.1)

        assert collector.total == 3
        assert collector.passed_count == 2
        assert collector.failed_count == 1
