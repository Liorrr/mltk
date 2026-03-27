"""Tests for mltk.monitor.gpu -- local GPU monitoring via nvidia-smi.

All tests mock subprocess.run since nvidia-smi is not available in CI.
"""

from unittest.mock import MagicMock, patch

import pytest

from mltk.core.assertion import MltkAssertionError
from mltk.monitor.gpu import assert_gpu_memory_local, assert_gpu_utilization_local


class TestAssertGpuUtilizationLocal:
    """GPU utilization checks via mocked nvidia-smi."""

    @patch("mltk.monitor.gpu.subprocess.run")
    def test_utilization_below_threshold(self, mock_run: MagicMock) -> None:
        """PASS: GPU at 60% utilization, threshold 95%."""
        mock_run.return_value = MagicMock(stdout="60\n", returncode=0)
        result = assert_gpu_utilization_local(max_util=0.95)
        assert result.passed is True

    @patch("mltk.monitor.gpu.subprocess.run")
    def test_utilization_above_threshold(self, mock_run: MagicMock) -> None:
        """FAIL: GPU at 98% utilization, threshold 95%."""
        mock_run.return_value = MagicMock(stdout="98\n", returncode=0)
        with pytest.raises(MltkAssertionError):
            assert_gpu_utilization_local(max_util=0.95)

    @patch("mltk.monitor.gpu.subprocess.run")
    def test_multiple_gpus(self, mock_run: MagicMock) -> None:
        """PASS: Two GPUs at 40% and 70%, threshold 80%."""
        mock_run.return_value = MagicMock(stdout="40\n70\n", returncode=0)
        result = assert_gpu_utilization_local(max_util=0.80)
        assert result.passed is True
        assert result.details["max_observed"] == 0.7

    @patch("mltk.monitor.gpu.subprocess.run")
    def test_multiple_gpus_one_over(self, mock_run: MagicMock) -> None:
        """FAIL: Two GPUs, one at 92% exceeds 90% threshold."""
        mock_run.return_value = MagicMock(stdout="50\n92\n", returncode=0)
        with pytest.raises(MltkAssertionError):
            assert_gpu_utilization_local(max_util=0.90)

    @patch("mltk.monitor.gpu.subprocess.run", side_effect=FileNotFoundError)
    def test_nvidia_smi_not_found(self, mock_run: MagicMock) -> None:
        """WARNING: nvidia-smi not installed — returns failed result (no raise)."""
        result = assert_gpu_utilization_local()
        assert result.passed is False
        assert "not found" in result.message

    @patch("mltk.monitor.gpu.subprocess.run")
    def test_empty_output(self, mock_run: MagicMock) -> None:
        """WARNING: nvidia-smi returns empty output — returns failed result."""
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        result = assert_gpu_utilization_local()
        assert result.passed is False
        assert "no GPU data" in result.message


class TestAssertGpuMemoryLocal:
    """GPU memory checks via mocked nvidia-smi."""

    @patch("mltk.monitor.gpu.subprocess.run")
    def test_memory_below_threshold(self, mock_run: MagicMock) -> None:
        """PASS: 4000 MiB used of 8000 MiB total = 50%, threshold 90%."""
        mock_run.return_value = MagicMock(stdout="4000, 8000\n", returncode=0)
        result = assert_gpu_memory_local(max_util=0.90)
        assert result.passed is True

    @patch("mltk.monitor.gpu.subprocess.run")
    def test_memory_above_threshold(self, mock_run: MagicMock) -> None:
        """FAIL: 7500 MiB used of 8000 MiB total = 93.75%, threshold 90%."""
        mock_run.return_value = MagicMock(stdout="7500, 8000\n", returncode=0)
        with pytest.raises(MltkAssertionError):
            assert_gpu_memory_local(max_util=0.90)

    @patch("mltk.monitor.gpu.subprocess.run")
    def test_multiple_gpus_memory(self, mock_run: MagicMock) -> None:
        """PASS: Two GPUs both under threshold."""
        mock_run.return_value = MagicMock(
            stdout="2000, 8000\n3000, 8000\n", returncode=0
        )
        result = assert_gpu_memory_local(max_util=0.90)
        assert result.passed is True
        assert len(result.details["gpu_memory"]) == 2

    @patch("mltk.monitor.gpu.subprocess.run", side_effect=FileNotFoundError)
    def test_nvidia_smi_not_found(self, mock_run: MagicMock) -> None:
        """WARNING: nvidia-smi not installed — returns failed result (no raise)."""
        result = assert_gpu_memory_local()
        assert result.passed is False
        assert "not found" in result.message

    @patch("mltk.monitor.gpu.subprocess.run")
    def test_empty_output(self, mock_run: MagicMock) -> None:
        """WARNING: nvidia-smi returns empty output — returns failed result."""
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        result = assert_gpu_memory_local()
        assert result.passed is False
        assert "no GPU memory data" in result.message
