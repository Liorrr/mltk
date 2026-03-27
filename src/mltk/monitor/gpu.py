"""GPU monitoring via nvidia-smi -- no Prometheus required.

Provides direct GPU health assertions by querying nvidia-smi locally.
Use these when you don't have a Prometheus/DCGM stack but need GPU checks.

Functions:
    assert_gpu_utilization_local — GPU compute utilization below threshold
    assert_gpu_memory_local      — GPU memory usage below threshold
"""

from __future__ import annotations

import subprocess

from mltk.core.assertion import assert_true, timed_assertion
from mltk.core.result import Severity, TestResult


def _run_nvidia_smi(query: str) -> str:
    """Run nvidia-smi with the given query and return stdout.

    Raises:
        FileNotFoundError: nvidia-smi not found on PATH.
        subprocess.CalledProcessError: nvidia-smi returned non-zero exit code.
    """
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return result.stdout.strip()


@timed_assertion
def assert_gpu_utilization_local(max_util: float = 0.95) -> TestResult:
    """Assert GPU utilization is below threshold using nvidia-smi.

    Queries ``nvidia-smi --query-gpu=utilization.gpu`` and checks that the
    highest utilization across all GPUs is below *max_util* (0-1 scale).

    Args:
        max_util: Maximum allowed GPU utilization (0.0-1.0). Default 0.95.

    Returns:
        TestResult with GPU utilization details.

    Example:
        >>> assert_gpu_utilization_local(max_util=0.90)
    """
    try:
        output = _run_nvidia_smi("utilization.gpu")
    except FileNotFoundError:
        return assert_true(
            False,
            name="monitor.gpu_utilization_local",
            message="nvidia-smi not found — cannot check GPU utilization",
            severity=Severity.WARNING,
        )
    except subprocess.CalledProcessError as exc:
        return assert_true(
            False,
            name="monitor.gpu_utilization_local",
            message=f"nvidia-smi failed: {exc}",
            severity=Severity.WARNING,
        )

    # Parse output — one line per GPU, value is percentage (0-100)
    gpu_utils = []
    for line in output.splitlines():
        line = line.strip()
        if line:
            gpu_utils.append(float(line) / 100.0)

    if not gpu_utils:
        return assert_true(
            False,
            name="monitor.gpu_utilization_local",
            message="nvidia-smi returned no GPU data",
            severity=Severity.WARNING,
        )

    max_observed = max(gpu_utils)
    passed = max_observed <= max_util
    message = (
        f"GPU utilization OK: {max_observed:.1%} <= {max_util:.1%}"
        if passed
        else f"GPU utilization high: {max_observed:.1%} > {max_util:.1%}"
    )

    return assert_true(
        passed,
        name="monitor.gpu_utilization_local",
        message=message,
        severity=Severity.CRITICAL,
        max_util=max_util,
        observed_utils=gpu_utils,
        max_observed=max_observed,
    )


@timed_assertion
def assert_gpu_memory_local(max_util: float = 0.90) -> TestResult:
    """Assert GPU memory usage is below threshold using nvidia-smi.

    Queries ``nvidia-smi --query-gpu=memory.used,memory.total`` and checks
    that the highest memory utilization across all GPUs is below *max_util*.

    Args:
        max_util: Maximum allowed memory utilization (0.0-1.0). Default 0.90.

    Returns:
        TestResult with GPU memory details.

    Example:
        >>> assert_gpu_memory_local(max_util=0.85)
    """
    try:
        output = _run_nvidia_smi("memory.used,memory.total")
    except FileNotFoundError:
        return assert_true(
            False,
            name="monitor.gpu_memory_local",
            message="nvidia-smi not found — cannot check GPU memory",
            severity=Severity.WARNING,
        )
    except subprocess.CalledProcessError as exc:
        return assert_true(
            False,
            name="monitor.gpu_memory_local",
            message=f"nvidia-smi failed: {exc}",
            severity=Severity.WARNING,
        )

    # Parse output — one line per GPU: "used, total" (MiB)
    gpu_mem_utils: list[float] = []
    gpu_mem_details: list[dict[str, float]] = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) != 2:
            continue
        used = float(parts[0].strip())
        total = float(parts[1].strip())
        util = used / total if total > 0 else 0.0
        gpu_mem_utils.append(util)
        gpu_mem_details.append({"used_mib": used, "total_mib": total, "util": util})

    if not gpu_mem_utils:
        return assert_true(
            False,
            name="monitor.gpu_memory_local",
            message="nvidia-smi returned no GPU memory data",
            severity=Severity.WARNING,
        )

    max_observed = max(gpu_mem_utils)
    passed = max_observed <= max_util
    message = (
        f"GPU memory OK: {max_observed:.1%} <= {max_util:.1%}"
        if passed
        else f"GPU memory high: {max_observed:.1%} > {max_util:.1%}"
    )

    return assert_true(
        passed,
        name="monitor.gpu_memory_local",
        message=message,
        severity=Severity.CRITICAL,
        max_util=max_util,
        gpu_memory=gpu_mem_details,
        max_observed=max_observed,
    )
