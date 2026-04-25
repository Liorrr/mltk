"""Tests for :mod:`mltk.container.trivy_adapter` and :mod:`mltk.container._binary`."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from mltk.container import _binary
from mltk.container.trivy_adapter import (
    TrivyAdapter,
    TrivyError,
    TrivyReport,
    TrivyTimeoutError,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "trivy_report.json"


def test_report_from_json_counts(trivy_report: TrivyReport) -> None:
    """Fixture report parses into the expected CVE + secret counts."""
    vulns = trivy_report.all_vulnerabilities()
    assert len(vulns) == 4

    counts = trivy_report.count_by_severity()
    assert counts.get("CRITICAL") == 1
    assert counts.get("HIGH") == 2
    assert counts.get("MEDIUM") == 1

    secrets = trivy_report.all_secrets()
    assert len(secrets) == 1
    assert secrets[0].rule_id == "aws-access-key-id"


def test_report_handles_null_vulnerabilities() -> None:
    """A result with ``Vulnerabilities: null`` is normalised to empty."""
    payload = {
        "SchemaVersion": 2,
        "ArtifactName": "empty:latest",
        "Results": [
            {"Target": "empty:latest", "Type": "alpine", "Vulnerabilities": None},
        ],
    }
    report = TrivyReport.from_json(payload)
    assert report.all_vulnerabilities() == []
    assert report.all_secrets() == []


def test_report_handles_missing_results_key() -> None:
    """A payload with no ``Results`` key still parses."""
    payload = {"SchemaVersion": 2, "ArtifactName": "empty:latest"}
    report = TrivyReport.from_json(payload)
    assert report.results == []


def test_vulnerability_fields_populated(trivy_report: TrivyReport) -> None:
    """Vulnerability objects expose CVE id, pkg, and fixed version."""
    vulns = trivy_report.all_vulnerabilities()
    critical = [v for v in vulns if v.severity == "CRITICAL"][0]
    assert critical.id == "CVE-2024-1234"
    assert critical.pkg_name == "openssl"
    assert critical.fixed_version == "3.1.5-r0"


def test_adapter_scan_image_invokes_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``scan_image`` runs Trivy with the expected argv and parses JSON."""
    monkeypatch.setattr(
        "mltk.container.trivy_adapter.find_trivy_binary",
        lambda: "/usr/bin/mock-trivy",
    )

    captured: dict[str, object] = {}
    fixture_text = FIXTURE_PATH.read_text(encoding="utf-8")

    def fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
        captured["args"] = args
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout=fixture_text, stderr=""
        )

    monkeypatch.setattr("mltk.container.trivy_adapter.subprocess.run", fake_run)

    adapter = TrivyAdapter()
    report = adapter.scan_image("alpine:3.18")

    argv = captured["args"]
    assert isinstance(argv, list)
    assert argv[0] == "/usr/bin/mock-trivy"
    assert argv[1] == "image"
    assert "--format" in argv
    assert "json" in argv
    assert argv[-1] == "alpine:3.18"
    assert captured["kwargs"].get("timeout") == 300  # type: ignore[union-attr]
    assert len(report.all_vulnerabilities()) == 4


def test_adapter_scan_image_passes_severity_and_scanners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Severity and scanner CLI flags are passed through."""
    monkeypatch.setattr(
        "mltk.container.trivy_adapter.find_trivy_binary",
        lambda: "/usr/bin/mock-trivy",
    )
    fixture_text = FIXTURE_PATH.read_text(encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
        captured["args"] = args
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout=fixture_text, stderr=""
        )

    monkeypatch.setattr("mltk.container.trivy_adapter.subprocess.run", fake_run)

    adapter = TrivyAdapter(cache_dir="/tmp/cache")
    adapter.scan_image(
        "alpine:3.18",
        severity=["CRITICAL", "HIGH"],
        scanners=["vuln", "secret"],
    )
    argv = captured["args"]
    assert isinstance(argv, list)
    assert "--severity" in argv
    idx = argv.index("--severity")
    assert argv[idx + 1] == "CRITICAL,HIGH"
    assert "--scanners" in argv
    sidx = argv.index("--scanners")
    assert argv[sidx + 1] == "vuln,secret"
    assert "--cache-dir" in argv


def test_adapter_timeout_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Subprocess timeout maps to :class:`TrivyTimeoutError`."""
    monkeypatch.setattr(
        "mltk.container.trivy_adapter.find_trivy_binary",
        lambda: "/usr/bin/mock-trivy",
    )

    def fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
        raise subprocess.TimeoutExpired(cmd=args, timeout=1)

    monkeypatch.setattr("mltk.container.trivy_adapter.subprocess.run", fake_run)

    adapter = TrivyAdapter()
    with pytest.raises(TrivyTimeoutError):
        adapter.scan_image("alpine:3.18", timeout_s=1)


def test_adapter_non_zero_exit_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-zero Trivy exit maps to :class:`TrivyError`."""
    monkeypatch.setattr(
        "mltk.container.trivy_adapter.find_trivy_binary",
        lambda: "/usr/bin/mock-trivy",
    )

    def fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
        return subprocess.CompletedProcess(
            args=args, returncode=2, stdout="", stderr="boom"
        )

    monkeypatch.setattr("mltk.container.trivy_adapter.subprocess.run", fake_run)
    adapter = TrivyAdapter()
    with pytest.raises(TrivyError, match="status 2"):
        adapter.scan_image("alpine:3.18")


def test_adapter_malformed_json_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed stdout maps to :class:`TrivyError`."""
    monkeypatch.setattr(
        "mltk.container.trivy_adapter.find_trivy_binary",
        lambda: "/usr/bin/mock-trivy",
    )

    def fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
        return subprocess.CompletedProcess(
            args=args, returncode=0, stdout="not json", stderr=""
        )

    monkeypatch.setattr("mltk.container.trivy_adapter.subprocess.run", fake_run)
    adapter = TrivyAdapter()
    with pytest.raises(TrivyError, match="parse"):
        adapter.scan_image("alpine:3.18")


def test_find_trivy_binary_missing_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing binary on PATH and in trivy-py raises ImportError."""
    monkeypatch.setattr(_binary.shutil, "which", lambda name: None)

    import sys as _sys  # noqa: PLC0415

    monkeypatch.setitem(_sys.modules, "trivy", None)

    with pytest.raises(ImportError, match="Trivy binary not found"):
        _binary.find_trivy_binary()


def test_find_trivy_binary_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Binary on PATH is returned verbatim."""
    monkeypatch.setattr(
        _binary.shutil, "which", lambda name: "/opt/bin/trivy"
    )
    assert _binary.find_trivy_binary() == "/opt/bin/trivy"


def test_scan_fs_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    """``scan_fs`` issues a ``trivy fs`` invocation with the given path."""
    monkeypatch.setattr(
        "mltk.container.trivy_adapter.find_trivy_binary",
        lambda: "/usr/bin/mock-trivy",
    )
    captured: dict[str, object] = {}

    def fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
        captured["args"] = args
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=json.dumps(
                {"SchemaVersion": 2, "ArtifactName": "/tmp", "Results": []}
            ),
            stderr="",
        )

    monkeypatch.setattr("mltk.container.trivy_adapter.subprocess.run", fake_run)
    adapter = TrivyAdapter()
    adapter.scan_fs("/tmp")
    argv = captured["args"]
    assert isinstance(argv, list)
    assert argv[1] == "fs"
    assert argv[-1] == "/tmp"
