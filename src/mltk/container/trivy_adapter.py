"""Subprocess wrapper and JSON parser for the Trivy CLI.

The :class:`TrivyAdapter` shells out to the ``trivy`` binary, parses
its JSON SchemaVersion 2 output, and returns strongly typed
dataclasses (:class:`TrivyReport`, :class:`TrivyResult`,
:class:`TrivyVulnerability`, :class:`TrivySecret`) for downstream use
by the scanner and pytest-native assertions.

Design notes:
  * All subprocess calls pass ``shell=False`` (default) with an
    argument list -- no string interpolation reaches the shell.
  * Trivy's schema allows ``Vulnerabilities`` and ``Secrets`` to be
    ``null`` (no findings) or a list; both are normalised to lists.
  * Timeouts bubble up as :class:`TrivyTimeoutError` so callers can
    distinguish "took too long" from "no Trivy binary".
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from typing import Any

from mltk.container._binary import find_trivy_binary

__all__ = [
    "TrivyAdapter",
    "TrivyError",
    "TrivyReport",
    "TrivyResult",
    "TrivySecret",
    "TrivyTimeoutError",
    "TrivyVulnerability",
]


class TrivyError(RuntimeError):
    """Raised when Trivy exits with a non-zero status or malformed output."""


class TrivyTimeoutError(TrivyError):
    """Raised when a Trivy subprocess exceeds the requested timeout."""


@dataclass
class TrivyVulnerability:
    """A single CVE entry from a Trivy scan."""

    id: str
    severity: str
    title: str = ""
    pkg_name: str = ""
    installed_version: str = ""
    fixed_version: str = ""
    description: str = ""

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> TrivyVulnerability:
        """Construct from a raw Trivy vulnerability dict."""
        return cls(
            id=str(payload.get("VulnerabilityID", "")),
            severity=str(payload.get("Severity", "UNKNOWN")).upper(),
            title=str(payload.get("Title", "")),
            pkg_name=str(payload.get("PkgName", "")),
            installed_version=str(payload.get("InstalledVersion", "")),
            fixed_version=str(payload.get("FixedVersion", "")),
            description=str(payload.get("Description", "")),
        )


@dataclass
class TrivySecret:
    """A single exposed-secret finding from a Trivy scan."""

    rule_id: str
    category: str
    severity: str
    title: str
    target: str = ""
    match: str = ""

    @classmethod
    def from_json(cls, payload: dict[str, Any], target: str = "") -> TrivySecret:
        """Construct from a raw Trivy secret dict."""
        return cls(
            rule_id=str(payload.get("RuleID", "")),
            category=str(payload.get("Category", "")),
            severity=str(payload.get("Severity", "UNKNOWN")).upper(),
            title=str(payload.get("Title", "")),
            target=target,
            match=str(payload.get("Match", "")),
        )


@dataclass
class TrivyResult:
    """One target (layer / package source) from a Trivy report."""

    target: str
    type: str = ""
    vulnerabilities: list[TrivyVulnerability] = field(default_factory=list)
    secrets: list[TrivySecret] = field(default_factory=list)

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> TrivyResult:
        """Construct from a raw Trivy result dict."""
        target = str(payload.get("Target", ""))
        vulns_raw = payload.get("Vulnerabilities") or []
        secrets_raw = payload.get("Secrets") or []
        return cls(
            target=target,
            type=str(payload.get("Type", "")),
            vulnerabilities=[TrivyVulnerability.from_json(v) for v in vulns_raw],
            secrets=[TrivySecret.from_json(s, target=target) for s in secrets_raw],
        )


@dataclass
class TrivyReport:
    """A full Trivy report, parsed from SchemaVersion 2 JSON."""

    schema_version: int
    artifact_name: str
    artifact_type: str = ""
    results: list[TrivyResult] = field(default_factory=list)

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> TrivyReport:
        """Construct a report from Trivy's raw JSON payload."""
        results_raw = payload.get("Results") or []
        return cls(
            schema_version=int(payload.get("SchemaVersion", 0)),
            artifact_name=str(payload.get("ArtifactName", "")),
            artifact_type=str(payload.get("ArtifactType", "")),
            results=[TrivyResult.from_json(r) for r in results_raw],
        )

    def all_vulnerabilities(self) -> list[TrivyVulnerability]:
        """Flatten vulnerabilities across every result target."""
        out: list[TrivyVulnerability] = []
        for result in self.results:
            out.extend(result.vulnerabilities)
        return out

    def all_secrets(self) -> list[TrivySecret]:
        """Flatten secret findings across every result target."""
        out: list[TrivySecret] = []
        for result in self.results:
            out.extend(result.secrets)
        return out

    def count_by_severity(self) -> dict[str, int]:
        """Return a mapping of severity -> count of vulnerabilities."""
        counts: dict[str, int] = {}
        for vuln in self.all_vulnerabilities():
            counts[vuln.severity] = counts.get(vuln.severity, 0) + 1
        return counts


class TrivyAdapter:
    """Thin wrapper around the Trivy CLI.

    Responsible for binary discovery, argument assembly, subprocess
    invocation, timeout handling, and JSON parsing.
    """

    def __init__(
        self,
        binary: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        """Create a new adapter.

        Args:
            binary: Explicit path to a Trivy binary. When ``None``,
                the adapter resolves it via :func:`find_trivy_binary`.
            cache_dir: Optional Trivy cache directory passed via
                ``--cache-dir``.
        """
        self._binary = binary or find_trivy_binary()
        self._cache_dir = cache_dir

    @property
    def binary(self) -> str:
        """Path to the Trivy binary in use."""
        return self._binary

    def scan_image(
        self,
        image_ref: str,
        *,
        severity: list[str] | None = None,
        scanners: list[str] | None = None,
        timeout_s: int = 300,
    ) -> TrivyReport:
        """Run ``trivy image`` against ``image_ref`` and parse the report.

        Args:
            image_ref: Image reference (e.g. ``"alpine:3.18"``).
            severity: Optional severity allowlist (e.g. ``["CRITICAL", "HIGH"]``).
            scanners: Optional list of Trivy scanner modules (e.g.
                ``["vuln"]`` or ``["vuln", "secret"]``).
            timeout_s: Maximum seconds to wait for Trivy to finish.

        Returns:
            Parsed :class:`TrivyReport`.
        """
        args = self._build_image_args(
            image_ref,
            severity=severity,
            scanners=scanners,
        )
        return self._run(args, timeout_s=timeout_s)

    def scan_fs(
        self,
        path: str,
        *,
        severity: list[str] | None = None,
        scanners: list[str] | None = None,
        timeout_s: int = 300,
    ) -> TrivyReport:
        """Run ``trivy fs`` against ``path`` and parse the report.

        Args:
            path: Filesystem path to scan.
            severity: Optional severity allowlist.
            scanners: Optional list of Trivy scanner modules.
            timeout_s: Maximum seconds to wait for Trivy to finish.

        Returns:
            Parsed :class:`TrivyReport`.
        """
        args = [
            self._binary,
            "fs",
            "--format", "json",
            "--quiet",
        ]
        if self._cache_dir:
            args.extend(["--cache-dir", self._cache_dir])
        if severity:
            args.extend(["--severity", ",".join(severity)])
        if scanners:
            args.extend(["--scanners", ",".join(scanners)])
        args.append(path)
        return self._run(args, timeout_s=timeout_s)

    def _build_image_args(
        self,
        image_ref: str,
        *,
        severity: list[str] | None,
        scanners: list[str] | None,
    ) -> list[str]:
        """Assemble the CLI argument list for ``trivy image``."""
        args = [
            self._binary,
            "image",
            "--format", "json",
            "--quiet",
        ]
        if self._cache_dir:
            args.extend(["--cache-dir", self._cache_dir])
        if severity:
            args.extend(["--severity", ",".join(severity)])
        if scanners:
            args.extend(["--scanners", ",".join(scanners)])
        args.append(image_ref)
        return args

    def _run(self, args: list[str], *, timeout_s: int) -> TrivyReport:
        """Invoke Trivy and parse stdout as a :class:`TrivyReport`."""
        try:
            completed = subprocess.run(  # noqa: S603 - args list, shell=False
                args,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TrivyTimeoutError(
                f"Trivy scan exceeded {timeout_s}s timeout"
            ) from exc
        except FileNotFoundError as exc:
            raise TrivyError(
                f"Trivy binary not executable at {self._binary!r}"
            ) from exc

        if completed.returncode != 0:
            raise TrivyError(
                f"Trivy exited with status {completed.returncode}: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )

        stdout = completed.stdout.strip()
        if not stdout:
            raise TrivyError("Trivy produced no output")

        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise TrivyError(
                f"Failed to parse Trivy JSON output: {exc}"
            ) from exc

        return TrivyReport.from_json(payload)
