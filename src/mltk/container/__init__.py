"""Container image security scanning for mltk.

This module integrates `Trivy <https://trivy.dev>`_ so users can
assert that their container images ship free of high-severity CVEs
and exposed secrets, directly from pytest or from an mltk scan.

Public API:
    :func:`assert_container_vulnerabilities`: pytest-native assertion
        that fails when CVE counts exceed configured thresholds.
    :func:`assert_no_secrets_in_image`: pytest-native assertion that
        fails when Trivy finds exposed credentials.
    :class:`ContainerScanner`: scanner-style entry point that returns
        :class:`~mltk.scan.finding.ScanFinding` objects for reports.
    :class:`TrivyAdapter`: lower-level subprocess wrapper, useful for
        custom workflows and for testing.
"""

from __future__ import annotations

from mltk.container.assertions import (
    assert_container_vulnerabilities,
    assert_no_secrets_in_image,
)
from mltk.container.scanner import ContainerScanner
from mltk.container.trivy_adapter import TrivyAdapter

__all__ = [
    "ContainerScanner",
    "TrivyAdapter",
    "assert_container_vulnerabilities",
    "assert_no_secrets_in_image",
]
