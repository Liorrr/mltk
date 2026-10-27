# mltk.container

## Overview
Container image security scanning built on the Trivy CLI: pytest-native assertions and a scanner that yields `ScanFinding` objects.

## Interface
- `assert_container_vulnerabilities(...)` - fails when CVE counts exceed configured severity thresholds.
- `assert_no_secrets_in_image(...)` - fails when exposed secrets are found.
- `ContainerScanner` - `scan(image_ref, **cfg) -> list[ScanFinding]`; `adapter` property.
- `TrivyAdapter` - subprocess wrapper (`scan_image`, `scan_fs`, `binary`); parsed types `TrivyReport`, `TrivyResult`, `TrivyVulnerability`, `TrivySecret`; errors `TrivyError`, `TrivyTimeoutError`.
- `_binary.find_trivy_binary()` - locates the binary with an actionable error message.

## Invariants
- Trivy is invoked with an argument list and `shell=False`.
- Severity order is CRITICAL, HIGH, MEDIUM, LOW, UNKNOWN.
- A timeout raises `TrivyTimeoutError`.

## Side Effects
- Runs the external `trivy` binary as a subprocess; it may use the network and its own cache.

## Verification
- Run: `python -m pytest tests/test_container -q`
