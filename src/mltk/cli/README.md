# mltk.cli

## Overview
Typer-based command line (`mltk = mltk.cli.app:main`). Commands are registered inside `main()` in `app.py`; helper modules hold the larger commands.

## Interface
- `app.main()` - entry point. Commands include `version`, `init`, `scan`, `drift`, `score`, `doctor`, `test`, `chat`, `server`, `scan-model`, plus sub-apps `contract`, `docs`, `registry`, `notify`, `container`.
- `_discovery.discover_assertions(...)` - collects public assertions for the assertion listing command.
- `importer.import_dataset(...)` - dataset import that builds a test suite and writes it to YAML.
- `security_scan.register_security_scan(app)` - attaches the security scan command.
- `container.app` - Typer sub-app; its `scan` command is `container.container_scan` (Trivy image scan, optional JUnit XML output).

## Invariants
- Heavy dependencies (typer, optional extras) are imported lazily inside command bodies.
- Commands delegate to library packages; they hold no scanning logic of their own.

## Side Effects
- `init` writes config and a sample test file; `contract init` writes `contract.yaml`.
- `docs serve/build/open` spawn subprocesses and may open a browser; honor `MLTK_DOCS_PORT` and `MLTK_DOCS_HOST`.
- `container scan`, `import`, and `notify` perform subprocess, file, or network I/O.

## Verification
- Run: `python -m pytest tests/test_cli -q`
