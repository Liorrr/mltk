# scan

## Overview
`mltk scan` auto-discovers model issues, produces findings and fix suggestions, and can generate runnable pytest files from them.

## Interface
- `register_scanner`, `scan` (package entry points)
- `ScanConfig`, `ScanContext` (`config.py`) - configuration and per-scan context
- `ScanEngine`, `ScanReport` (`engine.py`) - orchestrates scanners and builds the report
- `ScanFinding`, `FixSuggestion` (`finding.py`) - scanner output units
- `detect_feature_types`, `detect_model_type`, `detect_sensitive_columns` (`detect.py`)
- `LoadedModel`, `load_model` (`loader.py`) - CLI model loading
- `generate_test_file` (`codegen.py`) - findings to pytest code
- `format_console_output`, `format_fixes` (`console.py`) - terminal display
- Child: [scanners](./scanners/README.md) - built-in scanner implementations

## Invariants
- Scanners are registered via `register_scanner` and run by `ScanEngine`.
- Scanner output is always `ScanFinding`.

## Side Effects
- `load_model` reads model files from disk; `generate_test_file` writes a test file.

## Verification
- Run: `python -m pytest tests/test_scan -q`
