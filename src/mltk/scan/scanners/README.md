# scan/scanners

## Overview
Built-in scanners for `mltk scan`. Each scanner subclasses `Scanner` and inspects a `ScanContext`, returning `ScanFinding` objects.

## Interface
- `base.Scanner` - abstract base class for scanners
- `BUILTIN_SCANNERS` (`__init__.py`) - ordered list of the built-in scanner classes below
- `BiasScanner` (`bias.py`) - fairness violations across demographic groups
- `CalibrationScanner` (`calibration.py`) - probability miscalibration
- `DataScanner` (`data.py`) - data quality checks before evaluation
- `DriftScanner` (`drift.py`) - distribution shifts within the dataset
- `LeakageScanner` (`leakage.py`) - feature-to-target leakage
- `OverfitScanner` (`overfit.py`) - train-test performance gap
- `RobustnessScanner` (`robustness.py`) - prediction instability
- `SliceScanner` (`slice.py`) - underperforming subgroups
- Parent: [scan](../README.md)

## Invariants
- Scanners do not modify the model or data they receive.
- Findings use the `ScanFinding` schema defined in `../finding.py`.

## Side Effects
- None intended; computation only (may call model prediction methods).

## Verification
- Run: `python -m pytest tests/test_scan -q`
