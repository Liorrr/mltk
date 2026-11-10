"""Offline parity of stored product and scoring-harness runs."""

from __future__ import annotations

from mltk.parity.assertions import assert_parity
from mltk.parity.control import ControlFixture, check_positive_control, make_positive_control
from mltk.parity.engine import compare_product, compare_runs
from mltk.parity.harness import compare_harness
from mltk.parity.loaders import load_run
from mltk.parity.models import Item, LayerSpec, Manifest, ParityConfig, ParityReport, Run, Trial
from mltk.parity.output import ArtifactPaths, run_to_dict, write_report
from mltk.parity.states import reduce_item

__all__ = [
    "ArtifactPaths", "ControlFixture", "Item", "LayerSpec", "Manifest", "ParityConfig",
    "ParityReport", "Run", "Trial", "assert_parity", "check_positive_control", "compare_harness",
    "compare_product", "compare_runs", "load_run", "make_positive_control", "reduce_item",
    "run_to_dict", "write_report",
]
