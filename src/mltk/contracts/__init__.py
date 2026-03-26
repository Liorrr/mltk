"""Data contracts — YAML spec to auto-generate and validate ML data tests."""

from mltk.contracts.generator import generate_tests_from_contract
from mltk.contracts.schema import ColumnSpec, Contract, QualitySpec
from mltk.contracts.validator import validate_data

__all__ = [
    "validate_data",
    "generate_tests_from_contract",
    "Contract",
    "ColumnSpec",
    "QualitySpec",
]
