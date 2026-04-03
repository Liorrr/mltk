"""Tests for the mltk_dataset MCP tool.

Covers dataset lookup, version handling, quality metrics,
error responses, and response structure.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from ._helpers import (
    assert_error,
    assert_ok,
    assert_valid_json,
    registered_tools,
)


def _call_dataset(**kwargs):
    """Call mltk_dataset and return parsed JSON.

    Bypasses call_tool() because the tool's first parameter
    is also named ``name``, which would collide with
    call_tool(name=...).
    """
    fn = registered_tools["mltk_dataset"]
    raw = fn(**kwargs)
    return json.loads(raw)


def _call_dataset_raw(**kwargs):
    """Call mltk_dataset and return the raw JSON string."""
    fn = registered_tools["mltk_dataset"]
    return fn(**kwargs)


def _make_found_registry():
    """Build a mock DatasetRegistry where the dataset exists."""
    mock_sample_1 = MagicMock(input="q1")
    mock_sample_2 = MagicMock(input="q2")
    mock_ds = MagicMock()
    mock_ds.name = "mmlu"
    mock_ds.version = "1.0"
    mock_ds.card.to_dict.return_value = {"description": "MMLU benchmark"}
    mock_ds.samples = [mock_sample_1, mock_sample_2]
    mock_ds.sample_count = 2
    mock_ds.target_coverage = 0.95
    mock_ds.categories = ["science", "math"]
    mock_ds.fingerprint = "abc123def456ghij"

    mock_registry = MagicMock()
    mock_registry.exists.return_value = True
    mock_registry.load.return_value = mock_ds
    mock_registry.versions.return_value = ["1.0", "2.0"]
    return mock_registry


def _make_missing_registry():
    """Build a mock DatasetRegistry where the dataset is missing."""
    mock_registry = MagicMock()
    mock_registry.exists.return_value = False
    mock_item = MagicMock()
    mock_item.name = "squad"
    mock_registry.list.return_value = [mock_item]
    return mock_registry


def _patch_registry(mock_registry):
    """Patch DatasetRegistry at the lazy-import source module."""
    return patch(
        "mltk.eval.dataset.DatasetRegistry",
        return_value=mock_registry,
    )


class TestMltkDataset:
    """Tests for the mltk_dataset tool."""

    def test_dataset_exists(self):
        # SCENARIO: Look up a known dataset by name
        # WHY: Core happy path -- existing dataset returns info
        # EXPECTED: status=ok, info dict present
        registry = _make_found_registry()
        with _patch_registry(registry):
            result = _call_dataset(name="mmlu")

        assert_ok(result)
        assert "info" in result

    def test_dataset_not_found(self):
        # SCENARIO: Look up a dataset name that does not exist
        # WHY: Missing dataset must give a clear error with suggestions
        # EXPECTED: status=error, suggested_action lists available datasets
        registry = _make_missing_registry()
        with _patch_registry(registry):
            result = _call_dataset(name="nonexistent")

        assert_error(result)
        assert "not found" in result["error"].lower()
        assert "squad" in result["suggested_action"]

    def test_specific_version(self):
        # SCENARIO: Request a specific version of a dataset
        # WHY: Versioned lookup should pass version through to registry
        # EXPECTED: status=ok, registry.exists called with version
        registry = _make_found_registry()
        with _patch_registry(registry):
            result = _call_dataset(
                name="mmlu", version="2.0",
            )

        assert_ok(result)
        registry.exists.assert_called_once_with("mmlu", "2.0")

    def test_latest_version(self):
        # SCENARIO: Omit version parameter (defaults to empty string)
        # WHY: Empty version should resolve to None (latest)
        # EXPECTED: status=ok, registry.exists called with None
        registry = _make_found_registry()
        with _patch_registry(registry):
            result = _call_dataset(name="mmlu")

        assert_ok(result)
        registry.exists.assert_called_once_with("mmlu", None)

    def test_response_has_quality_metrics(self):
        # SCENARIO: Inspect quality block of a found dataset
        # WHY: Quality metrics drive dataset selection decisions
        # EXPECTED: quality dict has sample_count and duplicate_rate
        registry = _make_found_registry()
        with _patch_registry(registry):
            result = _call_dataset(name="mmlu")

        assert_ok(result)
        quality = result["quality"]
        assert "sample_count" in quality
        assert "duplicate_rate" in quality
        assert quality["sample_count"] == 2
        assert quality["duplicate_rate"] == 0.0

    def test_response_has_version_list(self):
        # SCENARIO: Check that versions are included in response
        # WHY: Agents need to know which versions are available
        # EXPECTED: versions is a list of strings
        registry = _make_found_registry()
        with _patch_registry(registry):
            result = _call_dataset(name="mmlu")

        assert_ok(result)
        assert isinstance(result["versions"], list)
        assert "1.0" in result["versions"]
        assert "2.0" in result["versions"]

    def test_error_has_suggested_action(self):
        # SCENARIO: Error response structure for missing dataset
        # WHY: Every error must give the agent a non-empty suggested action
        # EXPECTED: suggested_action is a non-empty string
        registry = _make_missing_registry()
        with _patch_registry(registry):
            result = _call_dataset(name="nonexistent")

        assert_error(result)
        assert isinstance(result["suggested_action"], str)
        assert len(result["suggested_action"]) > 0

    def test_returns_valid_json(self):
        # SCENARIO: Raw output format validation
        # WHY: MCP tools must always return well-formed JSON
        # EXPECTED: Raw string parses as JSON with status key
        registry = _make_found_registry()
        with _patch_registry(registry):
            raw = _call_dataset_raw(name="mmlu")

        data = assert_valid_json(raw)
        assert data["status"] == "ok"

    def test_error_recoverable_is_bool(self):
        # SCENARIO: Error response recoverable field type check
        # WHY: Agents branch on recoverable -- must be a real bool
        # EXPECTED: recoverable is a bool
        registry = _make_missing_registry()
        with _patch_registry(registry):
            result = _call_dataset(name="nonexistent")

        assert_error(result)
        assert isinstance(result["recoverable"], bool)

    def test_response_has_info(self):
        # SCENARIO: Inspect info block of a found dataset
        # WHY: Info is the primary metadata consumers need
        # EXPECTED: info dict has name and version keys
        registry = _make_found_registry()
        with _patch_registry(registry):
            result = _call_dataset(name="mmlu")

        assert_ok(result)
        info = result["info"]
        assert "name" in info
        assert "version" in info
        assert info["name"] == "mmlu"
        assert info["version"] == "1.0"
