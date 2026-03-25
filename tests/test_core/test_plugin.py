"""Tests for the mltk plugin / assertion registry system."""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import patch

# Import the module under test — use a fresh registry state per test via monkeypatch
import mltk.core.plugin as plugin_module
from mltk.core.plugin import (
    discover_plugins,
    get_registered_assertions,
    register_assertion,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clear_registry(monkeypatch):
    """Replace the global registry with a fresh empty dict for the test."""
    monkeypatch.setattr(plugin_module, "_ASSERTION_REGISTRY", {})


# ---------------------------------------------------------------------------
# register_assertion tests
# ---------------------------------------------------------------------------

class TestRegisterAssertion:
    # SCENARIO: decorator with explicit name registers the function under that name
    # WHY: users should be able to choose a registry key distinct from the function name
    # EXPECTED: function available under the given name, not under its __name__
    def test_register_with_name(self, monkeypatch):
        _clear_registry(monkeypatch)

        @register_assertion("custom_drift_check")
        def assert_drift(data, threshold=0.1):
            return True

        registry = get_registered_assertions()
        assert "custom_drift_check" in registry
        assert registry["custom_drift_check"] is assert_drift

    # SCENARIO: decorator without name uses the function's __name__
    # WHY: convenience form — name inferred automatically
    # EXPECTED: function available under "assert_my_check"
    def test_register_assertion_infers_name(self, monkeypatch):
        _clear_registry(monkeypatch)

        @register_assertion()
        def assert_my_check(data):
            return True

        registry = get_registered_assertions()
        assert "assert_my_check" in registry

    # SCENARIO: decorated function is still directly callable
    # WHY: decorator must return the original function unchanged
    # EXPECTED: calling the decorated function works as normal
    def test_register_decorator_returns_func(self, monkeypatch):
        _clear_registry(monkeypatch)

        @register_assertion("callable_test")
        def assert_something(value):
            return value * 2

        assert assert_something(21) == 42

    # SCENARIO: multiple assertions can be registered independently
    # WHY: registry must support accumulation without overwriting previous entries
    # EXPECTED: both names present in the registry
    def test_register_multiple_assertions(self, monkeypatch):
        _clear_registry(monkeypatch)

        @register_assertion("check_a")
        def fn_a():
            pass

        @register_assertion("check_b")
        def fn_b():
            pass

        registry = get_registered_assertions()
        assert "check_a" in registry
        assert "check_b" in registry

    # SCENARIO: get_registered_assertions returns a copy, not the live dict
    # WHY: callers should not be able to mutate the global registry via the returned dict
    # EXPECTED: modifying the returned dict does not affect the registry
    def test_get_registered_returns_copy(self, monkeypatch):
        _clear_registry(monkeypatch)

        @register_assertion("sentinel")
        def fn_sentinel():
            pass

        copy = get_registered_assertions()
        copy["injected"] = lambda: None  # mutate the copy

        live = get_registered_assertions()
        assert "injected" not in live


# ---------------------------------------------------------------------------
# get_registered_assertions tests
# ---------------------------------------------------------------------------

class TestGetRegisteredAssertions:
    # SCENARIO: empty registry returns empty dict
    # WHY: no plugins or custom assertions yet — must not error
    # EXPECTED: empty dict
    def test_get_registered_empty(self, monkeypatch):
        _clear_registry(monkeypatch)
        assert get_registered_assertions() == {}

    # SCENARIO: returns all previously registered assertions
    # WHY: registry is cumulative — all entries since module load should be present
    # EXPECTED: all registered names appear in the result
    def test_get_registered_all(self, monkeypatch):
        _clear_registry(monkeypatch)

        for i in range(3):
            register_assertion(f"check_{i}")(lambda: None)

        registry = get_registered_assertions()
        for i in range(3):
            assert f"check_{i}" in registry


# ---------------------------------------------------------------------------
# discover_plugins tests
# ---------------------------------------------------------------------------

class TestDiscoverPlugins:
    # SCENARIO: no mltk_plugin_* packages installed
    # WHY: most environments won't have plugins; should return empty list silently
    # EXPECTED: empty list
    def test_discover_plugins_empty(self):
        # Patch packages_distributions to return nothing with mltk_plugin_ prefix
        with patch(
            "importlib.metadata.packages_distributions",
            return_value={"some_pkg": ["some_pkg"], "other_lib": ["other_lib"]},
        ):
            result = discover_plugins()

        assert result == []

    # SCENARIO: one mltk_plugin_* package is installed and importable
    # WHY: happy path — plugin should be imported and name returned
    # EXPECTED: package name appears in returned list
    def test_discover_plugins_found(self, monkeypatch):
        # Create a dummy module in sys.modules
        dummy_mod = ModuleType("mltk_plugin_dummy")
        monkeypatch.setitem(sys.modules, "mltk_plugin_dummy", dummy_mod)

        with patch(
            "importlib.metadata.packages_distributions",
            return_value={"mltk_plugin_dummy": ["mltk_plugin_dummy"]},
        ):
            result = discover_plugins()

        assert "mltk_plugin_dummy" in result

    # SCENARIO: discover_plugins with custom prefix
    # WHY: allow non-standard prefixes for closed-source plugin ecosystems
    # EXPECTED: only packages matching the custom prefix are returned
    def test_discover_plugins_custom_prefix(self, monkeypatch):
        dummy_mod = ModuleType("acme_ml_plugin")
        monkeypatch.setitem(sys.modules, "acme_ml_plugin", dummy_mod)

        with patch(
            "importlib.metadata.packages_distributions",
            return_value={
                "acme_ml_plugin": ["acme_ml_plugin"],
                "mltk_plugin_other": ["mltk_plugin_other"],
            },
        ):
            result = discover_plugins(package_prefix="acme_ml_")

        assert "acme_ml_plugin" in result
        assert "mltk_plugin_other" not in result

    # SCENARIO: metadata lookup raises an unexpected exception
    # WHY: robustness — broken metadata should not crash the application
    # EXPECTED: returns empty list without raising
    def test_discover_plugins_metadata_error(self):
        with patch(
            "importlib.metadata.packages_distributions",
            side_effect=Exception("metadata broken"),
        ):
            result = discover_plugins()

        assert result == []

    # SCENARIO: plugin package is listed in metadata but fails to import
    # WHY: bad plugin should not crash discover_plugins for valid ones
    # EXPECTED: failed package not in list, no exception raised
    def test_discover_plugins_import_error_skipped(self):
        with patch(
            "importlib.metadata.packages_distributions",
            return_value={"mltk_plugin_broken": ["mltk_plugin_broken"]},
        ):
            with patch("importlib.import_module", side_effect=ImportError("missing dep")):
                result = discover_plugins()

        assert result == []
