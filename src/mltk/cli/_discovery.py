"""Assertion discovery — scan mltk subpackages for assert_* functions."""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
from typing import Any

logger = logging.getLogger(__name__)

# Human-readable category labels keyed by top-level subpackage.
_CATEGORY_LABELS: dict[str, str] = {
    "data": "Data Quality",
    "model": "Model Quality",
    "domains.llm": "LLM Evaluation",
    "domains.nlp": "NLP",
    "domains.cv": "Computer Vision",
    "domains.speech": "Speech",
    "domains.tabular": "Tabular",
    "domains.rl": "Reinforcement Learning",
    "domains.multimodal": "Multimodal",
    "monitor": "Monitoring",
    "inference": "Inference",
    "pipeline": "Pipeline",
    "training": "Training",
    "testing": "Testing",
}

# Ordered list of subpackage dotted paths to scan, relative to ``mltk.``.
# We use an explicit list instead of recursive walk so that heavy optional
# dependencies (e.g., ``mltk.server``, ``mltk.integrations``) are never
# imported.
_SCAN_TARGETS: list[str] = [
    "mltk.data",
    "mltk.model",
    "mltk.domains.llm",
    "mltk.domains.nlp",
    "mltk.domains.cv",
    "mltk.domains.speech",
    "mltk.domains.tabular",
    "mltk.domains.rl",
    "mltk.domains.multimodal",
    "mltk.monitor",
    "mltk.inference",
    "mltk.pipeline",
    "mltk.training",
    "mltk.testing",
]


def _short_key(full_module: str) -> str:
    """Turn ``mltk.domains.llm`` into ``domains.llm``."""
    return full_module.removeprefix("mltk.")


def _first_doc_line(func: Any) -> str:
    """Return the first non-empty line of a callable's docstring.

    ASCII-safe: replaces non-ASCII chars to prevent cp1252 encoding
    errors on Windows terminals.
    """
    doc = inspect.getdoc(func) or ""
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped.encode("ascii", errors="replace").decode()
    return ""


def _collect_from_package(
    pkg_name: str,
) -> list[dict[str, str]]:
    """Import *pkg_name* and collect ``assert_*`` entries.

    If the package exposes ``__all__``, only names listed there are
    considered.  Otherwise every public ``assert_*`` callable is
    included.  We then walk one level of child modules so that
    functions defined directly in submodules (but not re-exported
    in ``__init__``) are also discovered.
    """
    entries: list[dict[str, str]] = []
    seen_names: set[str] = set()

    try:
        pkg = importlib.import_module(pkg_name)
    except Exception:  # noqa: BLE001
        logger.debug("Could not import %s", pkg_name)
        return entries

    export_names: list[str] | None = getattr(pkg, "__all__", None)

    # --- pass 1: package-level names ---
    candidates = export_names if export_names is not None else dir(pkg)
    for name in candidates:
        if not name.startswith("assert_"):
            continue
        obj = getattr(pkg, name, None)
        if obj is None or not callable(obj):
            continue
        # Resolve the defining module for a nicer display path.
        src_mod = getattr(
            inspect.unwrap(obj), "__module__", pkg_name
        )
        entries.append({
            "name": name,
            "module": src_mod,
            "doc": _first_doc_line(obj),
        })
        seen_names.add(name)

    # --- pass 2: child modules (one level deep) ---
    pkg_path = getattr(pkg, "__path__", None)
    if pkg_path is not None:
        for _importer, mod_name, _is_pkg in pkgutil.iter_modules(
            pkg_path
        ):
            full = f"{pkg_name}.{mod_name}"
            try:
                child = importlib.import_module(full)
            except Exception:  # noqa: BLE001
                continue
            for attr in dir(child):
                if not attr.startswith("assert_"):
                    continue
                if attr in seen_names:
                    continue
                obj = getattr(child, attr, None)
                if obj is None or not callable(obj):
                    continue
                entries.append({
                    "name": attr,
                    "module": full,
                    "doc": _first_doc_line(obj),
                })
                seen_names.add(attr)

    # Stable sort by function name.
    entries.sort(key=lambda e: e["name"])
    return entries


def discover_assertions(
    filter_keyword: str = "",
) -> dict[str, list[dict[str, str]]]:
    """Scan all known mltk subpackages and return assertion metadata.

    Returns a dict mapping human-readable category labels to lists of
    ``{"name": ..., "module": ..., "doc": ...}`` dicts.

    Parameters
    ----------
    filter_keyword:
        If non-empty, only entries whose name, module path, or first
        docstring line contain this substring (case-insensitive) are
        included.
    """
    result: dict[str, list[dict[str, str]]] = {}
    kw = filter_keyword.strip().lower()

    for target in _SCAN_TARGETS:
        key = _short_key(target)
        label = _CATEGORY_LABELS.get(key, key)
        items = _collect_from_package(target)

        if kw:
            items = [
                e for e in items
                if kw in e["name"].lower()
                or kw in e["module"].lower()
                or kw in e["doc"].lower()
            ]

        if items:
            result[label] = items

    return result
