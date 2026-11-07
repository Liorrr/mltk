"""Path confinement helpers for the MCP server."""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = ["PathOutsideRootError", "allowed_roots", "confine_path"]

_UNRESTRICTED_LOGGED = False


class PathOutsideRootError(ValueError):
    """Raised when a path is outside the configured MCP roots."""


def allowed_roots() -> list[Path]:
    """Return resolved roots configured for MCP path access."""
    configured = os.environ.get("MLTK_MCP_ROOTS", "")
    entries = [entry.strip() for entry in configured.split(os.pathsep) if entry.strip()]
    return [
        Path(entry).expanduser().resolve() for entry in entries
    ] or [Path.cwd().resolve()]


def confine_path(raw: str) -> Path:
    """Resolve ``raw`` and require it to be within an allowed root."""
    global _UNRESTRICTED_LOGGED

    resolved = Path(raw).expanduser().resolve(strict=False)
    unrestricted = os.environ.get("MLTK_MCP_UNRESTRICTED", "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    if unrestricted:
        if not _UNRESTRICTED_LOGGED:
            print("[mltk-mcp] path confinement check is off", file=sys.stderr)
            _UNRESTRICTED_LOGGED = True
        return resolved

    roots = allowed_roots()
    if any(resolved == root or resolved.is_relative_to(root) for root in roots):
        return resolved
    roots_text = ", ".join(str(root) for root in roots)
    raise PathOutsideRootError(
        f"Path '{raw}' is outside the allowed roots ({roots_text}). "
        "Set MLTK_MCP_ROOTS to change the roots, or "
        "MLTK_MCP_UNRESTRICTED=1 to turn the check off."
    )
