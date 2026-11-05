"""mltk server — self-hosted ML test results platform."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mltk.server.app import create_app

__all__ = ["create_app"]


def __getattr__(name: str) -> object:
    """Lazily import the optional server application."""
    if name == "create_app":
        from mltk.server.app import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
