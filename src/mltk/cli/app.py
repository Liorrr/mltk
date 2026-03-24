"""mltk CLI — powered by Typer."""

from __future__ import annotations


def main() -> None:
    """Entry point for mltk CLI."""
    try:
        import typer
    except ImportError as err:
        print("CLI requires: pip install mltk[cli]")  # noqa: T201
        raise SystemExit(1) from err

    app = typer.Typer(
        name="mltk",
        help="ML Test Kit -- pytest for ML. Unified testing across the entire ML lifecycle.",
    )

    @app.command()
    def version() -> None:
        """Show mltk version."""
        from mltk import __version__

        print(f"mltk v{__version__}")  # noqa: T201

    @app.command()
    def init() -> None:
        """Scaffold mltk.yaml + example test file."""
        # TODO: Sprint 5
        print("mltk init: coming in Sprint 5")  # noqa: T201

    @app.command()
    def scan(path: str) -> None:
        """Quick data quality scan on a CSV/Parquet file."""
        # TODO: Sprint 5
        print(f"mltk scan {path}: coming in Sprint 5")  # noqa: T201

    app()
