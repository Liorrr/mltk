"""mltk CLI — powered by Typer."""

from __future__ import annotations

from pathlib import Path


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
        config_content = """\
# mltk configuration
drift_method: ks
drift_threshold: 0.05
report_dir: ./mltk-reports
seed: 42
"""
        test_content = """\
\"\"\"Example mltk test file.\"\"\"

import pandas as pd
import pytest

from mltk.data import assert_no_nulls, assert_row_count, assert_schema


@pytest.mark.ml_data
def test_data_quality():
    # Replace with your actual data path
    df = pd.DataFrame({"id": [1, 2, 3], "value": [1.0, 2.0, 3.0]})
    assert_schema(df, {"id": "int64", "value": "float64"})
    assert_no_nulls(df)
    assert_row_count(df, min_rows=1)
"""
        config_path = Path("mltk.yaml")
        test_dir = Path("tests")
        test_path = test_dir / "test_mltk_example.py"

        config_path.write_text(config_content)
        print(f"Created {config_path}")  # noqa: T201

        test_dir.mkdir(exist_ok=True)
        test_path.write_text(test_content)
        print(f"Created {test_path}")  # noqa: T201

    @app.command()
    def scan(path: str) -> None:
        """Quick data quality scan on a CSV/Parquet file."""
        import pandas as pd

        p = Path(path)
        if not p.exists():
            print(f"File not found: {path}")  # noqa: T201
            raise typer.Exit(1)

        if p.suffix == ".parquet":
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)

        print(f"MLTK Data Scan: {path}")  # noqa: T201
        print(f"  Rows: {len(df):,} | Columns: {len(df.columns)}")  # noqa: T201
        print(f"  Columns: {list(df.columns)}")  # noqa: T201
        print(f"  Dtypes: {dict(df.dtypes)}")  # noqa: T201

        # Null check
        nulls = df.isnull().sum()
        null_cols = nulls[nulls > 0]
        if len(null_cols) == 0:
            print("  [PASS] No null values")  # noqa: T201
        else:
            print(f"  [WARN] Nulls found: {dict(null_cols)}")  # noqa: T201

        # Row count
        print(f"  [INFO] Row count: {len(df):,}")  # noqa: T201

    @app.command()
    def drift(
        reference: str,
        current: str,
        method: str = "psi",
    ) -> None:
        """Compare two datasets for distribution drift."""
        import pandas as pd

        ref_path = Path(reference)
        cur_path = Path(current)

        if not ref_path.exists() or not cur_path.exists():
            print("Both files must exist")  # noqa: T201
            raise typer.Exit(1)

        ref_df = pd.read_csv(ref_path)
        cur_df = pd.read_csv(cur_path)

        from mltk.data.drift import assert_no_drift

        print(f"MLTK Drift Analysis: {reference} vs {current}")  # noqa: T201
        print(f"  Method: {method}")  # noqa: T201
        print()  # noqa: T201

        # Compare numeric columns
        common_cols = [
            c for c in ref_df.columns
            if c in cur_df.columns and pd.api.types.is_numeric_dtype(ref_df[c])
        ]

        for col in common_cols:
            try:
                result = assert_no_drift(
                    ref_df[col], cur_df[col], method=method
                )
                status = "OK" if result.passed else "DRIFT"
                stat = result.details.get("statistic", 0)
                print(f"  {col:20s} | {stat:.4f} | {status}")  # noqa: T201
            except Exception:
                # Drift detected (exception raised)
                print(f"  {col:20s} | DRIFT DETECTED")  # noqa: T201

    app()
