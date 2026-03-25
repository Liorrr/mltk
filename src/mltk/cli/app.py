"""mltk CLI — powered by Typer."""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    """Entry point for mltk CLI.

    Initializes the Typer application and registers all subcommands
    (version, init, scan, drift, score). Requires ``typer`` to be installed
    via the ``mltk[cli]`` extra.
    """
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

    @app.command()
    def score() -> None:
        """Show ML Test Score (run pytest first to generate results)."""
        print("ML Test Score")  # noqa: T201
        print("Run: pytest --mltk-report to generate scores")  # noqa: T201
        print()  # noqa: T201
        print("Categories (Google 28-test rubric):")  # noqa: T201
        print("  Data:           schema, distribution, drift, freshness, PII, labels")  # noqa: T201
        print("  Model:          metrics, regression, slicing, calibration, bias, adversarial")  # noqa: T201
        print("  Infrastructure: reproducibility, pipeline, contract, latency, throughput")  # noqa: T201
        print("  Monitoring:     drift monitoring, degradation, SLA, alerts")  # noqa: T201

    # Contract subcommands
    contract_app = typer.Typer(name="contract", help="Data contract operations")

    @contract_app.command("init")
    def contract_init() -> None:
        """Scaffold an example data contract YAML file."""
        example = """\
name: my_dataset
version: "1.0"

columns:
  id:
    type: int64
    nullable: false
    unique: true
  value:
    type: float64
    nullable: false
    range: [0, 100]
  label:
    type: int64
    nullable: false

quality:
  min_rows: 100
"""
        Path("contract.yaml").write_text(example)
        print("Created contract.yaml")  # noqa: T201

    @contract_app.command("validate")
    def contract_validate(
        data: str,
        contract: str = typer.Option("contract.yaml", help="Contract YAML path"),
    ) -> None:
        """Validate a data file against a contract."""
        import pandas as pd

        from mltk.contracts import validate_data

        p = Path(data)
        if not p.exists():
            print(f"File not found: {data}")  # noqa: T201
            raise typer.Exit(1)

        df = pd.read_csv(p) if p.suffix != ".parquet" else pd.read_parquet(p)
        suite = validate_data(df, contract)

        print(f"Contract validation: {suite.passed_count}/{suite.total} passed")  # noqa: T201
        for r in suite.results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  [{status}] {r.name}: {r.message}")  # noqa: T201

    @contract_app.command("generate-tests")
    def contract_generate(
        contract: str = typer.Argument("contract.yaml"),
        output: str = typer.Option("tests/test_contract_gen.py", help="Output test file"),
    ) -> None:
        """Generate pytest test file from a data contract."""
        from mltk.contracts.generator import generate_tests_from_contract

        cp = Path(contract)
        if not cp.exists():
            print(f"Contract not found: {contract}")  # noqa: T201
            raise typer.Exit(1)

        result = generate_tests_from_contract(cp, output)
        print(f"Generated: {result}")  # noqa: T201

    app.add_typer(contract_app)

    app()
