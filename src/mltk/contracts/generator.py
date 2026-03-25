"""Contract test generator — auto-generate pytest file from contract YAML."""

from __future__ import annotations

from pathlib import Path

from mltk.contracts.schema import Contract


def generate_tests_from_contract(
    contract_path: str | Path,
    output_path: str | Path,
    data_path: str = "data.csv",
) -> Path:
    """Generate a pytest test file from a data contract.

    Args:
        contract_path: Path to the contract YAML file.
        output_path: Path for the generated test file.
        data_path: Data file path to use in generated tests.

    Returns:
        Path to the generated test file.

    Example:
        >>> generate_tests_from_contract("contract.yaml", "tests/test_contract.py")
    """
    contract = Contract.from_yaml(contract_path)
    out = Path(output_path)

    lines = [
        '"""Auto-generated tests from data contract."""',
        "",
        "import pandas as pd",
        "import pytest",
        "",
    ]

    # Schema test
    schema_dict = {col.name: col.type for col in contract.columns}
    lines.extend([
        "",
        f'DATA_PATH = "{data_path}"',
        "",
        "",
        "@pytest.mark.ml_data",
        f'def test_{contract.name}_schema():',
        f'    """Validate schema matches contract {contract.name} v{contract.version}."""',
        "    df = pd.read_csv(DATA_PATH)",
        "    from mltk.data import assert_schema",
        f"    assert_schema(df, {schema_dict})",
    ])

    # Non-nullable columns
    non_nullable = [col.name for col in contract.columns if not col.nullable]
    if non_nullable:
        lines.extend([
            "",
            "",
            "@pytest.mark.ml_data",
            f'def test_{contract.name}_no_nulls():',
            '    """Validate non-nullable columns have no nulls."""',
            "    df = pd.read_csv(DATA_PATH)",
            "    from mltk.data import assert_no_nulls",
            f"    assert_no_nulls(df, columns={non_nullable})",
        ])

    # Range checks
    for col in contract.columns:
        if col.range is not None:
            lines.extend([
                "",
                "",
                "@pytest.mark.ml_data",
                f'def test_{contract.name}_{col.name}_range():',
                f'    """Validate {col.name} is in range {col.range}."""',
                "    df = pd.read_csv(DATA_PATH)",
                "    from mltk.data import assert_range",
                f'    assert_range(df["{col.name}"], '
                f"min_val={col.range[0]}, max_val={col.range[1]})",
            ])

    # Row count
    q = contract.quality
    if q.min_rows is not None:
        lines.extend([
            "",
            "",
            "@pytest.mark.ml_data",
            f'def test_{contract.name}_row_count():',
            '    """Validate minimum row count."""',
            "    df = pd.read_csv(DATA_PATH)",
            "    from mltk.data import assert_row_count",
            f"    assert_row_count(df, min_rows={q.min_rows})",
        ])

    lines.append("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")

    return out
