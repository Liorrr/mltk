"""Tests for mltk.contracts — data contract parsing, validation, and generation."""

from pathlib import Path

import pandas as pd

from mltk.contracts.generator import generate_tests_from_contract
from mltk.contracts.schema import Contract
from mltk.contracts.validator import validate_data

SAMPLE_CONTRACT = """\
name: test_dataset
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
  min_rows: 3
"""


class TestContractParsing:
    """Contract YAML parsing tests."""

    def test_parse_valid_contract(self, tmp_path: Path) -> None:
        """PASS: Valid YAML parses into Contract with correct fields."""
        path = tmp_path / "contract.yaml"
        path.write_text(SAMPLE_CONTRACT)
        contract = Contract.from_yaml(path)
        assert contract.name == "test_dataset"
        assert len(contract.columns) == 3
        assert contract.columns[1].range == (0.0, 100.0)
        assert contract.quality.min_rows == 3

    def test_column_properties(self, tmp_path: Path) -> None:
        """Columns have correct nullable/unique/range properties."""
        path = tmp_path / "contract.yaml"
        path.write_text(SAMPLE_CONTRACT)
        contract = Contract.from_yaml(path)
        id_col = contract.columns[0]
        assert id_col.nullable is False
        assert id_col.unique is True


class TestContractValidation:
    """Contract validation against DataFrames."""

    def test_valid_data_passes(self, tmp_path: Path) -> None:
        """PASS: DataFrame satisfies all contract requirements."""
        path = tmp_path / "contract.yaml"
        path.write_text(SAMPLE_CONTRACT)
        df = pd.DataFrame({"id": [1, 2, 3], "value": [10.0, 50.0, 90.0], "label": [0, 1, 0]})
        suite = validate_data(df, path)
        assert suite.passed is True

    def test_missing_column_fails(self, tmp_path: Path) -> None:
        """FAIL: DataFrame missing a required column."""
        path = tmp_path / "contract.yaml"
        path.write_text(SAMPLE_CONTRACT)
        df = pd.DataFrame({"id": [1, 2, 3], "value": [10.0, 50.0, 90.0]})
        suite = validate_data(df, path)
        assert suite.passed is False

    def test_null_in_non_nullable_fails(self, tmp_path: Path) -> None:
        """FAIL: Null value in non-nullable column."""
        path = tmp_path / "contract.yaml"
        path.write_text(SAMPLE_CONTRACT)
        df = pd.DataFrame({"id": [1, 2, 3], "value": [10.0, None, 90.0], "label": [0, 1, 0]})
        suite = validate_data(df, path)
        assert suite.passed is False

    def test_out_of_range_fails(self, tmp_path: Path) -> None:
        """FAIL: Value outside specified range."""
        path = tmp_path / "contract.yaml"
        path.write_text(SAMPLE_CONTRACT)
        df = pd.DataFrame({"id": [1, 2, 3], "value": [10.0, 200.0, 90.0], "label": [0, 1, 0]})
        suite = validate_data(df, path)
        assert suite.passed is False


class TestContractGeneration:
    """Test file generation from contracts."""

    def test_generates_pytest_file(self, tmp_path: Path) -> None:
        """Generate a valid Python test file from contract."""
        contract_path = tmp_path / "contract.yaml"
        contract_path.write_text(SAMPLE_CONTRACT)
        output_path = tmp_path / "test_generated.py"
        result = generate_tests_from_contract(contract_path, output_path)
        assert result.exists()
        content = result.read_text()
        assert "def test_test_dataset_schema" in content
        assert "assert_schema" in content

    def test_generated_file_has_range_test(self, tmp_path: Path) -> None:
        """Generated file includes range test for ranged columns."""
        contract_path = tmp_path / "contract.yaml"
        contract_path.write_text(SAMPLE_CONTRACT)
        output_path = tmp_path / "test_gen.py"
        generate_tests_from_contract(contract_path, output_path)
        content = output_path.read_text()
        assert "assert_range" in content
        assert "value" in content
