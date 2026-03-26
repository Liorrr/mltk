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

    def test_generated_file_has_no_nulls_test(self, tmp_path: Path) -> None:
        """Generated file includes assert_no_nulls for non-nullable columns.

        SCENARIO: contract has non-nullable columns (id, value, label)
        WHY: the generator emits test_{name}_no_nulls only when there are
             non-nullable columns; this path was not explicitly verified
        EXPECTED: assert_no_nulls present and references at least one column name
        """
        contract_path = tmp_path / "contract.yaml"
        contract_path.write_text(SAMPLE_CONTRACT)
        output_path = tmp_path / "test_no_nulls.py"
        generate_tests_from_contract(contract_path, output_path)
        content = output_path.read_text()
        assert "assert_no_nulls" in content
        assert "test_test_dataset_no_nulls" in content

    def test_no_nulls_test_omitted_when_all_nullable(self, tmp_path: Path) -> None:
        """No assert_no_nulls block when every column is nullable.

        SCENARIO: contract where all columns have nullable: true (the default)
        WHY: the generator should emit no_nulls only for non-nullable columns;
             generating it when non_nullable list is empty would produce broken code
             (assert_no_nulls(df, columns=[]) is meaningless)
        EXPECTED: assert_no_nulls absent from generated file
        """
        all_nullable_contract = """\
name: nullable_dataset
version: "1.0"
columns:
  score:
    type: float64
    nullable: true
  label:
    type: int64
    nullable: true
"""
        contract_path = tmp_path / "nullable.yaml"
        contract_path.write_text(all_nullable_contract)
        output_path = tmp_path / "test_nullable.py"
        generate_tests_from_contract(contract_path, output_path)
        content = output_path.read_text()
        assert "assert_no_nulls" not in content

    def test_row_count_test_generated_when_min_rows_set(self, tmp_path: Path) -> None:
        """Generated file includes assert_row_count when min_rows is in quality spec.

        SCENARIO: contract specifies quality.min_rows = 3
        WHY: row count test is only emitted when min_rows is not None; if the
             condition logic broke, users would silently lose their row-count checks
        EXPECTED: assert_row_count present with the correct min value
        """
        contract_path = tmp_path / "contract.yaml"
        contract_path.write_text(SAMPLE_CONTRACT)
        output_path = tmp_path / "test_rowcount.py"
        generate_tests_from_contract(contract_path, output_path)
        content = output_path.read_text()
        assert "assert_row_count" in content
        assert "min_rows=3" in content

    def test_row_count_test_omitted_when_no_min_rows(self, tmp_path: Path) -> None:
        """No assert_row_count block when quality.min_rows is absent.

        SCENARIO: contract with no quality block
        WHY: the generator should not emit a test for a constraint that was not
             declared; an empty quality spec must produce no row count test
        EXPECTED: assert_row_count absent from generated file
        """
        no_quality_contract = """\
name: no_quality
version: "1.0"
columns:
  x:
    type: float64
"""
        contract_path = tmp_path / "no_quality.yaml"
        contract_path.write_text(no_quality_contract)
        output_path = tmp_path / "test_no_quality.py"
        generate_tests_from_contract(contract_path, output_path)
        content = output_path.read_text()
        assert "assert_row_count" not in content

    def test_custom_data_path_written_into_generated_file(self, tmp_path: Path) -> None:
        """data_path argument appears as DATA_PATH in the generated file.

        SCENARIO: caller passes data_path='s3://bucket/data.parquet'
        WHY: generated tests must reference the user-supplied data path, not the
             default 'data.csv'; if the argument were ignored, generated tests
             would always point to a non-existent file
        EXPECTED: the custom path appears verbatim in the generated content
        """
        contract_path = tmp_path / "contract.yaml"
        contract_path.write_text(SAMPLE_CONTRACT)
        output_path = tmp_path / "test_custom_path.py"
        generate_tests_from_contract(
            contract_path, output_path, data_path="s3://bucket/data.parquet"
        )
        content = output_path.read_text()
        assert "s3://bucket/data.parquet" in content

    def test_output_parent_dirs_created(self, tmp_path: Path) -> None:
        """generate_tests_from_contract creates intermediate output directories.

        SCENARIO: output_path points to a file inside a non-existent subdirectory
        WHY: out.parent.mkdir(parents=True, exist_ok=True) must execute before
             the write; if it were removed, callers that pass nested paths would
             get FileNotFoundError
        EXPECTED: file exists at the nested path after generation
        """
        contract_path = tmp_path / "contract.yaml"
        contract_path.write_text(SAMPLE_CONTRACT)
        nested_output = tmp_path / "generated" / "subdir" / "test_gen.py"
        result = generate_tests_from_contract(contract_path, nested_output)
        assert result.exists()
        assert result.parent.is_dir()

    def test_generated_file_has_ml_data_marker(self, tmp_path: Path) -> None:
        """Generated tests are decorated with @pytest.mark.ml_data.

        SCENARIO: inspect generated file for marker decoration
        WHY: the marker is required so mltk's pytest plugin can selectively run or
             skip data contract tests; if it were absent, the contract test suite
             would lose its classification metadata
        EXPECTED: '@pytest.mark.ml_data' appears in the generated file
        """
        contract_path = tmp_path / "contract.yaml"
        contract_path.write_text(SAMPLE_CONTRACT)
        output_path = tmp_path / "test_markers.py"
        generate_tests_from_contract(contract_path, output_path)
        content = output_path.read_text()
        assert "@pytest.mark.ml_data" in content

    def test_returns_path_object(self, tmp_path: Path) -> None:
        """generate_tests_from_contract returns a Path, not a string.

        SCENARIO: check return type
        WHY: callers rely on the return value being a Path for further operations
             (e.g. .read_text(), .exists()); returning a str would break those uses
        EXPECTED: isinstance(result, Path) is True
        """
        contract_path = tmp_path / "contract.yaml"
        contract_path.write_text(SAMPLE_CONTRACT)
        output_path = tmp_path / "test_rtype.py"
        result = generate_tests_from_contract(contract_path, output_path)
        assert isinstance(result, Path)
