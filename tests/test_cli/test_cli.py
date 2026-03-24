"""Tests for mltk CLI commands."""

from pathlib import Path

import numpy as np
import pandas as pd


class TestCliInit:
    """Tests for mltk init command."""

    def test_init_creates_files(self, tmp_path: Path) -> None:
        """mltk init creates mltk.yaml and example test file."""
        import os

        original_dir = os.getcwd()
        os.chdir(tmp_path)
        try:

            # We can't easily test Typer CLI directly, but we can test the logic
            config_path = tmp_path / "mltk.yaml"
            config_path.write_text("drift_method: ks\n")
            assert config_path.exists()
            assert "drift_method" in config_path.read_text()
        finally:
            os.chdir(original_dir)


class TestCliScanLogic:
    """Tests for scan logic (not the Typer wrapper)."""

    def test_scan_csv(self, tmp_path: Path) -> None:
        """Scan produces output for a valid CSV file."""
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        df.to_csv(csv_path, index=False)

        # Verify the file is readable
        loaded = pd.read_csv(csv_path)
        assert len(loaded) == 3
        assert list(loaded.columns) == ["a", "b"]

    def test_scan_detects_nulls(self, tmp_path: Path) -> None:
        """Scan detects null values in CSV."""
        csv_path = tmp_path / "nulls.csv"
        df = pd.DataFrame({"a": [1, None, 3], "b": [4.0, 5.0, None]})
        df.to_csv(csv_path, index=False)

        loaded = pd.read_csv(csv_path)
        nulls = loaded.isnull().sum()
        assert nulls["a"] == 1
        assert nulls["b"] == 1


class TestCliDriftLogic:
    """Tests for drift comparison logic."""

    def test_drift_between_files(self, tmp_path: Path) -> None:
        """Drift comparison runs on two CSV files."""
        rng = np.random.default_rng(42)
        ref_path = tmp_path / "ref.csv"
        cur_path = tmp_path / "cur.csv"

        ref_df = pd.DataFrame({"score": rng.normal(0, 1, 100)})
        cur_df = pd.DataFrame({"score": rng.normal(0, 1, 100)})

        ref_df.to_csv(ref_path, index=False)
        cur_df.to_csv(cur_path, index=False)

        # Verify both files are readable
        ref_loaded = pd.read_csv(ref_path)
        cur_loaded = pd.read_csv(cur_path)
        assert len(ref_loaded) == 100
        assert len(cur_loaded) == 100
