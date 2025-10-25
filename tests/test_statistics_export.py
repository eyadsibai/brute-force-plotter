"""
Tests for statistical summary export functionality.
"""

import os

import pandas as pd
import pytest

from src.brute_force_plotter import export_statistical_summaries


class TestExportStatisticalSummaries:
    """Tests for export_statistical_summaries function."""

    @pytest.mark.unit
    def test_creates_statistics_directory(
        self, temp_dir, sample_parquet_file, mixed_dtypes
    ):
        """Test that statistics directory is created."""
        export_statistical_summaries(sample_parquet_file, mixed_dtypes, temp_dir)

        stats_dir = os.path.join(temp_dir, "statistics")
        assert os.path.exists(stats_dir)
        assert os.path.isdir(stats_dir)

    @pytest.mark.unit
    def test_exports_numeric_statistics(
        self, temp_dir, sample_parquet_file, mixed_dtypes
    ):
        """Test that numeric statistics are exported correctly."""
        export_statistical_summaries(sample_parquet_file, mixed_dtypes, temp_dir)

        stats_file = os.path.join(temp_dir, "statistics", "numeric_statistics.csv")
        assert os.path.exists(stats_file)

        # Load and verify the statistics
        stats = pd.read_csv(stats_file, index_col=0)

        # Check that expected columns are present
        assert "age" in stats.columns
        assert "income" in stats.columns

        # Check that expected statistics are present
        assert "mean" in stats.index
        assert "std" in stats.index
        assert "min" in stats.index
        assert "max" in stats.index
        assert "missing" in stats.index
        assert "missing_pct" in stats.index

    @pytest.mark.unit
    def test_exports_categorical_statistics(
        self, temp_dir, sample_parquet_file, mixed_dtypes
    ):
        """Test that categorical statistics are exported correctly."""
        export_statistical_summaries(sample_parquet_file, mixed_dtypes, temp_dir)

        # Check for gender category file
        gender_file = os.path.join(temp_dir, "statistics", "category_gender_counts.csv")
        assert os.path.exists(gender_file)

        # Load and verify
        gender_counts = pd.read_csv(gender_file)
        assert "value" in gender_counts.columns
        assert "count" in gender_counts.columns
        assert "percentage" in gender_counts.columns

        # Check for education category file
        education_file = os.path.join(
            temp_dir, "statistics", "category_education_counts.csv"
        )
        assert os.path.exists(education_file)

    @pytest.mark.unit
    def test_exports_missing_values_summary(
        self, temp_dir, sample_parquet_file, mixed_dtypes
    ):
        """Test that missing values summary is exported."""
        export_statistical_summaries(sample_parquet_file, mixed_dtypes, temp_dir)

        missing_file = os.path.join(
            temp_dir, "statistics", "missing_values_summary.csv"
        )
        assert os.path.exists(missing_file)

        # Load and verify
        missing_summary = pd.read_csv(missing_file)
        assert "column" in missing_summary.columns
        assert "missing_count" in missing_summary.columns
        assert "missing_percentage" in missing_summary.columns
        assert "total_count" in missing_summary.columns
        assert "non_missing_count" in missing_summary.columns

    @pytest.mark.unit
    def test_exports_overall_summary(self, temp_dir, sample_parquet_file, mixed_dtypes):
        """Test that overall summary is exported."""
        export_statistical_summaries(sample_parquet_file, mixed_dtypes, temp_dir)

        overall_file = os.path.join(temp_dir, "statistics", "overall_summary.csv")
        assert os.path.exists(overall_file)

        # Load and verify
        overall_summary = pd.read_csv(overall_file)
        assert "metric" in overall_summary.columns
        assert "value" in overall_summary.columns

        # Check for expected metrics
        metrics = overall_summary["metric"].tolist()
        assert "total_rows" in metrics
        assert "total_columns" in metrics
        assert "numeric_columns" in metrics
        assert "categorical_columns" in metrics

    @pytest.mark.unit
    def test_handles_data_with_missing_values(self, temp_dir, sample_data_with_missing):
        """Test that export handles missing values correctly."""
        import tempfile

        # Create parquet file
        with tempfile.NamedTemporaryFile(suffix=".parq", delete=False) as tmp:
            parquet_path = tmp.name
        sample_data_with_missing.to_parquet(parquet_path)

        try:
            dtypes = {"col1": "n", "col2": "n", "category": "c"}
            export_statistical_summaries(parquet_path, dtypes, temp_dir)

            # Check that numeric statistics include missing counts
            stats_file = os.path.join(temp_dir, "statistics", "numeric_statistics.csv")
            stats = pd.read_csv(stats_file, index_col=0)

            # Verify that missing values are tracked
            assert "missing" in stats.index
            assert stats.loc["missing", "col1"] > 0
            assert stats.loc["missing", "col2"] > 0
        finally:
            if os.path.exists(parquet_path):
                os.remove(parquet_path)

    @pytest.mark.unit
    def test_handles_ignored_columns(self, temp_dir, sample_parquet_file):
        """Test that ignored columns are not included in statistics."""
        dtypes = {
            "age": "n",
            "income": "i",  # ignored
            "gender": "c",
            "education": "i",  # ignored
        }

        export_statistical_summaries(sample_parquet_file, dtypes, temp_dir)

        # Check numeric statistics
        stats_file = os.path.join(temp_dir, "statistics", "numeric_statistics.csv")
        stats = pd.read_csv(stats_file, index_col=0)

        # income should not be present
        assert "age" in stats.columns
        assert "income" not in stats.columns

        # Check that education category file doesn't exist
        education_file = os.path.join(
            temp_dir, "statistics", "category_education_counts.csv"
        )
        assert not os.path.exists(education_file)

        # But gender should exist
        gender_file = os.path.join(temp_dir, "statistics", "category_gender_counts.csv")
        assert os.path.exists(gender_file)

    @pytest.mark.unit
    def test_handles_empty_dtypes(self, temp_dir, sample_parquet_file):
        """Test handling when no columns are specified."""
        dtypes = {}

        export_statistical_summaries(sample_parquet_file, dtypes, temp_dir)

        # Statistics directory should still be created but might be empty or minimal
        # stats_dir = os.path.join(temp_dir, "statistics")

    @pytest.mark.unit
    def test_handles_only_numeric_columns(self, temp_dir, sample_parquet_file):
        """Test export with only numeric columns."""
        dtypes = {"age": "n", "income": "n"}

        export_statistical_summaries(sample_parquet_file, dtypes, temp_dir)

        # Numeric stats should exist
        stats_file = os.path.join(temp_dir, "statistics", "numeric_statistics.csv")
        assert os.path.exists(stats_file)

        # No categorical files should exist
        stats_dir = os.path.join(temp_dir, "statistics")
        category_files = [f for f in os.listdir(stats_dir) if f.startswith("category_")]
        assert len(category_files) == 0

    @pytest.mark.unit
    def test_handles_only_categorical_columns(self, temp_dir, sample_parquet_file):
        """Test export with only categorical columns."""
        dtypes = {"gender": "c", "education": "c"}

        export_statistical_summaries(sample_parquet_file, dtypes, temp_dir)

        # Categorical files should exist
        gender_file = os.path.join(temp_dir, "statistics", "category_gender_counts.csv")
        education_file = os.path.join(
            temp_dir, "statistics", "category_education_counts.csv"
        )
        assert os.path.exists(gender_file)
        assert os.path.exists(education_file)

        # Numeric stats file should not exist
        stats_file = os.path.join(temp_dir, "statistics", "numeric_statistics.csv")
        assert not os.path.exists(stats_file)
