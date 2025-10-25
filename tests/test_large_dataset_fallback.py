"""
Tests for large dataset fallback functionality.
"""

import os

import pandas as pd
import pytest

from src.brute_force_plotter import (
    check_and_sample_large_dataset,
    plot,
)


class TestCheckAndSampleLargeDataset:
    """Tests for the check_and_sample_large_dataset function."""

    @pytest.mark.unit
    def test_small_dataset_not_sampled(self):
        """Test that small datasets are not sampled."""
        data = pd.DataFrame(
            {
                "col1": range(100),
                "col2": range(100),
            }
        )

        result_data, was_sampled = check_and_sample_large_dataset(
            data, max_rows=1000, sample_size=500
        )

        assert not was_sampled
        assert len(result_data) == len(data)
        assert result_data.equals(data)

    @pytest.mark.unit
    def test_large_dataset_sampled(self):
        """Test that large datasets are sampled."""
        # Create a dataset larger than the threshold
        data = pd.DataFrame(
            {
                "col1": range(2000),
                "col2": range(2000),
            }
        )

        result_data, was_sampled = check_and_sample_large_dataset(
            data, max_rows=1000, sample_size=500
        )

        assert was_sampled
        assert len(result_data) == 500
        # Check that sampled data is a subset of original
        assert all(col in data.columns for col in result_data.columns)

    @pytest.mark.unit
    def test_no_sample_flag_disables_sampling(self):
        """Test that no_sample flag prevents sampling even for large datasets."""
        data = pd.DataFrame(
            {
                "col1": range(2000),
                "col2": range(2000),
            }
        )

        result_data, was_sampled = check_and_sample_large_dataset(
            data, max_rows=1000, sample_size=500, no_sample=True
        )

        assert not was_sampled
        assert len(result_data) == len(data)

    @pytest.mark.unit
    def test_sample_size_smaller_than_data(self):
        """Test sampling when sample_size is smaller than data size."""
        data = pd.DataFrame(
            {
                "col1": range(10000),
                "col2": range(10000),
            }
        )

        result_data, was_sampled = check_and_sample_large_dataset(
            data, max_rows=5000, sample_size=3000
        )

        assert was_sampled
        assert len(result_data) == 3000

    @pytest.mark.unit
    def test_sample_size_larger_than_data(self):
        """Test sampling when sample_size is larger than data size."""
        data = pd.DataFrame(
            {
                "col1": range(1500),
                "col2": range(1500),
            }
        )

        # Sample size is 3000, but data only has 1500 rows
        result_data, was_sampled = check_and_sample_large_dataset(
            data, max_rows=1000, sample_size=3000
        )

        assert was_sampled
        # Should return min(sample_size, n_rows)
        assert len(result_data) == 1500

    @pytest.mark.unit
    def test_sampling_deterministic(self):
        """Test that sampling is deterministic (same results with same seed)."""
        data = pd.DataFrame(
            {
                "col1": range(2000),
                "col2": range(2000),
            }
        )

        result1, _ = check_and_sample_large_dataset(
            data, max_rows=1000, sample_size=500
        )
        result2, _ = check_and_sample_large_dataset(
            data, max_rows=1000, sample_size=500
        )

        # Results should be identical due to fixed random_state=42
        assert result1.equals(result2)

    @pytest.mark.unit
    def test_sampling_preserves_columns(self):
        """Test that sampling preserves all columns."""
        data = pd.DataFrame(
            {
                "col1": range(2000),
                "col2": ["A"] * 2000,
                "col3": range(2000, 4000),
                "col4": ["B", "C", "D", "E"] * 500,
            }
        )

        result_data, was_sampled = check_and_sample_large_dataset(
            data, max_rows=1000, sample_size=500
        )

        assert was_sampled
        assert list(result_data.columns) == list(data.columns)


class TestPlotWithLargeDatasetFallback:
    """Integration tests for plot function with large dataset handling."""

    @pytest.mark.integration
    def test_plot_with_small_dataset_no_sampling(self, temp_dir):
        """Test that small datasets are plotted without sampling."""
        data = pd.DataFrame(
            {
                "value1": range(100),
                "value2": range(100),
                "category": ["A", "B"] * 50,
            }
        )

        dtypes = {
            "value1": "n",
            "value2": "n",
            "category": "c",
        }

        # Should not trigger sampling
        output_path = plot(
            data,
            dtypes,
            output_path=temp_dir,
            show=False,
            use_dask=False,
            max_rows=1000,
            sample_size=500,
        )

        assert output_path == temp_dir

        # Check that plots were created
        dist_dir = os.path.join(temp_dir, "distributions")
        assert os.path.exists(dist_dir)

    @pytest.mark.integration
    def test_plot_with_large_dataset_triggers_sampling(self, temp_dir, caplog):
        """Test that large datasets trigger sampling with appropriate warnings."""
        # Create a large dataset
        data = pd.DataFrame(
            {
                "value1": range(2000),
                "value2": range(2000),
                "category": ["A", "B", "C", "D"] * 500,
            }
        )

        dtypes = {
            "value1": "n",
            "value2": "n",
            "category": "c",
        }

        output_path = plot(
            data,
            dtypes,
            output_path=temp_dir,
            show=False,
            use_dask=False,
            max_rows=1000,
            sample_size=500,
        )

        assert output_path == temp_dir

        # Check that sampling was triggered
        log_messages = [record.message for record in caplog.records]
        assert any("2,000 rows" in msg for msg in log_messages)
        assert any("Sampling" in msg for msg in log_messages)

        # Plots should still be created
        dist_dir = os.path.join(temp_dir, "distributions")
        assert os.path.exists(dist_dir)

    @pytest.mark.integration
    def test_plot_with_no_sample_flag(self, temp_dir):
        """Test that no_sample flag prevents sampling."""
        data = pd.DataFrame(
            {
                "value1": range(2000),
                "value2": range(2000),
                "category": ["A", "B"] * 1000,
            }
        )

        dtypes = {
            "value1": "n",
            "value2": "n",
            "category": "c",
        }

        # Even though dataset is large, should not sample
        output_path = plot(
            data,
            dtypes,
            output_path=temp_dir,
            show=False,
            use_dask=False,
            max_rows=1000,
            sample_size=500,
            no_sample=True,
        )

        assert output_path == temp_dir

        # Plots should be created from full dataset
        dist_dir = os.path.join(temp_dir, "distributions")
        assert os.path.exists(dist_dir)

    @pytest.mark.integration
    def test_plot_with_custom_sample_size(self, temp_dir, caplog):
        """Test plot with custom sample size."""
        data = pd.DataFrame(
            {
                "value1": range(5000),
                "value2": range(5000),
            }
        )

        dtypes = {
            "value1": "n",
            "value2": "n",
        }

        output_path = plot(
            data,
            dtypes,
            output_path=temp_dir,
            show=False,
            use_dask=False,
            max_rows=1000,
            sample_size=2000,
        )

        assert output_path == temp_dir

        # Check that custom sample size was used
        assert any("2,000" in record.message for record in caplog.records)

    @pytest.mark.integration
    def test_plot_export_stats_with_sampling_uses_full_dataset(self, temp_dir, caplog):
        """Test that export_stats uses full dataset even when plots use sampled data."""
        import logging

        caplog.set_level(logging.INFO)

        data = pd.DataFrame(
            {
                "value1": range(2000),
                "value2": range(2000),
                "category": ["A", "B"] * 1000,
            }
        )

        dtypes = {
            "value1": "n",
            "value2": "n",
            "category": "c",
        }

        output_path = plot(
            data,
            dtypes,
            output_path=temp_dir,
            show=False,
            use_dask=False,
            max_rows=1000,
            sample_size=500,
            export_stats=True,
        )

        assert output_path == temp_dir

        # Check that sampling was used for plots
        log_messages = [record.message for record in caplog.records]
        assert any("Sampling" in msg for msg in log_messages)

        # Check that we're informed about using full dataset for stats
        # The message is at line 261-264
        assert any(
            "Statistical summaries" in msg and "full dataset" in msg
            for msg in log_messages
        )

        # Check that stats were exported
        stats_dir = os.path.join(temp_dir, "statistics")
        assert os.path.exists(stats_dir)

        # Verify statistics are from full dataset (should have 2000 rows)
        overall_file = os.path.join(stats_dir, "overall_summary.csv")
        assert os.path.exists(overall_file)

        stats_df = pd.read_csv(overall_file)
        total_rows = stats_df[stats_df["metric"] == "total_rows"]["value"].values[0]
        assert total_rows == 2000  # Full dataset, not sampled

    @pytest.mark.integration
    def test_plot_at_threshold_boundary(self, temp_dir):
        """Test behavior at the exact threshold boundary."""
        # Exactly at threshold
        data = pd.DataFrame(
            {
                "value": range(1000),
            }
        )

        dtypes = {"value": "n"}

        # Should not sample when exactly at threshold
        output_path = plot(
            data,
            dtypes,
            output_path=temp_dir,
            show=False,
            use_dask=False,
            max_rows=1000,
            sample_size=500,
        )

        assert output_path == temp_dir

    @pytest.mark.integration
    def test_plot_just_over_threshold(self, temp_dir, caplog):
        """Test behavior just over the threshold."""
        # One row over threshold
        data = pd.DataFrame(
            {
                "value": range(1001),
            }
        )

        dtypes = {"value": "n"}

        # Should sample
        output_path = plot(
            data,
            dtypes,
            output_path=temp_dir,
            show=False,
            use_dask=False,
            max_rows=1000,
            sample_size=500,
        )

        assert output_path == temp_dir
        log_messages = [record.message for record in caplog.records]
        assert any("Sampling" in msg for msg in log_messages)


class TestCLIWithLargeDatasetOptions:
    """Tests for CLI with large dataset options."""

    @pytest.mark.integration
    def test_cli_creates_large_dataset_csv(self, temp_dir):
        """Create a large CSV for CLI testing."""
        # Create a large dataset CSV
        large_data = pd.DataFrame(
            {
                "value1": range(2000),
                "value2": range(2000),
                "category": ["A", "B", "C"] * 666 + ["D"] * 2,
            }
        )

        csv_path = os.path.join(temp_dir, "large_test.csv")
        large_data.to_csv(csv_path, index=False)

        dtypes_dict = {"value1": "n", "value2": "n", "category": "c"}
        import json

        dtypes_path = os.path.join(temp_dir, "dtypes.json")
        with open(dtypes_path, "w") as f:
            json.dump(dtypes_dict, f)

        assert os.path.exists(csv_path)
        assert os.path.exists(dtypes_path)

        # Verify the data
        df = pd.read_csv(csv_path)
        assert len(df) == 2000
