"""
Integration tests for the plot library function and CLI.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.brute_force_plotter import create_plots, plot


class TestPlotLibraryFunction:
    """Integration tests for the plot() library function."""

    @pytest.mark.integration
    def test_plot_with_output_path(self, sample_mixed_data, mixed_dtypes, temp_dir):
        """Test plotting with specified output path."""
        output_path = plot(
            sample_mixed_data,
            mixed_dtypes,
            output_path=temp_dir,
            show=False,
            use_dask=False,
        )
        
        assert output_path == temp_dir
        
        # Check that directories were created
        assert os.path.exists(os.path.join(temp_dir, "distributions"))
        assert os.path.exists(os.path.join(temp_dir, "2d_interactions"))

    @pytest.mark.integration
    def test_plot_creates_distribution_plots(self, sample_mixed_data, mixed_dtypes, temp_dir):
        """Test that distribution plots are created."""
        plot(
            sample_mixed_data,
            mixed_dtypes,
            output_path=temp_dir,
            show=False,
            use_dask=False,
        )
        
        dist_dir = os.path.join(temp_dir, "distributions")
        
        # Check for numeric distribution plots
        assert os.path.exists(os.path.join(dist_dir, "age-dist-plot.png"))
        assert os.path.exists(os.path.join(dist_dir, "income-dist-plot.png"))
        
        # Check for categorical bar plots
        assert os.path.exists(os.path.join(dist_dir, "gender-bar-plot.png"))
        assert os.path.exists(os.path.join(dist_dir, "education-bar-plot.png"))

    @pytest.mark.integration
    def test_plot_creates_correlation_matrices(self, sample_mixed_data, mixed_dtypes, temp_dir):
        """Test that correlation matrices are created."""
        plot(
            sample_mixed_data,
            mixed_dtypes,
            output_path=temp_dir,
            show=False,
            use_dask=False,
        )
        
        dist_dir = os.path.join(temp_dir, "distributions")
        
        # Check for correlation matrices
        assert os.path.exists(os.path.join(dist_dir, "correlation-pearson.png"))
        assert os.path.exists(os.path.join(dist_dir, "correlation-spearman.png"))

    @pytest.mark.integration
    def test_plot_creates_2d_interaction_plots(self, sample_mixed_data, mixed_dtypes, temp_dir):
        """Test that 2D interaction plots are created."""
        plot(
            sample_mixed_data,
            mixed_dtypes,
            output_path=temp_dir,
            show=False,
            use_dask=False,
        )
        
        interactions_dir = os.path.join(temp_dir, "2d_interactions")
        
        # Check for numeric-numeric scatter plot
        assert os.path.exists(os.path.join(interactions_dir, "age-income-scatter-plot.png"))
        
        # Check for category-category plots
        assert os.path.exists(os.path.join(interactions_dir, "education-gender-bar-plot.png"))
        assert os.path.exists(os.path.join(interactions_dir, "education-gender-heatmap.png"))
        
        # Check for category-numeric plots
        assert os.path.exists(os.path.join(interactions_dir, "gender-age-plot.png"))
        assert os.path.exists(os.path.join(interactions_dir, "gender-income-plot.png"))

    @pytest.mark.integration
    def test_plot_with_export_stats(self, sample_mixed_data, mixed_dtypes, temp_dir):
        """Test plotting with statistical export enabled."""
        plot(
            sample_mixed_data,
            mixed_dtypes,
            output_path=temp_dir,
            show=False,
            use_dask=False,
            export_stats=True,
        )
        
        # Check that statistics directory exists
        stats_dir = os.path.join(temp_dir, "statistics")
        assert os.path.exists(stats_dir)
        
        # Check for statistics files
        assert os.path.exists(os.path.join(stats_dir, "numeric_statistics.csv"))
        assert os.path.exists(os.path.join(stats_dir, "missing_values_summary.csv"))
        assert os.path.exists(os.path.join(stats_dir, "overall_summary.csv"))

    @pytest.mark.integration
    def test_plot_with_dask(self, sample_mixed_data, mixed_dtypes, temp_dir):
        """Test plotting with Dask enabled."""
        output_path = plot(
            sample_mixed_data,
            mixed_dtypes,
            output_path=temp_dir,
            show=False,
            use_dask=True,
            n_workers=2,
        )
        
        assert output_path == temp_dir
        
        # Check that some plots were created
        dist_dir = os.path.join(temp_dir, "distributions")
        assert os.path.exists(os.path.join(dist_dir, "age-dist-plot.png"))

    @pytest.mark.integration
    def test_plot_without_output_path(self, sample_mixed_data, mixed_dtypes):
        """Test plotting without specified output path (uses temp directory)."""
        output_path = plot(
            sample_mixed_data,
            mixed_dtypes,
            output_path=None,
            show=False,
            use_dask=False,
        )
        
        # Should return a temporary directory path
        assert output_path is not None
        assert os.path.exists(output_path)
        assert "brute_force_plotter" in output_path

    @pytest.mark.integration
    def test_plot_handles_ignored_columns(self, sample_mixed_data, temp_dir):
        """Test that ignored columns are properly handled."""
        dtypes = {
            "age": "n",
            "income": "i",  # ignored
            "gender": "c",
            "education": "i",  # ignored
        }
        
        plot(
            sample_mixed_data,
            dtypes,
            output_path=temp_dir,
            show=False,
            use_dask=False,
        )
        
        dist_dir = os.path.join(temp_dir, "distributions")
        
        # Age plot should exist
        assert os.path.exists(os.path.join(dist_dir, "age-dist-plot.png"))
        
        # Income plot should NOT exist (ignored)
        assert not os.path.exists(os.path.join(dist_dir, "income-dist-plot.png"))
        
        # Gender plot should exist
        assert os.path.exists(os.path.join(dist_dir, "gender-bar-plot.png"))
        
        # Education plot should NOT exist (ignored)
        assert not os.path.exists(os.path.join(dist_dir, "education-bar-plot.png"))

    @pytest.mark.integration
    def test_plot_with_missing_values(self, sample_data_with_missing, temp_dir):
        """Test plotting with data containing missing values."""
        dtypes = {"col1": "n", "col2": "n", "category": "c"}
        
        plot(
            sample_data_with_missing,
            dtypes,
            output_path=temp_dir,
            show=False,
            use_dask=False,
        )
        
        dist_dir = os.path.join(temp_dir, "distributions")
        
        # Missing values heatmap should be created
        assert os.path.exists(os.path.join(dist_dir, "missing-values-heatmap.png"))

    @pytest.mark.integration
    @pytest.mark.slow
    def test_plot_end_to_end_with_titanic_data(self, titanic_data, titanic_dtypes, temp_dir):
        """Test end-to-end plotting with the Titanic dataset."""
        if titanic_data is None:
            pytest.skip("Titanic data not available")
        
        output_path = plot(
            titanic_data,
            titanic_dtypes,
            output_path=temp_dir,
            show=False,
            use_dask=False,
            export_stats=True,
        )
        
        assert output_path == temp_dir
        
        # Verify directories exist
        assert os.path.exists(os.path.join(temp_dir, "distributions"))
        assert os.path.exists(os.path.join(temp_dir, "2d_interactions"))
        assert os.path.exists(os.path.join(temp_dir, "statistics"))
        
        # Check for some expected plots
        dist_dir = os.path.join(temp_dir, "distributions")
        assert os.path.exists(os.path.join(dist_dir, "Age-dist-plot.png"))
        assert os.path.exists(os.path.join(dist_dir, "Survived-bar-plot.png"))


class TestCreatePlotsFunction:
    """Tests for the create_plots function."""

    @pytest.mark.integration
    def test_create_plots_returns_delayed_tasks(self, sample_parquet_file, mixed_dtypes):
        """Test that create_plots returns Dask delayed tasks."""
        output_dir = tempfile.mkdtemp()
        
        try:
            plots = create_plots(sample_parquet_file, mixed_dtypes, output_dir, use_dask=True)
            
            # Should return a list of delayed objects
            assert isinstance(plots, list)
            assert len(plots) > 0
        finally:
            # Cleanup
            import shutil
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)

    @pytest.mark.integration
    def test_create_plots_with_use_dask_false(self, sample_parquet_file, mixed_dtypes):
        """Test create_plots with use_dask=False."""
        output_dir = tempfile.mkdtemp()
        
        try:
            from src import brute_force_plotter
            brute_force_plotter._save_plots = True
            brute_force_plotter._show_plots = False
            
            plots = create_plots(sample_parquet_file, mixed_dtypes, output_dir, use_dask=False)
            
            # Should still return a list (though items are executed immediately)
            assert isinstance(plots, list)
            
            # Check that some plots were created
            dist_dir = os.path.join(output_dir, "distributions")
            assert os.path.exists(os.path.join(dist_dir, "age-dist-plot.png"))
        finally:
            # Cleanup
            import shutil
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)


class TestCLIInterface:
    """Integration tests for the command-line interface."""

    @pytest.mark.integration
    @pytest.mark.cli
    def test_cli_basic_execution(self, sample_csv_file, sample_dtypes_json, temp_dir):
        """Test basic CLI execution."""
        import sys
        
        # Run the CLI
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.brute_force_plotter",
                sample_csv_file,
                sample_dtypes_json,
                temp_dir,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        # Check that it ran successfully
        assert result.returncode == 0
        
        # Check that output directories were created
        assert os.path.exists(os.path.join(temp_dir, "distributions"))
        assert os.path.exists(os.path.join(temp_dir, "2d_interactions"))

    @pytest.mark.integration
    @pytest.mark.cli
    def test_cli_with_export_stats(self, sample_csv_file, sample_dtypes_json, temp_dir):
        """Test CLI with --export-stats flag."""
        import sys
        
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.brute_force_plotter",
                sample_csv_file,
                sample_dtypes_json,
                temp_dir,
                "--export-stats",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        assert result.returncode == 0
        
        # Check that statistics were exported
        stats_dir = os.path.join(temp_dir, "statistics")
        assert os.path.exists(stats_dir)

    @pytest.mark.integration
    @pytest.mark.cli
    def test_cli_with_theme_option(self, sample_csv_file, sample_dtypes_json, temp_dir):
        """Test CLI with --theme option."""
        import sys
        
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.brute_force_plotter",
                sample_csv_file,
                sample_dtypes_json,
                temp_dir,
                "--theme",
                "whitegrid",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        assert result.returncode == 0
        
        # Check that plots were created
        assert os.path.exists(os.path.join(temp_dir, "distributions"))

    @pytest.mark.integration
    @pytest.mark.cli
    def test_cli_with_n_workers_option(self, sample_csv_file, sample_dtypes_json, temp_dir):
        """Test CLI with --n-workers option."""
        import sys
        
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.brute_force_plotter",
                sample_csv_file,
                sample_dtypes_json,
                temp_dir,
                "--n-workers",
                "2",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        assert result.returncode == 0
