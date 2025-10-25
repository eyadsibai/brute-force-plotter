"""
Unit tests for core plotting functions in brute_force_plotter.
"""

import os
from unittest.mock import patch

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.brute_force_plotter import (
    bar_box_violin_dot_plots,
    bar_plot,
    correlation_heatmap,
    heatmap,
    histogram_violin_plots,
    missing_plot,
    plot_category_category_sync,
    plot_category_numeric_sync,
    plot_correlation_matrix,
    plot_missing_values,
    plot_numeric_numeric_sync,
    plot_single_category_sync,
    plot_single_numeric_sync,
    scatter_plot,
)


class TestSingleNumericPlot:
    """Tests for plot_single_numeric function."""

    @pytest.mark.unit
    @pytest.mark.plotting
    def test_creates_histogram_and_violin_plot(self, temp_dir, sample_parquet_file, mixed_dtypes):
        """Test that numeric plot creates histogram and violin plots."""
        from src import brute_force_plotter
        brute_force_plotter._save_plots = True
        brute_force_plotter._show_plots = False
        
        plot_single_numeric_sync(sample_parquet_file, "age", temp_dir)
        
        expected_file = os.path.join(temp_dir, "age-dist-plot.png")
        assert os.path.exists(expected_file)

    @pytest.mark.unit
    @pytest.mark.plotting
    def test_handles_missing_values_in_numeric(self, temp_dir, sample_data_with_missing):
        """Test that numeric plot handles missing values correctly."""
        import tempfile
        from src import brute_force_plotter
        
        brute_force_plotter._save_plots = True
        brute_force_plotter._show_plots = False
        
        # Create parquet file
        with tempfile.NamedTemporaryFile(suffix=".parq", delete=False) as tmp:
            parquet_path = tmp.name
        sample_data_with_missing.to_parquet(parquet_path)
        
        try:
            plot_single_numeric_sync(parquet_path, "col1", temp_dir)
            
            expected_file = os.path.join(temp_dir, "col1-dist-plot.png")
            assert os.path.exists(expected_file)
        finally:
            if os.path.exists(parquet_path):
                os.remove(parquet_path)


class TestSingleCategoryPlot:
    """Tests for plot_single_category function."""

    @pytest.mark.unit
    @pytest.mark.plotting
    def test_creates_bar_plot_for_category(self, temp_dir, sample_parquet_file):
        """Test that categorical plot creates a bar plot."""
        from src import brute_force_plotter
        brute_force_plotter._save_plots = True
        brute_force_plotter._show_plots = False
        
        plot_single_category_sync(sample_parquet_file, "gender", temp_dir)
        
        expected_file = os.path.join(temp_dir, "gender-bar-plot.png")
        assert os.path.exists(expected_file)

    @pytest.mark.unit
    @pytest.mark.plotting
    def test_ignores_columns_with_many_categories(self, temp_dir, sample_data_many_categories):
        """Test that columns with >50 categories are ignored."""
        import tempfile
        from src import brute_force_plotter
        
        brute_force_plotter._save_plots = True
        brute_force_plotter._show_plots = False
        brute_force_plotter.ignore.clear()
        
        # Create parquet file
        with tempfile.NamedTemporaryFile(suffix=".parq", delete=False) as tmp:
            parquet_path = tmp.name
        sample_data_many_categories.to_parquet(parquet_path)
        
        try:
            plot_single_category_sync(parquet_path, "many_cats", temp_dir)
            
            # Check that the column was added to ignore set
            assert "many_cats" in brute_force_plotter.ignore
            
            # Check that no plot was created
            expected_file = os.path.join(temp_dir, "many_cats-bar-plot.png")
            assert not os.path.exists(expected_file)
        finally:
            if os.path.exists(parquet_path):
                os.remove(parquet_path)


class TestCategoryCategoryPlot:
    """Tests for plot_category_category function."""

    @pytest.mark.unit
    @pytest.mark.plotting
    def test_creates_bar_and_heatmap(self, temp_dir, sample_parquet_file):
        """Test that category-category plot creates bar plot and heatmap."""
        from src import brute_force_plotter
        brute_force_plotter._save_plots = True
        brute_force_plotter._show_plots = False
        
        plot_category_category_sync(sample_parquet_file, "gender", "education", temp_dir)
        
        # Check for bar plot
        bar_file = os.path.join(temp_dir, "education-gender-bar-plot.png")
        assert os.path.exists(bar_file)
        
        # Check for heatmap
        heatmap_file = os.path.join(temp_dir, "education-gender-heatmap.png")
        assert os.path.exists(heatmap_file)


class TestNumericNumericPlot:
    """Tests for plot_numeric_numeric function."""

    @pytest.mark.unit
    @pytest.mark.plotting
    def test_creates_scatter_plot(self, temp_dir, sample_parquet_file):
        """Test that numeric-numeric plot creates a scatter plot."""
        from src import brute_force_plotter
        brute_force_plotter._save_plots = True
        brute_force_plotter._show_plots = False
        
        plot_numeric_numeric_sync(sample_parquet_file, "age", "income", temp_dir)
        
        expected_file = os.path.join(temp_dir, "age-income-scatter-plot.png")
        assert os.path.exists(expected_file)


class TestCategoryNumericPlot:
    """Tests for plot_category_numeric function."""

    @pytest.mark.unit
    @pytest.mark.plotting
    def test_creates_multi_plot_panel(self, temp_dir, sample_parquet_file):
        """Test that category-numeric plot creates multi-panel plot."""
        from src import brute_force_plotter
        brute_force_plotter._save_plots = True
        brute_force_plotter._show_plots = False
        
        plot_category_numeric_sync(sample_parquet_file, "gender", "age", temp_dir)
        
        expected_file = os.path.join(temp_dir, "gender-age-plot.png")
        assert os.path.exists(expected_file)


class TestCorrelationMatrix:
    """Tests for correlation matrix plotting."""

    @pytest.mark.unit
    @pytest.mark.plotting
    def test_creates_correlation_matrices(self, temp_dir, sample_parquet_file):
        """Test that correlation matrices are created."""
        from src import brute_force_plotter
        brute_force_plotter._save_plots = True
        brute_force_plotter._show_plots = False
        
        dtypes = {"age": "n", "income": "n"}
        
        # This is a delayed function, so we need to compute it
        result = plot_correlation_matrix(sample_parquet_file, dtypes, temp_dir)
        result.compute()
        
        # Check for Pearson correlation
        pearson_file = os.path.join(temp_dir, "correlation-pearson.png")
        assert os.path.exists(pearson_file)
        
        # Check for Spearman correlation
        spearman_file = os.path.join(temp_dir, "correlation-spearman.png")
        assert os.path.exists(spearman_file)

    @pytest.mark.unit
    def test_skips_when_insufficient_numeric_columns(self, temp_dir, sample_parquet_file):
        """Test that correlation matrix is skipped when <2 numeric columns."""
        from src import brute_force_plotter
        brute_force_plotter._save_plots = True
        brute_force_plotter._show_plots = False
        
        # Only one numeric column
        dtypes = {"age": "n", "gender": "c"}
        
        result = plot_correlation_matrix(sample_parquet_file, dtypes, temp_dir)
        result.compute()
        
        # No files should be created
        pearson_file = os.path.join(temp_dir, "correlation-pearson.png")
        spearman_file = os.path.join(temp_dir, "correlation-spearman.png")
        assert not os.path.exists(pearson_file)
        assert not os.path.exists(spearman_file)


class TestMissingValuesPlot:
    """Tests for missing values plotting."""

    @pytest.mark.unit
    @pytest.mark.plotting
    def test_creates_missing_values_heatmap(self, temp_dir, sample_data_with_missing):
        """Test that missing values heatmap is created."""
        import tempfile
        from src import brute_force_plotter
        
        brute_force_plotter._save_plots = True
        brute_force_plotter._show_plots = False
        
        # Create parquet file
        with tempfile.NamedTemporaryFile(suffix=".parq", delete=False) as tmp:
            parquet_path = tmp.name
        sample_data_with_missing.to_parquet(parquet_path)
        
        try:
            dtypes = {"col1": "n", "col2": "n", "category": "c"}
            
            result = plot_missing_values(parquet_path, dtypes, temp_dir)
            result.compute()
            
            expected_file = os.path.join(temp_dir, "missing-values-heatmap.png")
            assert os.path.exists(expected_file)
        finally:
            if os.path.exists(parquet_path):
                os.remove(parquet_path)

    @pytest.mark.unit
    def test_skips_when_no_missing_values(self, temp_dir, sample_parquet_file):
        """Test that plot is skipped when there are no missing values."""
        from src import brute_force_plotter
        brute_force_plotter._save_plots = True
        brute_force_plotter._show_plots = False
        
        dtypes = {"age": "n", "income": "n", "gender": "c"}
        
        result = plot_missing_values(sample_parquet_file, dtypes, temp_dir)
        result.compute()
        
        # No file should be created since there are no missing values
        expected_file = os.path.join(temp_dir, "missing-values-heatmap.png")
        assert not os.path.exists(expected_file)


class TestPlotHelperFunctions:
    """Tests for helper plotting functions."""

    @pytest.mark.unit
    @pytest.mark.plotting
    def test_histogram_violin_plots(self, temp_dir, sample_numeric_data):
        """Test histogram_violin_plots helper function."""
        from src import brute_force_plotter
        brute_force_plotter._save_plots = True
        brute_force_plotter._show_plots = False
        
        file_path = os.path.join(temp_dir, "test-histogram-violin.png")
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
        
        histogram_violin_plots(sample_numeric_data["col1"], axes, file_name=file_path)
        
        assert os.path.exists(file_path)

    @pytest.mark.unit
    @pytest.mark.plotting
    def test_bar_plot(self, temp_dir, sample_categorical_data):
        """Test bar_plot helper function."""
        from src import brute_force_plotter
        brute_force_plotter._save_plots = True
        brute_force_plotter._show_plots = False
        
        file_path = os.path.join(temp_dir, "test-bar.png")
        
        bar_plot(sample_categorical_data, "category1", file_name=file_path)
        
        assert os.path.exists(file_path)

    @pytest.mark.unit
    @pytest.mark.plotting
    def test_scatter_plot(self, temp_dir, sample_numeric_data):
        """Test scatter_plot helper function."""
        from src import brute_force_plotter
        brute_force_plotter._save_plots = True
        brute_force_plotter._show_plots = False
        
        file_path = os.path.join(temp_dir, "test-scatter.png")
        
        scatter_plot(sample_numeric_data, "col1", "col2", file_name=file_path)
        
        assert os.path.exists(file_path)

    @pytest.mark.unit
    @pytest.mark.plotting
    def test_heatmap(self, temp_dir):
        """Test heatmap helper function."""
        from src import brute_force_plotter
        brute_force_plotter._save_plots = True
        brute_force_plotter._show_plots = False
        
        file_path = os.path.join(temp_dir, "test-heatmap.png")
        
        # Create a simple cross-tabulation data
        data = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"], index=["X", "Y"])
        
        heatmap(data, file_name=file_path)
        
        assert os.path.exists(file_path)

    @pytest.mark.unit
    @pytest.mark.plotting
    def test_correlation_heatmap(self, temp_dir, sample_numeric_data):
        """Test correlation_heatmap helper function."""
        from src import brute_force_plotter
        brute_force_plotter._save_plots = True
        brute_force_plotter._show_plots = False
        
        file_path = os.path.join(temp_dir, "test-correlation.png")
        
        corr_matrix = sample_numeric_data.corr()
        correlation_heatmap(corr_matrix, file_name=file_path, title="Test Correlation")
        
        assert os.path.exists(file_path)

    @pytest.mark.unit
    @pytest.mark.plotting
    def test_missing_plot(self, temp_dir, sample_data_with_missing):
        """Test missing_plot helper function."""
        from src import brute_force_plotter
        brute_force_plotter._save_plots = True
        brute_force_plotter._show_plots = False
        
        file_path = os.path.join(temp_dir, "test-missing.png")
        
        missing_data = sample_data_with_missing.isnull()
        missing_plot(missing_data, file_name=file_path)
        
        assert os.path.exists(file_path)

    @pytest.mark.unit
    @pytest.mark.plotting
    def test_bar_box_violin_dot_plots(self, temp_dir, sample_mixed_data):
        """Test bar_box_violin_dot_plots helper function."""
        from itertools import chain
        from src import brute_force_plotter
        
        brute_force_plotter._save_plots = True
        brute_force_plotter._show_plots = False
        
        file_path = os.path.join(temp_dir, "test-multi-panel.png")
        fig, axes = plt.subplots(2, 2, sharex="col", sharey="row", figsize=(8, 6))
        axes = list(chain.from_iterable(axes))
        
        bar_box_violin_dot_plots(sample_mixed_data, "gender", "age", axes, file_name=file_path)
        
        assert os.path.exists(file_path)
