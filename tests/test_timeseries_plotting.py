"""
Unit tests for time series plotting functions in brute_force_plotter.
"""

import os
import tempfile

import pandas as pd
import pytest

from src.brute_force_plotter import (
    multiple_time_series_plot,
    plot_single_timeseries_sync,
    plot_timeseries_category_numeric_sync,
    plot_timeseries_numeric_sync,
    plot_timeseries_timeseries_sync,
    time_series_category_plot,
    time_series_line_plot,
    time_series_numeric_plot,
)


@pytest.fixture
def sample_timeseries_data():
    """Create a sample DataFrame with time series data."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "value1": range(100),
            "value2": [x * 2 for x in range(100)],
            "category": ["A", "B"] * 50,
        }
    )


@pytest.fixture
def sample_timeseries_parquet(temp_dir, sample_timeseries_data):
    """Create a temporary parquet file with time series data."""
    parquet_path = os.path.join(temp_dir, "timeseries_data.parq")
    sample_timeseries_data.to_parquet(parquet_path)
    return parquet_path


@pytest.fixture
def temp_parquet_file(temp_dir):
    """Create a temporary parquet file from a DataFrame and clean it up after test."""
    parquet_files = []

    def _create_parquet(data):
        with tempfile.NamedTemporaryFile(
            suffix=".parq", delete=False, dir=temp_dir
        ) as tmp:
            parquet_path = tmp.name
        data.to_parquet(parquet_path)
        parquet_files.append(parquet_path)
        return parquet_path

    yield _create_parquet

    # Cleanup
    for parquet_path in parquet_files:
        if os.path.exists(parquet_path):
            os.remove(parquet_path)


class TestSingleTimeSeriesPlot:
    """Tests for single time series plotting."""

    @pytest.mark.unit
    @pytest.mark.plotting
    def test_creates_timeseries_plot(self, temp_dir, sample_timeseries_parquet):
        """Test that single time series plot is created."""
        from src import brute_force_plotter

        brute_force_plotter._save_plots = True
        brute_force_plotter._show_plots = False

        plot_single_timeseries_sync(sample_timeseries_parquet, "date", temp_dir)

        expected_file = os.path.join(temp_dir, "date-timeseries-plot.png")
        assert os.path.exists(expected_file)

    @pytest.mark.unit
    @pytest.mark.plotting
    def test_handles_non_datetime_timeseries(self, temp_dir, temp_parquet_file):
        """Test that time series plot handles non-datetime columns by converting them."""
        from src import brute_force_plotter

        brute_force_plotter._save_plots = True
        brute_force_plotter._show_plots = False

        # Create data with string dates
        data = pd.DataFrame(
            {
                "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "value": [1, 2, 3],
            }
        )

        parquet_path = temp_parquet_file(data)
        plot_single_timeseries_sync(parquet_path, "date", temp_dir)

        expected_file = os.path.join(temp_dir, "date-timeseries-plot.png")
        assert os.path.exists(expected_file)


class TestTimeSeriesNumericPlot:
    """Tests for time series vs numeric plotting."""

    @pytest.mark.unit
    @pytest.mark.plotting
    def test_creates_timeseries_numeric_plot(self, temp_dir, sample_timeseries_parquet):
        """Test that time series-numeric plot is created."""
        from src import brute_force_plotter

        brute_force_plotter._save_plots = True
        brute_force_plotter._show_plots = False

        plot_timeseries_numeric_sync(
            sample_timeseries_parquet, "date", "value1", temp_dir
        )

        expected_file = os.path.join(temp_dir, "date-value1-timeseries-plot.png")
        assert os.path.exists(expected_file)

    @pytest.mark.unit
    @pytest.mark.plotting
    def test_handles_missing_values_in_timeseries_numeric(
        self, temp_dir, temp_parquet_file
    ):
        """Test that time series-numeric plot handles missing values."""
        from src import brute_force_plotter

        brute_force_plotter._save_plots = True
        brute_force_plotter._show_plots = False

        # Create data with missing values
        dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
        data = pd.DataFrame(
            {
                "date": dates,
                "value": [1, 2, None, 4, 5, None, 7, 8, 9, 10],
            }
        )

        parquet_path = temp_parquet_file(data)
        plot_timeseries_numeric_sync(parquet_path, "date", "value", temp_dir)

        expected_file = os.path.join(temp_dir, "date-value-timeseries-plot.png")
        assert os.path.exists(expected_file)


class TestTimeSeriesTimeSeriesPlot:
    """Tests for multiple time series overlay plotting."""

    @pytest.mark.unit
    @pytest.mark.plotting
    def test_creates_timeseries_comparison_plot(self, temp_dir, temp_parquet_file):
        """Test that time series comparison plot is created."""
        from src import brute_force_plotter

        brute_force_plotter._save_plots = True
        brute_force_plotter._show_plots = False

        # Create data with two time series
        dates1 = pd.date_range(start="2023-01-01", periods=10, freq="D")
        dates2 = pd.date_range(start="2023-01-05", periods=10, freq="D")
        data = pd.DataFrame({"date1": dates1, "date2": dates2})

        parquet_path = temp_parquet_file(data)
        plot_timeseries_timeseries_sync(parquet_path, "date1", "date2", temp_dir)

        expected_file = os.path.join(temp_dir, "date1-date2-timeseries-comparison.png")
        assert os.path.exists(expected_file)


class TestTimeSeriesCategoryNumericPlot:
    """Tests for time series grouped by category plotting."""

    @pytest.mark.unit
    @pytest.mark.plotting
    def test_creates_grouped_timeseries_plot(self, temp_dir, sample_timeseries_parquet):
        """Test that grouped time series plot is created."""
        from src import brute_force_plotter

        brute_force_plotter._save_plots = True
        brute_force_plotter._show_plots = False

        plot_timeseries_category_numeric_sync(
            sample_timeseries_parquet, "date", "category", "value1", temp_dir
        )

        expected_file = os.path.join(temp_dir, "date-value1-by-category-timeseries.png")
        assert os.path.exists(expected_file)

    @pytest.mark.unit
    @pytest.mark.plotting
    def test_handles_multiple_categories(self, temp_dir, temp_parquet_file):
        """Test that grouped time series handles multiple categories."""
        from src import brute_force_plotter

        brute_force_plotter._save_plots = True
        brute_force_plotter._show_plots = False

        # Create data with multiple categories
        dates = pd.date_range(start="2023-01-01", periods=20, freq="D")
        data = pd.DataFrame(
            {
                "date": dates,
                "category": ["A", "B", "C", "D"] * 5,
                "value": range(20),
            }
        )

        parquet_path = temp_parquet_file(data)
        plot_timeseries_category_numeric_sync(
            parquet_path, "date", "category", "value", temp_dir
        )

        expected_file = os.path.join(temp_dir, "date-value-by-category-timeseries.png")
        assert os.path.exists(expected_file)


class TestTimeSeriesHelperFunctions:
    """Tests for time series helper plotting functions."""

    @pytest.mark.unit
    @pytest.mark.plotting
    def test_time_series_line_plot(self, temp_dir, sample_timeseries_data):
        """Test time_series_line_plot helper function."""
        from src import brute_force_plotter

        brute_force_plotter._save_plots = True
        brute_force_plotter._show_plots = False

        file_path = os.path.join(temp_dir, "test-timeseries-line.png")

        time_series_line_plot(sample_timeseries_data, "date", file_name=file_path)

        assert os.path.exists(file_path)

    @pytest.mark.unit
    @pytest.mark.plotting
    def test_time_series_numeric_plot(self, temp_dir, sample_timeseries_data):
        """Test time_series_numeric_plot helper function."""
        from src import brute_force_plotter

        brute_force_plotter._save_plots = True
        brute_force_plotter._show_plots = False

        file_path = os.path.join(temp_dir, "test-timeseries-numeric.png")

        time_series_numeric_plot(
            sample_timeseries_data, "date", "value1", file_name=file_path
        )

        assert os.path.exists(file_path)

    @pytest.mark.unit
    @pytest.mark.plotting
    def test_time_series_category_plot(self, temp_dir, sample_timeseries_data):
        """Test time_series_category_plot helper function."""
        from src import brute_force_plotter

        brute_force_plotter._save_plots = True
        brute_force_plotter._show_plots = False

        file_path = os.path.join(temp_dir, "test-timeseries-category.png")

        time_series_category_plot(
            sample_timeseries_data,
            "date",
            "value1",
            "category",
            file_name=file_path,
        )

        assert os.path.exists(file_path)

    @pytest.mark.unit
    @pytest.mark.plotting
    def test_multiple_time_series_plot(self, temp_dir, sample_timeseries_data):
        """Test multiple_time_series_plot helper function."""
        from src import brute_force_plotter

        brute_force_plotter._save_plots = True
        brute_force_plotter._show_plots = False

        file_path = os.path.join(temp_dir, "test-multiple-timeseries.png")

        multiple_time_series_plot(
            sample_timeseries_data,
            "date",
            ["value1", "value2"],
            file_name=file_path,
        )

        assert os.path.exists(file_path)
