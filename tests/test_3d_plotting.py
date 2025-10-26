"""
Unit tests for 3-variable plotting functions in brute_force_plotter.
"""

import os

import pandas as pd
import pytest

from src.brute_force_plotter import (
    contour_plot,
    grouped_bar_violin_plot,
    multi_level_heatmap,
    plot_category_category_category_sync,
    plot_numeric_category_category_sync,
    plot_numeric_numeric_category_sync,
    plot_numeric_numeric_numeric_sync,
    scatter_plot_3d,
    scatter_plot_with_hue,
)


@pytest.fixture
def sample_3d_numeric_data(temp_dir):
    """Create a parquet file with 3 numeric columns for testing."""
    df = pd.DataFrame(
        {
            "var1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 5,
            "var2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] * 5,
            "var3": [5, 15, 25, 35, 45, 55, 65, 75, 85, 95] * 5,
        }
    )
    file_path = os.path.join(temp_dir, "3d_numeric.parquet")
    df.to_parquet(file_path)
    return file_path


@pytest.fixture
def sample_3d_categorical_data(temp_dir):
    """Create a parquet file with 3 categorical columns for testing."""
    df = pd.DataFrame(
        {
            "cat1": ["A", "B", "C"] * 20,
            "cat2": ["X", "Y", "Z"] * 20,
            "cat3": ["P", "Q"] * 30,
        }
    )
    file_path = os.path.join(temp_dir, "3d_categorical.parquet")
    df.to_parquet(file_path)
    return file_path


@pytest.fixture
def sample_2num_1cat_data(temp_dir):
    """Create a parquet file with 2 numeric and 1 categorical column."""
    df = pd.DataFrame(
        {
            "num1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 5,
            "num2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] * 5,
            "category": ["A", "B", "C", "A", "B"] * 10,
        }
    )
    file_path = os.path.join(temp_dir, "2num_1cat.parquet")
    df.to_parquet(file_path)
    return file_path


@pytest.fixture
def sample_1num_2cat_data(temp_dir):
    """Create a parquet file with 1 numeric and 2 categorical columns."""
    df = pd.DataFrame(
        {
            "value": [
                10,
                20,
                30,
                40,
                50,
                60,
                70,
                80,
                90,
                100,
                15,
                25,
                35,
                45,
                55,
                65,
                75,
                85,
                95,
                105,
            ]
            * 3,
            "cat1": ["A", "B", "C", "A", "B", "C"] * 10,
            "cat2": ["X", "Y", "X", "Y", "X", "Y"] * 10,
        }
    )
    file_path = os.path.join(temp_dir, "1num_2cat.parquet")
    df.to_parquet(file_path)
    return file_path


class TestNumericNumericCategory:
    """Tests for plot_numeric_numeric_category function."""

    @pytest.mark.unit
    @pytest.mark.plotting
    def test_creates_scatter_plot_with_hue(self, temp_dir, sample_2num_1cat_data):
        """Test that 2 numeric + 1 category creates scatter with hue."""
        from src import brute_force_plotter

        brute_force_plotter._save_plots = True
        brute_force_plotter._show_plots = False

        plot_numeric_numeric_category_sync(
            sample_2num_1cat_data, "num1", "num2", "category", temp_dir
        )

        expected_file = os.path.join(temp_dir, "num1-num2-category-scatter-3d.png")
        assert os.path.exists(expected_file)

    @pytest.mark.unit
    @pytest.mark.plotting
    def test_scatter_with_hue_helper(self, temp_dir, sample_2num_1cat_data):
        """Test the scatter_plot_with_hue helper function."""
        from src import brute_force_plotter

        brute_force_plotter._save_plots = True
        brute_force_plotter._show_plots = False

        df = pd.read_parquet(sample_2num_1cat_data)
        file_name = os.path.join(temp_dir, "test_scatter_hue.png")

        scatter_plot_with_hue(df, "num1", "num2", "category", file_name=file_name)

        assert os.path.exists(file_name)


class TestNumericNumericNumeric:
    """Tests for plot_numeric_numeric_numeric function."""

    @pytest.mark.unit
    @pytest.mark.plotting
    def test_creates_3d_scatter_and_contour(self, temp_dir, sample_3d_numeric_data):
        """Test that 3 numeric variables create 3D scatter and contour plots."""
        from src import brute_force_plotter

        brute_force_plotter._save_plots = True
        brute_force_plotter._show_plots = False

        plot_numeric_numeric_numeric_sync(
            sample_3d_numeric_data, "var1", "var2", "var3", temp_dir
        )

        # Check for 3D scatter plot
        scatter_file = os.path.join(temp_dir, "var1-var2-var3-3d-scatter.png")
        assert os.path.exists(scatter_file)

        # Check for contour plot
        contour_file = os.path.join(temp_dir, "var1-var2-contour-var3.png")
        assert os.path.exists(contour_file)

    @pytest.mark.unit
    @pytest.mark.plotting
    def test_3d_scatter_helper(self, temp_dir, sample_3d_numeric_data):
        """Test the scatter_plot_3d helper function."""
        from src import brute_force_plotter

        brute_force_plotter._save_plots = True
        brute_force_plotter._show_plots = False

        df = pd.read_parquet(sample_3d_numeric_data)
        file_name = os.path.join(temp_dir, "test_3d_scatter.png")

        scatter_plot_3d(df, "var1", "var2", "var3", file_name=file_name)

        assert os.path.exists(file_name)

    @pytest.mark.unit
    @pytest.mark.plotting
    def test_contour_plot_helper(self, temp_dir, sample_3d_numeric_data):
        """Test the contour_plot helper function."""
        from src import brute_force_plotter

        brute_force_plotter._save_plots = True
        brute_force_plotter._show_plots = False

        df = pd.read_parquet(sample_3d_numeric_data)
        file_name = os.path.join(temp_dir, "test_contour.png")

        contour_plot(df, "var1", "var2", "var3", file_name=file_name)

        assert os.path.exists(file_name)

    @pytest.mark.unit
    @pytest.mark.plotting
    def test_handles_insufficient_data_for_contour(self, temp_dir):
        """Test contour plot handles insufficient data gracefully."""
        from src import brute_force_plotter

        brute_force_plotter._save_plots = True
        brute_force_plotter._show_plots = False

        # Create very small dataset
        df = pd.DataFrame(
            {
                "x": [1, 2],
                "y": [3, 4],
                "z": [5, 6],
            }
        )
        file_path = os.path.join(temp_dir, "small_data.parquet")
        df.to_parquet(file_path)

        file_name = os.path.join(temp_dir, "test_small_contour.png")

        # Should not raise an error
        contour_plot(df, "x", "y", "z", file_name=file_name)


class TestCategoryCategoryCategory:
    """Tests for plot_category_category_category function."""

    @pytest.mark.unit
    @pytest.mark.plotting
    def test_creates_multi_level_heatmap(self, temp_dir, sample_3d_categorical_data):
        """Test that 3 categorical variables create multi-level heatmap."""
        from src import brute_force_plotter

        brute_force_plotter._save_plots = True
        brute_force_plotter._show_plots = False

        plot_category_category_category_sync(
            sample_3d_categorical_data, "cat1", "cat2", "cat3", temp_dir
        )

        expected_file = os.path.join(temp_dir, "cat1-cat2-by-cat3-heatmap.png")
        assert os.path.exists(expected_file)

    @pytest.mark.unit
    @pytest.mark.plotting
    def test_multi_level_heatmap_helper(self, temp_dir, sample_3d_categorical_data):
        """Test the multi_level_heatmap helper function."""
        from src import brute_force_plotter

        brute_force_plotter._save_plots = True
        brute_force_plotter._show_plots = False

        df = pd.read_parquet(sample_3d_categorical_data)
        file_name = os.path.join(temp_dir, "test_multilevel.png")

        multi_level_heatmap(df, "cat1", "cat2", "cat3", file_name=file_name)

        assert os.path.exists(file_name)

    @pytest.mark.unit
    @pytest.mark.plotting
    def test_handles_many_categories(self, temp_dir):
        """Test multi-level heatmap with many categories."""
        from src import brute_force_plotter

        brute_force_plotter._save_plots = True
        brute_force_plotter._show_plots = False

        # Create data with many categories in third dimension
        df = pd.DataFrame(
            {
                "cat1": ["A", "B", "C"] * 50,
                "cat2": ["X", "Y"] * 75,
                "cat3": [f"Cat{i}" for i in range(15)] * 10,  # 15 categories
            }
        )
        file_path = os.path.join(temp_dir, "many_cats.parquet")
        df.to_parquet(file_path)

        file_name = os.path.join(temp_dir, "test_many_cats.png")

        # Should limit to first 10 categories
        multi_level_heatmap(df, "cat1", "cat2", "cat3", file_name=file_name)

        assert os.path.exists(file_name)


class TestNumericCategoryCategory:
    """Tests for plot_numeric_category_category function."""

    @pytest.mark.unit
    @pytest.mark.plotting
    def test_creates_grouped_visualizations(self, temp_dir, sample_1num_2cat_data):
        """Test that 1 numeric + 2 categories creates grouped plots."""
        from src import brute_force_plotter

        brute_force_plotter._save_plots = True
        brute_force_plotter._show_plots = False

        plot_numeric_category_category_sync(
            sample_1num_2cat_data, "value", "cat1", "cat2", temp_dir
        )

        expected_file = os.path.join(temp_dir, "value-cat1-cat2-grouped.png")
        assert os.path.exists(expected_file)

    @pytest.mark.unit
    @pytest.mark.plotting
    def test_grouped_bar_violin_helper(self, temp_dir, sample_1num_2cat_data):
        """Test the grouped_bar_violin_plot helper function."""
        from src import brute_force_plotter

        brute_force_plotter._save_plots = True
        brute_force_plotter._show_plots = False

        df = pd.read_parquet(sample_1num_2cat_data)
        file_name = os.path.join(temp_dir, "test_grouped.png")

        grouped_bar_violin_plot(df, "value", "cat1", "cat2", file_name=file_name)

        assert os.path.exists(file_name)

    @pytest.mark.unit
    @pytest.mark.plotting
    def test_handles_many_categories_in_second_cat(self, temp_dir):
        """Test grouped plot with many categories in second categorical variable."""
        from src import brute_force_plotter

        brute_force_plotter._save_plots = True
        brute_force_plotter._show_plots = False

        # Create data with many categories in second categorical variable
        # Need to ensure all arrays have the same length
        n = 120
        df = pd.DataFrame(
            {
                "value": list(range(n)),
                "cat1": ["A", "B", "C"] * (n // 3),
                "cat2": [f"Cat{i}" for i in range(20)] * (n // 20),  # 20 categories
            }
        )
        file_path = os.path.join(temp_dir, "many_cat2.parquet")
        df.to_parquet(file_path)

        file_name = os.path.join(temp_dir, "test_many_cat2.png")

        # Should limit to first 10 categories
        grouped_bar_violin_plot(df, "value", "cat1", "cat2", file_name=file_name)

        assert os.path.exists(file_name)


class TestThreeVariableIntegration:
    """Integration tests for 3-variable plotting in create_plots."""

    @pytest.mark.integration
    def test_create_plots_generates_3d_plots(self, temp_dir):
        """Test that create_plots generates 3-variable plots."""
        import src.brute_force_plotter
        from src.brute_force_plotter import create_plots

        src.brute_force_plotter._save_plots = True
        src.brute_force_plotter._show_plots = False

        # Create test data with multiple variables
        # Ensure all arrays have the same length
        n = 51
        df = pd.DataFrame(
            {
                "num1": list(range(n)),
                "num2": [x * 2 for x in range(n)],
                "num3": [x * 3 for x in range(n)],
                "cat1": (["A", "B", "C"] * 17)[:n],
                "cat2": (["X", "Y"] * 26)[:n],
            }
        )

        input_file = os.path.join(temp_dir, "test_input.parquet")
        df.to_parquet(input_file)

        dtypes = {
            "num1": "n",
            "num2": "n",
            "num3": "n",
            "cat1": "c",
            "cat2": "c",
        }

        # Create plots without dask for simplicity
        create_plots(input_file, dtypes, temp_dir, use_dask=False)

        # Check that 3D interactions directory was created
        three_d_path = os.path.join(temp_dir, "3d_interactions")
        assert os.path.exists(three_d_path)

        # Check for some expected 3-variable plots
        # We should have at least one 3-numeric plot
        files_in_3d = os.listdir(three_d_path)
        assert len(files_in_3d) > 0, "No 3D interaction plots were created"

    @pytest.mark.integration
    def test_create_plots_with_dask_generates_3d_plots(self, temp_dir):
        """Test that create_plots with Dask generates 3-variable plots."""
        import dask
        from dask.distributed import Client, LocalCluster

        import src.brute_force_plotter
        from src.brute_force_plotter import create_plots

        src.brute_force_plotter._save_plots = True
        src.brute_force_plotter._show_plots = False

        # Create test data
        df = pd.DataFrame(
            {
                "a": list(range(30)),
                "b": [x * 2 for x in range(30)],
                "c": [x * 3 for x in range(30)],
            }
        )

        input_file = os.path.join(temp_dir, "test_dask.parquet")
        df.to_parquet(input_file)

        dtypes = {
            "a": "n",
            "b": "n",
            "c": "n",
        }

        # Create plots with Dask
        cluster = LocalCluster(n_workers=2, silence_logs=40)
        client = Client(cluster)

        try:
            plots = create_plots(input_file, dtypes, temp_dir, use_dask=True)
            dask.compute(*plots)

            # Check that 3D interactions directory was created
            three_d_path = os.path.join(temp_dir, "3d_interactions")
            assert os.path.exists(three_d_path)

            # Should have at least the 3D scatter and contour for a-b-c
            files_in_3d = os.listdir(three_d_path)
            assert len(files_in_3d) > 0, (
                "No 3D interaction plots were created with Dask"
            )
        finally:
            client.close()
            cluster.close()
