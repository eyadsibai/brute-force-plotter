"""
Edge case tests for brute-force-plotter.
"""

import os
import tempfile

import pandas as pd
import pytest

from src.brute_force_plotter import (
    create_plots,
    plot,
    plot_single_category_sync,
)


class TestEmptyDataFrames:
    """Tests for handling empty DataFrames."""

    @pytest.mark.edge_case
    def test_plot_with_empty_dataframe(self, sample_empty_data, temp_dir):
        """Test that plotting with empty DataFrame doesn't crash."""
        dtypes = {}
        
        # Should not crash
        try:
            output_path = plot(
                sample_empty_data,
                dtypes,
                output_path=temp_dir,
                show=False,
                use_dask=False,
            )
            assert output_path == temp_dir
        except Exception as e:
            pytest.fail(f"Plotting empty DataFrame should not raise: {e}")

    @pytest.mark.edge_case
    def test_plot_with_empty_dtypes(self, sample_mixed_data, temp_dir):
        """Test plotting when all columns are ignored."""
        dtypes = {
            "age": "i",
            "income": "i",
            "gender": "i",
            "education": "i",
        }
        
        output_path = plot(
            sample_mixed_data,
            dtypes,
            output_path=temp_dir,
            show=False,
            use_dask=False,
        )
        
        assert output_path == temp_dir


class TestManyCategories:
    """Tests for columns with many categories (>50)."""

    @pytest.mark.edge_case
    def test_ignores_column_with_many_categories(self, sample_data_many_categories, temp_dir):
        """Test that columns with >50 categories are ignored."""
        import tempfile as tmp
        from src import brute_force_plotter
        
        brute_force_plotter._save_plots = True
        brute_force_plotter._show_plots = False
        brute_force_plotter.ignore.clear()
        
        # Create parquet file
        with tmp.NamedTemporaryFile(suffix=".parq", delete=False) as tmpfile:
            parquet_path = tmpfile.name
        sample_data_many_categories.to_parquet(parquet_path)
        
        try:
            plot_single_category_sync(parquet_path, "many_cats", temp_dir)
            
            # Check that column was added to ignore set
            assert "many_cats" in brute_force_plotter.ignore
            
            # Check that no plot was created
            plot_file = os.path.join(temp_dir, "many_cats-bar-plot.png")
            assert not os.path.exists(plot_file)
        finally:
            if os.path.exists(parquet_path):
                os.remove(parquet_path)

    @pytest.mark.edge_case
    def test_plot_skips_ignored_columns_in_interactions(self, temp_dir):
        """Test that ignored columns are not used in interaction plots."""
        from src import brute_force_plotter
        
        # Create data with one column that will be ignored
        data = pd.DataFrame(
            {
                "many_cats": [f"cat_{i}" for i in range(100)],
                "value": list(range(100)),
                "category": ["A", "B"] * 50,
            }
        )
        
        dtypes = {
            "many_cats": "c",
            "value": "n",
            "category": "c",
        }
        
        brute_force_plotter.ignore.clear()
        
        plot(
            data,
            dtypes,
            output_path=temp_dir,
            show=False,
            use_dask=False,
        )
        
        # many_cats should be in ignore set
        assert "many_cats" in brute_force_plotter.ignore
        
        # No interaction plots with many_cats should exist
        interactions_dir = os.path.join(temp_dir, "2d_interactions")
        if os.path.exists(interactions_dir):
            files = os.listdir(interactions_dir)
            many_cats_files = [f for f in files if "many_cats" in f]
            assert len(many_cats_files) == 0


class TestAllMissingValues:
    """Tests for columns with all missing values."""

    @pytest.mark.edge_case
    def test_plot_column_with_all_missing_values(self, temp_dir):
        """Test plotting a numeric column with all missing values."""
        import tempfile as tmp
        from src import brute_force_plotter
        
        brute_force_plotter._save_plots = True
        brute_force_plotter._show_plots = False
        
        data = pd.DataFrame(
            {
                "all_missing": [None] * 10,
                "valid_col": list(range(10)),
            }
        )
        
        dtypes = {
            "all_missing": "n",
            "valid_col": "n",
        }
        
        # Should not crash
        try:
            output_path = plot(
                data,
                dtypes,
                output_path=temp_dir,
                show=False,
                use_dask=False,
            )
            assert output_path == temp_dir
        except Exception as e:
            pytest.fail(f"Plotting column with all missing values should not crash: {e}")


class TestVariousDataTypeCombinations:
    """Tests for various combinations of data types."""

    @pytest.mark.edge_case
    def test_plot_only_numeric_columns(self, sample_numeric_data, simple_dtypes, temp_dir):
        """Test plotting with only numeric columns."""
        output_path = plot(
            sample_numeric_data,
            simple_dtypes,
            output_path=temp_dir,
            show=False,
            use_dask=False,
        )
        
        assert output_path == temp_dir
        
        # Check that numeric plots were created
        dist_dir = os.path.join(temp_dir, "distributions")
        assert os.path.exists(os.path.join(dist_dir, "col1-dist-plot.png"))
        
        # Check that numeric-numeric scatter plots were created
        interactions_dir = os.path.join(temp_dir, "2d_interactions")
        assert os.path.exists(os.path.join(interactions_dir, "col1-col2-scatter-plot.png"))

    @pytest.mark.edge_case
    def test_plot_only_categorical_columns(self, sample_categorical_data, temp_dir):
        """Test plotting with only categorical columns."""
        dtypes = {
            "category1": "c",
            "category2": "c",
            "category3": "c",
        }
        
        output_path = plot(
            sample_categorical_data,
            dtypes,
            output_path=temp_dir,
            show=False,
            use_dask=False,
        )
        
        assert output_path == temp_dir
        
        # Check that categorical plots were created
        dist_dir = os.path.join(temp_dir, "distributions")
        assert os.path.exists(os.path.join(dist_dir, "category1-bar-plot.png"))
        
        # Check that category-category plots were created
        interactions_dir = os.path.join(temp_dir, "2d_interactions")
        assert os.path.exists(os.path.join(interactions_dir, "category1-category2-bar-plot.png"))

    @pytest.mark.edge_case
    def test_plot_single_numeric_column(self, temp_dir):
        """Test plotting with just one numeric column."""
        data = pd.DataFrame({"single_col": [1, 2, 3, 4, 5]})
        dtypes = {"single_col": "n"}
        
        output_path = plot(
            data,
            dtypes,
            output_path=temp_dir,
            show=False,
            use_dask=False,
        )
        
        assert output_path == temp_dir
        
        # Distribution plot should exist
        dist_dir = os.path.join(temp_dir, "distributions")
        assert os.path.exists(os.path.join(dist_dir, "single_col-dist-plot.png"))
        
        # No correlation matrix (need at least 2 numeric columns)
        assert not os.path.exists(os.path.join(dist_dir, "correlation-pearson.png"))

    @pytest.mark.edge_case
    def test_plot_single_categorical_column(self, temp_dir):
        """Test plotting with just one categorical column."""
        data = pd.DataFrame({"single_cat": ["A", "B", "C", "A", "B"]})
        dtypes = {"single_cat": "c"}
        
        output_path = plot(
            data,
            dtypes,
            output_path=temp_dir,
            show=False,
            use_dask=False,
        )
        
        assert output_path == temp_dir
        
        # Bar plot should exist
        dist_dir = os.path.join(temp_dir, "distributions")
        assert os.path.exists(os.path.join(dist_dir, "single_cat-bar-plot.png"))

    @pytest.mark.edge_case
    def test_plot_with_mixed_missing_patterns(self, temp_dir):
        """Test plotting with various missing value patterns."""
        data = pd.DataFrame(
            {
                "complete": [1, 2, 3, 4, 5],
                "some_missing": [1, None, 3, None, 5],
                "all_missing": [None, None, None, None, None],
                "category": ["A", "B", None, "A", "B"],
            }
        )
        
        dtypes = {
            "complete": "n",
            "some_missing": "n",
            "all_missing": "n",
            "category": "c",
        }
        
        # Should not crash
        try:
            output_path = plot(
                data,
                dtypes,
                output_path=temp_dir,
                show=False,
                use_dask=False,
            )
            assert output_path == temp_dir
        except Exception as e:
            pytest.fail(f"Plotting with mixed missing patterns should not crash: {e}")

    @pytest.mark.edge_case
    def test_plot_with_constant_column(self, temp_dir):
        """Test plotting with a column that has only one unique value."""
        data = pd.DataFrame(
            {
                "constant_numeric": [5] * 10,
                "constant_category": ["A"] * 10,
                "variable": list(range(10)),
            }
        )
        
        dtypes = {
            "constant_numeric": "n",
            "constant_category": "c",
            "variable": "n",
        }
        
        # Should not crash
        try:
            output_path = plot(
                data,
                dtypes,
                output_path=temp_dir,
                show=False,
                use_dask=False,
            )
            assert output_path == temp_dir
        except Exception as e:
            pytest.fail(f"Plotting with constant columns should not crash: {e}")

    @pytest.mark.edge_case
    def test_plot_with_very_small_dataset(self, temp_dir):
        """Test plotting with a very small dataset (2 rows)."""
        data = pd.DataFrame(
            {
                "num1": [1, 2],
                "num2": [10, 20],
                "cat": ["A", "B"],
            }
        )
        
        dtypes = {
            "num1": "n",
            "num2": "n",
            "cat": "c",
        }
        
        # Should not crash even with very little data
        try:
            output_path = plot(
                data,
                dtypes,
                output_path=temp_dir,
                show=False,
                use_dask=False,
            )
            assert output_path == temp_dir
        except Exception as e:
            pytest.fail(f"Plotting with very small dataset should not crash: {e}")

    @pytest.mark.edge_case
    def test_plot_with_unicode_column_names(self, temp_dir):
        """Test plotting with Unicode characters in column names."""
        import warnings
        
        data = pd.DataFrame(
            {
                "数値": [1, 2, 3, 4, 5],
                "カテゴリ": ["A", "B", "C", "A", "B"],
            }
        )
        
        dtypes = {
            "数値": "n",
            "カテゴリ": "c",
        }
        
        # Should handle Unicode column names
        # May raise UserWarnings about missing glyphs, but should not crash
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, message=".*Glyph.*missing from font.*")
                output_path = plot(
                    data,
                    dtypes,
                    output_path=temp_dir,
                    show=False,
                    use_dask=False,
                )
                assert output_path == temp_dir
        except UserWarning:
            # Font rendering warnings are acceptable
            pass
        except Exception as e:
            # Other exceptions are not acceptable
            pytest.fail(f"Plotting with Unicode column names should not crash with non-font errors: {e}")
