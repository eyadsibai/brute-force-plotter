"""
Tests for target variable support.
"""

import os

import pandas as pd
import pytest

from src.brute_force_plotter import plot
from src.core import config


class TestTargetVariableSupport:
    """Tests for target variable highlighting in plots."""

    @pytest.fixture
    def classification_data(self):
        """Create sample classification dataset."""
        return pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "feature2": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
                "category": ["A", "B", "A", "B", "A", "B", "A", "B"],
                "target": ["yes", "no", "yes", "no", "yes", "no", "yes", "no"],
            }
        )

    @pytest.fixture
    def regression_data(self):
        """Create sample regression dataset."""
        return pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "feature2": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
                "category": ["A", "B", "A", "B", "A", "B", "A", "B"],
                "target": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            }
        )

    @pytest.fixture
    def dtypes_classification(self):
        """Data types for classification data."""
        return {
            "feature1": "n",
            "feature2": "n",
            "category": "c",
            "target": "c",
        }

    @pytest.fixture
    def dtypes_regression(self):
        """Data types for regression data."""
        return {
            "feature1": "n",
            "feature2": "n",
            "category": "c",
            "target": "n",
        }

    @pytest.mark.integration
    def test_plot_with_target_variable(
        self, classification_data, dtypes_classification, temp_dir
    ):
        """Test that plots are created with target variable highlighting."""
        output_path, _ = plot(
            classification_data,
            dtypes_classification,
            output_path=temp_dir,
            show=False,
            use_dask=False,
            target="target",
        )

        # Check that plots were created
        assert os.path.exists(os.path.join(temp_dir, "distributions"))
        assert os.path.exists(os.path.join(temp_dir, "2d_interactions"))

        # Check for distribution plots
        dist_dir = os.path.join(temp_dir, "distributions")
        assert os.path.exists(os.path.join(dist_dir, "feature1-dist-plot.png"))
        assert os.path.exists(os.path.join(dist_dir, "feature2-dist-plot.png"))

        # Check for interaction plots with target highlighting
        int_dir = os.path.join(temp_dir, "2d_interactions")
        assert os.path.exists(
            os.path.join(int_dir, "feature1-feature2-scatter-plot.png")
        )

    @pytest.mark.integration
    def test_plot_without_target_variable(
        self, classification_data, dtypes_classification, temp_dir
    ):
        """Test that plots work without target variable (backward compatibility)."""
        output_path, _ = plot(
            classification_data,
            dtypes_classification,
            output_path=temp_dir,
            show=False,
            use_dask=False,
        )

        # Should still create plots
        assert os.path.exists(os.path.join(temp_dir, "distributions"))
        assert os.path.exists(os.path.join(temp_dir, "2d_interactions"))

    @pytest.mark.integration
    def test_invalid_target_variable(
        self, classification_data, dtypes_classification, temp_dir
    ):
        """Test handling of invalid target variable."""
        output_path, _ = plot(
            classification_data,
            dtypes_classification,
            output_path=temp_dir,
            show=False,
            use_dask=False,
            target="nonexistent",
        )

        # Should still create plots with warning
        assert os.path.exists(os.path.join(temp_dir, "distributions"))

    @pytest.mark.integration
    def test_target_variable_config_set(self, classification_data, dtypes_classification):
        """Test that target variable is properly set in config."""
        # Reset config
        config.set_target_variable(None)
        assert config.get_target_variable() is None

        # Plot with target
        plot(
            classification_data,
            dtypes_classification,
            show=False,
            use_dask=False,
            target="target",
        )

        # Config should be set
        assert config.get_target_variable() == "target"

    @pytest.mark.integration
    def test_regression_target_variable(
        self, regression_data, dtypes_regression, temp_dir
    ):
        """Test target variable with numeric target (regression)."""
        output_path, _ = plot(
            regression_data,
            dtypes_regression,
            output_path=temp_dir,
            show=False,
            use_dask=False,
            target="target",
        )

        # Check that plots were created
        assert os.path.exists(os.path.join(temp_dir, "distributions"))
        assert os.path.exists(os.path.join(temp_dir, "2d_interactions"))

        # Check for scatter plot with target coloring
        int_dir = os.path.join(temp_dir, "2d_interactions")
        assert os.path.exists(
            os.path.join(int_dir, "feature1-feature2-scatter-plot.png")
        )
