"""
Unit tests for utility functions in brute_force_plotter.
"""

import os
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import pytest

from src.brute_force_plotter import (
    ignore_if_exist_or_save,
    make_sure_path_exists,
)


class TestMakeSurePathExists:
    """Tests for the make_sure_path_exists function."""

    @pytest.mark.unit
    def test_creates_new_directory(self, temp_dir):
        """Test that a new directory is created successfully."""
        new_path = os.path.join(temp_dir, "new_directory")
        assert not os.path.exists(new_path)

        result = make_sure_path_exists(new_path)

        assert result is True
        assert os.path.exists(new_path)
        assert os.path.isdir(new_path)

    @pytest.mark.unit
    def test_handles_existing_directory(self, temp_dir):
        """Test that existing directory doesn't cause errors."""
        # Create the directory first
        os.makedirs(temp_dir, exist_ok=True)

        result = make_sure_path_exists(temp_dir)

        assert result is True
        assert os.path.exists(temp_dir)

    @pytest.mark.unit
    def test_creates_nested_directories(self, temp_dir):
        """Test that nested directories are created."""
        nested_path = os.path.join(temp_dir, "level1", "level2", "level3")
        assert not os.path.exists(nested_path)

        result = make_sure_path_exists(nested_path)

        assert result is True
        assert os.path.exists(nested_path)
        assert os.path.isdir(nested_path)

    @pytest.mark.unit
    def test_handles_permission_error(self, temp_dir):
        """Test handling of permission errors."""
        # Mock os.makedirs to raise a permission error
        with patch("src.brute_force_plotter.os.makedirs") as mock_makedirs:
            mock_makedirs.side_effect = OSError(13, "Permission denied")

            result = make_sure_path_exists("/invalid/path")

            assert result is False


class TestIgnoreIfExistOrSaveDecorator:
    """Tests for the ignore_if_exist_or_save decorator."""

    @pytest.mark.unit
    def test_decorator_saves_plot_when_file_doesnt_exist(self, temp_dir):
        """Test that plot is saved when file doesn't exist."""
        from src import brute_force_plotter

        # Set up to save plots
        brute_force_plotter._show_plots = False
        brute_force_plotter._save_plots = True

        file_path = os.path.join(temp_dir, "test_plot.png")

        @ignore_if_exist_or_save
        def create_test_plot(file_name=None):
            plt.figure()
            plt.plot([1, 2, 3], [1, 2, 3])

        create_test_plot(file_name=file_path)

        assert os.path.exists(file_path)

    @pytest.mark.unit
    def test_decorator_skips_when_file_exists(self, temp_dir):
        """Test that plot creation is skipped when file exists."""
        from src import brute_force_plotter

        brute_force_plotter._show_plots = False
        brute_force_plotter._save_plots = True

        file_path = os.path.join(temp_dir, "test_plot.png")

        # Create the file first
        Path(file_path).touch()
        initial_mtime = os.path.getmtime(file_path)

        call_count = 0

        @ignore_if_exist_or_save
        def create_test_plot(file_name=None):
            nonlocal call_count
            call_count += 1
            plt.figure()
            plt.plot([1, 2, 3], [1, 2, 3])

        create_test_plot(file_name=file_path)

        # Function should still be called but shouldn't modify the file
        assert os.path.exists(file_path)
        assert os.path.getmtime(file_path) == initial_mtime

    @pytest.mark.unit
    def test_decorator_shows_plot_when_configured(self, temp_dir):
        """Test that plot is shown when show_plots is True."""
        from src import brute_force_plotter

        brute_force_plotter._show_plots = True
        brute_force_plotter._save_plots = False

        with patch("matplotlib.pyplot.show") as mock_show:

            @ignore_if_exist_or_save
            def create_test_plot(file_name=None):
                plt.figure()
                plt.plot([1, 2, 3], [1, 2, 3])

            create_test_plot(file_name=None)

            # Check that show was called
            mock_show.assert_called_once()

    @pytest.mark.unit
    def test_decorator_saves_and_shows_when_both_configured(self, temp_dir):
        """Test that plot is both saved and shown when both are configured."""
        from src import brute_force_plotter

        brute_force_plotter._show_plots = True
        brute_force_plotter._save_plots = True

        file_path = os.path.join(temp_dir, "test_plot.png")

        with patch("matplotlib.pyplot.show") as mock_show:

            @ignore_if_exist_or_save
            def create_test_plot(file_name=None):
                plt.figure()
                plt.plot([1, 2, 3], [1, 2, 3])

            create_test_plot(file_name=file_path)

            # Check that both save and show happened
            assert os.path.exists(file_path)
            mock_show.assert_called_once()

    @pytest.mark.unit
    def test_decorator_closes_figures(self, temp_dir):
        """Test that figures are properly closed after plotting."""
        from src import brute_force_plotter

        brute_force_plotter._show_plots = False
        brute_force_plotter._save_plots = True

        file_path = os.path.join(temp_dir, "test_plot.png")

        @ignore_if_exist_or_save
        def create_test_plot(file_name=None):
            plt.figure()
            plt.plot([1, 2, 3], [1, 2, 3])

        # Check that no figures are open before
        assert len(plt.get_fignums()) == 0

        create_test_plot(file_name=file_path)

        # Check that all figures are closed after
        assert len(plt.get_fignums()) == 0
