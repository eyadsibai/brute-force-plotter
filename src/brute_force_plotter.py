#!/usr/bin/env python

"""
Brute Force Plotter
-----------------
Library and Command Line Interface

This is the main entry point that provides backward compatibility
while using the new modular structure.
"""

# Re-export the public API from the new modules
from .cli.commands import main
from .cli.orchestration import create_plots
from .core import config
from .core.config import (
    ignore,
    skip_existing_plots,
)
from .core.data_types import infer_dtypes
from .core.utils import (
    autolabel,
    check_and_sample_large_dataset,
    ignore_if_exist_or_save,
    make_sure_path_exists,
)
from .library import plot
from .plotting.base import (
    bar_box_violin_dot_plots,
    bar_plot,
    box_violin_plots,
    correlation_heatmap,
    heatmap,
    histogram_violin_plots,
    missing_plot,
    scatter_plot,
)
from .plotting.maps import (
    create_map_visualization,
    detect_geocoordinate_pairs,
    plot_map_visualization,
    plot_map_visualization_sync,
)
from .plotting.single_variable import (
    plot_single_category,
    plot_single_category_sync,
    plot_single_numeric,
    plot_single_numeric_sync,
)
from .plotting.summary import (
    plot_correlation_matrix,
    plot_correlation_matrix_minimal,
    plot_missing_values,
)
from .plotting.three_variable import (
    contour_plot,
    grouped_bar_violin_plot,
    multi_level_heatmap,
    plot_category_category_category,
    plot_category_category_category_sync,
    plot_numeric_category_category,
    plot_numeric_category_category_sync,
    plot_numeric_numeric_category,
    plot_numeric_numeric_category_sync,
    plot_numeric_numeric_numeric,
    plot_numeric_numeric_numeric_sync,
    scatter_plot_3d,
    scatter_plot_with_hue,
)
from .plotting.timeseries import (
    multiple_time_series_plot,
    plot_single_timeseries,
    plot_single_timeseries_sync,
    plot_timeseries_category_numeric,
    plot_timeseries_category_numeric_sync,
    plot_timeseries_numeric,
    plot_timeseries_numeric_sync,
    plot_timeseries_timeseries,
    plot_timeseries_timeseries_sync,
    time_series_category_plot,
    time_series_line_plot,
    time_series_numeric_plot,
)
from .plotting.two_variable import (
    plot_category_category,
    plot_category_category_minimal,
    plot_category_category_minimal_sync,
    plot_category_category_sync,
    plot_category_numeric,
    plot_category_numeric_minimal,
    plot_category_numeric_minimal_sync,
    plot_category_numeric_sync,
    plot_numeric_numeric,
    plot_numeric_numeric_sync,
)
from .stats.export import export_statistical_summaries

__all__ = [
    "plot",
    "infer_dtypes",
    "ignore",
    "skip_existing_plots",
    "main",
    "create_plots",
    "check_and_sample_large_dataset",
    "ignore_if_exist_or_save",
    "make_sure_path_exists",
    "autolabel",
    "export_statistical_summaries",
    # Base plotting
    "histogram_violin_plots",
    "bar_plot",
    "scatter_plot",
    "bar_box_violin_dot_plots",
    "box_violin_plots",
    "heatmap",
    "correlation_heatmap",
    "missing_plot",
    # Single variable
    "plot_single_numeric",
    "plot_single_numeric_sync",
    "plot_single_category",
    "plot_single_category_sync",
    # Two variable
    "plot_numeric_numeric",
    "plot_numeric_numeric_sync",
    "plot_category_category",
    "plot_category_category_sync",
    "plot_category_category_minimal",
    "plot_category_category_minimal_sync",
    "plot_category_numeric",
    "plot_category_numeric_sync",
    "plot_category_numeric_minimal",
    "plot_category_numeric_minimal_sync",
    # Three variable
    "scatter_plot_with_hue",
    "scatter_plot_3d",
    "contour_plot",
    "multi_level_heatmap",
    "grouped_bar_violin_plot",
    "plot_numeric_numeric_category",
    "plot_numeric_numeric_category_sync",
    "plot_numeric_numeric_numeric",
    "plot_numeric_numeric_numeric_sync",
    "plot_category_category_category",
    "plot_category_category_category_sync",
    "plot_numeric_category_category",
    "plot_numeric_category_category_sync",
    # Summary
    "plot_correlation_matrix",
    "plot_correlation_matrix_minimal",
    "plot_missing_values",
    # Timeseries
    "time_series_line_plot",
    "time_series_numeric_plot",
    "time_series_category_plot",
    "multiple_time_series_plot",
    "plot_single_timeseries",
    "plot_single_timeseries_sync",
    "plot_timeseries_numeric",
    "plot_timeseries_numeric_sync",
    "plot_timeseries_timeseries",
    "plot_timeseries_timeseries_sync",
    "plot_timeseries_category_numeric",
    "plot_timeseries_category_numeric_sync",
    # Maps
    "detect_geocoordinate_pairs",
    "create_map_visualization",
    "plot_map_visualization",
    "plot_map_visualization_sync",
]

# Backward compatibility aliases
_detect_geocoordinate_pairs = detect_geocoordinate_pairs


# Backward compatibility: expose config module attributes
_show_plots = config._show_plots
_save_plots = config._save_plots
