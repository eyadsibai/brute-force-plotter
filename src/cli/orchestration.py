"""
Plot orchestration module.

This module handles the main logic for creating all plots based on data types.
"""

from itertools import combinations
import logging
import os

from ..core.config import ignore
from ..core.utils import make_sure_path_exists
from ..plotting.maps import (
    detect_geocoordinate_pairs,
    plot_map_visualization,
    plot_map_visualization_sync,
)
from ..plotting.single_variable import (
    plot_single_category,
    plot_single_category_sync,
    plot_single_numeric,
    plot_single_numeric_sync,
)
from ..plotting.summary import (
    plot_correlation_matrix,
    plot_correlation_matrix_minimal,
    plot_missing_values,
)
from ..plotting.three_variable import (
    plot_category_category_category,
    plot_category_category_category_sync,
    plot_numeric_category_category,
    plot_numeric_category_category_sync,
    plot_numeric_numeric_category,
    plot_numeric_numeric_category_sync,
    plot_numeric_numeric_numeric,
    plot_numeric_numeric_numeric_sync,
)
from ..plotting.timeseries import (
    plot_single_timeseries,
    plot_single_timeseries_sync,
    plot_timeseries_category_numeric,
    plot_timeseries_category_numeric_sync,
    plot_timeseries_numeric,
    plot_timeseries_numeric_sync,
    plot_timeseries_timeseries,
    plot_timeseries_timeseries_sync,
)
from ..plotting.two_variable import (
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

logger = logging.getLogger(__name__)

__all__ = ["create_plots"]


def _create_directories(output_path):
    distribution_path = os.path.join(output_path, "distributions")
    two_d_interaction_path = os.path.join(output_path, "2d_interactions")
    three_d_interaction_path = os.path.join(output_path, "3d_interactions")

    make_sure_path_exists(distribution_path)
    make_sure_path_exists(two_d_interaction_path)
    make_sure_path_exists(three_d_interaction_path)
    return distribution_path, two_d_interaction_path, three_d_interaction_path


def create_plots(input_file, dtypes, output_path, use_dask=True, minimal=False):
    distributions_path, two_d_interactions_path, three_d_interactions_path = (
        _create_directories(output_path)
    )
    plots = []

    # Add summary plots
    logger.info("Adding correlation matrix and missing values plots...")
    if minimal:
        plots.append(
            plot_correlation_matrix_minimal(input_file, dtypes, distributions_path)
        )
    else:
        plots.append(plot_correlation_matrix(input_file, dtypes, distributions_path))
    plots.append(plot_missing_values(input_file, dtypes, distributions_path))

    for col, dtype in dtypes.items():
        print(col)
        if dtype == "i":
            continue
        if dtype == "n":
            if use_dask:
                plots.append(plot_single_numeric(input_file, col, distributions_path))
            else:
                plot_single_numeric_sync(input_file, col, distributions_path)
        if dtype == "c":
            if use_dask:
                plots.append(plot_single_category(input_file, col, distributions_path))
            else:
                plot_single_category_sync(input_file, col, distributions_path)
        if dtype == "t":
            if use_dask:
                plots.append(
                    plot_single_timeseries(input_file, col, distributions_path)
                )
            else:
                plot_single_timeseries_sync(input_file, col, distributions_path)

    for (col1, dtype1), (col2, dtype2) in combinations(dtypes.items(), 2):
        print(col1, col2)
        if dtype1 == "i" or dtype2 == "i":
            continue
        if any(col in ignore for col in [col1, col2]):
            continue
        if dtype1 == dtype2 == "n":
            if use_dask:
                plots.append(
                    plot_numeric_numeric(
                        input_file, col1, col2, two_d_interactions_path
                    )
                )
            else:
                plot_numeric_numeric_sync(
                    input_file, col1, col2, two_d_interactions_path
                )
        if dtype1 == dtype2 == "c":
            if minimal:
                # Minimal mode: only heatmap
                if use_dask:
                    plots.append(
                        plot_category_category_minimal(
                            input_file, col1, col2, two_d_interactions_path
                        )
                    )
                else:
                    plot_category_category_minimal_sync(
                        input_file, col1, col2, two_d_interactions_path
                    )
            else:
                # Full mode: bar plot + heatmap
                if use_dask:
                    plots.append(
                        plot_category_category(
                            input_file, col1, col2, two_d_interactions_path
                        )
                    )
                else:
                    plot_category_category_sync(
                        input_file, col1, col2, two_d_interactions_path
                    )
        if dtype1 == "c" and dtype2 == "n":
            if minimal:
                # Minimal mode: only box + violin
                if use_dask:
                    plots.append(
                        plot_category_numeric_minimal(
                            input_file, col1, col2, two_d_interactions_path
                        )
                    )
                else:
                    plot_category_numeric_minimal_sync(
                        input_file, col1, col2, two_d_interactions_path
                    )
            else:
                # Full mode: all 4 plots (bar, strip, box, violin)
                if use_dask:
                    plots.append(
                        plot_category_numeric(
                            input_file, col1, col2, two_d_interactions_path
                        )
                    )
                else:
                    plot_category_numeric_sync(
                        input_file, col1, col2, two_d_interactions_path
                    )
        if dtype1 == "n" and dtype2 == "c":
            if minimal:
                # Minimal mode: only box + violin
                if use_dask:
                    plots.append(
                        plot_category_numeric_minimal(
                            input_file, col2, col1, two_d_interactions_path
                        )
                    )
                else:
                    plot_category_numeric_minimal_sync(
                        input_file, col2, col1, two_d_interactions_path
                    )
            else:
                # Full mode: all 4 plots (bar, strip, box, violin)
                if use_dask:
                    plots.append(
                        plot_category_numeric(
                            input_file, col2, col1, two_d_interactions_path
                        )
                    )
                else:
                    plot_category_numeric_sync(
                        input_file, col2, col1, two_d_interactions_path
                    )
                plot_category_numeric_sync(
                    input_file, col2, col1, two_d_interactions_path
                )
        # Time series interactions
        if dtype1 == "t" and dtype2 == "n":
            if use_dask:
                plots.append(
                    plot_timeseries_numeric(
                        input_file, col1, col2, two_d_interactions_path
                    )
                )
            else:
                plot_timeseries_numeric_sync(
                    input_file, col1, col2, two_d_interactions_path
                )
        if dtype1 == "n" and dtype2 == "t":
            if use_dask:
                plots.append(
                    plot_timeseries_numeric(
                        input_file, col2, col1, two_d_interactions_path
                    )
                )
            else:
                plot_timeseries_numeric_sync(
                    input_file, col2, col1, two_d_interactions_path
                )
        if dtype1 == dtype2 == "t":
            if use_dask:
                plots.append(
                    plot_timeseries_timeseries(
                        input_file, col1, col2, two_d_interactions_path
                    )
                )
            else:
                plot_timeseries_timeseries_sync(
                    input_file, col1, col2, two_d_interactions_path
                )

    # 3-variable interactions
    logger.info("Adding 3-variable interaction plots...")
    # 3-variable interactions
    logger.info("Adding 3-variable interaction plots...")
    for (col1, dtype1), (col2, dtype2), (col3, dtype3) in combinations(
        dtypes.items(), 3
    ):
        if dtype1 == "i" or dtype2 == "i" or dtype3 == "i":
            continue
        if any(col in ignore for col in [col1, col2, col3]):
            continue

        # All numeric: 3D scatter and contour plots
        if dtype1 == "n" and dtype2 == "n" and dtype3 == "n":
            if use_dask:
                plots.append(
                    plot_numeric_numeric_numeric(
                        input_file, col1, col2, col3, three_d_interactions_path
                    )
                )
            else:
                plot_numeric_numeric_numeric_sync(
                    input_file, col1, col2, col3, three_d_interactions_path
                )

        # All categorical: multi-level heatmap
        elif dtype1 == "c" and dtype2 == "c" and dtype3 == "c":
            if use_dask:
                plots.append(
                    plot_category_category_category(
                        input_file, col1, col2, col3, three_d_interactions_path
                    )
                )
            else:
                plot_category_category_category_sync(
                    input_file, col1, col2, col3, three_d_interactions_path
                )

        # Two numeric, one categorical: colored scatter
        elif dtype1 == "n" and dtype2 == "n" and dtype3 == "c":
            if use_dask:
                plots.append(
                    plot_numeric_numeric_category(
                        input_file, col1, col2, col3, three_d_interactions_path
                    )
                )
            else:
                plot_numeric_numeric_category_sync(
                    input_file, col1, col2, col3, three_d_interactions_path
                )
        elif dtype1 == "n" and dtype2 == "c" and dtype3 == "n":
            if use_dask:
                plots.append(
                    plot_numeric_numeric_category(
                        input_file, col1, col3, col2, three_d_interactions_path
                    )
                )
            else:
                plot_numeric_numeric_category_sync(
                    input_file, col1, col3, col2, three_d_interactions_path
                )
        elif dtype1 == "c" and dtype2 == "n" and dtype3 == "n":
            if use_dask:
                plots.append(
                    plot_numeric_numeric_category(
                        input_file, col2, col3, col1, three_d_interactions_path
                    )
                )
            else:
                plot_numeric_numeric_category_sync(
                    input_file, col2, col3, col1, three_d_interactions_path
                )

        # One numeric, two categorical: grouped visualizations
        elif dtype1 == "n" and dtype2 == "c" and dtype3 == "c":
            if use_dask:
                plots.append(
                    plot_numeric_category_category(
                        input_file, col1, col2, col3, three_d_interactions_path
                    )
                )
            else:
                plot_numeric_category_category_sync(
                    input_file, col1, col2, col3, three_d_interactions_path
                )
        elif dtype1 == "c" and dtype2 == "n" and dtype3 == "c":
            if use_dask:
                plots.append(
                    plot_numeric_category_category(
                        input_file, col2, col1, col3, three_d_interactions_path
                    )
                )
            else:
                plot_numeric_category_category_sync(
                    input_file, col2, col1, col3, three_d_interactions_path
                )
        elif dtype1 == "c" and dtype2 == "c" and dtype3 == "n":
            if use_dask:
                plots.append(
                    plot_numeric_category_category(
                        input_file, col3, col1, col2, three_d_interactions_path
                    )
                )
            else:
                plot_numeric_category_category_sync(
                    input_file, col3, col1, col2, three_d_interactions_path
                )

    # 3-way interactions: time series + category + numeric
    for (col1, dtype1), (col2, dtype2), (col3, dtype3) in combinations(
        dtypes.items(), 3
    ):
        if dtype1 == "i" or dtype2 == "i" or dtype3 == "i":
            continue
        if any(col in ignore for col in [col1, col2, col3]):
            continue

        # Find time, category, and numeric columns
        time_col = None
        category_col = None
        numeric_col = None

        if dtype1 == "t":
            time_col = col1
        elif dtype2 == "t":
            time_col = col2
        elif dtype3 == "t":
            time_col = col3

        if dtype1 == "c":
            category_col = col1
        elif dtype2 == "c":
            category_col = col2
        elif dtype3 == "c":
            category_col = col3

        if dtype1 == "n":
            numeric_col = col1
        elif dtype2 == "n":
            numeric_col = col2
        elif dtype3 == "n":
            numeric_col = col3

        # Plot if we have time + category + numeric
        if time_col and category_col and numeric_col:
            if use_dask:
                plots.append(
                    plot_timeseries_category_numeric(
                        input_file,
                        time_col,
                        category_col,
                        numeric_col,
                        two_d_interactions_path,
                    )
                )
            else:
                plot_timeseries_category_numeric_sync(
                    input_file,
                    time_col,
                    category_col,
                    numeric_col,
                    two_d_interactions_path,
                )

    # Generate map visualizations for geocoordinate pairs
    logger.info("Checking for geocoordinate pairs...")
    geo_pairs = detect_geocoordinate_pairs(dtypes)

    if geo_pairs:
        logger.info(
            f"Generating map visualizations for {len(geo_pairs)} geocoordinate pairs..."
        )

        # Create maps directory
        maps_path = os.path.join(output_path, "maps")
        make_sure_path_exists(maps_path)

        for lat_col, lon_col in geo_pairs:
            # Create a simple map without categories
            if use_dask:
                plots.append(
                    plot_map_visualization(input_file, lat_col, lon_col, maps_path)
                )
            else:
                plot_map_visualization_sync(input_file, lat_col, lon_col, maps_path)

            # Create maps with categorical overlays
            category_cols = [col for col, dtype in dtypes.items() if dtype == "c"]
            for cat_col in category_cols:
                if cat_col not in ignore:
                    if use_dask:
                        plots.append(
                            plot_map_visualization(
                                input_file, lat_col, lon_col, maps_path, cat_col
                            )
                        )
                    else:
                        plot_map_visualization_sync(
                            input_file, lat_col, lon_col, maps_path, cat_col
                        )

            # for (col1, dtype1), (col2, dtype2), (col3, dtype3) in combinations(
            # dtypes.items(), 3):
            #     print(col1, col2, col3)
            #     dtypes_array = [dtype1, dtype2, dtype3]
            #     all_categories = all(dtype == 'c' for dtype in dtypes_array)
            #     all_numeric = all(dtype == 'n' for dtype in dtypes_array)
            #
            #     if any(col in ignore for col in [col1, col2, col3]):
            #         continue
            #     if all_categories:
            #         plot_categorical_categorical_categorical(three_d_interactions_path)
            #     if all_numeric:
            #         plot_numeric_numeric_numeric(three_d_interactions_path)
            # if dtype1 == 'c' and dtype2 == 'n' and dtype3 == 'n':
            #     plot_numeric_numeric_category(df, col2, col3, col1,
            #                                   three_d_interactions_path)
            #
            # if dtype1 == 'c' and dtype2 == 'c' and dtype3 == 'n':
            #     plot_numeric_category_category(df, col3, col1, col3,
            #                                    three_d_interactions_path)
            # if dtype1 == 'c' and dtype2 == 'n' and dtype3 == 'c':
            #     plot_numeric_category_category(df, col2, col1, col3,
            #                                    three_d_interactions_path)
            # if dtype1 == 'n' and dtype2 == 'n' and dtype3 == 'c':
            #     plot_numeric_numeric_category(df, col1, col2, col3,
            #                                   three_d_interactions_path)
            # if dtype1 == 'n' and dtype2 == 'c' and dtype3 == 'c':
            #     plot_numeric_category_category(df, col1, col2, col3,
            #                                    three_d_interactions_path)
            # if dtype1 == 'n' and dtype2 == 'c' and dtype3 == 'n':
            #     plot_numeric_numeric_category(df, col1, col3, col2,
            #                                   three_d_interactions_path)
    return plots
