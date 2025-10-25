#!/usr/bin/env python

"""
Brute Force Plotter
-----------------
Library and Command Line Interface

"""

import errno
from itertools import chain, combinations
import json
import logging
import math
import os
import tempfile

import click
import dask
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


ignore = set()
skip_existing_plots = True  # Global flag for skipping existing plots

# Global configuration
_show_plots = False
_save_plots = True

sns.set_style("darkgrid")
sns.set_context("paper")

sns.set(rc={"figure.figsize": (8, 6)})


@click.command()
@click.argument("input_file")
@click.argument("dtypes")
@click.argument("output_path")
@click.option(
    "--skip-existing",
    is_flag=True,
    default=True,
    help="Skip generating plots that already exist",
)
@click.option(
    "--theme",
    type=click.Choice(["darkgrid", "whitegrid", "dark", "white", "ticks"]),
    default="darkgrid",
    help="Seaborn plot style theme",
)
@click.option(
    "--n-workers",
    type=int,
    default=4,
    help="Number of parallel workers for plot generation",
)
@click.option(
    "--export-stats",
    is_flag=True,
    default=False,
    help="Export statistical summary to CSV",
)
def main(
    input_file, dtypes, output_path, skip_existing, theme, n_workers, export_stats
):
    """Create Plots From data in input"""
    # Set matplotlib backend for CLI (non-interactive)
    matplotlib.use("agg")

    from dask.distributed import Client, LocalCluster

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Set global skip_existing flag
    global skip_existing_plots
    skip_existing_plots = skip_existing

    # Apply theme
    sns.set_style(theme)

    cluster = LocalCluster(n_workers=n_workers, silence_logs=logging.WARNING)
    _client = Client(cluster)  # noqa: F841 - Client instance needed to enable dask cluster

    # Load dtypes JSON first to know which columns to ignore
    with open(dtypes) as f:
        data_types = json.load(f)

    # Filter out columns with dtype "i" (ignore)
    columns_to_load = [col for col, dtype in data_types.items() if dtype != "i"]

    # Only load non-ignored columns from CSV
    data = pd.read_csv(input_file, usecols=columns_to_load)
    new_file_name = f"{input_file}.parq"
    data.to_parquet(new_file_name)

    plots = create_plots(new_file_name, data_types, output_path)
    dask.compute(*plots)

    # Export statistical summaries if requested
    if export_stats:
        export_statistical_summaries(new_file_name, data_types, output_path)


def plot(
    data,
    dtypes,
    output_path=None,
    show=False,
    use_dask=False,
    n_workers=4,
    export_stats=False,
):
    """
    Create plots from a pandas DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        The data to plot
    dtypes : dict
        Dictionary mapping column names to data types:
        - 'n' for numeric
        - 'c' for category
        - 'i' for ignore
    output_path : str, optional
        Path to save plots. If None and show=False, uses a temporary directory.
        Defaults to None.
    show : bool, optional
        If True, display plots interactively. If False, save to disk.
        Defaults to False.
    use_dask : bool, optional
        If True, use Dask for parallel processing. Defaults to False.
    n_workers : int, optional
        Number of workers for Dask (only used if use_dask=True). Defaults to 4.
    export_stats : bool, optional
        If True, export statistical summaries to CSV files. Defaults to False.

    Returns
    -------
    str
        Path where plots were saved (if saved)

    Examples
    --------
    >>> import pandas as pd
    >>> import brute_force_plotter as bfp
    >>>
    >>> data = pd.read_csv('data.csv')
    >>> dtypes = {'age': 'n', 'gender': 'c', 'id': 'i'}
    >>>
    >>> # Save plots to directory
    >>> bfp.plot(data, dtypes, output_path='./plots')
    >>>
    >>> # Show plots interactively
    >>> bfp.plot(data, dtypes, show=True)
    >>>
    >>> # Export statistical summaries
    >>> bfp.plot(data, dtypes, output_path='./plots', export_stats=True)
    """
    global _show_plots, _save_plots

    # Set matplotlib backend based on show parameter
    if show:
        try:
            matplotlib.use("TkAgg")
        except Exception:
            try:
                matplotlib.use("Qt5Agg")
            except Exception:
                logger.warning("Could not set interactive backend, using default")
    else:
        matplotlib.use("agg")

    _show_plots = show
    _save_plots = not show or output_path is not None

    # Determine output path
    if output_path is None and not show:
        output_path = tempfile.mkdtemp(prefix="brute_force_plotter_")
        logger.info(
            f"No output path specified, using temporary directory: {output_path}"
        )
    elif output_path is None and show:
        # Create a temp directory anyway for potential saving
        output_path = tempfile.mkdtemp(prefix="brute_force_plotter_")

    # Create temporary parquet file for efficient processing
    temp_parquet = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".parq", delete=False) as tmp:
            temp_parquet = tmp.name
        data.to_parquet(temp_parquet)

        if use_dask:
            from dask.distributed import Client, LocalCluster

            cluster = LocalCluster(n_workers=n_workers)
            client = Client(cluster)
            try:
                plots = create_plots(temp_parquet, dtypes, output_path)
                dask.compute(*plots)
            finally:
                client.close()
                cluster.close()
        else:
            plots = create_plots(temp_parquet, dtypes, output_path, use_dask=False)
            if plots:
                for plot_task in plots:
                    plot_task.compute()

        # Export statistical summaries if requested
        if export_stats:
            export_statistical_summaries(temp_parquet, dtypes, output_path)
    finally:
        # Clean up temporary parquet file
        if temp_parquet and os.path.exists(temp_parquet):
            os.remove(temp_parquet)

    return output_path


def ignore_if_exist_or_save(func):
    """Decorator to handle plot saving/showing logic"""

    def wrapper(*args, **kwargs):
        file_name = kwargs.get("file_name")

        # If saving is disabled and showing is enabled, just create and show
        if _show_plots and not _save_plots:
            func(*args, **kwargs)
            plt.gcf().set_tight_layout(True)
            plt.show()
            plt.close("all")
        # If file exists and we're saving, skip
        elif file_name and os.path.isfile(file_name) and _save_plots:
            plt.close("all")
        # Otherwise, create the plot
        else:
            func(*args, **kwargs)
            plt.gcf().set_tight_layout(True)

            # Save if we should save
            if _save_plots and file_name:
                plt.gcf().savefig(file_name, dpi=120)

            # Show if we should show
            if _show_plots:
                plt.show()

            plt.close("all")

    return wrapper


def make_sure_path_exists(path):
    logger.debug(f"Make sure {path} exists")
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            return False
    return True


@dask.delayed
def plot_single_numeric(input_file, col, path):
    df = pd.read_parquet(input_file, columns=[col])
    file_name = os.path.join(path, f"{col}-dist-plot.png")
    data = df[col].dropna()
    f, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    histogram_violin_plots(data, axes, file_name=file_name)


def plot_single_numeric_sync(input_file, col, path):
    """Non-delayed version for synchronous execution"""
    df = pd.read_parquet(input_file, columns=[col])
    file_name = os.path.join(path, f"{col}-dist-plot.png")
    data = df[col].dropna()
    f, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    histogram_violin_plots(data, axes, file_name=file_name)

    # TODO plot log transformation too?
    # file_path = path + col + '-log-dist-plot.png'
    # if not os.path.isfile(file_path):
    # if data.min() < 1:
    # tmp = data + data.min()
    # logged = tmp.map(np.arcsinh)
    # else:
    # logged = data.map(np.log)
    # sns.distplot(logged, axlabel='log distribution of {}'.format(logged.name))
    # plt.tight_layout()
    # plt.savefig(file_path, dpi=120)
    # plt.close()
    #
    # file_path = path + col + '-sqrroot-dist-plot.png'
    # if not os.path.isfile(file_path):
    # square_rooted = data.map(np.sqrt)
    # sns.distplot(square_rooted, axlabel='sqrrot distribution of {}'.format(square_rooted.name))
    # plt.tight_layout()
    # plt.savefig(file_path, dpi=120)
    # plt.close()


@dask.delayed
def plot_single_category(input_file, col, path):
    df = pd.read_parquet(input_file, columns=[col])
    value_counts = df[col].value_counts(dropna=False)
    # if the categories are more than 50 then this should be ignored
    # TODO find a better way to visualize this
    if len(value_counts) > 50:
        ignore.add(col)
    else:
        file_name = os.path.join(path, col + "-bar-plot.png")
        bar_plot(df, col, file_name=file_name)


def plot_single_category_sync(input_file, col, path):
    """Non-delayed version for synchronous execution"""
    df = pd.read_parquet(input_file, columns=[col])
    value_counts = df[col].value_counts(dropna=False)
    if len(value_counts) > 50:
        ignore.add(col)
    else:
        file_name = os.path.join(path, col + "-bar-plot.png")
        bar_plot(df, col, file_name=file_name)


@dask.delayed
def plot_category_category(input_file, col1, col2, path):
    df = pd.read_parquet(input_file, columns=[col1, col2])
    if len(df[col1].unique()) < len(df[col2].unique()):
        col1, col2 = col2, col1
    file_name = os.path.join(path, f"{col1}-{col2}-bar-plot.png")
    bar_plot(df, col1, hue=col2, file_name=file_name)

    file_name = os.path.join(path, f"{col1}-{col2}-heatmap.png")
    heatmap(pd.crosstab(df[col1], df[col2]), file_name=file_name)


def plot_category_category_sync(input_file, col1, col2, path):
    """Non-delayed version for synchronous execution"""
    df = pd.read_parquet(input_file, columns=[col1, col2])
    if len(df[col1].unique()) < len(df[col2].unique()):
        col1, col2 = col2, col1
    file_name = os.path.join(path, f"{col1}-{col2}-bar-plot.png")
    bar_plot(df, col1, hue=col2, file_name=file_name)

    file_name = os.path.join(path, f"{col1}-{col2}-heatmap.png")
    heatmap(pd.crosstab(df[col1], df[col2]), file_name=file_name)


@dask.delayed
def plot_numeric_numeric(input_file, col1, col2, path):
    df = pd.read_parquet(input_file, columns=[col1, col2])
    file_name = os.path.join(path, f"{col1}-{col2}-scatter-plot.png")
    scatter_plot(df, col1, col2, file_name=file_name)


def plot_numeric_numeric_sync(input_file, col1, col2, path):
    """Non-delayed version for synchronous execution"""
    df = pd.read_parquet(input_file, columns=[col1, col2])
    file_name = os.path.join(path, f"{col1}-{col2}-scatter-plot.png")
    scatter_plot(df, col1, col2, file_name=file_name)


@dask.delayed
def plot_category_numeric(input_file, category_col, numeric_col, path):
    df = pd.read_parquet(input_file, columns=[category_col, numeric_col])
    f, axes = plt.subplots(2, 2, sharex="col", sharey="row", figsize=(8, 6))
    axes = list(chain.from_iterable(axes))
    file_name = os.path.join(path, f"{category_col}-{numeric_col}-plot.png")
    bar_box_violin_dot_plots(df, category_col, numeric_col, axes, file_name=file_name)


def plot_category_numeric_sync(input_file, category_col, numeric_col, path):
    """Non-delayed version for synchronous execution"""
    df = pd.read_parquet(input_file, columns=[category_col, numeric_col])
    f, axes = plt.subplots(2, 2, sharex="col", sharey="row", figsize=(8, 6))
    axes = list(chain.from_iterable(axes))
    file_name = os.path.join(path, f"{category_col}-{numeric_col}-plot.png")
    bar_box_violin_dot_plots(df, category_col, numeric_col, axes, file_name=file_name)


# ============================================================================
# 3-Variable Plotting Functions
# ============================================================================


@dask.delayed
def plot_numeric_numeric_category(input_file, num_col1, num_col2, cat_col, path):
    """Create 3D scatter plot with category as hue (2D scatter colored by category)"""
    df = pd.read_parquet(input_file, columns=[num_col1, num_col2, cat_col])
    file_name = os.path.join(path, f"{num_col1}-{num_col2}-{cat_col}-scatter-3d.png")
    scatter_plot_with_hue(df, num_col1, num_col2, cat_col, file_name=file_name)


def plot_numeric_numeric_category_sync(input_file, num_col1, num_col2, cat_col, path):
    """Non-delayed version for synchronous execution"""
    df = pd.read_parquet(input_file, columns=[num_col1, num_col2, cat_col])
    file_name = os.path.join(path, f"{num_col1}-{num_col2}-{cat_col}-scatter-3d.png")
    scatter_plot_with_hue(df, num_col1, num_col2, cat_col, file_name=file_name)


@dask.delayed
def plot_numeric_numeric_numeric(input_file, col1, col2, col3, path):
    """Create 3D scatter plot and contour plots for 3 numeric variables"""
    df = pd.read_parquet(input_file, columns=[col1, col2, col3])

    # 3D scatter plot
    file_name = os.path.join(path, f"{col1}-{col2}-{col3}-3d-scatter.png")
    scatter_plot_3d(df, col1, col2, col3, file_name=file_name)

    # Contour plots for each pair with third variable as color
    file_name = os.path.join(path, f"{col1}-{col2}-contour-{col3}.png")
    contour_plot(df, col1, col2, col3, file_name=file_name)


def plot_numeric_numeric_numeric_sync(input_file, col1, col2, col3, path):
    """Non-delayed version for synchronous execution"""
    df = pd.read_parquet(input_file, columns=[col1, col2, col3])

    # 3D scatter plot
    file_name = os.path.join(path, f"{col1}-{col2}-{col3}-3d-scatter.png")
    scatter_plot_3d(df, col1, col2, col3, file_name=file_name)

    # Contour plots
    file_name = os.path.join(path, f"{col1}-{col2}-contour-{col3}.png")
    contour_plot(df, col1, col2, col3, file_name=file_name)


@dask.delayed
def plot_category_category_category(input_file, col1, col2, col3, path):
    """Create multi-level heatmap for 3 categorical variables"""
    df = pd.read_parquet(input_file, columns=[col1, col2, col3])

    # Create heatmaps for each level of the third category
    file_name = os.path.join(path, f"{col1}-{col2}-by-{col3}-heatmap.png")
    multi_level_heatmap(df, col1, col2, col3, file_name=file_name)


def plot_category_category_category_sync(input_file, col1, col2, col3, path):
    """Non-delayed version for synchronous execution"""
    df = pd.read_parquet(input_file, columns=[col1, col2, col3])
    file_name = os.path.join(path, f"{col1}-{col2}-by-{col3}-heatmap.png")
    multi_level_heatmap(df, col1, col2, col3, file_name=file_name)


@dask.delayed
def plot_numeric_category_category(input_file, num_col, cat_col1, cat_col2, path):
    """Create grouped visualizations for numeric vs two categories"""
    df = pd.read_parquet(input_file, columns=[num_col, cat_col1, cat_col2])
    file_name = os.path.join(path, f"{num_col}-{cat_col1}-{cat_col2}-grouped.png")
    grouped_bar_violin_plot(df, num_col, cat_col1, cat_col2, file_name=file_name)


def plot_numeric_category_category_sync(input_file, num_col, cat_col1, cat_col2, path):
    """Non-delayed version for synchronous execution"""
    df = pd.read_parquet(input_file, columns=[num_col, cat_col1, cat_col2])
    file_name = os.path.join(path, f"{num_col}-{cat_col1}-{cat_col2}-grouped.png")
    grouped_bar_violin_plot(df, num_col, cat_col1, cat_col2, file_name=file_name)


def create_plots(input_file, dtypes, output_path, use_dask=True):
    distributions_path, two_d_interactions_path, three_d_interactions_path = (
        _create_directories(output_path)
    )
    plots = []

    # Add summary plots
    logger.info("Adding correlation matrix and missing values plots...")
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

    # 3-variable interactions
    logger.info("Adding 3-variable interaction plots...")
    for (col1, dtype1), (col2, dtype2), (col3, dtype3) in combinations(
        dtypes.items(), 3
    ):
        logger.debug(f"Processing 3-variable interaction: {col1}, {col2}, {col3}")

        # Skip if any column should be ignored
        if any(dtype == "i" for dtype in [dtype1, dtype2, dtype3]):
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

    return plots


def _create_directories(output_path):
    distribution_path = os.path.join(output_path, "distributions")
    two_d_interaction_path = os.path.join(output_path, "2d_interactions")
    three_d_interaction_path = os.path.join(output_path, "3d_interactions")

    make_sure_path_exists(distribution_path)
    make_sure_path_exists(two_d_interaction_path)
    make_sure_path_exists(three_d_interaction_path)
    return distribution_path, two_d_interaction_path, three_d_interaction_path


# def plot_data_frame(df):
# file_path = path + 'corr-spearman-plot.png'
# if not os.path.isfile(file_path):
# sns.corrplot(df, cmap_range='full', method='spearman')
# plt.savefig(file_path, dpi=120)
# plt.close()

# file_path = path + 'corr-pearson-plot.png'
# if not os.path.isfile(file_path):
# sns.corrplot(df, cmap_range='full')
# plt.savefig(file_path, dpi=120)
# plt.close()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        if not math.isnan(height) and height > 0:
            plt.annotate(
                f"{int(height)}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )


@ignore_if_exist_or_save
def histogram_violin_plots(data, axes, file_name=None):
    # histogram
    sns.histplot(data, ax=axes[0], kde=True)
    sns.violinplot(x=data, ax=axes[1], inner="quartile", density_norm="count")
    sns.despine(left=True)


@ignore_if_exist_or_save
def bar_plot(data, col, hue=None, file_name=None):
    ax = sns.countplot(x=col, hue=hue, data=data.sort_values(col))
    sns.despine(left=True)

    autolabel(ax.patches)


@ignore_if_exist_or_save
def scatter_plot(data, col1, col2, file_name=None):
    sns.regplot(x=col1, y=col2, data=data, fit_reg=False)
    sns.despine(left=True)


@ignore_if_exist_or_save
def bar_box_violin_dot_plots(data, category_col, numeric_col, axes, file_name=None):
    sns.barplot(x=category_col, y=numeric_col, data=data, ax=axes[0])
    sns.stripplot(x=category_col, y=numeric_col, data=data, jitter=True, ax=axes[1])
    sns.boxplot(
        x=category_col,
        y=numeric_col,
        data=data[data[numeric_col].notnull()],
        ax=axes[2],
    )
    sns.violinplot(
        x=category_col,
        y=numeric_col,
        data=data,
        inner="quartile",
        density_norm="count",
        ax=axes[3],
    )
    sns.despine(left=True)


@ignore_if_exist_or_save
def heatmap(data, file_name=None):
    cmap = "BuGn" if (data.values >= 0).all() else "coolwarm"
    sns.heatmap(data=data, annot=True, fmt="d", cmap=cmap)
    sns.despine(left=True)


@ignore_if_exist_or_save
def correlation_heatmap(data, file_name=None, title="Correlation Matrix"):
    """Create a correlation matrix heatmap"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        data=data,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
    )
    plt.title(title)
    sns.despine(left=True)


@ignore_if_exist_or_save
def missing_plot(data, file_name=None):
    """Create a heatmap showing missing values"""
    plt.figure(figsize=(12, 6))
    sns.heatmap(data, cbar=True, yticklabels=False, cmap="viridis")
    plt.title("Missing Values Heatmap")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    sns.despine(left=True)


# ============================================================================
# 3-Variable Plotting Helper Functions
# ============================================================================


@ignore_if_exist_or_save
def scatter_plot_with_hue(data, col1, col2, hue_col, file_name=None):
    """Create 2D scatter plot with category as hue/color"""
    plt.figure(figsize=(10, 8))

    # Remove rows with NaN in any of the columns
    clean_data = data[[col1, col2, hue_col]].dropna()

    if len(clean_data) == 0:
        logger.warning(
            f"No valid data points for scatter plot: {col1}, {col2}, {hue_col}"
        )
        plt.text(
            0.5,
            0.5,
            "No valid data points",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )
        plt.title(f"{col1} vs {col2} by {hue_col}")
    else:
        sns.scatterplot(
            x=col1,
            y=col2,
            hue=hue_col,
            data=clean_data,
            palette="deep",
            s=100,
            alpha=0.7,
        )
        plt.title(f"{col1} vs {col2} by {hue_col}")
        # Only add legend if there are artists with labels
        if plt.gca().get_legend_handles_labels()[0]:
            plt.legend(title=hue_col, bbox_to_anchor=(1.05, 1), loc="upper left")

    sns.despine(left=True)


@ignore_if_exist_or_save
def scatter_plot_3d(data, col1, col2, col3, file_name=None):
    """Create 3D scatter plot for three numeric variables"""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Remove NaN values
    clean_data = data[[col1, col2, col3]].dropna()

    scatter = ax.scatter(
        clean_data[col1],
        clean_data[col2],
        clean_data[col3],
        c=clean_data[col3],
        cmap="viridis",
        s=50,
        alpha=0.6,
        edgecolors="w",
        linewidth=0.5,
    )

    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    ax.set_zlabel(col3)
    ax.set_title(f"3D Scatter: {col1}, {col2}, {col3}")

    plt.colorbar(scatter, ax=ax, label=col3, shrink=0.5)


@ignore_if_exist_or_save
def contour_plot(data, col1, col2, col3, file_name=None):
    """Create contour plot showing relationship between 3 numeric variables"""
    import numpy as np
    from scipy.interpolate import griddata

    # Remove NaN values
    clean_data = data[[col1, col2, col3]].dropna()

    if len(clean_data) < 4:
        logger.warning(
            f"Not enough data points for contour plot: {col1}, {col2}, {col3}"
        )
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create grid
    x = clean_data[col1].values
    y = clean_data[col2].values
    z = clean_data[col3].values

    # Create grid for interpolation
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    # Interpolate z values on grid
    try:
        zi = griddata((x, y), z, (xi_grid, yi_grid), method="cubic")

        # Create contour plot
        contour = ax.contourf(
            xi_grid, yi_grid, zi, levels=15, cmap="viridis", alpha=0.7
        )
        ax.contour(
            xi_grid, yi_grid, zi, levels=15, colors="black", alpha=0.3, linewidths=0.5
        )

        # Add scatter points
        ax.scatter(
            x,
            y,
            c=z,
            cmap="viridis",
            s=20,
            edgecolors="black",
            linewidth=0.5,
            alpha=0.8,
        )

        plt.colorbar(contour, ax=ax, label=col3)
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
        ax.set_title(f"Contour Plot: {col1} vs {col2} (colored by {col3})")
        sns.despine(left=True)
    except Exception as e:
        logger.warning(f"Could not create contour plot for {col1}, {col2}, {col3}: {e}")
        # Fallback to simple scatter plot
        ax.scatter(x, y, c=z, cmap="viridis", s=50, alpha=0.6)
        plt.colorbar(ax.collections[0], ax=ax, label=col3)
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
        ax.set_title(f"Scatter Plot: {col1} vs {col2} (colored by {col3})")
        sns.despine(left=True)


@ignore_if_exist_or_save
def multi_level_heatmap(data, col1, col2, col3, file_name=None):
    """Create faceted heatmaps for 3 categorical variables"""
    # Get unique values of the third category
    unique_vals = sorted(data[col3].dropna().unique())

    if len(unique_vals) > 10:
        logger.warning(
            f"Too many categories in {col3} ({len(unique_vals)}), limiting to first 10"
        )
        unique_vals = unique_vals[:10]

    # Calculate grid dimensions
    n_plots = len(unique_vals)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_plots > 1 else [axes]

    for idx, val in enumerate(unique_vals):
        ax = axes[idx]
        subset = data[data[col3] == val]

        # Create crosstab for heatmap
        ct = pd.crosstab(subset[col1], subset[col2])

        if ct.size > 0:
            sns.heatmap(ct, annot=True, fmt="d", cmap="YlOrRd", ax=ax, cbar=True)
            ax.set_title(f"{col3} = {val}")
        else:
            ax.text(
                0.5,
                0.5,
                f"No data for {col3}={val}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])

    # Hide extra subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f"{col1} vs {col2} by {col3}", fontsize=16, y=1.00)
    plt.tight_layout()


@ignore_if_exist_or_save
def grouped_bar_violin_plot(data, num_col, cat_col1, cat_col2, file_name=None):
    """Create grouped bar and violin plots for numeric vs two categories"""
    # Get unique values
    unique_cat2 = sorted(data[cat_col2].dropna().unique())

    if len(unique_cat2) > 10:
        logger.warning(
            f"Too many categories in {cat_col2} ({len(unique_cat2)}), limiting to first 10"
        )
        unique_cat2 = unique_cat2[:10]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Grouped bar plot
    ax1 = axes[0]
    sns.barplot(x=cat_col1, y=num_col, hue=cat_col2, data=data, ax=ax1)
    ax1.set_title(f"Mean {num_col} by {cat_col1} and {cat_col2}")
    ax1.legend(title=cat_col2, bbox_to_anchor=(1.05, 1), loc="upper left")

    # Grouped violin plot
    ax2 = axes[1]
    sns.violinplot(
        x=cat_col1,
        y=num_col,
        hue=cat_col2,
        data=data,
        ax=ax2,
        split=False,
        inner="quartile",
    )
    ax2.set_title(f"Distribution of {num_col} by {cat_col1} and {cat_col2}")
    ax2.legend(title=cat_col2, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    sns.despine(left=True)


@dask.delayed
def plot_correlation_matrix(input_file, dtypes, path):
    """
    Generate correlation matrix plots (Pearson and Spearman) for numeric columns
    """
    # Get only numeric columns
    numeric_cols = [col for col, dtype in dtypes.items() if dtype == "n"]

    if len(numeric_cols) < 2:
        logger.info(
            "Not enough numeric columns for correlation matrix (need at least 2)"
        )
        return

    # Read only numeric columns
    df = pd.read_parquet(input_file, columns=numeric_cols)

    # Pearson correlation
    pearson_corr = df.corr(method="pearson")
    file_name = os.path.join(path, "correlation-pearson.png")
    correlation_heatmap(
        pearson_corr, file_name=file_name, title="Pearson Correlation Matrix"
    )

    # Spearman correlation
    spearman_corr = df.corr(method="spearman")
    file_name = os.path.join(path, "correlation-spearman.png")
    correlation_heatmap(
        spearman_corr, file_name=file_name, title="Spearman Correlation Matrix"
    )


@dask.delayed
def plot_missing_values(input_file, dtypes, path):
    """
    Generate missing values heatmap and analysis
    """
    # Get all non-ignored columns
    cols = [col for col, dtype in dtypes.items() if dtype != "i"]

    if not cols:
        logger.info("No columns to analyze for missing values")
        return

    # Read data
    df = pd.read_parquet(input_file, columns=cols)

    # Create missing values pattern (True where value is missing)
    missing_data = df.isnull()

    # Only create plot if there are any missing values
    if missing_data.any().any():
        file_name = os.path.join(path, "missing-values-heatmap.png")
        missing_plot(missing_data, file_name=file_name)
    else:
        logger.info("No missing values found in the dataset")


def export_statistical_summaries(input_file, dtypes, output_path):
    """
    Export statistical summaries to CSV files

    Parameters
    ----------
    input_file : str
        Path to the parquet file
    dtypes : dict
        Dictionary mapping column names to data types
    output_path : str
        Directory where CSV files will be saved
    """
    logger.info("Exporting statistical summaries...")

    # Get non-ignored columns
    cols = [col for col, dtype in dtypes.items() if dtype != "i"]

    if not cols:
        logger.info("No columns to export statistics for")
        return

    # Read data
    df = pd.read_parquet(input_file, columns=cols)

    # Create stats directory
    stats_path = os.path.join(output_path, "statistics")
    make_sure_path_exists(stats_path)

    # 1. Numeric statistics
    numeric_cols = [col for col, dtype in dtypes.items() if dtype == "n"]
    if numeric_cols:
        numeric_stats = df[numeric_cols].describe()
        # Add missing count
        numeric_stats.loc["missing"] = df[numeric_cols].isnull().sum()
        numeric_stats.loc["missing_pct"] = (
            df[numeric_cols].isnull().sum() / len(df)
        ) * 100

        stats_file = os.path.join(stats_path, "numeric_statistics.csv")
        numeric_stats.to_csv(stats_file)
        logger.info(f"Numeric statistics saved to: {stats_file}")

    # 2. Categorical statistics (value counts for each categorical column)
    category_cols = [col for col, dtype in dtypes.items() if dtype == "c"]
    if category_cols:
        for col in category_cols:
            value_counts = df[col].value_counts(dropna=False)
            value_counts_df = pd.DataFrame(
                {
                    "value": value_counts.index,
                    "count": value_counts.values,
                    "percentage": (value_counts.values / len(df)) * 100,
                }
            )

            stats_file = os.path.join(stats_path, f"category_{col}_counts.csv")
            value_counts_df.to_csv(stats_file, index=False)

        logger.info(f"Categorical statistics saved for {len(category_cols)} columns")

    # 3. Missing values analysis
    missing_summary = pd.DataFrame(
        {
            "column": cols,
            "missing_count": [df[col].isnull().sum() for col in cols],
            "missing_percentage": [
                (df[col].isnull().sum() / len(df)) * 100 for col in cols
            ],
            "total_count": len(df),
            "non_missing_count": [df[col].notnull().sum() for col in cols],
        }
    )

    missing_file = os.path.join(stats_path, "missing_values_summary.csv")
    missing_summary.to_csv(missing_file, index=False)
    logger.info(f"Missing values summary saved to: {missing_file}")

    # 4. Overall dataset summary
    overall_summary = pd.DataFrame(
        {
            "metric": [
                "total_rows",
                "total_columns",
                "numeric_columns",
                "categorical_columns",
                "columns_with_missing",
                "total_missing_cells",
                "missing_percentage",
            ],
            "value": [
                len(df),
                len(cols),
                len(numeric_cols),
                len(category_cols),
                missing_summary[missing_summary["missing_count"] > 0].shape[0],
                missing_summary["missing_count"].sum(),
                (missing_summary["missing_count"].sum() / (len(df) * len(cols))) * 100,
            ],
        }
    )

    overall_file = os.path.join(stats_path, "overall_summary.csv")
    overall_summary.to_csv(overall_file, index=False)
    logger.info(f"Overall summary saved to: {overall_file}")


if __name__ == "__main__":
    main()
