#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Brute Force Plotter
-----------------
Library and Command Line Interface

"""

import errno
import json
import logging
import math
import os
import tempfile
from itertools import chain, combinations

import click
import dask
from dask.diagnostics import ProgressBar
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


ignore = set()

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
def main(input_file, dtypes, output_path):
    """Create Plots From data in input"""
    # Set matplotlib backend for CLI (non-interactive)
    matplotlib.use("agg")
    
    from dask.distributed import Client, LocalCluster
    cluster = LocalCluster(n_workers=10)
    client = Client(cluster)

    data = pd.read_csv(input_file)
    new_file_name = f"{input_file}.parq"
    data.to_parquet(new_file_name)

    with open(dtypes, "r") as f:
        data_types = json.load(f)

    plots = create_plots(new_file_name, data_types, output_path)
    dask.compute(*plots)


def plot(data, dtypes, output_path=None, show=False, use_dask=False, n_workers=4):
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
    """
    global _show_plots, _save_plots
    
    # Set matplotlib backend based on show parameter
    if show:
        try:
            matplotlib.use('TkAgg')
        except:
            try:
                matplotlib.use('Qt5Agg')
            except:
                logger.warning("Could not set interactive backend, using default")
    else:
        matplotlib.use('agg')
    
    _show_plots = show
    _save_plots = not show or output_path is not None
    
    # Determine output path
    if output_path is None and not show:
        output_path = tempfile.mkdtemp(prefix='brute_force_plotter_')
        logger.info(f"No output path specified, using temporary directory: {output_path}")
    elif output_path is None and show:
        # Create a temp directory anyway for potential saving
        output_path = tempfile.mkdtemp(prefix='brute_force_plotter_')
    
    # Create temporary parquet file for efficient processing
    temp_parquet = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.parq', delete=False) as tmp:
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


def create_plots(input_file, dtypes, output_path, use_dask=True):
    distributions_path, two_d_interactions_path, three_d_interactions_path = _create_directories(
        output_path
    )
    plots = []
    for col, dtype in dtypes.items():
        print(col)
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
        if any(col in ignore for col in [dtype1, dtype2]):
            continue
        if dtype1 == dtype2 == "n":
            if use_dask:
                plots.append(
                    plot_numeric_numeric(input_file, col1, col2, two_d_interactions_path)
                )
            else:
                plot_numeric_numeric_sync(input_file, col1, col2, two_d_interactions_path)
        if dtype1 == dtype2 == "c":
            if use_dask:
                plots.append(
                    plot_category_category(input_file, col1, col2, two_d_interactions_path)
                )
            else:
                plot_category_category_sync(input_file, col1, col2, two_d_interactions_path)
        if dtype1 == "c" and dtype2 == "n":
            if use_dask:
                plots.append(
                    plot_category_numeric(input_file, col1, col2, two_d_interactions_path)
                )
            else:
                plot_category_numeric_sync(input_file, col1, col2, two_d_interactions_path)
        if dtype1 == "n" and dtype2 == "c":
            if use_dask:
                plots.append(
                    plot_category_numeric(input_file, col2, col1, two_d_interactions_path)
                )
            else:
                plot_category_numeric_sync(input_file, col2, col1, two_d_interactions_path)

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
            plt.annotate(f'{int(height)}',
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom')


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
        x=category_col, y=numeric_col, data=data[data[numeric_col].notnull()], ax=axes[2]
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


if __name__ == "__main__":
    main()
