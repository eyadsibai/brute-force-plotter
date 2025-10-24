#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Brute Force Plotter
-----------------
Command Line Interface

"""

import errno
import json
import logging
import math
import os
from itertools import chain, combinations

import click
import dask
from dask.diagnostics import ProgressBar
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


matplotlib.use("agg")


ignore = set()
skip_existing_plots = True  # Global flag for skipping existing plots

sns.set_style("darkgrid")
sns.set_context("paper")

sns.set(rc={"figure.figsize": (8, 6)})


@click.command()
@click.argument("input_file")
@click.argument("dtypes")
@click.argument("output_path")
@click.option("--skip-existing", is_flag=True, default=True, help="Skip generating plots that already exist")
@click.option("--theme", type=click.Choice(["darkgrid", "whitegrid", "dark", "white", "ticks"]), default="darkgrid", help="Seaborn plot style theme")
@click.option("--n-workers", type=int, default=4, help="Number of parallel workers for plot generation")
@click.option("--export-stats", is_flag=True, default=False, help="Export statistical summary to CSV")
def main(input_file, dtypes, output_path, skip_existing, theme, n_workers, export_stats):
    """Create Plots From data in input"""
    from dask.distributed import Client, LocalCluster
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set global skip_existing flag
    global skip_existing_plots
    skip_existing_plots = skip_existing
    
    # Apply theme
    sns.set_style(theme)
    
    cluster = LocalCluster(n_workers=n_workers, silence_logs=logging.WARNING)
    client = Client(cluster)

    data = pd.read_csv(input_file)
    new_file_name = f"{input_file}.parq"
    data.to_parquet(new_file_name)

    with open(dtypes, "r") as f:
        data_types = json.load(f)
    
    # Export statistical summary if requested
    if export_stats:
        export_statistical_summary(data, data_types, output_path)

    plots = create_plots(new_file_name, data_types, output_path, skip_existing)
    
    logger.info(f"Generating {len(plots)} plots...")
    with ProgressBar():
        dask.compute(*plots)
    
    logger.info("All plots generated successfully!")
    client.close()
    cluster.close()


def ignore_if_exist_or_save(func):
    def wrapper(*args, **kwargs):
        file_name = kwargs["file_name"]
        
        if skip_existing_plots and os.path.isfile(file_name):
            logger.debug(f"Skipping existing plot: {file_name}")
            plt.close("all")
        else:
            func(*args, **kwargs)
            plt.gcf().set_tight_layout(True)
            plt.gcf().savefig(file_name, dpi=120)
            logger.debug(f"Saved plot: {file_name}")
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


@dask.delayed
def plot_category_category(input_file, col1, col2, path):
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


@dask.delayed
def plot_category_numeric(input_file, category_col, numeric_col, path):
    df = pd.read_parquet(input_file, columns=[category_col, numeric_col])
    f, axes = plt.subplots(2, 2, sharex="col", sharey="row", figsize=(8, 6))
    axes = list(chain.from_iterable(axes))
    file_name = os.path.join(path, f"{category_col}-{numeric_col}-plot.png")
    bar_box_violin_dot_plots(df, category_col, numeric_col, axes, file_name=file_name)


@dask.delayed
def plot_missing_values(input_file, dtypes, path):
    """Create a heatmap showing missing values across the dataset"""
    # Read all columns except ignored ones
    cols_to_read = [col for col, dtype in dtypes.items() if dtype != 'i']
    df = pd.read_parquet(input_file, columns=cols_to_read)
    
    file_name = os.path.join(path, "missing-values-heatmap.png")
    missing_data = df.isnull()
    
    if missing_data.sum().sum() > 0:  # Only create if there are missing values
        missing_plot(missing_data, file_name=file_name)
    else:
        logger.info("No missing values found in dataset")


@dask.delayed
def plot_correlation_matrix(input_file, dtypes, path):
    """Create correlation matrix for numeric variables"""
    numeric_cols = [col for col, dtype in dtypes.items() if dtype == 'n']
    
    if len(numeric_cols) < 2:
        logger.info("Not enough numeric columns for correlation matrix")
        return
    
    df = pd.read_parquet(input_file, columns=numeric_cols)
    
    # Pearson correlation
    file_name = os.path.join(path, "correlation-matrix-pearson.png")
    correlation_heatmap(df.corr(method='pearson'), file_name=file_name, title="Pearson Correlation Matrix")
    
    # Spearman correlation
    file_name = os.path.join(path, "correlation-matrix-spearman.png")
    correlation_heatmap(df.corr(method='spearman'), file_name=file_name, title="Spearman Correlation Matrix")


def export_statistical_summary(data, dtypes, output_path):
    """Export statistical summary of the dataset to CSV"""
    logger.info("Generating statistical summary...")
    
    numeric_cols = [col for col, dtype in dtypes.items() if dtype == 'n']
    category_cols = [col for col, dtype in dtypes.items() if dtype == 'c']
    
    summary_file = os.path.join(output_path, "statistical_summary.csv")
    
    # Numeric statistics
    if numeric_cols:
        numeric_stats = data[numeric_cols].describe()
        numeric_stats.to_csv(summary_file)
        logger.info(f"Statistical summary saved to {summary_file}")
    
    # Category value counts
    if category_cols:
        category_summary_file = os.path.join(output_path, "category_summary.csv")
        category_summaries = []
        for col in category_cols:
            value_counts = data[col].value_counts()
            category_summaries.append(pd.DataFrame({
                'column': col,
                'value': value_counts.index,
                'count': value_counts.values
            }))
        if category_summaries:
            pd.concat(category_summaries).to_csv(category_summary_file, index=False)
            logger.info(f"Category summary saved to {category_summary_file}")
    
    # Missing values summary
    missing_summary_file = os.path.join(output_path, "missing_values_summary.csv")
    missing_counts = data.isnull().sum()
    missing_pct = (missing_counts / len(data)) * 100
    missing_df = pd.DataFrame({
        'column': missing_counts.index,
        'missing_count': missing_counts.values,
        'missing_percentage': missing_pct.values
    })
    missing_df.to_csv(missing_summary_file, index=False)
    logger.info(f"Missing values summary saved to {missing_summary_file}")


def create_plots(input_file, dtypes, output_path, skip_existing=True):
    distributions_path, two_d_interactions_path, three_d_interactions_path = _create_directories(
        output_path
    )
    plots = []
    
    # Add summary plots
    logger.info("Adding correlation matrix and missing values plots...")
    plots.append(plot_correlation_matrix(input_file, dtypes, distributions_path))
    plots.append(plot_missing_values(input_file, dtypes, distributions_path))
    
    for col, dtype in dtypes.items():
        logger.info(f"Processing column: {col}")
        if dtype == "n":
            plots.append(plot_single_numeric(input_file, col, distributions_path))
        if dtype == "c":
            plots.append(plot_single_category(input_file, col, distributions_path))

    for (col1, dtype1), (col2, dtype2) in combinations(dtypes.items(), 2):
        logger.debug(f"Processing pair: {col1}, {col2}")
        if any(col in ignore for col in [dtype1, dtype2]):
            continue
        if dtype1 == dtype2 == "n":
            plots.append(
                plot_numeric_numeric(input_file, col1, col2, two_d_interactions_path)
            )
        if dtype1 == dtype2 == "c":
            plots.append(
                plot_category_category(input_file, col1, col2, two_d_interactions_path)
            )
        if dtype1 == "c" and dtype2 == "n":
            plots.append(
                plot_category_numeric(input_file, col1, col2, two_d_interactions_path)
            )
        if dtype1 == "n" and dtype2 == "c":
            plots.append(
                plot_category_numeric(input_file, col2, col1, two_d_interactions_path)
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
    sns.violinplot(x=data, ax=axes[1], inner="quartile", scale="count")
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
        scale="count",
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
    sns.heatmap(data=data, annot=True, fmt=".2f", cmap="coolwarm", center=0, 
                vmin=-1, vmax=1, square=True, linewidths=0.5)
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


if __name__ == "__main__":
    main()
