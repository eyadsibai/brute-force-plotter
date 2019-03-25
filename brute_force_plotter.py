#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Brute Force Plotter
-----------------
Command Line Interface

"""

from __future__ import unicode_literals

import errno
import json
import logging
import math
import os
from itertools import chain, combinations

import click
import dask
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


@click.command()
@click.argument("input-file")
@click.argument("dtypes")
@click.argument("output-path")
def main(input_file, dtypes, output_path):
    """Create Plots From data in input"""

    data = pd.read_csv(input_file)

    data_types = json.load(open(dtypes, "r"))
    plots = create_plots(data, data_types, output_path)
    dask.compute(*plots, scheduler="synchronous")


matplotlib.use("agg")


ignore = set()

sns.set_style("darkgrid")
sns.set_context("paper")

sns.set(rc={"figure.figsize": (8, 6)})


def ignore_if_exist_or_save(func):
    def wrapper(*args, **kwargs):

        file_name = kwargs["file_name"]

        if os.path.isfile(file_name):
            plt.close("all")
        else:
            func(*args, **kwargs)
            plt.gcf().set_tight_layout(True)
            plt.gcf().savefig(file_name, dpi=120)
            plt.close("all")

    return wrapper


def make_sure_path_exists(path):
    logging.debug(f"Make sure {path} exists")
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            return False
    return True


@dask.delayed
def plot_single_numeric(df, col, path):
    file_name = os.path.join(path, col + "-dist-plot.png")
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
def plot_single_category(df, col, path):
    value_counts = df[col].value_counts(dropna=False)
    # if the categories are more than 50 then this should be ignored
    # TODO find a better way to visualize this
    if len(value_counts) > 50:
        ignore.add(col)
    else:
        file_name = os.path.join(path, col + "-bar-plot.png")
        bar_plot(df, col, file_name=file_name)


@dask.delayed
def plot_category_category(df, col1, col2, path):
    if len(df[col1].unique()) < len(df[col2].unique()):
        col1, col2 = col2, col1
    file_name = os.path.join(path, f"{col1}-{col2}-bar-plot.png")
    bar_plot(df, col1, hue=col2, file_name=file_name)

    file_name = os.path.join(path, f"{col1}-{col2}-heatmap.png")
    heatmap(pd.crosstab(df[col1], df[col2]), file_name=file_name)


@dask.delayed
def plot_numeric_numeric(df, col1, col2, path):
    file_name = os.path.join(path, f"{col1}-{col2}-scatter-plot.png")
    scatter_plot(df, col1, col2, file_name=file_name)


@dask.delayed
def plot_category_numeric(df, category_col, numeric_col, path):
    f, axes = plt.subplots(2, 2, sharex="col", sharey="row", figsize=(8, 6))
    axes = list(chain.from_iterable(axes))
    file_name = os.path.join(path, f"{category_col}-{numeric_col}-plot.png")
    bar_box_violin_dot_plots(df, category_col, numeric_col, axes, file_name=file_name)


def create_plots(df, dtypes, output_path):
    data = df[list(dtypes.keys())]
    distributions_path, two_d_interactions_path, three_d_interactions_path = _create_directories(
        output_path
    )
    plots = []
    for col, dtype in dtypes.items():
        print(col)
        if dtype == "n":
            plots.append(plot_single_numeric(data, col, distributions_path))
        if dtype == "c":
            plots.append(plot_single_category(data, col, distributions_path))

    for (col1, dtype1), (col2, dtype2) in combinations(dtypes.items(), 2):
        print(col1, col2)
        if any(col in ignore for col in [dtype1, dtype2]):
            continue
        if dtype1 == "n" and dtype2 == "n":
            plots.append(
                plot_numeric_numeric(data, col1, col2, two_d_interactions_path)
            )
        if dtype1 == "c" and dtype2 == "c":
            plots.append(
                plot_category_category(data, col1, col2, two_d_interactions_path)
            )
        if dtype1 == "c" and dtype2 == "n":
            plots.append(
                plot_category_numeric(data, col1, col2, two_d_interactions_path)
            )
        if dtype1 == "n" and dtype2 == "c":
            plots.append(
                plot_category_numeric(data, col2, col1, two_d_interactions_path)
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
    for rect in rects:
        height = rect.get_height()
        width = rect.get_width()

        if math.isnan(height):
            height = 0.0
        if (height, width) == (1.0, 1.0):
            continue
        plt.text(
            rect.get_x() + width / 2.0,
            1.0 + height,
            "%d" % int(height),
            ha="center",
            va="bottom",
        )


@ignore_if_exist_or_save
def histogram_violin_plots(data, axes, file_name=None):
    # histogram
    sns.distplot(data, ax=axes[0], axlabel="")
    sns.violinplot(data, ax=axes[1], inner="quartile", scale="count")
    sns.despine(left=True)


@ignore_if_exist_or_save
def bar_plot(data, col, hue=None, file_name=None):
    sns.countplot(col, hue=hue, data=data.sort_values(col))
    sns.despine(left=True)

    subplots = [
        x for x in plt.gcf().get_children() if isinstance(x, matplotlib.axes.Subplot)
    ]
    for plot in subplots:
        rectangles = [
            x
            for x in plot.get_children()
            if isinstance(x, matplotlib.patches.Rectangle)
        ]
    autolabel(rectangles)


@ignore_if_exist_or_save
def scatter_plot(data, col1, col2, file_name=None):
    sns.regplot(x=col1, y=col2, data=data, fit_reg=False)
    sns.despine(left=True)


@ignore_if_exist_or_save
def bar_box_violin_dot_plots(data, category_col, numeric_col, axes, file_name=None):
    sns.barplot(category_col, numeric_col, data=data, ax=axes[0])
    sns.boxplot(
        category_col, numeric_col, data=data[data[numeric_col].notnull()], ax=axes[2]
    )
    sns.violinplot(
        category_col,
        numeric_col,
        data=data,
        kind="violin",
        inner="quartile",
        scale="count",
        split=True,
        ax=axes[3],
    )
    sns.stripplot(category_col, numeric_col, data=data, jitter=True, ax=axes[1])
    sns.despine(left=True)


@ignore_if_exist_or_save
def heatmap(data, file_name=None):
    cmap = "BuGn" if (data.values >= 0).all() else "coolwarm"
    sns.heatmap(data=data, annot=True, fmt="d", cmap=cmap)
    sns.despine(left=True)


if __name__ == "__main__":
    main()
