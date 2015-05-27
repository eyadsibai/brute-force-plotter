# coding=utf-8
from itertools import chain, combinations
import os

import pandas as pd
import matplotlib


matplotlib.use('agg')

import matplotlib.pyplot as plt
import seaborn as sns

from plot_types import bar_plot, scatter_plot, \
    histogram_violin_plots, bar_box_violin_dot_plots, heatmap
from utils import make_sure_path_exists


ignore = set()

sns.set_style('darkgrid')
sns.set_context("paper")

sns.set(rc={"figure.figsize": (8, 6)})


def plot_single_numeric(df, col, path):
    file_name = os.path.join(path, col + '-dist-plot.png')
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


def plot_single_category(df, col, path):
    value_counts = df[col].value_counts()
    # if the categories are more than 50 then this should be ignored
    # TODO find a better way to visualize this
    if len(value_counts) > 50:
        ignore.add(col)
    else:
        file_name = os.path.join(path, col + '-bar-plot.png')
        bar_plot(df, col, file_name=file_name)


def plot_category_category(df, col1, col2, path):
    if len(df[col1].unique()) < len(df[col2].unique()):
        col1, col2 = col2, col1
    file_name = os.path.join(path, col1 + '-' + col2 + '-bar-plot.png')
    bar_plot(df, col1, hue=col2, file_name=file_name)

    file_name = os.path.join(path, col1 + '-' + col2 + '-heatmap.png')
    heatmap(pd.crosstab(df[col1], df[col2]), file_name=file_name)


def plot_numeric_numeric(df, col1, col2, path):
    file_name = os.path.join(path, col1 + '-' + col2 + '-scatter-plot.png')
    scatter_plot(df, col1, col2, file_name=file_name)


def plot_category_numeric(df, category_col, numeric_col, path):
    f, axes = plt.subplots(2, 2, sharex='col', sharey='row',
                           figsize=(8, 6))
    axes = list(chain.from_iterable(axes))
    file_name = os.path.join(
        path, category_col + '-' + numeric_col + '-plot.png')
    bar_box_violin_dot_plots(df, category_col, numeric_col, axes,
                             file_name=file_name)


def create_plots(df, dtypes, output_path):
    data = df[list(dtypes.keys())]
    distributions_path, two_d_interactions_path, three_d_interactions_path = \
        _create_directories(output_path)
    for col, dtype in dtypes.items():
        print(col)
        if dtype == 'n':
            plot_single_numeric(data, col, distributions_path)
        if dtype == 'c':
            plot_single_category(data, col, distributions_path)

    for (col1, dtype1), (col2, dtype2) in combinations(dtypes.items(), 2):
        print(col1, col2)
        if any(col in ignore for col in [dtype1, dtype2]):
            continue
        if dtype1 == 'n' and dtype2 == 'n':
            plot_numeric_numeric(data, col1, col2,
                                 two_d_interactions_path)
        if dtype1 == 'c' and dtype2 == 'c':
            plot_category_category(data, col1, col2,
                                   two_d_interactions_path)
        if dtype1 == 'c' and dtype2 == 'n':
            plot_category_numeric(data, col1, col2,
                                  two_d_interactions_path)
        if dtype1 == 'n' and dtype2 == 'c':
            plot_category_numeric(data, col2, col1,
                                  two_d_interactions_path)

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


def _create_directories(output_path):
    distribution_path = os.path.join(output_path, 'distributions')
    two_d_interaction_path = os.path.join(output_path, '2d_interactions')
    three_d_interaction_path = os.path.join(output_path, '3d_interactions')

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
