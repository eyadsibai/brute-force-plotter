"""
Base plotting functions and helpers.

This module contains common plotting utilities used across all plot types.
"""

import matplotlib.pyplot as plt
import seaborn as sns

from ..core.utils import autolabel, ignore_if_exist_or_save

__all__ = [
    "histogram_violin_plots",
    "bar_plot",
    "scatter_plot",
    "bar_box_violin_dot_plots",
    "box_violin_plots",
    "heatmap",
    "correlation_heatmap",
    "missing_plot",
]


@ignore_if_exist_or_save
def histogram_violin_plots(data, axes, file_name=None):
    """
    Create histogram with KDE and violin plot for numeric data.

    Parameters
    ----------
    data : pandas.Series
        Numeric data to plot
    axes : list of matplotlib.axes.Axes
        Two axes for histogram and violin plot
    file_name : str, optional
        Output file path
    """
    # histogram
    sns.histplot(data, ax=axes[0], kde=True)
    sns.violinplot(x=data, ax=axes[1], inner="quartile", density_norm="count")
    sns.despine(left=True)


@ignore_if_exist_or_save
def bar_plot(data, col, hue=None, file_name=None):
    """
    Create bar plot for categorical data.

    Parameters
    ----------
    data : pandas.DataFrame
        Data containing the column
    col : str
        Column name to plot
    hue : str, optional
        Column name for grouping
    file_name : str, optional
        Output file path
    """
    ax = sns.countplot(x=col, hue=hue, data=data.sort_values(col))
    sns.despine(left=True)
    autolabel(ax.patches)


@ignore_if_exist_or_save
def scatter_plot(data, col1, col2, file_name=None):
    """
    Create scatter plot for two numeric variables.

    Parameters
    ----------
    data : pandas.DataFrame
        Data containing the columns
    col1 : str
        First column name (x-axis)
    col2 : str
        Second column name (y-axis)
    file_name : str, optional
        Output file path
    """
    sns.regplot(x=col1, y=col2, data=data, fit_reg=False)
    sns.despine(left=True)


@ignore_if_exist_or_save
def bar_box_violin_dot_plots(data, category_col, numeric_col, axes, file_name=None):
    """
    Create four plots showing category vs numeric relationship.

    Parameters
    ----------
    data : pandas.DataFrame
        Data containing the columns
    category_col : str
        Categorical column name
    numeric_col : str
        Numeric column name
    axes : list of matplotlib.axes.Axes
        Four axes for the plots
    file_name : str, optional
        Output file path
    """
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
def box_violin_plots(data, category_col, numeric_col, axes, file_name=None):
    """
    Create box and violin plots (minimal version).

    Parameters
    ----------
    data : pandas.DataFrame
        Data containing the columns
    category_col : str
        Categorical column name
    numeric_col : str
        Numeric column name
    axes : list of matplotlib.axes.Axes
        Two axes for the plots
    file_name : str, optional
        Output file path
    """
    sns.boxplot(
        x=category_col,
        y=numeric_col,
        data=data[data[numeric_col].notnull()],
        ax=axes[0],
    )
    sns.violinplot(
        x=category_col,
        y=numeric_col,
        data=data,
        inner="quartile",
        density_norm="count",
        ax=axes[1],
    )
    sns.despine(left=True)


@ignore_if_exist_or_save
def heatmap(data, file_name=None):
    """
    Create heatmap for categorical cross-tabulation.

    Parameters
    ----------
    data : pandas.DataFrame
        Cross-tabulated data
    file_name : str, optional
        Output file path
    """
    cmap = "BuGn" if (data.values >= 0).all() else "coolwarm"
    sns.heatmap(data=data, annot=True, fmt="d", cmap=cmap)
    sns.despine(left=True)


@ignore_if_exist_or_save
def correlation_heatmap(data, file_name=None, title="Correlation Matrix"):
    """
    Create correlation matrix heatmap.

    Parameters
    ----------
    data : pandas.DataFrame
        Correlation matrix
    file_name : str, optional
        Output file path
    title : str, optional
        Plot title
    """
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
    """
    Create heatmap showing missing values.

    Parameters
    ----------
    data : pandas.DataFrame
        Boolean DataFrame indicating missing values
    file_name : str, optional
        Output file path
    """
    plt.figure(figsize=(12, 6))
    sns.heatmap(data, cbar=True, yticklabels=False, cmap="viridis")
    plt.title("Missing Values Heatmap")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    sns.despine(left=True)
