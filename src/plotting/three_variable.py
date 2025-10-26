"""
Three-variable (3D) interaction plotting functions.

This module handles plots showing relationships between three variables.
"""

import logging
import os

import dask
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import griddata

from ..core.utils import ignore_if_exist_or_save

logger = logging.getLogger(__name__)

__all__ = [
    "plot_numeric_numeric_category",
    "plot_numeric_numeric_category_sync",
    "plot_numeric_numeric_numeric",
    "plot_numeric_numeric_numeric_sync",
    "plot_category_category_category",
    "plot_category_category_category_sync",
    "plot_numeric_category_category",
    "plot_numeric_category_category_sync",
    "scatter_plot_with_hue",
    "scatter_plot_3d",
    "contour_plot",
    "multi_level_heatmap",
    "grouped_bar_violin_plot",
]


@ignore_if_exist_or_save
def scatter_plot_with_hue(data, col1, col2, hue_col, file_name=None):
    """
    Create 2D scatter plot with category as hue/color.

    Parameters
    ----------
    data : pandas.DataFrame
        Data containing the columns
    col1 : str
        First numeric column (x-axis)
    col2 : str
        Second numeric column (y-axis)
    hue_col : str
        Categorical column for coloring
    file_name : str, optional
        Output file path
    """
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
    """
    Create 3D scatter plot for three numeric variables.

    Parameters
    ----------
    data : pandas.DataFrame
        Data containing the columns
    col1 : str
        First numeric column (x-axis)
    col2 : str
        Second numeric column (y-axis)
    col3 : str
        Third numeric column (z-axis)
    file_name : str, optional
        Output file path
    """
    # Import Axes3D for its side effect: registers the '3d' projection for matplotlib.
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
    """
    Create contour plot showing relationship between 3 numeric variables.

    Parameters
    ----------
    data : pandas.DataFrame
        Data containing the columns
    col1 : str
        First numeric column (x-axis)
    col2 : str
        Second numeric column (y-axis)
    col3 : str
        Third numeric column (color)
    file_name : str, optional
        Output file path
    """
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
    """
    Create faceted heatmaps for 3 categorical variables.

    Parameters
    ----------
    data : pandas.DataFrame
        Data containing the columns
    col1 : str
        First categorical column
    col2 : str
        Second categorical column
    col3 : str
        Third categorical column
    file_name : str, optional
        Output file path
    """
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
    axes = [axes] if n_plots == 1 else axes.flatten() if n_plots > 1 else [axes]

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
    """
    Create grouped bar and violin plots for numeric vs two categories.

    Parameters
    ----------
    data : pandas.DataFrame
        Data containing the columns
    num_col : str
        Numeric column
    cat_col1 : str
        First categorical column
    cat_col2 : str
        Second categorical column
    file_name : str, optional
        Output file path
    """
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


# Delayed versions of 3D plotting functions
@dask.delayed
def plot_numeric_numeric_category(input_file, num_col1, num_col2, cat_col, path):
    """
    Create 3D scatter plot with category as hue (delayed).

    Parameters
    ----------
    input_file : str
        Path to parquet file
    num_col1 : str
        First numeric column
    num_col2 : str
        Second numeric column
    cat_col : str
        Categorical column
    path : str
        Output directory path
    """
    df = pd.read_parquet(input_file, columns=[num_col1, num_col2, cat_col])
    file_name = os.path.join(path, f"{num_col1}-{num_col2}-{cat_col}-scatter-3d.png")
    scatter_plot_with_hue(df, num_col1, num_col2, cat_col, file_name=file_name)


def plot_numeric_numeric_category_sync(input_file, num_col1, num_col2, cat_col, path):
    """
    Create 3D scatter plot with category as hue (synchronous).

    Parameters
    ----------
    input_file : str
        Path to parquet file
    num_col1 : str
        First numeric column
    num_col2 : str
        Second numeric column
    cat_col : str
        Categorical column
    path : str
        Output directory path
    """
    df = pd.read_parquet(input_file, columns=[num_col1, num_col2, cat_col])
    file_name = os.path.join(path, f"{num_col1}-{num_col2}-{cat_col}-scatter-3d.png")
    scatter_plot_with_hue(df, num_col1, num_col2, cat_col, file_name=file_name)


@dask.delayed
def plot_numeric_numeric_numeric(input_file, col1, col2, col3, path):
    """
    Create 3D scatter plot and contour plots for 3 numeric variables (delayed).

    Parameters
    ----------
    input_file : str
        Path to parquet file
    col1 : str
        First numeric column
    col2 : str
        Second numeric column
    col3 : str
        Third numeric column
    path : str
        Output directory path
    """
    df = pd.read_parquet(input_file, columns=[col1, col2, col3])

    # 3D scatter plot
    file_name = os.path.join(path, f"{col1}-{col2}-{col3}-3d-scatter.png")
    scatter_plot_3d(df, col1, col2, col3, file_name=file_name)

    # Contour plots for each pair with third variable as color
    file_name = os.path.join(path, f"{col1}-{col2}-contour-{col3}.png")
    contour_plot(df, col1, col2, col3, file_name=file_name)


def plot_numeric_numeric_numeric_sync(input_file, col1, col2, col3, path):
    """
    Create 3D scatter plot and contour plots for 3 numeric variables (synchronous).

    Parameters
    ----------
    input_file : str
        Path to parquet file
    col1 : str
        First numeric column
    col2 : str
        Second numeric column
    col3 : str
        Third numeric column
    path : str
        Output directory path
    """
    df = pd.read_parquet(input_file, columns=[col1, col2, col3])

    # 3D scatter plot
    file_name = os.path.join(path, f"{col1}-{col2}-{col3}-3d-scatter.png")
    scatter_plot_3d(df, col1, col2, col3, file_name=file_name)

    # Contour plots
    file_name = os.path.join(path, f"{col1}-{col2}-contour-{col3}.png")
    contour_plot(df, col1, col2, col3, file_name=file_name)


@dask.delayed
def plot_category_category_category(input_file, col1, col2, col3, path):
    """
    Create multi-level heatmap for 3 categorical variables (delayed).

    Parameters
    ----------
    input_file : str
        Path to parquet file
    col1 : str
        First categorical column
    col2 : str
        Second categorical column
    col3 : str
        Third categorical column
    path : str
        Output directory path
    """
    df = pd.read_parquet(input_file, columns=[col1, col2, col3])
    file_name = os.path.join(path, f"{col1}-{col2}-by-{col3}-heatmap.png")
    multi_level_heatmap(df, col1, col2, col3, file_name=file_name)


def plot_category_category_category_sync(input_file, col1, col2, col3, path):
    """
    Create multi-level heatmap for 3 categorical variables (synchronous).

    Parameters
    ----------
    input_file : str
        Path to parquet file
    col1 : str
        First categorical column
    col2 : str
        Second categorical column
    col3 : str
        Third categorical column
    path : str
        Output directory path
    """
    df = pd.read_parquet(input_file, columns=[col1, col2, col3])
    file_name = os.path.join(path, f"{col1}-{col2}-by-{col3}-heatmap.png")
    multi_level_heatmap(df, col1, col2, col3, file_name=file_name)


@dask.delayed
def plot_numeric_category_category(input_file, num_col, cat_col1, cat_col2, path):
    """
    Create grouped visualizations for numeric vs two categories (delayed).

    Parameters
    ----------
    input_file : str
        Path to parquet file
    num_col : str
        Numeric column
    cat_col1 : str
        First categorical column
    cat_col2 : str
        Second categorical column
    path : str
        Output directory path
    """
    df = pd.read_parquet(input_file, columns=[num_col, cat_col1, cat_col2])
    file_name = os.path.join(path, f"{num_col}-{cat_col1}-{cat_col2}-grouped.png")
    grouped_bar_violin_plot(df, num_col, cat_col1, cat_col2, file_name=file_name)


def plot_numeric_category_category_sync(input_file, num_col, cat_col1, cat_col2, path):
    """
    Create grouped visualizations for numeric vs two categories (synchronous).

    Parameters
    ----------
    input_file : str
        Path to parquet file
    num_col : str
        Numeric column
    cat_col1 : str
        First categorical column
    cat_col2 : str
        Second categorical column
    path : str
        Output directory path
    """
    df = pd.read_parquet(input_file, columns=[num_col, cat_col1, cat_col2])
    file_name = os.path.join(path, f"{num_col}-{cat_col1}-{cat_col2}-grouped.png")
    grouped_bar_violin_plot(df, num_col, cat_col1, cat_col2, file_name=file_name)
