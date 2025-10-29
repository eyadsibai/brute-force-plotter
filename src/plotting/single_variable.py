"""
Single variable (1D) plotting functions.

This module handles distribution plots for individual variables.
"""

import os

import dask
import matplotlib.pyplot as plt
import pandas as pd

from ..core.config import ignore
from .base import bar_plot, histogram_violin_plots

__all__ = [
    "plot_single_numeric",
    "plot_single_numeric_sync",
    "plot_single_category",
    "plot_single_category_sync",
]


@dask.delayed
def plot_single_numeric(input_file, col, path):
    """
    Create distribution plots for a single numeric variable (delayed).

    Parameters
    ----------
    input_file : str
        Path to parquet file
    col : str
        Column name
    path : str
        Output directory path
    """
    from ..core.config import get_target_variable

    target = get_target_variable()

    # Load columns including target if available
    columns = [col]
    if target and target != col:
        columns.append(target)

    df = pd.read_parquet(input_file, columns=columns)
    file_name = os.path.join(path, f"{col}-dist-plot.png")

    # Use target for grouping if available and categorical
    if target and target != col and target in df.columns:
        # Keep both columns for grouped visualization
        data = df[[col, target]].dropna()
        _, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
        histogram_violin_plots(data, axes, file_name=file_name, hue=target)
    else:
        # Original single-variable plot
        data = df[col].dropna()
        _, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
        histogram_violin_plots(data, axes, file_name=file_name)


def plot_single_numeric_sync(input_file, col, path):
    """
    Create distribution plots for a single numeric variable (synchronous).

    Parameters
    ----------
    input_file : str
        Path to parquet file
    col : str
        Column name
    path : str
        Output directory path
    """
    from ..core.config import get_target_variable

    target = get_target_variable()

    # Load columns including target if available
    columns = [col]
    if target and target != col:
        columns.append(target)

    df = pd.read_parquet(input_file, columns=columns)
    file_name = os.path.join(path, f"{col}-dist-plot.png")

    # Use target for grouping if available and categorical
    if target and target != col and target in df.columns:
        # Keep both columns for grouped visualization
        data = df[[col, target]].dropna()
        _, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
        histogram_violin_plots(data, axes, file_name=file_name, hue=target)
    else:
        # Original single-variable plot
        data = df[col].dropna()
        _, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
        histogram_violin_plots(data, axes, file_name=file_name)


@dask.delayed
def plot_single_category(input_file, col, path):
    """
    Create bar plot for a single categorical variable (delayed).

    Parameters
    ----------
    input_file : str
        Path to parquet file
    col : str
        Column name
    path : str
        Output directory path
    """
    from ..core.config import get_target_variable

    target = get_target_variable()

    # Load columns including target if available
    columns = [col]
    if target and target != col:
        columns.append(target)

    df = pd.read_parquet(input_file, columns=columns)
    value_counts = df[col].value_counts(dropna=False)
    # if the categories are more than 50 then this should be ignored
    if len(value_counts) > 50:
        ignore.add(col)
    else:
        file_name = os.path.join(path, col + "-bar-plot.png")
        # Use target as hue if it's categorical and different from col
        hue = target if target and target != col and target in df.columns else None
        bar_plot(df, col, hue=hue, file_name=file_name)


def plot_single_category_sync(input_file, col, path):
    """
    Create bar plot for a single categorical variable (synchronous).

    Parameters
    ----------
    input_file : str
        Path to parquet file
    col : str
        Column name
    path : str
        Output directory path
    """
    from ..core.config import get_target_variable

    target = get_target_variable()

    # Load columns including target if available
    columns = [col]
    if target and target != col:
        columns.append(target)

    df = pd.read_parquet(input_file, columns=columns)
    value_counts = df[col].value_counts(dropna=False)
    if len(value_counts) > 50:
        ignore.add(col)
    else:
        file_name = os.path.join(path, col + "-bar-plot.png")
        # Use target as hue if it's categorical and different from col
        hue = target if target and target != col and target in df.columns else None
        bar_plot(df, col, hue=hue, file_name=file_name)
