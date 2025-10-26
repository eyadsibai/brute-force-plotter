"""
Two-variable (2D) interaction plotting functions.

This module handles plots showing relationships between two variables.
"""

import os
from itertools import chain

import dask
import matplotlib.pyplot as plt
import pandas as pd

from .base import (
    bar_box_violin_dot_plots,
    bar_plot,
    box_violin_plots,
    heatmap,
    scatter_plot,
)

__all__ = [
    "plot_category_category",
    "plot_category_category_sync",
    "plot_category_category_minimal",
    "plot_category_category_minimal_sync",
    "plot_numeric_numeric",
    "plot_numeric_numeric_sync",
    "plot_category_numeric",
    "plot_category_numeric_sync",
    "plot_category_numeric_minimal",
    "plot_category_numeric_minimal_sync",
]


@dask.delayed
def plot_category_category(input_file, col1, col2, path):
    """
    Create bar plot and heatmap for two categorical variables (delayed).

    Parameters
    ----------
    input_file : str
        Path to parquet file
    col1 : str
        First column name
    col2 : str
        Second column name
    path : str
        Output directory path
    """
    df = pd.read_parquet(input_file, columns=[col1, col2])
    if len(df[col1].unique()) < len(df[col2].unique()):
        col1, col2 = col2, col1
    file_name = os.path.join(path, f"{col1}-{col2}-bar-plot.png")
    bar_plot(df, col1, hue=col2, file_name=file_name)

    file_name = os.path.join(path, f"{col1}-{col2}-heatmap.png")
    heatmap(pd.crosstab(df[col1], df[col2]), file_name=file_name)


def plot_category_category_sync(input_file, col1, col2, path):
    """
    Create bar plot and heatmap for two categorical variables (synchronous).

    Parameters
    ----------
    input_file : str
        Path to parquet file
    col1 : str
        First column name
    col2 : str
        Second column name
    path : str
        Output directory path
    """
    df = pd.read_parquet(input_file, columns=[col1, col2])
    if len(df[col1].unique()) < len(df[col2].unique()):
        col1, col2 = col2, col1
    file_name = os.path.join(path, f"{col1}-{col2}-bar-plot.png")
    bar_plot(df, col1, hue=col2, file_name=file_name)

    file_name = os.path.join(path, f"{col1}-{col2}-heatmap.png")
    heatmap(pd.crosstab(df[col1], df[col2]), file_name=file_name)


@dask.delayed
def plot_category_category_minimal(input_file, col1, col2, path):
    """
    Create only heatmap for two categorical variables (delayed, minimal version).

    Parameters
    ----------
    input_file : str
        Path to parquet file
    col1 : str
        First column name
    col2 : str
        Second column name
    path : str
        Output directory path
    """
    df = pd.read_parquet(input_file, columns=[col1, col2])
    if len(df[col1].unique()) < len(df[col2].unique()):
        col1, col2 = col2, col1
    file_name = os.path.join(path, f"{col1}-{col2}-heatmap.png")
    heatmap(pd.crosstab(df[col1], df[col2]), file_name=file_name)


def plot_category_category_minimal_sync(input_file, col1, col2, path):
    """
    Create only heatmap for two categorical variables (synchronous, minimal version).

    Parameters
    ----------
    input_file : str
        Path to parquet file
    col1 : str
        First column name
    col2 : str
        Second column name
    path : str
        Output directory path
    """
    df = pd.read_parquet(input_file, columns=[col1, col2])
    if len(df[col1].unique()) < len(df[col2].unique()):
        col1, col2 = col2, col1
    file_name = os.path.join(path, f"{col1}-{col2}-heatmap.png")
    heatmap(pd.crosstab(df[col1], df[col2]), file_name=file_name)


@dask.delayed
def plot_numeric_numeric(input_file, col1, col2, path):
    """
    Create scatter plot for two numeric variables (delayed).

    Parameters
    ----------
    input_file : str
        Path to parquet file
    col1 : str
        First column name
    col2 : str
        Second column name
    path : str
        Output directory path
    """
    df = pd.read_parquet(input_file, columns=[col1, col2])
    file_name = os.path.join(path, f"{col1}-{col2}-scatter-plot.png")
    scatter_plot(df, col1, col2, file_name=file_name)


def plot_numeric_numeric_sync(input_file, col1, col2, path):
    """
    Create scatter plot for two numeric variables (synchronous).

    Parameters
    ----------
    input_file : str
        Path to parquet file
    col1 : str
        First column name
    col2 : str
        Second column name
    path : str
        Output directory path
    """
    df = pd.read_parquet(input_file, columns=[col1, col2])
    file_name = os.path.join(path, f"{col1}-{col2}-scatter-plot.png")
    scatter_plot(df, col1, col2, file_name=file_name)


@dask.delayed
def plot_category_numeric(input_file, category_col, numeric_col, path):
    """
    Create four plots for category vs numeric relationship (delayed).

    Parameters
    ----------
    input_file : str
        Path to parquet file
    category_col : str
        Categorical column name
    numeric_col : str
        Numeric column name
    path : str
        Output directory path
    """
    df = pd.read_parquet(input_file, columns=[category_col, numeric_col])
    _, axes = plt.subplots(2, 2, sharex="col", sharey="row", figsize=(8, 6))
    axes = list(chain.from_iterable(axes))
    file_name = os.path.join(path, f"{category_col}-{numeric_col}-plot.png")
    bar_box_violin_dot_plots(df, category_col, numeric_col, axes, file_name=file_name)


def plot_category_numeric_sync(input_file, category_col, numeric_col, path):
    """
    Create four plots for category vs numeric relationship (synchronous).

    Parameters
    ----------
    input_file : str
        Path to parquet file
    category_col : str
        Categorical column name
    numeric_col : str
        Numeric column name
    path : str
        Output directory path
    """
    df = pd.read_parquet(input_file, columns=[category_col, numeric_col])
    _, axes = plt.subplots(2, 2, sharex="col", sharey="row", figsize=(8, 6))
    axes = list(chain.from_iterable(axes))
    file_name = os.path.join(path, f"{category_col}-{numeric_col}-plot.png")
    bar_box_violin_dot_plots(df, category_col, numeric_col, axes, file_name=file_name)


@dask.delayed
def plot_category_numeric_minimal(input_file, category_col, numeric_col, path):
    """
    Create box and violin plots for category vs numeric (delayed, minimal version).

    Parameters
    ----------
    input_file : str
        Path to parquet file
    category_col : str
        Categorical column name
    numeric_col : str
        Numeric column name
    path : str
        Output directory path
    """
    df = pd.read_parquet(input_file, columns=[category_col, numeric_col])
    _, axes = plt.subplots(1, 2, figsize=(8, 4))
    file_name = os.path.join(path, f"{category_col}-{numeric_col}-minimal-plot.png")
    box_violin_plots(df, category_col, numeric_col, axes, file_name=file_name)


def plot_category_numeric_minimal_sync(input_file, category_col, numeric_col, path):
    """
    Create box and violin plots for category vs numeric (synchronous, minimal version).

    Parameters
    ----------
    input_file : str
        Path to parquet file
    category_col : str
        Categorical column name
    numeric_col : str
        Numeric column name
    path : str
        Output directory path
    """
    df = pd.read_parquet(input_file, columns=[category_col, numeric_col])
    _, axes = plt.subplots(1, 2, figsize=(8, 4))
    file_name = os.path.join(path, f"{category_col}-{numeric_col}-minimal-plot.png")
    box_violin_plots(df, category_col, numeric_col, axes, file_name=file_name)
