"""
Time series plotting functions.

This module handles time series visualizations.
"""

import os

import dask
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..core.utils import ignore_if_exist_or_save

__all__ = [
    "plot_single_timeseries",
    "plot_single_timeseries_sync",
    "plot_timeseries_numeric",
    "plot_timeseries_numeric_sync",
    "plot_timeseries_timeseries",
    "plot_timeseries_timeseries_sync",
    "plot_timeseries_category_numeric",
    "plot_timeseries_category_numeric_sync",
    "time_series_line_plot",
    "time_series_numeric_plot",
    "time_series_category_plot",
    "multiple_time_series_plot",
]


@ignore_if_exist_or_save
def time_series_line_plot(data, time_col, file_name=None):
    """
    Create a timeline plot showing the distribution of datetime values.

    Parameters
    ----------
    data : pandas.DataFrame
        Data containing the time column
    time_col : str
        Time series column name
    file_name : str, optional
        Output file path
    """
    plt.figure(figsize=(12, 6))
    # Plot datetime index as a timeline
    plt.plot(data[time_col], range(len(data)), linewidth=1.5)
    plt.xlabel("Time")
    plt.ylabel("Observation Index")
    plt.title(f"Timeline: {time_col}")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    sns.despine()


@ignore_if_exist_or_save
def time_series_numeric_plot(data, time_col, numeric_col, file_name=None):
    """
    Create a time series plot with numeric values on y-axis.

    Parameters
    ----------
    data : pandas.DataFrame
        Data containing the columns
    time_col : str
        Time series column name
    numeric_col : str
        Numeric column name
    file_name : str, optional
        Output file path
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data[time_col], data[numeric_col], linewidth=1.5, marker="o", markersize=2)
    plt.xlabel(time_col)
    plt.ylabel(numeric_col)
    plt.title(f"{numeric_col} over {time_col}")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    sns.despine()


@ignore_if_exist_or_save
def time_series_category_plot(
    data, time_col, numeric_col, category_col, file_name=None
):
    """
    Create a time series plot grouped by category.

    Parameters
    ----------
    data : pandas.DataFrame
        Data containing the columns
    time_col : str
        Time series column name
    numeric_col : str
        Numeric column name
    category_col : str
        Categorical column name
    file_name : str, optional
        Output file path
    """
    plt.figure(figsize=(12, 6))
    for category in data[category_col].unique():
        subset = data[data[category_col] == category]
        plt.plot(
            subset[time_col],
            subset[numeric_col],
            linewidth=1.5,
            marker="o",
            markersize=2,
            label=category,
            alpha=0.7,
        )
    plt.xlabel(time_col)
    plt.ylabel(numeric_col)
    plt.title(f"{numeric_col} over {time_col} by {category_col}")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    sns.despine()


@ignore_if_exist_or_save
def multiple_time_series_plot(data, time_col, numeric_cols, file_name=None):
    """
    Create an overlay plot for multiple time series.

    Parameters
    ----------
    data : pandas.DataFrame
        Data containing the columns
    time_col : str
        Time series column name
    numeric_cols : list of str
        List of numeric column names
    file_name : str, optional
        Output file path
    """
    plt.figure(figsize=(12, 6))
    for col in numeric_cols:
        plt.plot(
            data[time_col],
            data[col],
            linewidth=1.5,
            marker="o",
            markersize=2,
            label=col,
            alpha=0.7,
        )
    plt.xlabel(time_col)
    plt.ylabel("Value")
    plt.title(f"Multiple Time Series over {time_col}")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    sns.despine()


@dask.delayed
def plot_single_timeseries(input_file, time_col, path):
    """
    Plot a single time series column (delayed).

    Parameters
    ----------
    input_file : str
        Path to parquet file
    time_col : str
        Time series column name
    path : str
        Output directory path
    """
    df = pd.read_parquet(input_file, columns=[time_col])
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    file_name = os.path.join(path, f"{time_col}-timeseries-plot.png")
    time_series_line_plot(df, time_col, file_name=file_name)


def plot_single_timeseries_sync(input_file, time_col, path):
    """
    Plot a single time series column (synchronous).

    Parameters
    ----------
    input_file : str
        Path to parquet file
    time_col : str
        Time series column name
    path : str
        Output directory path
    """
    df = pd.read_parquet(input_file, columns=[time_col])
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    file_name = os.path.join(path, f"{time_col}-timeseries-plot.png")
    time_series_line_plot(df, time_col, file_name=file_name)


@dask.delayed
def plot_timeseries_numeric(input_file, time_col, numeric_col, path):
    """
    Plot numeric values over time (delayed).

    Parameters
    ----------
    input_file : str
        Path to parquet file
    time_col : str
        Time series column name
    numeric_col : str
        Numeric column name
    path : str
        Output directory path
    """
    df = pd.read_parquet(input_file, columns=[time_col, numeric_col])
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    file_name = os.path.join(path, f"{time_col}-{numeric_col}-timeseries-plot.png")
    time_series_numeric_plot(df, time_col, numeric_col, file_name=file_name)


def plot_timeseries_numeric_sync(input_file, time_col, numeric_col, path):
    """
    Plot numeric values over time (synchronous).

    Parameters
    ----------
    input_file : str
        Path to parquet file
    time_col : str
        Time series column name
    numeric_col : str
        Numeric column name
    path : str
        Output directory path
    """
    df = pd.read_parquet(input_file, columns=[time_col, numeric_col])
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    file_name = os.path.join(path, f"{time_col}-{numeric_col}-timeseries-plot.png")
    time_series_numeric_plot(df, time_col, numeric_col, file_name=file_name)


@dask.delayed
def plot_timeseries_timeseries(input_file, time_col1, time_col2, path):
    """
    Plot two datetime series showing their temporal coverage and overlap (delayed).

    Parameters
    ----------
    input_file : str
        Path to parquet file
    time_col1 : str
        First time series column name
    time_col2 : str
        Second time series column name
    path : str
        Output directory path
    """
    df = pd.read_parquet(input_file, columns=[time_col1, time_col2])
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[time_col1]):
        df[time_col1] = pd.to_datetime(df[time_col1])
    if not pd.api.types.is_datetime64_any_dtype(df[time_col2]):
        df[time_col2] = pd.to_datetime(df[time_col2])

    # Create a temporal coverage comparison plot
    file_name = os.path.join(path, f"{time_col1}-{time_col2}-timeseries-comparison.png")

    # Create figure with two subplots showing both timelines
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot first time series
    ax1.plot(df[time_col1], range(len(df)), linewidth=1.5, marker="o", markersize=2)
    ax1.set_ylabel("Observation Index")
    ax1.set_title(f"Timeline: {time_col1}")
    ax1.grid(True, alpha=0.3)

    # Plot second time series
    ax2.plot(df[time_col2], range(len(df)), linewidth=1.5, marker="o", markersize=2)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Observation Index")
    ax2.set_title(f"Timeline: {time_col2}")
    ax2.grid(True, alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()

    if file_name:
        plt.savefig(file_name, dpi=120)
    plt.close("all")


def plot_timeseries_timeseries_sync(input_file, time_col1, time_col2, path):
    """
    Plot two datetime series showing their temporal coverage and overlap (synchronous).

    Parameters
    ----------
    input_file : str
        Path to parquet file
    time_col1 : str
        First time series column name
    time_col2 : str
        Second time series column name
    path : str
        Output directory path
    """
    df = pd.read_parquet(input_file, columns=[time_col1, time_col2])
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[time_col1]):
        df[time_col1] = pd.to_datetime(df[time_col1])
    if not pd.api.types.is_datetime64_any_dtype(df[time_col2]):
        df[time_col2] = pd.to_datetime(df[time_col2])

    # Create a temporal coverage comparison plot
    file_name = os.path.join(path, f"{time_col1}-{time_col2}-timeseries-comparison.png")

    # Create figure with two subplots showing both timelines
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot first time series
    ax1.plot(df[time_col1], range(len(df)), linewidth=1.5, marker="o", markersize=2)
    ax1.set_ylabel("Observation Index")
    ax1.set_title(f"Timeline: {time_col1}")
    ax1.grid(True, alpha=0.3)

    # Plot second time series
    ax2.plot(df[time_col2], range(len(df)), linewidth=1.5, marker="o", markersize=2)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Observation Index")
    ax2.set_title(f"Timeline: {time_col2}")
    ax2.grid(True, alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()

    if file_name:
        plt.savefig(file_name, dpi=120)
    plt.close("all")


@dask.delayed
def plot_timeseries_category_numeric(
    input_file, time_col, category_col, numeric_col, path
):
    """
    Plot numeric values over time grouped by category (delayed).

    Parameters
    ----------
    input_file : str
        Path to parquet file
    time_col : str
        Time series column name
    category_col : str
        Categorical column name
    numeric_col : str
        Numeric column name
    path : str
        Output directory path
    """
    df = pd.read_parquet(input_file, columns=[time_col, category_col, numeric_col])
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    file_name = os.path.join(
        path, f"{time_col}-{numeric_col}-by-{category_col}-timeseries.png"
    )
    time_series_category_plot(
        df, time_col, numeric_col, category_col, file_name=file_name
    )


def plot_timeseries_category_numeric_sync(
    input_file, time_col, category_col, numeric_col, path
):
    """
    Plot numeric values over time grouped by category (synchronous).

    Parameters
    ----------
    input_file : str
        Path to parquet file
    time_col : str
        Time series column name
    category_col : str
        Categorical column name
    numeric_col : str
        Numeric column name
    path : str
        Output directory path
    """
    df = pd.read_parquet(input_file, columns=[time_col, category_col, numeric_col])
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    file_name = os.path.join(
        path, f"{time_col}-{numeric_col}-by-{category_col}-timeseries.png"
    )
    time_series_category_plot(
        df, time_col, numeric_col, category_col, file_name=file_name
    )
