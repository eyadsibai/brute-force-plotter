"""
Summary plotting functions (correlation matrices, missing values).

This module handles summary visualizations.
"""

import logging
import os

import dask
import pandas as pd

from .base import correlation_heatmap, missing_plot

logger = logging.getLogger(__name__)

__all__ = [
    "plot_correlation_matrix",
    "plot_correlation_matrix_minimal",
    "plot_missing_values",
]


@dask.delayed
def plot_correlation_matrix(input_file, dtypes, path):
    """
    Generate correlation matrix plots (Pearson and Spearman) for numeric columns.

    Parameters
    ----------
    input_file : str
        Path to parquet file
    dtypes : dict
        Dictionary mapping column names to data types
    path : str
        Output directory path
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
def plot_correlation_matrix_minimal(input_file, dtypes, path):
    """
    Generate only Spearman correlation matrix (minimal version).

    Parameters
    ----------
    input_file : str
        Path to parquet file
    dtypes : dict
        Dictionary mapping column names to data types
    path : str
        Output directory path
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

    # Only Spearman correlation (more robust to outliers)
    spearman_corr = df.corr(method="spearman")
    file_name = os.path.join(path, "correlation-spearman.png")
    correlation_heatmap(
        spearman_corr, file_name=file_name, title="Spearman Correlation Matrix"
    )


@dask.delayed
def plot_missing_values(input_file, dtypes, path):
    """
    Generate missing values heatmap and analysis.

    Parameters
    ----------
    input_file : str
        Path to parquet file
    dtypes : dict
        Dictionary mapping column names to data types
    path : str
        Output directory path
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
