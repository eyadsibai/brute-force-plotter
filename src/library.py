"""
Library interface for Brute Force Plotter.

This module provides the programmatic Python API for creating plots.
"""

import logging
import os
import tempfile

import dask
import matplotlib

from .cli.orchestration import create_plots
from .core import config
from .core.config import (
    DEFAULT_MAX_ROWS,
    DEFAULT_SAMPLE_SIZE,
)
from .core.data_types import infer_dtypes
from .core.utils import check_and_sample_large_dataset
from .stats.export import export_statistical_summaries

logger = logging.getLogger(__name__)

__all__ = ["plot"]


def plot(
    data,
    dtypes=None,
    output_path=None,
    show=False,
    use_dask=False,
    n_workers=4,
    export_stats=False,
    minimal=False,
    max_rows=DEFAULT_MAX_ROWS,
    sample_size=DEFAULT_SAMPLE_SIZE,
    no_sample=False,
    target=None,
):
    """
    Create plots from a pandas DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        The data to plot
    dtypes : dict, optional
        Dictionary mapping column names to data types:
        - 'n' for numeric
        - 'c' for category
        - 'g' for geocoordinate (latitude/longitude)
        - 't' for time series (datetime)
        - 'i' for ignore
        If None, data types will be automatically inferred.
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
    export_stats : bool, optional
        If True, export statistical summaries to CSV files. Defaults to False.
    minimal : bool, optional
        If True, generate minimal set of plots (reduces redundant visualizations).
        Defaults to False.
    max_rows : int, optional
        Maximum number of rows before sampling is applied. Defaults to 100,000.
    sample_size : int, optional
        Number of rows to sample for large datasets. Defaults to 50,000.
    no_sample : bool, optional
        If True, disable sampling for large datasets. Defaults to False.
    target : str, optional
        Name of the target variable for highlighting in plots. When specified,
        plots will use the target variable for coloring/grouping where appropriate.
        Useful for classification/regression tasks. Defaults to None.

    Returns
    -------
    tuple
        A tuple of (output_path, dtypes) where output_path is the directory where
        plots were saved and dtypes is the dictionary of data types used for plotting.

    Examples
    --------
    >>> import pandas as pd
    >>> import brute_force_plotter as bfp
    >>>
    >>> data = pd.read_csv('data.csv')
    >>>
    >>> # Automatic type inference
    >>> output_path, dtypes = bfp.plot(data)
    >>> print(f"Inferred types: {dtypes}")
    >>>
    >>> # Manual type specification
    >>> dtypes = {'age': 'n', 'gender': 'c', 'id': 'i'}
    >>> output_path, dtypes_used = bfp.plot(data, dtypes, output_path='./plots')
    >>>
    >>> # Show plots interactively
    >>> bfp.plot(data, dtypes, show=True)
    >>>
    >>> # Export statistical summaries
    >>> bfp.plot(data, dtypes, output_path='./plots', export_stats=True)
    >>>
    >>> # Create maps from geocoordinate data
    >>> geo_data = pd.read_csv('cities.csv')
    >>> geo_dtypes = {'latitude': 'g', 'longitude': 'g', 'category': 'c'}
    >>> bfp.plot(geo_data, geo_dtypes, output_path='./maps')
    >>> # Generate minimal set of plots
    >>> bfp.plot(data, dtypes, output_path='./plots', minimal=True)
    >>> # Handle large datasets with sampling
    >>> bfp.plot(data, dtypes, output_path='./plots', max_rows=50000, sample_size=25000)
    >>> # Use target variable for highlighting
    >>> dtypes = {'age': 'n', 'gender': 'c', 'survived': 'c'}
    >>> bfp.plot(data, dtypes, output_path='./plots', target='survived')
    """
    global _show_plots, _save_plots

    # Infer dtypes if not provided
    if dtypes is None:
        logger.info("No dtypes provided, automatically inferring data types...")
        dtypes = infer_dtypes(data)

        # Log inferred types
        logger.info("Inferred data types:")
        for col, dtype in sorted(dtypes.items()):
            dtype_name = {"n": "numeric", "c": "categorical", "i": "ignore"}[dtype]
            logger.info(f"  {col}: {dtype_name}")

    # Set matplotlib backend based on show parameter
    if show:
        try:
            matplotlib.use("TkAgg")
        except Exception:
            try:
                matplotlib.use("Qt5Agg")
            except Exception:
                logger.warning("Could not set interactive backend, using default")
    else:
        matplotlib.use("agg")

    config._show_plots = show
    config._save_plots = not show or output_path is not None

    # Set target variable if provided
    if target:
        if target not in dtypes:
            logger.warning(
                f"Target variable '{target}' not found in data types. "
                "Target highlighting will be disabled."
            )
            config.set_target_variable(None)
        elif target not in data.columns:
            logger.warning(
                f"Target variable '{target}' not found in data. "
                "Target highlighting will be disabled."
            )
            config.set_target_variable(None)
        else:
            logger.info(f"Using '{target}' as target variable for plot highlighting")
            config.set_target_variable(target)
    else:
        config.set_target_variable(None)

    # Check and sample large datasets if necessary
    original_data = data
    data, was_sampled = check_and_sample_large_dataset(
        data, max_rows=max_rows, sample_size=sample_size, no_sample=no_sample
    )

    if was_sampled:
        logger.info(
            "Note: Plots are generated from sampled data. "
            "Statistical summaries (if enabled) will use the full dataset."
        )

    # Determine output path
    if output_path is None and not show:
        output_path = tempfile.mkdtemp(prefix="brute_force_plotter_")
        logger.info(
            f"No output path specified, using temporary directory: {output_path}"
        )
    elif output_path is None and show:
        # Create a temp directory anyway for potential saving
        output_path = tempfile.mkdtemp(prefix="brute_force_plotter_")

    # Create temporary parquet file for efficient processing
    temp_parquet = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".parq", delete=False) as tmp:
            temp_parquet = tmp.name
        data.to_parquet(temp_parquet)

        if use_dask:
            from dask.distributed import Client, LocalCluster

            cluster = LocalCluster(n_workers=n_workers)
            client = Client(cluster)
            try:
                plots = create_plots(temp_parquet, dtypes, output_path, minimal=minimal)
                dask.compute(*plots)
            finally:
                client.close()
                cluster.close()
        else:
            plots = create_plots(
                temp_parquet, dtypes, output_path, use_dask=False, minimal=minimal
            )
            if plots:
                for plot_task in plots:
                    plot_task.compute()

        # Export statistical summaries if requested
        # If data was sampled, use full dataset for accurate statistics
        if export_stats:
            if was_sampled:
                logger.info("Using full dataset for statistical summary export...")
                temp_full_parquet = None
                try:
                    with tempfile.NamedTemporaryFile(
                        suffix=".parq", delete=False
                    ) as tmp:
                        temp_full_parquet = tmp.name
                    original_data.to_parquet(temp_full_parquet)
                    export_statistical_summaries(temp_full_parquet, dtypes, output_path)
                finally:
                    if temp_full_parquet and os.path.exists(temp_full_parquet):
                        os.remove(temp_full_parquet)
            else:
                export_statistical_summaries(temp_parquet, dtypes, output_path)
    finally:
        # Clean up temporary parquet file
        if temp_parquet and os.path.exists(temp_parquet):
            os.remove(temp_parquet)

    # Always return both output_path and dtypes for consistency
    return output_path, dtypes
