#!/usr/bin/env python

"""
Brute Force Plotter
-----------------
Library and Command Line Interface

"""

import errno
from itertools import chain, combinations
import json
import logging
import math
import os
import tempfile

import click
import dask
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


ignore = set()
skip_existing_plots = True  # Global flag for skipping existing plots

# Global configuration
_show_plots = False
_save_plots = True

# Large dataset configuration
DEFAULT_MAX_ROWS = 100000  # Default threshold for sampling
DEFAULT_SAMPLE_SIZE = 50000  # Default sample size for large datasets

sns.set_style("darkgrid")
sns.set_context("paper")

sns.set(rc={"figure.figsize": (8, 6)})


def infer_dtypes(data, max_categorical_ratio=0.05, max_categorical_unique=50):
    """
    Automatically infer data types for columns in a DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        The data to infer types for
    max_categorical_ratio : float, optional
        Maximum ratio of unique values to total values for a column to be
        considered categorical. Default is 0.05 (5%).
    max_categorical_unique : int, optional
        Maximum number of unique values for a column to be considered
        categorical. Default is 50.

    Returns
    -------
    dict
        Dictionary mapping column names to inferred data types:
        - 'n' for numeric
        - 'c' for category
        - 'i' for ignore (e.g., unique identifiers, text fields)

    Notes
    -----
    The inference logic:
    - Numeric dtypes (int, float) with few unique values -> categorical
    - Numeric dtypes with many unique values -> numeric
    - Object/string dtypes with few unique values -> categorical
    - Object/string dtypes with many unique values -> ignore
    - Boolean dtypes -> categorical
    - Datetime dtypes -> ignore (for now)
    """
    inferred_dtypes = {}

    for col in data.columns:
        dtype = data[col].dtype
        n_unique = data[col].nunique()
        n_total = len(data[col].dropna())
        unique_ratio = n_unique / n_total if n_total > 0 else 0

        # Boolean columns -> categorical
        if dtype == "bool":
            inferred_dtypes[col] = "c"

        # Numeric columns (int, float)
        elif pd.api.types.is_numeric_dtype(dtype):
            # Check if it's likely an ID column (all unique or nearly all unique)
            if unique_ratio > 0.95 and n_unique > max_categorical_unique:
                inferred_dtypes[col] = "i"
            # Check if it should be categorical
            # Use AND for the conditions: both must be true
            elif (
                n_unique <= max_categorical_unique
                and unique_ratio <= max_categorical_ratio
            ):
                inferred_dtypes[col] = "c"
            else:
                inferred_dtypes[col] = "n"

        # Datetime columns -> ignore (for now, could be enhanced later)
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            inferred_dtypes[col] = "i"

        # Object/string columns
        elif dtype == "object" or pd.api.types.is_string_dtype(dtype):
            # Check if all values are unique (likely an ID or name column)
            if unique_ratio > 0.95:
                inferred_dtypes[col] = "i"
            # Check if it has few unique values (categorical)
            elif n_unique <= max_categorical_unique:
                inferred_dtypes[col] = "c"
            else:
                # Too many unique text values -> ignore
                inferred_dtypes[col] = "i"

        # Other types -> ignore
        else:
            inferred_dtypes[col] = "i"

    return inferred_dtypes


@click.command()
@click.argument("input_file")
@click.argument("dtypes", required=False)
@click.argument("output_path", required=False)
@click.option(
    "--skip-existing",
    is_flag=True,
    default=True,
    help="Skip generating plots that already exist",
)
@click.option(
    "--theme",
    type=click.Choice(["darkgrid", "whitegrid", "dark", "white", "ticks"]),
    default="darkgrid",
    help="Seaborn plot style theme",
)
@click.option(
    "--n-workers",
    type=int,
    default=4,
    help="Number of parallel workers for plot generation",
)
@click.option(
    "--export-stats",
    is_flag=True,
    default=False,
    help="Export statistical summary to CSV",
)
@click.option(
    "--infer-dtypes",
    "infer_types",  # Use a different parameter name
    is_flag=True,
    default=False,
    help="Automatically infer data types from the data",
)
@click.option(
    "--save-dtypes",
    type=click.Path(),
    default=None,
    help="Save inferred or used dtypes to a JSON file",
)
@click.option(
    "--max-rows",
    type=int,
    default=DEFAULT_MAX_ROWS,
    help="Maximum number of rows before sampling is applied",
)
@click.option(
    "--sample-size",
    type=int,
    default=DEFAULT_SAMPLE_SIZE,
    help="Number of rows to sample for large datasets",
)
@click.option(
    "--no-sample",
    is_flag=True,
    default=False,
    help="Disable sampling for large datasets (may cause memory issues)",
)
def main(
    input_file,
    dtypes,
    output_path,
    skip_existing,
    theme,
    n_workers,
    export_stats,
    infer_types,
    save_dtypes,
    max_rows,
    sample_size,
    no_sample,
):
    """Create Plots From data in input

    INPUT_FILE: Path to CSV file containing the data

    DTYPES: (Optional) Path to JSON file with data types. If not provided,
    data types will be automatically inferred.

    OUTPUT_PATH: (Optional) Directory for output plots. Defaults to './output'
    """
    # Set matplotlib backend for CLI (non-interactive)
    matplotlib.use("agg")

    from dask.distributed import Client, LocalCluster

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Set global skip_existing flag
    global skip_existing_plots
    skip_existing_plots = skip_existing

    # Apply theme
    sns.set_style(theme)

    # Set default output path if not provided
    if output_path is None:
        output_path = "./output"
        logger.info(f"No output path specified, using: {output_path}")

    # Load or infer data types
    if dtypes is None or infer_types:
        logger.info("Inferring data types from the data...")
        # Load the full dataset to infer types
        data_full = pd.read_csv(input_file)
        data_types = infer_dtypes(data_full)

        # Log inferred types
        logger.info("Inferred data types:")
        for col, dtype in sorted(data_types.items()):
            dtype_name = {"n": "numeric", "c": "categorical", "i": "ignore"}[dtype]
            logger.info(f"  {col}: {dtype_name}")

        # Save dtypes if requested
        if save_dtypes:
            with open(save_dtypes, "w") as f:
                json.dump(data_types, f, indent=2)
            logger.info(f"Saved inferred data types to: {save_dtypes}")
    else:
        # Load dtypes from JSON file
        logger.info(f"Loading data types from: {dtypes}")
        with open(dtypes) as f:
            data_types = json.load(f)

        # Save dtypes if requested (even when loaded from file)
        if save_dtypes and save_dtypes != dtypes:
            with open(save_dtypes, "w") as f:
                json.dump(data_types, f, indent=2)
            logger.info(f"Saved data types to: {save_dtypes}")

    # Filter out columns with dtype "i" (ignore)
    columns_to_load = [col for col, dtype in data_types.items() if dtype != "i"]

    # Only load non-ignored columns from CSV
    data = pd.read_csv(input_file, usecols=columns_to_load)

    # Check and sample large datasets if necessary
    data, was_sampled = check_and_sample_large_dataset(
        data, max_rows=max_rows, sample_size=sample_size, no_sample=no_sample
    )

    if was_sampled:
        logger.info(
            "Note: Plots are generated from sampled data. "
            "Statistical summaries (--export-stats) will still use the full dataset."
        )

    new_file_name = f"{input_file}.parq"
    data.to_parquet(new_file_name)

    cluster = LocalCluster(n_workers=n_workers, silence_logs=logging.WARNING)
    _client = Client(cluster)  # noqa: F841 - Client instance needed to enable dask cluster

    plots = create_plots(new_file_name, data_types, output_path)
    dask.compute(*plots)

    # Export statistical summaries if requested
    # Note: For stats, we reload the full dataset to ensure accuracy
    if export_stats:
        if was_sampled:
            # Reload full dataset for statistics
            logger.info("Loading full dataset for statistical summary export...")
            full_data = pd.read_csv(input_file, usecols=columns_to_load)
            full_parquet = f"{input_file}.full.parq"
            full_data.to_parquet(full_parquet)
            export_statistical_summaries(full_parquet, data_types, output_path)
            # Clean up full dataset parquet
            if os.path.exists(full_parquet):
                os.remove(full_parquet)
        else:
            export_statistical_summaries(new_file_name, data_types, output_path)


def plot(
    data,
    dtypes=None,
    output_path=None,
    show=False,
    use_dask=False,
    n_workers=4,
    export_stats=False,
    max_rows=DEFAULT_MAX_ROWS,
    sample_size=DEFAULT_SAMPLE_SIZE,
    no_sample=False,
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
    max_rows : int, optional
        Maximum number of rows before sampling is applied. Defaults to 100,000.
    sample_size : int, optional
        Number of rows to sample for large datasets. Defaults to 50,000.
    no_sample : bool, optional
        If True, disable sampling for large datasets. Defaults to False.

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
    >>> # Handle large datasets with sampling
    >>> bfp.plot(data, dtypes, output_path='./plots', max_rows=50000, sample_size=25000)
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

    _show_plots = show
    _save_plots = not show or output_path is not None

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
                plots = create_plots(temp_parquet, dtypes, output_path)
                dask.compute(*plots)
            finally:
                client.close()
                cluster.close()
        else:
            plots = create_plots(temp_parquet, dtypes, output_path, use_dask=False)
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


def check_and_sample_large_dataset(
    data, max_rows=DEFAULT_MAX_ROWS, sample_size=DEFAULT_SAMPLE_SIZE, no_sample=False
):
    """
    Check if dataset is large and sample if necessary.

    Parameters
    ----------
    data : pandas.DataFrame
        The input data
    max_rows : int
        Maximum number of rows before sampling is applied
    sample_size : int
        Number of rows to sample for large datasets
    no_sample : bool
        If True, disable sampling even for large datasets

    Returns
    -------
    pandas.DataFrame
        Original or sampled DataFrame
    bool
        True if data was sampled, False otherwise
    """
    n_rows = len(data)

    # Check if sampling is needed
    if no_sample or n_rows <= max_rows:
        return data, False

    # Log warning about large dataset
    logger.warning(
        f"Dataset has {n_rows:,} rows, which exceeds the threshold of {max_rows:,} rows. "
        f"Sampling {sample_size:,} rows for visualization to improve performance."
    )
    logger.info(
        "To disable sampling, use --no-sample flag (may cause memory issues). "
        "To adjust sample size, use --sample-size parameter."
    )

    # Calculate memory usage estimate
    memory_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
    logger.info(f"Original dataset memory usage: {memory_mb:.2f} MB")

    # Perform stratified sampling if possible, otherwise random sampling
    sampled_data = data.sample(n=min(sample_size, n_rows), random_state=42)

    # Log result
    sampled_memory_mb = sampled_data.memory_usage(deep=True).sum() / 1024 / 1024
    logger.info(
        f"Sampled dataset: {len(sampled_data):,} rows, {sampled_memory_mb:.2f} MB"
    )

    return sampled_data, True


def ignore_if_exist_or_save(func):
    """Decorator to handle plot saving/showing logic"""

    def wrapper(*args, **kwargs):
        file_name = kwargs.get("file_name")

        # If saving is disabled and showing is enabled, just create and show
        if _show_plots and not _save_plots:
            func(*args, **kwargs)
            plt.gcf().set_tight_layout(True)
            plt.show()
            plt.close("all")
        # If file exists and we're saving, skip
        elif file_name and os.path.isfile(file_name) and _save_plots:
            plt.close("all")
        # Otherwise, create the plot
        else:
            func(*args, **kwargs)
            plt.gcf().set_tight_layout(True)

            # Save if we should save
            if _save_plots and file_name:
                plt.gcf().savefig(file_name, dpi=120)

            # Show if we should show
            if _show_plots:
                plt.show()

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


def plot_single_numeric_sync(input_file, col, path):
    """Non-delayed version for synchronous execution"""
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


def plot_single_category_sync(input_file, col, path):
    """Non-delayed version for synchronous execution"""
    df = pd.read_parquet(input_file, columns=[col])
    value_counts = df[col].value_counts(dropna=False)
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


def plot_category_category_sync(input_file, col1, col2, path):
    """Non-delayed version for synchronous execution"""
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


def plot_numeric_numeric_sync(input_file, col1, col2, path):
    """Non-delayed version for synchronous execution"""
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


def plot_category_numeric_sync(input_file, category_col, numeric_col, path):
    """Non-delayed version for synchronous execution"""
    df = pd.read_parquet(input_file, columns=[category_col, numeric_col])
    f, axes = plt.subplots(2, 2, sharex="col", sharey="row", figsize=(8, 6))
    axes = list(chain.from_iterable(axes))
    file_name = os.path.join(path, f"{category_col}-{numeric_col}-plot.png")
    bar_box_violin_dot_plots(df, category_col, numeric_col, axes, file_name=file_name)


@dask.delayed
def plot_single_timeseries(input_file, time_col, path):
    """Plot a single time series column"""
    df = pd.read_parquet(input_file, columns=[time_col])
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    file_name = os.path.join(path, f"{time_col}-timeseries-plot.png")
    time_series_line_plot(df, time_col, file_name=file_name)


def plot_single_timeseries_sync(input_file, time_col, path):
    """Non-delayed version for synchronous execution"""
    df = pd.read_parquet(input_file, columns=[time_col])
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    file_name = os.path.join(path, f"{time_col}-timeseries-plot.png")
    time_series_line_plot(df, time_col, file_name=file_name)


@dask.delayed
def plot_timeseries_numeric(input_file, time_col, numeric_col, path):
    """Plot numeric values over time"""
    df = pd.read_parquet(input_file, columns=[time_col, numeric_col])
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    file_name = os.path.join(path, f"{time_col}-{numeric_col}-timeseries-plot.png")
    time_series_numeric_plot(df, time_col, numeric_col, file_name=file_name)


def plot_timeseries_numeric_sync(input_file, time_col, numeric_col, path):
    """Non-delayed version for synchronous execution"""
    df = pd.read_parquet(input_file, columns=[time_col, numeric_col])
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col])
    file_name = os.path.join(path, f"{time_col}-{numeric_col}-timeseries-plot.png")
    time_series_numeric_plot(df, time_col, numeric_col, file_name=file_name)


@dask.delayed
def plot_timeseries_timeseries(input_file, time_col1, time_col2, path):
    """
    Plot two datetime series showing their temporal coverage and overlap.
    Creates a timeline visualization showing when each time series has data.
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
    """Non-delayed version for synchronous execution"""
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
    """Plot numeric values over time grouped by category"""
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
    """Non-delayed version for synchronous execution"""
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


def create_plots(input_file, dtypes, output_path, use_dask=True):
    distributions_path, two_d_interactions_path, three_d_interactions_path = (
        _create_directories(output_path)
    )
    plots = []

    # Add summary plots
    logger.info("Adding correlation matrix and missing values plots...")
    plots.append(plot_correlation_matrix(input_file, dtypes, distributions_path))
    plots.append(plot_missing_values(input_file, dtypes, distributions_path))

    for col, dtype in dtypes.items():
        print(col)
        if dtype == "i":
            continue
        if dtype == "n":
            if use_dask:
                plots.append(plot_single_numeric(input_file, col, distributions_path))
            else:
                plot_single_numeric_sync(input_file, col, distributions_path)
        if dtype == "c":
            if use_dask:
                plots.append(plot_single_category(input_file, col, distributions_path))
            else:
                plot_single_category_sync(input_file, col, distributions_path)
        if dtype == "t":
            if use_dask:
                plots.append(
                    plot_single_timeseries(input_file, col, distributions_path)
                )
            else:
                plot_single_timeseries_sync(input_file, col, distributions_path)

    for (col1, dtype1), (col2, dtype2) in combinations(dtypes.items(), 2):
        print(col1, col2)
        if dtype1 == "i" or dtype2 == "i":
            continue
        if any(col in ignore for col in [col1, col2]):
            continue
        if dtype1 == dtype2 == "n":
            if use_dask:
                plots.append(
                    plot_numeric_numeric(
                        input_file, col1, col2, two_d_interactions_path
                    )
                )
            else:
                plot_numeric_numeric_sync(
                    input_file, col1, col2, two_d_interactions_path
                )
        if dtype1 == dtype2 == "c":
            if use_dask:
                plots.append(
                    plot_category_category(
                        input_file, col1, col2, two_d_interactions_path
                    )
                )
            else:
                plot_category_category_sync(
                    input_file, col1, col2, two_d_interactions_path
                )
        if dtype1 == "c" and dtype2 == "n":
            if use_dask:
                plots.append(
                    plot_category_numeric(
                        input_file, col1, col2, two_d_interactions_path
                    )
                )
            else:
                plot_category_numeric_sync(
                    input_file, col1, col2, two_d_interactions_path
                )
        if dtype1 == "n" and dtype2 == "c":
            if use_dask:
                plots.append(
                    plot_category_numeric(
                        input_file, col2, col1, two_d_interactions_path
                    )
                )
            else:
                plot_category_numeric_sync(
                    input_file, col2, col1, two_d_interactions_path
                )
        # Time series interactions
        if dtype1 == "t" and dtype2 == "n":
            if use_dask:
                plots.append(
                    plot_timeseries_numeric(
                        input_file, col1, col2, two_d_interactions_path
                    )
                )
            else:
                plot_timeseries_numeric_sync(
                    input_file, col1, col2, two_d_interactions_path
                )
        if dtype1 == "n" and dtype2 == "t":
            if use_dask:
                plots.append(
                    plot_timeseries_numeric(
                        input_file, col2, col1, two_d_interactions_path
                    )
                )
            else:
                plot_timeseries_numeric_sync(
                    input_file, col2, col1, two_d_interactions_path
                )
        if dtype1 == dtype2 == "t":
            if use_dask:
                plots.append(
                    plot_timeseries_timeseries(
                        input_file, col1, col2, two_d_interactions_path
                    )
                )
            else:
                plot_timeseries_timeseries_sync(
                    input_file, col1, col2, two_d_interactions_path
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

    # 3-way interactions: time series + category + numeric
    for (col1, dtype1), (col2, dtype2), (col3, dtype3) in combinations(
        dtypes.items(), 3
    ):
        if dtype1 == "i" or dtype2 == "i" or dtype3 == "i":
            continue
        if any(col in ignore for col in [col1, col2, col3]):
            continue

        # Find time, category, and numeric columns
        time_col = None
        category_col = None
        numeric_col = None

        if dtype1 == "t":
            time_col = col1
        elif dtype2 == "t":
            time_col = col2
        elif dtype3 == "t":
            time_col = col3

        if dtype1 == "c":
            category_col = col1
        elif dtype2 == "c":
            category_col = col2
        elif dtype3 == "c":
            category_col = col3

        if dtype1 == "n":
            numeric_col = col1
        elif dtype2 == "n":
            numeric_col = col2
        elif dtype3 == "n":
            numeric_col = col3

        # Plot if we have time + category + numeric
        if time_col and category_col and numeric_col:
            if use_dask:
                plots.append(
                    plot_timeseries_category_numeric(
                        input_file,
                        time_col,
                        category_col,
                        numeric_col,
                        two_d_interactions_path,
                    )
                )
            else:
                plot_timeseries_category_numeric_sync(
                    input_file,
                    time_col,
                    category_col,
                    numeric_col,
                    two_d_interactions_path,
                )

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
            plt.annotate(
                f"{int(height)}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )


@ignore_if_exist_or_save
def histogram_violin_plots(data, axes, file_name=None):
    # histogram
    sns.histplot(data, ax=axes[0], kde=True)
    sns.violinplot(x=data, ax=axes[1], inner="quartile", density_norm="count")
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
def heatmap(data, file_name=None):
    cmap = "BuGn" if (data.values >= 0).all() else "coolwarm"
    sns.heatmap(data=data, annot=True, fmt="d", cmap=cmap)
    sns.despine(left=True)


@ignore_if_exist_or_save
def correlation_heatmap(data, file_name=None, title="Correlation Matrix"):
    """Create a correlation matrix heatmap"""
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
    """Create a heatmap showing missing values"""
    plt.figure(figsize=(12, 6))
    sns.heatmap(data, cbar=True, yticklabels=False, cmap="viridis")
    plt.title("Missing Values Heatmap")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    sns.despine(left=True)


@ignore_if_exist_or_save
def time_series_line_plot(data, time_col, file_name=None):
    """Create a timeline plot showing the distribution of datetime values"""
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
    """Create a time series plot with numeric values on y-axis"""
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
    """Create a time series plot grouped by category"""
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
    """Create an overlay plot for multiple time series"""
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
def plot_correlation_matrix(input_file, dtypes, path):
    """
    Generate correlation matrix plots (Pearson and Spearman) for numeric columns
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
def plot_missing_values(input_file, dtypes, path):
    """
    Generate missing values heatmap and analysis
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


def export_statistical_summaries(input_file, dtypes, output_path):
    """
    Export statistical summaries to CSV files

    Parameters
    ----------
    input_file : str
        Path to the parquet file
    dtypes : dict
        Dictionary mapping column names to data types
    output_path : str
        Directory where CSV files will be saved
    """
    logger.info("Exporting statistical summaries...")

    # Get non-ignored columns
    cols = [col for col, dtype in dtypes.items() if dtype != "i"]

    if not cols:
        logger.info("No columns to export statistics for")
        return

    # Read data
    df = pd.read_parquet(input_file, columns=cols)

    # Create stats directory
    stats_path = os.path.join(output_path, "statistics")
    make_sure_path_exists(stats_path)

    # 1. Numeric statistics
    numeric_cols = [col for col, dtype in dtypes.items() if dtype == "n"]
    if numeric_cols:
        numeric_stats = df[numeric_cols].describe()
        # Add missing count
        numeric_stats.loc["missing"] = df[numeric_cols].isnull().sum()
        numeric_stats.loc["missing_pct"] = (
            df[numeric_cols].isnull().sum() / len(df)
        ) * 100

        stats_file = os.path.join(stats_path, "numeric_statistics.csv")
        numeric_stats.to_csv(stats_file)
        logger.info(f"Numeric statistics saved to: {stats_file}")

    # 2. Categorical statistics (value counts for each categorical column)
    category_cols = [col for col, dtype in dtypes.items() if dtype == "c"]
    if category_cols:
        for col in category_cols:
            value_counts = df[col].value_counts(dropna=False)
            value_counts_df = pd.DataFrame(
                {
                    "value": value_counts.index,
                    "count": value_counts.values,
                    "percentage": (value_counts.values / len(df)) * 100,
                }
            )

            stats_file = os.path.join(stats_path, f"category_{col}_counts.csv")
            value_counts_df.to_csv(stats_file, index=False)

        logger.info(f"Categorical statistics saved for {len(category_cols)} columns")

    # 3. Missing values analysis
    missing_summary = pd.DataFrame(
        {
            "column": cols,
            "missing_count": [df[col].isnull().sum() for col in cols],
            "missing_percentage": [
                (df[col].isnull().sum() / len(df)) * 100 for col in cols
            ],
            "total_count": len(df),
            "non_missing_count": [df[col].notnull().sum() for col in cols],
        }
    )

    missing_file = os.path.join(stats_path, "missing_values_summary.csv")
    missing_summary.to_csv(missing_file, index=False)
    logger.info(f"Missing values summary saved to: {missing_file}")

    # 4. Overall dataset summary
    overall_summary = pd.DataFrame(
        {
            "metric": [
                "total_rows",
                "total_columns",
                "numeric_columns",
                "categorical_columns",
                "columns_with_missing",
                "total_missing_cells",
                "missing_percentage",
            ],
            "value": [
                len(df),
                len(cols),
                len(numeric_cols),
                len(category_cols),
                missing_summary[missing_summary["missing_count"] > 0].shape[0],
                missing_summary["missing_count"].sum(),
                (missing_summary["missing_count"].sum() / (len(df) * len(cols))) * 100,
            ],
        }
    )

    overall_file = os.path.join(stats_path, "overall_summary.csv")
    overall_summary.to_csv(overall_file, index=False)
    logger.info(f"Overall summary saved to: {overall_file}")


if __name__ == "__main__":
    main()
