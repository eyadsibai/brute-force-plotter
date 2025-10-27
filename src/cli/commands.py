"""
Command-line interface for Brute Force Plotter.

This module contains the Click-based CLI commands.
"""

import json
import logging
import os

import click
import dask
import matplotlib
import pandas as pd
import seaborn as sns

from ..core import config
from ..core.config import DEFAULT_MAX_ROWS, DEFAULT_SAMPLE_SIZE
from ..core.data_types import infer_dtypes
from ..core.utils import check_and_sample_large_dataset
from ..stats.export import export_statistical_summaries
from .orchestration import create_plots

logger = logging.getLogger(__name__)

__all__ = ["main"]


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
    "--minimal",
    is_flag=True,
    default=False,
    help="Generate minimal set of plots (reduces redundant visualizations)",
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
    minimal,
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
    config.skip_existing_plots = skip_existing

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

    plots = create_plots(new_file_name, data_types, output_path, minimal=minimal)
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
