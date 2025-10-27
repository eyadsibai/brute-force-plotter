"""
Utility functions for Brute Force Plotter.
"""

import errno
import logging
import math
import os

import matplotlib.pyplot as plt

from .config import get_save_plots, get_show_plots

logger = logging.getLogger(__name__)


def make_sure_path_exists(path):
    """
    Create a directory if it doesn't exist.

    Parameters
    ----------
    path : str
        Directory path to create

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    logger.debug(f"Make sure {path} exists")
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            return False
    return True


def autolabel(rects):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    Parameters
    ----------
    rects : list
        List of matplotlib Rectangle objects
    """
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


def ignore_if_exist_or_save(func):
    """
    Decorator to handle plot saving/showing logic.

    Parameters
    ----------
    func : callable
        Plotting function to wrap

    Returns
    -------
    callable
        Wrapped function
    """

    def wrapper(*args, **kwargs):
        file_name = kwargs.get("file_name")

        # Get current plot settings
        show_plots = get_show_plots()
        save_plots = get_save_plots()

        # If saving is disabled and showing is enabled, just create and show
        if show_plots and not save_plots:
            func(*args, **kwargs)
            plt.gcf().set_tight_layout(True)
            plt.show()
            plt.close("all")
        # If file exists and we're saving, skip
        elif file_name and os.path.isfile(file_name) and save_plots:
            plt.close("all")
        # Otherwise, create the plot
        else:
            func(*args, **kwargs)
            plt.gcf().set_tight_layout(True)

            # Save if we should save
            if save_plots and file_name:
                plt.gcf().savefig(file_name, dpi=120)

            # Show if we should show
            if show_plots:
                plt.show()

            plt.close("all")

    return wrapper


def check_and_sample_large_dataset(
    data, max_rows=100000, sample_size=50000, no_sample=False
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
