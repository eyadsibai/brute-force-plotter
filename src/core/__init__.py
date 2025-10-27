"""
Core modules for Brute Force Plotter.

This package contains core functionality including:
- Configuration and constants
- Data type inference
- Utility functions
"""

from .config import (
    DEFAULT_MAX_ROWS,
    DEFAULT_SAMPLE_SIZE,
    get_save_plots,
    get_show_plots,
    set_save_plots,
    set_show_plots,
)
from .data_types import infer_dtypes
from .utils import autolabel, make_sure_path_exists

__all__ = [
    "infer_dtypes",
    "make_sure_path_exists",
    "autolabel",
    "DEFAULT_MAX_ROWS",
    "DEFAULT_SAMPLE_SIZE",
    "get_show_plots",
    "set_show_plots",
    "get_save_plots",
    "set_save_plots",
]
