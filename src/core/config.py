"""
Global configuration and constants for Brute Force Plotter.
"""

import logging

logger = logging.getLogger(__name__)

# Global state for ignored columns
ignore = set()

# Global state for target variable
target_variable = None

# Global flag for skipping existing plots
skip_existing_plots = True

# Global configuration for plot display and saving
_show_plots = False
_save_plots = True

# Large dataset configuration
DEFAULT_MAX_ROWS = 100000  # Default threshold for sampling
DEFAULT_SAMPLE_SIZE = 50000  # Default sample size for large datasets


def get_show_plots():
    """Get the current value of show_plots flag."""
    return _show_plots


def set_show_plots(value):
    """Set the show_plots flag."""
    global _show_plots
    _show_plots = value


def get_save_plots():
    """Get the current value of save_plots flag."""
    return _save_plots


def set_save_plots(value):
    """Set the save_plots flag."""
    global _save_plots
    _save_plots = value


def get_target_variable():
    """Get the current target variable."""
    return target_variable


def set_target_variable(value):
    """Set the target variable."""
    global target_variable
    target_variable = value
