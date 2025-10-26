"""
Brute Force Plotter
===================

A tool to visualize data quickly with no brain usage for plot creation.

Basic usage:
    >>> import brute_force_plotter as bfp
    >>> import pandas as pd
    >>>
    >>> # Load your data
    >>> data = pd.read_csv('data.csv')
    >>>
    >>> # Option 1: Automatic type inference
    >>> output_path, dtypes = bfp.plot(data)
    >>> print(f"Inferred types: {dtypes}")
    >>>
    >>> # Option 2: Manual type definition (c=category, n=numeric, i=ignore)
    >>> dtypes = {
    >>>     'column1': 'n',  # numeric
    >>>     'column2': 'c',  # category
    >>>     'column3': 'i'   # ignore
    >>> }
    >>>
    >>> # Create plots
    >>> bfp.plot(data, dtypes, output_path='./output', show=False)
    >>>
    >>> # Option 3: Infer types manually
    >>> dtypes = bfp.infer_dtypes(data)

"""

from .brute_force_plotter import infer_dtypes, plot

__version__ = "0.1.0"
__all__ = ["plot", "infer_dtypes"]
