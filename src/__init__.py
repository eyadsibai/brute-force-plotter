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
    >>> # Define data types (c=category, n=numeric, i=ignore)
    >>> dtypes = {
    >>>     'column1': 'n',  # numeric
    >>>     'column2': 'c',  # category
    >>>     'column3': 'i'   # ignore
    >>> }
    >>> 
    >>> # Create plots
    >>> bfp.plot(data, dtypes, output_path='./output', show=False)

"""

from .brute_force_plotter import plot

__version__ = '0.1.0'
__all__ = ['plot']
