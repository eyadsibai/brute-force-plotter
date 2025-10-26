"""
Data type inference functionality.
"""

import pandas as pd


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
