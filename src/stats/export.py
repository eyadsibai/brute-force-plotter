"""
Statistical summary export functionality.
"""

import logging
import os

import pandas as pd

from ..core.utils import make_sure_path_exists

logger = logging.getLogger(__name__)

__all__ = ["export_statistical_summaries"]


def export_statistical_summaries(input_file, dtypes, output_path):
    """
    Export statistical summaries to CSV files.

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
