#!/usr/bin/env python3
"""
Example of using brute-force-plotter as a Python library
"""

import os
import sys

import pandas as pd

# Add the src directory to the path (only needed if not installed via pip)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import brute_force_plotter as bfp

# Example 1: Automatic data type inference (NEW!)
print("Example 1: Using automatic data type inference")
data = pd.read_csv(os.path.join(os.path.dirname(__file__), "titanic.csv"))

# Let the library automatically infer data types
output_dir, inferred_dtypes = bfp.plot(
    data,
    output_path="./output_auto_inference",
    show=False,
    use_dask=False,
    export_stats=True,
)

print(f"✓ Plots saved to: {output_dir}")
print(f"✓ Inferred data types:")
for col, dtype in sorted(inferred_dtypes.items()):
    dtype_name = {"n": "numeric", "c": "categorical", "i": "ignore"}[dtype]
    print(f"    {col:15} -> {dtype_name}")

# Example 2: Manual data type definition
print("\nExample 2: Creating plots with manual data types")

# Define data types manually
# 'n' = numeric, 'c' = category, 'i' = ignore
dtypes = {
    "Survived": "c",
    "Pclass": "c",
    "Sex": "c",
    "Age": "n",
    "SibSp": "n",
    "Parch": "n",
    "Fare": "n",
    "Embarked": "c",
    "PassengerId": "i",
    "Ticket": "i",
    "Cabin": "i",
    "Name": "i",
}

# Create plots and save to directory
output_dir = bfp.plot(
    data,
    dtypes,
    output_path="./output_from_library",
    show=False,  # Set to True to display plots interactively
    use_dask=False,  # Set to True to use parallel processing with Dask
    export_stats=True,  # Export statistical summaries to CSV files
)

print(f"✓ Plots saved to: {output_dir}")
print(f"✓ Statistical summaries exported to: {output_dir}/statistics/")

# Example 3: Infer types first, then modify if needed
print("\nExample 3: Infer types, then customize")

# Infer data types
auto_dtypes = bfp.infer_dtypes(data)

# Customize some inferred types if needed
auto_dtypes["Age"] = "c"  # Treat age as categorical instead of numeric
auto_dtypes["Fare"] = "c"  # Treat fare as categorical instead of numeric

output_dir3 = bfp.plot(
    data,
    auto_dtypes,
    output_path="./output_custom",
    show=False,
    use_dask=False,
)

print(f"✓ Custom plots saved to: {output_dir3}")

# Example 4: Creating plots from a simple DataFrame and showing them interactively
# (Uncomment to test interactive display)
# print("\nExample 4: Showing plots interactively")
# simple_dtypes = {
#     'Age': 'n',
#     'Fare': 'n',
#     'Survived': 'c',
#     'Pclass': 'c'
# }
#
# bfp.plot(
#     data,
#     simple_dtypes,
#     show=True  # Display plots instead of saving
# )

# Example 5: Creating a simple dataset with automatic inference
print("\nExample 5: Simple DataFrame with auto inference")

# Create a simple dataset
simple_data = pd.DataFrame(
    {
        "height": [165, 170, 175, 180, 185, 160, 155, 172, 178, 168],
        "weight": [65, 70, 75, 80, 85, 60, 55, 72, 78, 68],
        "gender": ["F", "M", "M", "M", "M", "F", "F", "M", "M", "F"],
        "age_group": [
            "young",
            "young",
            "adult",
            "adult",
            "adult",
            "young",
            "young",
            "adult",
            "adult",
            "young",
        ],
    }
)

# Use automatic inference
output_dir5, simple_dtypes = bfp.plot(
    simple_data,
    output_path="./simple_output",
    show=False,
    use_dask=False,
)

print(f"✓ Simple plots saved to: {output_dir5}")
print(f"✓ Auto-inferred types: {simple_dtypes}")

print("\n✓ All examples completed successfully!")

