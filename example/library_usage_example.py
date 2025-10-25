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

# Example 1: Load data from CSV and create plots
print("Example 1: Creating plots from CSV data")
data = pd.read_csv(os.path.join(os.path.dirname(__file__), "titanic.csv"))

# Define data types
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

# Example 2: Creating plots from a DataFrame and showing them interactively
# (Uncomment to test interactive display)
# print("\nExample 2: Showing plots interactively")
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

# Example 3: Creating a simple dataset and plotting
print("\nExample 3: Creating plots from a simple DataFrame")

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

simple_dtypes = {"height": "n", "weight": "n", "gender": "c", "age_group": "c"}

output_dir2 = bfp.plot(
    simple_data,
    simple_dtypes,
    output_path="./simple_output",
    show=False,
    use_dask=False,
)

print(f"✓ Simple plots saved to: {output_dir2}")

# Example 4: Creating map visualizations from geocoordinate data
print("\nExample 4: Creating interactive maps from geocoordinate data")

# Create a dataset with latitude and longitude
geo_data = pd.DataFrame(
    {
        "city": [
            "New York",
            "Los Angeles",
            "Chicago",
            "Houston",
            "Phoenix",
            "Philadelphia",
            "San Antonio",
            "San Diego",
        ],
        "latitude": [40.7128, 34.0522, 41.8781, 29.7604, 33.4484, 39.9526, 29.4241, 32.7157],
        "longitude": [-74.0060, -118.2437, -87.6298, -95.3698, -112.0740, -75.1652, -98.4936, -117.1611],
        "population": [8804190, 3979576, 2695598, 2325502, 1660272, 1584138, 1547253, 1423851],
        "size_category": ["Large", "Large", "Large", "Large", "Large", "Large", "Medium", "Medium"],
    }
)

# Define data types with 'g' for geocoordinate columns
geo_dtypes = {
    "city": "i",           # ignore city names in plots
    "latitude": "g",       # geocoordinate (latitude)
    "longitude": "g",      # geocoordinate (longitude)
    "population": "n",     # numeric
    "size_category": "c",  # category
}

output_dir3 = bfp.plot(
    geo_data,
    geo_dtypes,
    output_path="./geo_output",
    show=False,
    use_dask=False,
)

print(f"✓ Maps and plots saved to: {output_dir3}")
print(f"✓ Interactive HTML maps saved to: {output_dir3}/maps/")

print("\n✓ All examples completed successfully!")
