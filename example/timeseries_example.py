#!/usr/bin/env python3
"""
Example script to generate time series data for testing brute-force-plotter
"""

from datetime import datetime, timedelta
import json

import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Generate dates
start_date = datetime(2023, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(365)]

# Generate multiple time series
# Temperature: seasonal pattern with noise
day_of_year = np.arange(365)
temperature = (
    15 + 10 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 2, 365)
)

# Sales: trend + seasonality + noise
trend = np.linspace(100, 150, 365)
seasonality = 20 * np.sin(2 * np.pi * day_of_year / 7)  # weekly seasonality
sales = trend + seasonality + np.random.normal(0, 5, 365)

# Stock price: random walk
stock_price = 100 + np.cumsum(np.random.normal(0.1, 2, 365))

# Customer count: another pattern
customer_count = (
    50 + 5 * np.sin(2 * np.pi * day_of_year / 30) + np.random.normal(0, 3, 365)
)

# Category: store location (for grouping)
store_location = np.random.choice(["North", "South", "East", "West"], 365)

# Region: another categorical
region = np.random.choice(["Urban", "Suburban", "Rural"], 365)

# Create DataFrame
df = pd.DataFrame(
    {
        "date": dates,
        "temperature": temperature,
        "sales": sales,
        "stock_price": stock_price,
        "customer_count": customer_count,
        "store_location": store_location,
        "region": region,
        "id": range(365),  # column to ignore
    }
)

# Save to CSV
df.to_csv("example/timeseries_data.csv", index=False)
print("Generated timeseries_data.csv with 365 daily observations")

# Create dtypes JSON
dtypes = {
    "date": "t",  # time series
    "temperature": "n",  # numeric
    "sales": "n",  # numeric
    "stock_price": "n",  # numeric
    "customer_count": "n",  # numeric
    "store_location": "c",  # categorical
    "region": "c",  # categorical
    "id": "i",  # ignore
}

with open("example/timeseries_dtypes.json", "w") as f:
    json.dump(dtypes, f, indent=2)

print("Generated timeseries_dtypes.json")
print(f"\nDataset shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
