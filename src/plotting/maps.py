"""
Geographic map visualization functions.

This module handles interactive map creation for geocoordinate data.
"""

import logging
import os

import dask
import folium
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd

from ..core.config import skip_existing_plots

logger = logging.getLogger(__name__)

# Common patterns for latitude and longitude column names
# Avoid single-letter patterns to prevent false positives
LAT_PATTERNS = ["lat", "latitude", "y_coord", "lat_coord"]
LON_PATTERNS = ["lon", "long", "longitude", "lng", "x_coord", "lon_coord"]

__all__ = [
    "detect_geocoordinate_pairs",
    "create_map_visualization",
    "plot_map_visualization",
    "plot_map_visualization_sync",
]


def detect_geocoordinate_pairs(dtypes):
    """
    Detect pairs of columns that represent geocoordinates (latitude/longitude).

    This function looks for:
    - Columns with 'g' data type (explicit geocoordinate marking)
    - Common lat/lon naming patterns in numeric columns

    Parameters
    ----------
    dtypes : dict
        Dictionary mapping column names to data types

    Returns
    -------
    list of tuple
        List of (latitude_col, longitude_col) pairs
    """
    geo_pairs = []

    # Get columns marked as geocoordinates
    geo_cols = [col for col, dtype in dtypes.items() if dtype == "g"]

    # First, try to pair explicitly marked geocoordinate columns
    if len(geo_cols) >= 2:
        # Try to identify lat/lon by name patterns
        for lat_col in geo_cols:
            for lon_col in geo_cols:
                if lat_col != lon_col:
                    lat_lower = lat_col.lower()
                    lon_lower = lon_col.lower()

                    # Check if names match common patterns
                    is_lat = any(pattern in lat_lower for pattern in LAT_PATTERNS)
                    is_lon = any(pattern in lon_lower for pattern in LON_PATTERNS)

                    if is_lat and is_lon and (lat_col, lon_col) not in geo_pairs:
                        geo_pairs.append((lat_col, lon_col))
                        break

    # Also check for common lat/lon patterns in numeric columns
    # This is a fallback for backward compatibility
    numeric_cols = [col for col, dtype in dtypes.items() if dtype == "n"]

    for col1 in numeric_cols:
        for col2 in numeric_cols:
            if col1 != col2 and (col1, col2) not in geo_pairs:
                col1_lower = col1.lower()
                col2_lower = col2.lower()

                is_lat1 = any(pattern in col1_lower for pattern in LAT_PATTERNS)
                is_lon2 = any(pattern in col2_lower for pattern in LON_PATTERNS)

                if is_lat1 and is_lon2:
                    geo_pairs.append((col1, col2))

    logger.info(f"Detected {len(geo_pairs)} geocoordinate pairs: {geo_pairs}")
    return geo_pairs


def create_map_visualization(input_file, lat_col, lon_col, path, category_col=None):
    """
    Core logic for creating an interactive map visualization using geocoordinates.

    Parameters
    ----------
    input_file : str
        Path to the parquet file
    lat_col : str
        Name of the latitude column
    lon_col : str
        Name of the longitude column
    path : str
        Output directory path
    category_col : str, optional
        Name of a categorical column to use for coloring points
    """
    columns_to_read = [lat_col, lon_col]
    if category_col:
        columns_to_read.append(category_col)

    df = pd.read_parquet(input_file, columns=columns_to_read)

    # Remove rows with missing geocoordinates
    df_clean = df.dropna(subset=[lat_col, lon_col])

    if len(df_clean) == 0:
        logger.warning(f"No valid geocoordinates found in {lat_col} and {lon_col}")
        return

    # Validate coordinate ranges
    lat_valid = df_clean[(df_clean[lat_col] >= -90) & (df_clean[lat_col] <= 90)]
    lon_valid = lat_valid[(lat_valid[lon_col] >= -180) & (lat_valid[lon_col] <= 180)]

    if len(lon_valid) == 0:
        logger.warning(f"No valid coordinate ranges in {lat_col} and {lon_col}")
        return

    df_clean = lon_valid

    # Create base filename
    if category_col:
        base_filename = f"{lat_col}-{lon_col}-{category_col}-map.html"
    else:
        base_filename = f"{lat_col}-{lon_col}-map.html"

    file_name = os.path.join(path, base_filename)

    # Skip if file exists and skip_existing_plots is True
    if skip_existing_plots and os.path.isfile(file_name):
        logger.info(f"Skipping existing map: {file_name}")
        return

    # Calculate center of the map
    center_lat = df_clean[lat_col].mean()
    center_lon = df_clean[lon_col].mean()

    # Create folium map
    m = folium.Map(
        location=[center_lat, center_lon], zoom_start=10, tiles="OpenStreetMap"
    )

    # Add markers
    if category_col and category_col in df_clean.columns:
        # Color code by category using matplotlib colormap for unlimited categories
        categories = df_clean[category_col].unique()
        n_categories = len(categories)

        # Use tab20 colormap for up to 20 categories, then tab20b/tab20c for more
        if n_categories <= 20:
            cmap = plt.cm.get_cmap("tab20")
        else:
            # For more than 20 categories, use a continuous colormap
            cmap = plt.cm.get_cmap("hsv")

        # Generate colors and convert to hex format for folium
        colors = [
            mcolors.rgb2hex(cmap(i / max(n_categories - 1, 1))[:3])
            for i in range(n_categories)
        ]

        # Create a mapping from category to color
        cat_to_color = {cat: colors[i] for i, cat in enumerate(categories)}

        # Add markers with category-based colors
        for _idx, row in df_clean.iterrows():
            folium.CircleMarker(
                location=[row[lat_col], row[lon_col]],
                radius=5,
                popup=f"{category_col}: {row[category_col]}",
                color=cat_to_color.get(row[category_col], "blue"),
                fill=True,
                fillColor=cat_to_color.get(row[category_col], "blue"),
                fillOpacity=0.6,
            ).add_to(m)

        # Add legend
        legend_html = f"""
        <div style="position: fixed;
                    bottom: 50px; right: 50px;
                    border:2px solid grey; z-index:9999;
                    background-color:white;
                    padding: 10px;
                    font-size:14px;">
        <p style="margin:0; font-weight:bold;">{category_col}</p>
        """

        for cat, color in cat_to_color.items():
            legend_html += f"""
            <p style="margin:5px 0;">
                <span style="background-color:{color};
                            width:15px; height:15px;
                            display:inline-block;
                            margin-right:5px;"></span>
                {cat}
            </p>
            """
        legend_html += "</div>"
        m.get_root().html.add_child(folium.Element(legend_html))
    else:
        # Simple blue markers
        for _idx, row in df_clean.iterrows():
            folium.CircleMarker(
                location=[row[lat_col], row[lon_col]],
                radius=5,
                popup=f"Lat: {row[lat_col]:.4f}, Lon: {row[lon_col]:.4f}",
                color="blue",
                fill=True,
                fillColor="blue",
                fillOpacity=0.6,
            ).add_to(m)

    # Save map
    m.save(file_name)
    logger.info(f"Map saved to: {file_name}")


@dask.delayed
def plot_map_visualization(input_file, lat_col, lon_col, path, category_col=None):
    """
    Create an interactive map visualization using geocoordinates (Dask delayed version).

    Parameters
    ----------
    input_file : str
        Path to the parquet file
    lat_col : str
        Name of the latitude column
    lon_col : str
        Name of the longitude column
    path : str
        Output directory path
    category_col : str, optional
        Name of a categorical column to use for coloring points
    """
    create_map_visualization(input_file, lat_col, lon_col, path, category_col)


def plot_map_visualization_sync(input_file, lat_col, lon_col, path, category_col=None):
    """
    Create an interactive map visualization using geocoordinates (synchronous).

    Parameters
    ----------
    input_file : str
        Path to the parquet file
    lat_col : str
        Name of the latitude column
    lon_col : str
        Name of the longitude column
    path : str
        Output directory path
    category_col : str, optional
        Name of a categorical column to use for coloring points
    """
    create_map_visualization(input_file, lat_col, lon_col, path, category_col)
