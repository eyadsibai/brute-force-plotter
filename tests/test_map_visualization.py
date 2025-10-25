"""
Unit tests for map visualization functions in brute_force_plotter.
"""

import os
import tempfile

import pandas as pd
import pytest

from src.brute_force_plotter import (
    _detect_geocoordinate_pairs,
    plot_map_visualization_sync,
)


class TestDetectGeocoordinatePairs:
    """Tests for geocoordinate pair detection."""

    @pytest.mark.unit
    def test_detects_explicit_geo_columns(self):
        """Test detection of explicitly marked geocoordinate columns."""
        dtypes = {
            "latitude": "g",
            "longitude": "g",
            "value": "n",
        }

        pairs = _detect_geocoordinate_pairs(dtypes)
        assert len(pairs) == 1
        assert ("latitude", "longitude") in pairs

    @pytest.mark.unit
    def test_detects_common_lat_lon_patterns_in_numeric(self):
        """Test detection of lat/lon patterns in numeric columns."""
        dtypes = {
            "lat": "n",
            "lon": "n",
            "value": "n",
        }

        pairs = _detect_geocoordinate_pairs(dtypes)
        assert len(pairs) >= 1
        assert ("lat", "lon") in pairs

    @pytest.mark.unit
    def test_detects_latitude_longitude_patterns(self):
        """Test detection of latitude/longitude named columns."""
        dtypes = {
            "latitude": "n",
            "longitude": "n",
            "value": "n",
        }

        pairs = _detect_geocoordinate_pairs(dtypes)
        assert len(pairs) >= 1
        assert ("latitude", "longitude") in pairs

    @pytest.mark.unit
    def test_no_detection_without_geo_columns(self):
        """Test that no pairs are detected without geocoordinate columns."""
        dtypes = {
            "age": "n",
            "income": "n",
            "gender": "c",
        }

        pairs = _detect_geocoordinate_pairs(dtypes)
        assert len(pairs) == 0

    @pytest.mark.unit
    def test_detects_x_y_coord_patterns(self):
        """Test detection of x/y coordinate patterns."""
        dtypes = {
            "y_coord": "n",
            "x_coord": "n",
            "value": "n",
        }

        pairs = _detect_geocoordinate_pairs(dtypes)
        assert len(pairs) >= 1
        # Should detect y_coord as lat and x_coord as lon
        assert ("y_coord", "x_coord") in pairs


class TestMapVisualization:
    """Tests for map visualization functions."""

    @pytest.fixture
    def geo_data(self):
        """Create sample data with geocoordinates."""
        return pd.DataFrame(
            {
                "latitude": [40.7128, 34.0522, 41.8781, 29.7604, 33.4484],
                "longitude": [-74.0060, -118.2437, -87.6298, -95.3698, -112.0740],
                "city": ["NYC", "LA", "Chicago", "Houston", "Phoenix"],
                "population": [8000000, 4000000, 2700000, 2300000, 1600000],
            }
        )

    @pytest.fixture
    def geo_parquet_file(self, temp_dir, geo_data):
        """Create a temporary parquet file with geocoordinate data."""
        parquet_path = os.path.join(temp_dir, "geo_data.parq")
        geo_data.to_parquet(parquet_path)
        return parquet_path

    @pytest.mark.unit
    def test_creates_simple_map(self, temp_dir, geo_parquet_file):
        """Test that a simple map is created."""
        plot_map_visualization_sync(geo_parquet_file, "latitude", "longitude", temp_dir)

        expected_file = os.path.join(temp_dir, "latitude-longitude-map.html")
        assert os.path.exists(expected_file)

        # Check that file has content
        with open(expected_file) as f:
            content = f.read()
            assert len(content) > 0
            assert "OpenStreetMap" in content

    @pytest.mark.unit
    def test_creates_map_with_category(self, temp_dir, geo_parquet_file):
        """Test that a map with categorical overlay is created."""
        plot_map_visualization_sync(
            geo_parquet_file, "latitude", "longitude", temp_dir, "city"
        )

        expected_file = os.path.join(temp_dir, "latitude-longitude-city-map.html")
        assert os.path.exists(expected_file)

        # Check that file has content
        with open(expected_file) as f:
            content = f.read()
            assert len(content) > 0
            # Should contain city names in the legend
            assert "city" in content.lower()

    @pytest.mark.unit
    def test_handles_missing_coordinates(self, temp_dir):
        """Test handling of data with missing coordinates."""
        data = pd.DataFrame(
            {
                "latitude": [40.7128, None, 41.8781, None, 33.4484],
                "longitude": [-74.0060, -118.2437, None, -95.3698, -112.0740],
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".parq", delete=False) as tmp:
            parquet_path = tmp.name
        data.to_parquet(parquet_path)

        try:
            plot_map_visualization_sync(parquet_path, "latitude", "longitude", temp_dir)

            # Should still create a map with valid coordinates
            expected_file = os.path.join(temp_dir, "latitude-longitude-map.html")
            assert os.path.exists(expected_file)
        finally:
            if os.path.exists(parquet_path):
                os.remove(parquet_path)

    @pytest.mark.unit
    def test_handles_invalid_coordinate_ranges(self, temp_dir):
        """Test handling of data with invalid coordinate ranges."""
        data = pd.DataFrame(
            {
                "latitude": [40.7128, 200.0, 41.8781, -100.0, 33.4484],
                "longitude": [-74.0060, -118.2437, -87.6298, -300.0, -112.0740],
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".parq", delete=False) as tmp:
            parquet_path = tmp.name
        data.to_parquet(parquet_path)

        try:
            plot_map_visualization_sync(parquet_path, "latitude", "longitude", temp_dir)

            # Should create a map only with valid coordinates
            expected_file = os.path.join(temp_dir, "latitude-longitude-map.html")
            assert os.path.exists(expected_file)
        finally:
            if os.path.exists(parquet_path):
                os.remove(parquet_path)

    @pytest.mark.unit
    def test_skips_existing_map(self, temp_dir, geo_parquet_file):
        """Test that existing maps are skipped when flag is set."""
        from src import brute_force_plotter

        brute_force_plotter.skip_existing_plots = True

        # Create initial map
        plot_map_visualization_sync(geo_parquet_file, "latitude", "longitude", temp_dir)

        expected_file = os.path.join(temp_dir, "latitude-longitude-map.html")
        assert os.path.exists(expected_file)

        # Get initial file size
        initial_size = os.path.getsize(expected_file)

        # Try to create again - should skip
        plot_map_visualization_sync(geo_parquet_file, "latitude", "longitude", temp_dir)

        # File size should be the same (not recreated)
        assert os.path.getsize(expected_file) == initial_size

    @pytest.mark.unit
    def test_handles_all_invalid_coordinates(self, temp_dir):
        """Test handling when all coordinates are invalid."""
        data = pd.DataFrame(
            {
                "latitude": [None, None, None],
                "longitude": [None, None, None],
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".parq", delete=False) as tmp:
            parquet_path = tmp.name
        data.to_parquet(parquet_path)

        try:
            # Should not crash, just log warning
            plot_map_visualization_sync(parquet_path, "latitude", "longitude", temp_dir)

            # No map file should be created
            expected_file = os.path.join(temp_dir, "latitude-longitude-map.html")
            assert not os.path.exists(expected_file)
        finally:
            if os.path.exists(parquet_path):
                os.remove(parquet_path)


class TestMapIntegration:
    """Integration tests for map visualization in the full plotting workflow."""

    @pytest.mark.integration
    def test_creates_maps_directory(self, temp_dir):
        """Test that maps directory is created when geocoordinates are detected."""
        from src import brute_force_plotter as bfp

        data = pd.DataFrame(
            {
                "latitude": [40.7128, 34.0522, 41.8781],
                "longitude": [-74.0060, -118.2437, -87.6298],
                "category": ["A", "B", "C"],
            }
        )

        dtypes = {
            "latitude": "g",
            "longitude": "g",
            "category": "c",
        }

        bfp.plot(data, dtypes, output_path=temp_dir, use_dask=False)

        # Check that maps directory was created
        maps_dir = os.path.join(temp_dir, "maps")
        assert os.path.exists(maps_dir)

        # Check that at least one map was created
        map_files = [f for f in os.listdir(maps_dir) if f.endswith(".html")]
        assert len(map_files) > 0

    @pytest.mark.integration
    def test_creates_maps_with_categories(self, temp_dir):
        """Test that maps with category overlays are created."""
        from src import brute_force_plotter as bfp

        data = pd.DataFrame(
            {
                "lat": [40.7128, 34.0522, 41.8781],
                "lon": [-74.0060, -118.2437, -87.6298],
                "city_type": ["A", "B", "C"],
            }
        )

        dtypes = {
            "lat": "g",
            "lon": "g",
            "city_type": "c",
        }

        bfp.plot(data, dtypes, output_path=temp_dir, use_dask=False)

        maps_dir = os.path.join(temp_dir, "maps")

        # Should have both simple map and map with category overlay
        map_files = [f for f in os.listdir(maps_dir) if f.endswith(".html")]
        assert len(map_files) >= 2

        # Check for category overlay map
        category_maps = [f for f in map_files if "city_type" in f]
        assert len(category_maps) > 0
