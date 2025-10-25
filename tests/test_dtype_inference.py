"""
Unit tests for automatic data type inference in brute_force_plotter.
"""

import pandas as pd
import pytest

from src.brute_force_plotter import infer_dtypes


class TestInferDtypes:
    """Tests for the infer_dtypes function."""

    @pytest.mark.unit
    def test_infers_numeric_columns(self):
        """Test that numeric columns are correctly identified."""
        data = pd.DataFrame(
            {
                "int_col": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "float_col": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0],
            }
        )

        dtypes = infer_dtypes(data)

        assert dtypes["int_col"] == "n"
        assert dtypes["float_col"] == "n"

    @pytest.mark.unit
    def test_infers_categorical_from_few_unique_values(self):
        """Test that columns with few unique values are categorical."""
        data = pd.DataFrame(
            {
                "status": ["active", "inactive", "active", "active", "inactive"]
                * 10,  # 2 unique
                "category": ["A", "B", "C", "A", "B"] * 10,  # 3 unique
            }
        )

        dtypes = infer_dtypes(data)

        assert dtypes["status"] == "c"
        assert dtypes["category"] == "c"

    @pytest.mark.unit
    def test_infers_categorical_from_numeric_with_few_values(self):
        """Test that numeric columns with few unique values are categorical."""
        data = pd.DataFrame(
            {
                "rating": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1] * 10,  # 3 unique values
                "binary": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 10,  # 2 unique values
            }
        )

        dtypes = infer_dtypes(data, max_categorical_unique=5)

        assert dtypes["rating"] == "c"
        assert dtypes["binary"] == "c"

    @pytest.mark.unit
    def test_infers_boolean_as_categorical(self):
        """Test that boolean columns are categorical."""
        data = pd.DataFrame(
            {
                "is_active": [True, False, True, True, False] * 10,
                "has_feature": [False, False, True, False, True] * 10,
            }
        )

        dtypes = infer_dtypes(data)

        assert dtypes["is_active"] == "c"
        assert dtypes["has_feature"] == "c"

    @pytest.mark.unit
    def test_infers_id_columns_as_ignore(self):
        """Test that ID columns (all unique values) are ignored."""
        data = pd.DataFrame(
            {
                "user_id": range(100),  # All unique
                "transaction_id": [f"TXN{i:04d}" for i in range(100)],  # All unique
            }
        )

        dtypes = infer_dtypes(data)

        assert dtypes["user_id"] == "i"
        assert dtypes["transaction_id"] == "i"

    @pytest.mark.unit
    def test_infers_text_with_many_unique_as_ignore(self):
        """Test that text columns with many unique values are ignored."""
        data = pd.DataFrame(
            {
                "name": [f"Person {i}" for i in range(100)],
                "description": [f"Description {i}" for i in range(100)],
            }
        )

        dtypes = infer_dtypes(data)

        assert dtypes["name"] == "i"
        assert dtypes["description"] == "i"

    @pytest.mark.unit
    def test_infers_datetime_as_ignore(self):
        """Test that datetime columns are ignored."""
        data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=100),
                "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"] * 33 + ["2024-01-01"]),
            }
        )

        dtypes = infer_dtypes(data)

        assert dtypes["timestamp"] == "i"
        assert dtypes["date"] == "i"

    @pytest.mark.unit
    def test_handles_mixed_data(self):
        """Test inference on a realistic mixed dataset."""
        data = pd.DataFrame(
            {
                "id": range(50),
                "age": [25, 30, 35, 40, 45] * 10,
                "income": [50000.5, 60000.3, 70000.2, 80000.1, 90000.0] * 10,
                "gender": ["M", "F", "M", "F", "M"] * 10,
                "city": ["NYC", "LA", "Chicago", "NYC", "LA"] * 10,
                "score": [1, 2, 3, 4, 5] * 10,
                "name": [f"Person {i}" for i in range(50)],
                "timestamp": pd.date_range("2024-01-01", periods=50),
            }
        )

        dtypes = infer_dtypes(data)

        assert dtypes["id"] == "n"  # 50 unique values, ratio is 100%, so numeric
        assert dtypes["age"] == "n"  # 5 unique, but 10% ratio > 5%, so numeric
        assert dtypes["income"] == "n"  # 5 unique, but 10% ratio > 5%, so numeric
        assert dtypes["gender"] == "c"  # Few unique values, low ratio
        assert dtypes["city"] == "c"  # Few unique values, low ratio
        assert dtypes["score"] == "n"  # 5 unique, but 10% ratio > 5%, so numeric
        assert dtypes["name"] == "i"  # Text with many unique values
        assert dtypes["timestamp"] == "i"  # Datetime

    @pytest.mark.unit
    def test_handles_missing_values(self):
        """Test that missing values are handled correctly."""
        data = pd.DataFrame(
            {
                "col_with_nulls": [1, 2, None, 4, 5, 6, 7, 8, 9, 10],
                "mostly_nulls": [1, None, None, None, None, None, None, None, None, 10],
            }
        )

        dtypes = infer_dtypes(data)

        # Should still infer correctly despite nulls
        assert dtypes["col_with_nulls"] == "n"
        assert dtypes["mostly_nulls"] == "n"

    @pytest.mark.unit
    def test_custom_thresholds(self):
        """Test that custom thresholds work correctly."""
        n_rows = 100
        data = pd.DataFrame(
            {
                "col1": list(range(20)) * (n_rows // 20),  # 20 unique values, 100 total = 20% ratio
                "col2": list(range(60)) + list(range(40)),  # 60 unique values, 100 total = 60% ratio
            }
        )

        # With default threshold (max 50 unique, max 5% ratio)
        dtypes_default = infer_dtypes(data)
        assert dtypes_default["col1"] == "n"  # 20 unique < 50, but 20% ratio > 5%
        assert dtypes_default["col2"] == "n"  # 60 unique > 50

        # With higher unique threshold and ratio (max 70 unique, max 25% ratio)
        dtypes_high = infer_dtypes(data, max_categorical_unique=70, max_categorical_ratio=0.25)
        assert dtypes_high["col1"] == "c"  # 20 unique < 70 AND 20% ratio < 25%
        assert dtypes_high["col2"] == "n"  # 60 unique < 70 BUT 60% ratio > 25%

        # With lower threshold (max 10 unique)
        dtypes_low = infer_dtypes(data, max_categorical_unique=10)
        assert dtypes_low["col1"] == "n"  # 20 unique > 10
        assert dtypes_low["col2"] == "n"  # 60 unique > 10

    @pytest.mark.unit
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        data = pd.DataFrame()

        dtypes = infer_dtypes(data)

        assert dtypes == {}

    @pytest.mark.unit
    def test_single_row_dataframe(self):
        """Test handling of DataFrame with single row."""
        data = pd.DataFrame(
            {
                "col1": [1],
                "col2": ["text"],
                "col3": [True],
            }
        )

        dtypes = infer_dtypes(data)

        # With one row, all values are unique (ratio = 1.0), which exceeds max_categorical_ratio
        assert dtypes["col1"] == "n"  # Numeric with 100% unique ratio -> numeric
        assert dtypes["col2"] == "i"  # Text with 100% unique ratio -> ignore
        assert dtypes["col3"] == "c"  # Boolean -> categorical

    @pytest.mark.unit
    def test_all_same_value(self):
        """Test columns where all values are the same."""
        data = pd.DataFrame(
            {
                "constant_num": [42] * 100,
                "constant_str": ["same"] * 100,
            }
        )

        dtypes = infer_dtypes(data)

        # Single unique value, low ratio -> categorical
        assert dtypes["constant_num"] == "c"
        assert dtypes["constant_str"] == "c"

    @pytest.mark.unit
    def test_ratio_threshold(self):
        """Test the unique ratio threshold."""
        # Create data with specific unique ratio
        data = pd.DataFrame(
            {
                "low_ratio": [1, 1, 1, 2, 2] * 20,  # 2 unique / 100 total = 2% ratio
                "high_ratio": list(range(50))
                * 2,  # 50 unique / 100 total = 50% ratio
            }
        )

        # Default ratio threshold is 5%
        dtypes = infer_dtypes(data, max_categorical_ratio=0.05)
        assert dtypes["low_ratio"] == "c"  # 2% < 5% AND 2 <= 50
        assert dtypes["high_ratio"] == "n"  # 50% > 5%, so numeric even though 50 unique <= 50

        # Higher ratio threshold
        dtypes_high = infer_dtypes(data, max_categorical_ratio=0.60)
        assert dtypes_high["low_ratio"] == "c"  # 2% < 60% AND 2 <= 50
        assert dtypes_high["high_ratio"] == "c"  # 50% < 60% AND 50 <= 50


class TestInferDtypesIntegration:
    """Integration tests for infer_dtypes with real-world-like data."""

    @pytest.mark.integration
    def test_titanic_like_dataset(self):
        """Test inference on a Titanic-like dataset."""
        data = pd.DataFrame(
            {
                "PassengerId": range(1, 101),
                "Survived": [0, 1] * 50,
                "Pclass": [1, 2, 3] * 33 + [1],
                "Name": [f"Passenger {i}" for i in range(100)],
                "Sex": ["male", "female"] * 50,
                "Age": [20, 25, 30, 35, 40] * 20,
                "SibSp": [0, 1, 2] * 33 + [0],
                "Parch": [0, 1, 2] * 33 + [0],
                "Ticket": [f"TICKET{i}" for i in range(100)],
                "Fare": [10.5, 20.3, 30.7, 40.2, 50.1] * 20,
                "Cabin": [f"C{i}" if i % 3 == 0 else None for i in range(100)],
                "Embarked": ["S", "C", "Q"] * 33 + ["S"],
            }
        )

        dtypes = infer_dtypes(data)

        assert dtypes["PassengerId"] == "i"  # ID column
        assert dtypes["Survived"] == "c"  # Binary outcome
        assert dtypes["Pclass"] == "c"  # Few classes
        assert dtypes["Name"] == "i"  # Text with many unique
        assert dtypes["Sex"] == "c"  # Binary category
        assert dtypes["Age"] == "c"  # 5 unique and 5% ratio - categorical
        assert dtypes["SibSp"] == "c"  # Few unique numeric
        assert dtypes["Parch"] == "c"  # Few unique numeric
        assert dtypes["Ticket"] == "i"  # ID-like
        assert dtypes["Fare"] == "c"  # 5 unique and 5% ratio - categorical
        assert dtypes["Cabin"] in ["c", "i"]  # Could be either
        assert dtypes["Embarked"] == "c"  # Few categories
