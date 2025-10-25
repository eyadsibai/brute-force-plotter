"""
Pytest configuration and shared fixtures for brute-force-plotter tests.
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_numeric_data():
    """Create a simple DataFrame with numeric columns."""
    return pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "col2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "col3": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
        }
    )


@pytest.fixture
def sample_categorical_data():
    """Create a simple DataFrame with categorical columns."""
    return pd.DataFrame(
        {
            "category1": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"],
            "category2": ["X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "Y"],
            "category3": ["P", "Q", "R", "P", "Q", "R", "P", "Q", "R", "P"],
        }
    )


@pytest.fixture
def sample_mixed_data():
    """Create a DataFrame with both numeric and categorical columns."""
    return pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
            "income": [30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000],
            "gender": ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F"],
            "education": ["HS", "BS", "MS", "PhD", "BS", "MS", "PhD", "BS", "MS", "PhD"],
        }
    )


@pytest.fixture
def sample_data_with_missing():
    """Create a DataFrame with missing values."""
    return pd.DataFrame(
        {
            "col1": [1, 2, None, 4, 5, None, 7, 8, 9, 10],
            "col2": [10, None, 30, 40, None, 60, 70, None, 90, 100],
            "category": ["A", "B", None, "A", "B", "C", None, "B", "C", "A"],
        }
    )


@pytest.fixture
def sample_empty_data():
    """Create an empty DataFrame."""
    return pd.DataFrame()


@pytest.fixture
def sample_data_many_categories():
    """Create a DataFrame with a column having >50 unique categories."""
    return pd.DataFrame(
        {
            "many_cats": [f"cat_{i}" for i in range(100)],
            "value": list(range(100)),
        }
    )


@pytest.fixture
def simple_dtypes():
    """Simple dtype mapping for testing."""
    return {
        "col1": "n",  # numeric
        "col2": "n",  # numeric
        "col3": "n",  # numeric
    }


@pytest.fixture
def mixed_dtypes():
    """Mixed dtype mapping for testing."""
    return {
        "age": "n",
        "income": "n",
        "gender": "c",
        "education": "c",
    }


@pytest.fixture
def ignore_dtypes():
    """Dtype mapping with ignored columns."""
    return {
        "col1": "n",
        "col2": "i",  # ignore
        "col3": "c",
    }


@pytest.fixture
def sample_parquet_file(temp_dir, sample_mixed_data):
    """Create a temporary parquet file from sample data."""
    parquet_path = os.path.join(temp_dir, "test_data.parq")
    sample_mixed_data.to_parquet(parquet_path)
    return parquet_path


@pytest.fixture
def sample_csv_file(temp_dir, sample_mixed_data):
    """Create a temporary CSV file from sample data."""
    csv_path = os.path.join(temp_dir, "test_data.csv")
    sample_mixed_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_dtypes_json(temp_dir, mixed_dtypes):
    """Create a temporary JSON file with dtypes."""
    import json
    
    json_path = os.path.join(temp_dir, "dtypes.json")
    with open(json_path, "w") as f:
        json.dump(mixed_dtypes, f)
    return json_path


@pytest.fixture(autouse=True)
def reset_matplotlib():
    """Reset matplotlib state between tests."""
    import matplotlib.pyplot as plt
    
    yield
    plt.close("all")


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global state in brute_force_plotter module."""
    from src import brute_force_plotter
    
    # Store original values
    original_ignore = brute_force_plotter.ignore.copy()
    original_skip_existing = brute_force_plotter.skip_existing_plots
    original_show_plots = brute_force_plotter._show_plots
    original_save_plots = brute_force_plotter._save_plots
    
    yield
    
    # Restore original values
    brute_force_plotter.ignore = original_ignore
    brute_force_plotter.skip_existing_plots = original_skip_existing
    brute_force_plotter._show_plots = original_show_plots
    brute_force_plotter._save_plots = original_save_plots


@pytest.fixture
def titanic_data():
    """Load the titanic example dataset if available."""
    titanic_path = Path(__file__).parent.parent / "example" / "titanic.csv"
    if titanic_path.exists():
        return pd.read_csv(titanic_path)
    return None


@pytest.fixture
def titanic_dtypes():
    """Return the dtypes for the titanic dataset."""
    return {
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
