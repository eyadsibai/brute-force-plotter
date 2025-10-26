# Brute Force Plotter - Copilot Instructions

## Project Overview

Brute Force Plotter is a Python tool designed to visualize data quickly with minimal configuration. It automatically generates various plots based on data types (categorical, numeric) from CSV files.

## Language and Dependencies

- **Python Version**: Python 3.10+ (tested on Python 3.10, 3.11, and 3.12)
- **Key Dependencies**:
  - matplotlib (3.10.7) - for plotting
  - pandas (2.3.3) - for data manipulation
  - seaborn (0.13.2) - for statistical visualizations
  - dask (2025.10.0) - for parallel processing
  - click (8.3.0) - for CLI interface
  - pyarrow (22.0.0) - for parquet file handling

## Project Structure

- `src/` - Main source code directory
  - `__main__.py` - Entry point for the module
  - `brute_force_plotter.py` - Core plotting logic and CLI
- `example/` - Example data and output
  - `titanic.csv` - Sample CSV data
  - `titanic_dtypes.json` - Data type definitions
  - `output/` - Generated plots directory
- `pyproject.toml` - Python project configuration and dependencies
- `uv.lock` - Locked dependency versions for reproducibility
- `README.md` - Documentation in Markdown format

## Running the Tool

The tool is executed as a Python module:

```bash
python3 -m src <input_file.csv> <dtypes.json> <output_directory>
```

### Arguments:
1. **input_file**: CSV file containing the data
2. **dtypes**: JSON file mapping column names to data types:
   - `"c"` - categorical variable
   - `"n"` - numeric variable
   - `"i"` - ignore this column
3. **output_path**: Directory where plots will be saved

### Example:
```bash
uv run python3 -m src example/titanic.csv example/titanic_dtypes.json example/output
```

## Installation

```bash
uv sync
```

## Code Style and Conventions

- Follow PEP 8 Python style guidelines
- Use 4 spaces for indentation
- Include docstrings for modules and functions
- Use snake_case for function and variable names
- Use descriptive variable names
- Keep functions focused on a single responsibility

## Plot Generation Logic

The tool generates three types of visualizations:

1. **Distribution Plots** (1D):
   - Numeric columns: histogram with KDE + violin plot
   - Categorical columns: bar plots (skipped if >50 unique values)

2. **2D Interaction Plots**:
   - Numeric vs Numeric: scatter plots
   - Categorical vs Categorical: bar plots and heatmaps
   - Categorical vs Numeric: bar, box, violin, and strip plots

3. **3D Interaction Plots**: (currently commented out in code)

## Technical Details

- Uses `matplotlib.use("agg")` for non-interactive backend
- Employs Dask for parallel plot generation with LocalCluster (10 workers)
- Converts CSV to Parquet format for efficient parallel reading
- Implements `@dask.delayed` decorator for lazy evaluation
- Uses `@ignore_if_exist_or_save` decorator to skip existing plots
- Seaborn theme: darkgrid style, paper context
- Default figure size: 8x6 inches, 120 DPI

## Output Structure

Generated plots are organized in subdirectories:
- `distributions/` - Single variable plots
- `2d_interactions/` - Two variable relationship plots
- `3d_interactions/` - (reserved for future use)

## Testing

The project includes a comprehensive test suite with 81+ tests covering:
- Unit tests for core plotting functions
- Integration tests for CLI and library interfaces
- Edge case tests (empty data, missing values, unicode, etc.)
- Large dataset handling tests

**Running Tests:**
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m edge_case     # Edge case tests only
```

**Test Coverage:** ~96% code coverage across the codebase.

## Known Limitations and TODOs

- No support for target variable highlighting
- Categorical columns with >50 unique values are ignored
- 3D visualizations are not implemented
- No automatic data type inference
- No time series support
- No geographic visualization support

## Making Changes

When modifying this codebase:
- Maintain backward compatibility with existing dtypes JSON format
- Ensure parallel processing with Dask continues to work
- Keep plot generation functions as `@dask.delayed` for performance
- Test with the example titanic dataset
- Update README.md for user-facing changes
- Consider matplotlib/seaborn version compatibility

## Code Quality and Linting

**IMPORTANT: Always run ruff before creating a PR to ensure tests pass in CI.**

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
# Check for linting issues
ruff check .

# Auto-fix linting issues
ruff check --fix .

# Format code
ruff format .
```

**Pre-commit Hooks:**
The project uses pre-commit hooks to automatically enforce code quality on every commit:

```bash
# Install pre-commit hooks
pre-commit install

# Run all hooks manually
pre-commit run --all-files
```

**Before Making a PR:**
1. Run `ruff check --fix .` to fix any linting issues
2. Run `ruff format .` to format code
3. Run `pytest` to ensure all tests pass
4. Review your changes carefully

The CI pipeline runs these checks automatically:
- Ruff linting (`ruff check .`)
- Ruff formatting (`ruff format --check .`)
- Pytest tests

**Note:** PRs that fail linting or formatting checks will not be merged. Always run ruff locally before pushing.
