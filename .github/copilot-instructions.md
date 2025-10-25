# Brute Force Plotter - Copilot Instructions

## Project Overview

Brute Force Plotter is a Python tool designed to visualize data quickly with minimal configuration. It automatically generates various plots based on data types (categorical, numeric) from CSV files.

## Language and Dependencies

- **Python Version**: Python 3 (tested on Python 3 only)
- **Key Dependencies**:
  - matplotlib (3.10.0) - for plotting
  - pandas (2.3.3) - for data manipulation
  - seaborn (0.13.2) - for statistical visualizations
  - dask (2024.4.1) - for parallel processing
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
- `requirements.txt` - Python dependencies
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
python3 -m src example/titanic.csv example/titanic_dtypes.json example/output
```

## Installation

```bash
pip3 install -r requirements.txt
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

Currently, the project does not have a formal test suite (see TODO in README).

## Known Limitations and TODOs

- No support for target variable highlighting
- No tests
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
