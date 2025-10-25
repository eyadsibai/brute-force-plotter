# Brute Force Plotter

[Work in progress]
Tool to visualize data quickly with no brain usage for plot creation

## Installation

**Using UV (Recommended)**

UV is a fast Python package installer and resolver. First, install UV:

```bash
$ pip install uv
```

Then install the project using:

```bash
$ git clone https://github.com/eyadsibai/brute_force_plotter.git
$ cd brute_force_plotter
$ uv sync
```

This will create a virtual environment (.venv) and install all dependencies with locked versions for reproducibility.

**Useful UV Commands:**

- `uv sync` - Install dependencies and sync the environment
- `uv add <package>` - Add a new dependency
- `uv remove <package>` - Remove a dependency
- `uv lock` - Update the lockfile
- `uv run <command>` - Run a command in the virtual environment

## Usage

**As a Python Library (NEW!)**

You can now use brute-force-plotter directly in your Python scripts:

```python
import pandas as pd
import brute_force_plotter as bfp

# Load your data
data = pd.read_csv('data.csv')

# Define data types (c=category, n=numeric, i=ignore)
dtypes = {
    'column1': 'n',  # numeric
    'column2': 'c',  # category
    'column3': 'i'   # ignore
}

# Create and save plots
bfp.plot(data, dtypes, output_path='./plots')

# Or show plots interactively
bfp.plot(data, dtypes, show=True)
```

See [example/library_usage_example.py](https://github.com/eyadsibai/brute-force-plotter/blob/master/example/library_usage_example.py) for more examples.

**As a Command-Line Tool**

## Example

It was tested on python3 only (Python 3.10+ required)

**Using UV:**

```bash
$ git clone https://github.com/eyadsibai/brute_force_plotter.git
$ cd brute_force_plotter
$ uv sync
$ uv run python -m src example/titanic.csv example/titanic_dtypes.json example/output

# Or use the brute-force-plotter command:
$ uv run brute-force-plotter example/titanic.csv example/titanic_dtypes.json example/output
```

## Command Line Options

- `--skip-existing`: Skip generating plots that already exist (default: True)
- `--theme`: Choose plot style theme (darkgrid, whitegrid, dark, white, ticks) (default: darkgrid)
- `--n-workers`: Number of parallel workers for plot generation (default: 4)
- `--export-stats`: Export statistical summary to CSV files
- `--max-rows`: Maximum number of rows before sampling is applied (default: 100,000)
- `--sample-size`: Number of rows to sample for large datasets (default: 50,000)
- `--no-sample`: Disable sampling for large datasets (may cause memory issues)

**Using UV:**

```bash
$ uv run brute-force-plotter example/titanic.csv example/titanic_dtypes.json example/output --theme whitegrid --n-workers 8 --export-stats
```


## Large Dataset Handling

For datasets exceeding 100,000 rows, brute-force-plotter automatically samples the data to improve performance and reduce memory usage. This ensures plots are generated quickly even with millions of rows.

**Default Behavior:**
- Datasets with ≤ 100,000 rows: No sampling, all data is used
- Datasets with > 100,000 rows: Automatically samples 50,000 rows for visualization
- Statistical exports (`--export-stats`) always use the full dataset for accuracy

**Customization:**

```bash
# Increase sampling threshold to 200,000 rows
$ python3 -m src data.csv dtypes.json output --max-rows 200000

# Use a larger sample size (75,000 rows)
$ python3 -m src data.csv dtypes.json output --sample-size 75000

# Disable sampling entirely (use with caution for very large datasets)
$ python3 -m src data.csv dtypes.json output --no-sample
```

**Library Usage:**

```python
import pandas as pd
import brute_force_plotter as bfp

# Load a large dataset
data = pd.read_csv('large_data.csv')  # e.g., 500,000 rows

dtypes = {'col1': 'n', 'col2': 'c'}

# Automatic sampling (default: max_rows=100000, sample_size=50000)
bfp.plot(data, dtypes, output_path='./plots')

# Custom sampling parameters
bfp.plot(data, dtypes, output_path='./plots', max_rows=200000, sample_size=75000)

# Disable sampling
bfp.plot(data, dtypes, output_path='./plots', no_sample=True)
```

**Note:** Sampling uses a fixed random seed (42) for reproducibility, ensuring consistent results across multiple runs.

## Arguments

- json.dump({k:v.name for k,v in df.dtypes.to_dict().items()},open('dtypes.json','w'))  
- the first argument is the input file (csv file with data) [example/titanic.csv](https://github.com/eyadsibai/brute-force-plotter/blob/master/example/titanic.csv)
- second argument is a json file with the data types of each columns (c for category, n for numeric, i for ignore) [example/titanic_dtypes.json](https://github.com/eyadsibai/brute-force-plotter/blob/master/example/titanic_dtypes.json)

```json
{
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
"Name": "i"
}	
```

- third argument is the output directory
- c stands for category, i stands for ignore, n for numeric

## Features

The tool automatically generates:

**Distribution Plots:**

- Histogram with KDE for numeric variables
- Violin plots for numeric variables
- Bar plots for categorical variables
- Correlation matrices (Pearson and Spearman)
- Missing values heatmap

**2D Interaction Plots:**

- Scatter plots for numeric vs numeric
- Heatmaps for categorical vs categorical
- Bar/Box/Violin/Strip plots for categorical vs numeric

**Statistical Summaries (with --export-stats):**

- Numeric statistics (mean, std, min, max, quartiles)
- Category value counts
- Missing values analysis

## Example Plots

![Age Distribution (Histogram with Kernel Density Estimation, Violin Plot)](https://raw.githubusercontent.com/eyadsibai/brute_force_plotter/master/example/output/distributions/Age-dist-plot.png)

![Heatmap for Sex and Pclass](https://github.com/eyadsibai/brute_force_plotter/blob/master/example/output/2d_interactions/Pclass-Sex-heatmap.png)

![Pclass vs Survived](https://github.com/eyadsibai/brute_force_plotter/blob/master/example/output/2d_interactions/Pclass-Survived-bar-plot.png)

![Survived vs Age](https://github.com/eyadsibai/brute_force_plotter/blob/master/example/output/2d_interactions/Survived-Age-plot.png)

![Age vs Fare](https://github.com/eyadsibai/brute_force_plotter/blob/master/example/output/2d_interactions/Age-Fare-scatter-plot.png)

## TODO

- target variable support
- ~~Tests?~~ ✅ Comprehensive test suite added!
- Support 3 variables (contour plots/ etc)
- ~~Fallback for large datasets~~ ✅ Automatic sampling for datasets >100k rows!
- Figure out the data type or suggest some
- Map visualization (if geocoordinates)
- Minimize the number of plots
- Support for Time Series

## Testing

The project includes a comprehensive test suite with 81+ tests covering unit tests, integration tests, and edge cases.

**Running Tests**

```bash
# Run all tests
$ pytest

# Run with coverage report
$ pytest --cov=src --cov-report=html

# Run specific test categories
$ pytest -m unit          # Unit tests only
$ pytest -m integration   # Integration tests only
$ pytest -m edge_case     # Edge case tests only

# Run tests in parallel (faster)
$ pytest -n auto

# Run with verbose output
$ pytest -v
```

**Test Coverage**

The test suite achieves ~96% code coverage and includes:

- **Unit tests**: Core plotting functions, utilities, statistical exports, large dataset handling
- **Integration tests**: CLI interface, library interface, end-to-end workflows
- **Edge case tests**: Empty data, missing values, many categories, Unicode support

**Writing Tests**

When contributing, please:
1. Add tests for new features in the appropriate test file
2. Ensure tests pass locally before submitting PR
3. Aim for >90% code coverage for new code
4. Use the fixtures in `conftest.py` for test data

## Development

### Setting Up for Development

When developing for this project, it's important to set up code quality tools to ensure consistency:

**1. Install Development Dependencies**

Using UV:
```bash
$ uv sync  # Installs all dependencies including dev tools
```

**2. Install Pre-commit Hooks (REQUIRED)**

This project uses [pre-commit](https://pre-commit.com/) hooks to automatically enforce code quality standards on every commit:

```bash
$ pre-commit install
```

After installation, the hooks will run automatically on `git commit` and check:
- ✅ Ruff linting (with auto-fix)
- ✅ Ruff formatting
- ✅ Trailing whitespace removal
- ✅ End-of-file fixes
- ✅ YAML/JSON/TOML validation
- ✅ Large file detection

**3. Manual Code Quality Checks**

You can also run these checks manually:

```bash
# Lint code (check for issues)
$ ruff check .

# Lint and auto-fix issues
$ ruff check --fix .

# Format code
$ ruff format .

# Run all pre-commit hooks on all files
$ pre-commit run --all-files
```

**4. Running Tests**

Always run tests before submitting changes:

```bash
$ pytest
```

### Why Pre-commit Hooks?

Pre-commit hooks ensure that:
- All code follows consistent style guidelines
- Linting issues are caught before they reach CI
- Code quality is maintained automatically
- Review cycles are faster (no style nitpicks)

**Note:** If you try to commit code that doesn't pass the checks, the commit will be blocked. Fix the issues reported and commit again.

## Recent Updates (2025)

✅ Updated all dependencies to latest stable versions
✅ Added correlation matrix plots (Pearson and Spearman)
✅ Added missing values visualization
✅ Added statistical summary export
✅ Added configurable plot themes
✅ Added parallel processing controls
✅ Added skip-existing-plots option
✅ Improved logging and progress indicators
✅ Code cleanup and better error handling
✅ **Comprehensive test suite with 96% coverage (81+ tests)**
✅ **Large dataset fallback with automatic sampling**

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on:

- Setting up your development environment
- Using code quality tools (Ruff, pre-commit)
- Submitting pull requests
- Coding standards and best practices

## Contributors

### Code Contributors

- Eyad Sibai / [@eyadsibai](https://github.com/eyadsibai)

### Special Thanks

The following haven't provided code directly, but have provided guidance and advice:

- Andreas Meisingseth / [@AndreasMeisingseth](https://github.com/AndreasMeisingseth)
- Tom Baylis / [@tbaylis](https://github.com/tbaylis)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
