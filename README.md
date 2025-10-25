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

**Using pip (Traditional)**

You can also use pip to install dependencies:

```bash
$ git clone https://github.com/eyadsibai/brute_force_plotter.git
$ cd brute_force_plotter
$ pip3 install -r requirements.txt
```

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

See [example/library_usage_example.py](https://github.com/eyadsibai/brute_force_plotter/example/library_usage_example.py) for more examples.

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

**Using pip:**

```bash
$ git clone https://github.com/eyadsibai/brute_force_plotter.git
$ cd brute_force_plotter
$ pip3 install -r requirements.txt
$ python3 -m src example/titanic.csv example/titanic_dtypes.json example/output
```

## Command Line Options

- `--skip-existing`: Skip generating plots that already exist (default: True)
- `--theme`: Choose plot style theme (darkgrid, whitegrid, dark, white, ticks) (default: darkgrid)
- `--n-workers`: Number of parallel workers for plot generation (default: 4)
- `--export-stats`: Export statistical summary to CSV files

**Using UV:**

```bash
$ uv run brute-force-plotter example/titanic.csv example/titanic_dtypes.json example/output --theme whitegrid --n-workers 8 --export-stats
```

**Using pip:**

```bash
$ python3 -m src example/titanic.csv example/titanic_dtypes.json example/output --theme whitegrid --n-workers 8 --export-stats
```

## Arguments

- json.dump({k:v.name for k,v in df.dtypes.to_dict().items()},open('dtypes.json','w'))  
- the first argument is the input file (csv file with data) [example/titanic.csv](https://github.com/eyadsibai/brute_force_plotter/example/titanic.csv)
- second argument is a json file with the data types of each columns (c for category, n for numeric, i for ignore) [example/titanic_dtypes.json](https://github.com/eyadsibai/brute_force_plotter/example/titanic_dtypes.json)

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
- Tests?
- Support 3 variables (contour plots/ etc)
- Fallback for large datasets
- Figure out the data type or suggest some
- Map visualization (if geocoordinates)
- Minimize the number of plots
- Support for Time Series

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

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on:

- Setting up your development environment
- Using code quality tools (Ruff, pre-commit)
- Submitting pull requests
- Coding standards and best practices

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
